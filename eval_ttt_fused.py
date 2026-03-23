"""
eval_ttt_fused.py — Fused Sliding-Window TTT for No-Injection Depth Recurrence.

Combines sliding-window evaluation with test-time training in a single left-to-right
pass. Each chunk is scored with full sliding-window context BEFORE being used for
gradient updates. Supports adaptive depth selection: starts at base depth and
switches to deeper when deeper becomes better.

Architecture: 1 entry + 1 recurrent (x0 residual, no injection) + 1 exit.

SAFETY: score-before-train on every chunk. No token is ever scored after training.

Usage:
    torchrun --nproc_per_node=4 eval_ttt_fused.py

Key env vars:
    CHECKPOINT_PATH       ttt_checkpoint_noinject.pt
    TTT_LR                Learning rate (default: 1e-4)
    TTT_BASE_DEPTH        Starting depth (default: from checkpoint)
    TTT_MAX_DEPTH         Max depth to try (default: 20)
    TTT_STRIDE            Tokens scored per window (default: 256)
    TTT_BATCH_SEQS        Windows per gradient step (default: 8)
    TTT_DEPTH_PROBE_EVERY Batches between depth probes (default: 50)
    TTT_LOG_EVERY         Log every N steps (default: 50)
    TTT_WARMUP_STEPS      Steps at base depth before probing (default: 100)
"""

from __future__ import annotations

import math
import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# Import from the no-injection training script
from train_gpt_noinject import (
    GPT,
    CastedLinear,
    Hyperparameters,
    build_sentencepiece_luts,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
    eval_val,
    CONTROL_TENSOR_NAME_PATTERNS,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

class TTTConfig:
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "ttt_checkpoint_noinject.pt")
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

    ttt_lr = float(os.environ.get("TTT_LR", 1e-4))
    ttt_wd = float(os.environ.get("TTT_WD", 0.0))
    ttt_base_depth = int(os.environ.get("TTT_BASE_DEPTH", 0))  # 0 = from checkpoint
    ttt_max_depth = int(os.environ.get("TTT_MAX_DEPTH", 20))
    ttt_stride = int(os.environ.get("TTT_STRIDE", 256))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 8))
    ttt_log_every = int(os.environ.get("TTT_LOG_EVERY", 50))
    ttt_warmup_steps = int(os.environ.get("TTT_WARMUP_STEPS", 100))
    ttt_depth_probe_every = int(os.environ.get("TTT_DEPTH_PROBE_EVERY", 50))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

    # What to freeze (for no-injection arch: freeze entry+exit, train recurrent only)
    freeze_entry = bool(int(os.environ.get("TTT_FREEZE_ENTRY", "1")))
    freeze_exit = bool(int(os.environ.get("TTT_FREEZE_EXIT", "1")))
    freeze_embeddings = bool(int(os.environ.get("TTT_FREEZE_EMBEDDINGS", "1")))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TTTConfig()

    # --- Distributed + CUDA ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/ttt_fused_{int(time.time())}.txt" if master_process else None

    def log0(msg: str, console: bool = True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(f"eval_ttt_fused.py rank={rank} world_size={world_size}")

    # --- Load checkpoint ---
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    base_depth = cfg.ttt_base_depth if cfg.ttt_base_depth > 0 else model_cfg["eval_recurrent_depth"]
    max_depth = cfg.ttt_max_depth
    seq_len = cfg.eval_seq_len
    stride = cfg.ttt_stride
    vocab_size = model_cfg["vocab_size"]

    log0(f"Model: dim={model_cfg['model_dim']} arch={model_cfg.get('architecture', 'unknown')}")
    log0(f"Depth: base={base_depth} max={max_depth} stride={stride}")

    # --- Build model ---
    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        model_dim=model_cfg["model_dim"],
        num_heads=model_cfg["num_heads"],
        num_kv_heads=model_cfg["num_kv_heads"],
        mlp_mult=model_cfg["mlp_mult"],
        tie_embeddings=model_cfg["tie_embeddings"],
        tied_embed_init_std=model_cfg["tied_embed_init_std"],
        logit_softcap=model_cfg["logit_softcap"],
        rope_base=model_cfg["rope_base"],
        qk_gain_init=model_cfg["qk_gain_init"],
        bigram_vocab_size=model_cfg.get("bigram_vocab_size", 0),
        bigram_dim=model_cfg.get("bigram_dim", 128),
        rope_dims=model_cfg.get("rope_dims", 0),
        ln_scale=model_cfg.get("ln_scale", False),
        mean_depth=model_cfg.get("mean_depth", 8),
    ).to(device).bfloat16()

    for mod in model.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)

    n_params = sum(p.numel() for p in model.parameters())
    log0(f"Model loaded: {n_params} params")

    # --- Freeze ---
    trainable_names = []
    for name, param in model.named_parameters():
        freeze = False
        if cfg.freeze_entry and name.startswith("entry_block."):
            freeze = True
        elif cfg.freeze_exit and name.startswith("exit_block."):
            freeze = True
        elif cfg.freeze_embeddings and (
            name.startswith("tok_emb.") or name.startswith("bigram.") or name.startswith("smear.")
        ):
            freeze = True
        elif name.startswith("final_norm."):
            freeze = True
        if freeze:
            param.requires_grad_(False)
        else:
            trainable_names.append(name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log0(f"Trainable: {n_trainable} ({100*n_trainable/n_params:.1f}%) — {trainable_names[:5]}...")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.ttt_lr, weight_decay=cfg.ttt_wd, betas=(0.9, 0.999), fused=True,
    )

    # --- Load val data ---
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    val_tokens = load_validation_tokens(cfg.val_files, seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, vocab_size, device
    )
    total_val_tokens = val_tokens.numel() - 1
    log0(f"Val tokens: {total_val_tokens}")

    # --- Build sliding windows and partition across ranks ---
    window_starts = [ws for ws in range(0, total_val_tokens, stride)
                     if min(ws + seq_len, total_val_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    log0(f"Rank {rank}: windows [{my_s}, {my_e}) = {len(my_windows)} windows")

    # --- Accumulators ---
    nll_accum = torch.zeros((), device=device, dtype=torch.float64)
    bytes_accum = torch.zeros((), device=device, dtype=torch.float64)
    tokens_accum = torch.zeros((), device=device, dtype=torch.float64)

    # Per-depth tracking
    depth_nll: dict[int, float] = defaultdict(float)
    depth_bytes: dict[int, float] = defaultdict(float)
    depth_tokens: dict[int, int] = defaultdict(int)

    # Adaptive depth state
    current_depth = base_depth
    # Recent batch losses for depth probing
    recent_losses: list[float] = []

    step = 0
    t_start = time.perf_counter()
    model.train()

    for bi in range(0, len(my_windows), cfg.ttt_batch_seqs):
        batch_ws = my_windows[bi:bi + cfg.ttt_batch_seqs]
        bsz = len(batch_ws)
        t_step = time.perf_counter()

        # --- Build batch ---
        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens: list[int] = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_val_tokens)
            wlen = end - ws
            wlens.append(wlen)
            chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]

        # --- Adaptive depth probing ---
        if (step > cfg.ttt_warmup_steps
                and step % cfg.ttt_depth_probe_every == 0
                and current_depth < max_depth):
            # Try current depth +2 on this batch (no grad)
            probe_depth = min(current_depth + 2, max_depth)
            with torch.no_grad():
                model.n_recurrent_iters = probe_depth
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    probe_logits = model.forward_logits(x_batch)
                probe_nll = F.cross_entropy(
                    probe_logits.reshape(-1, vocab_size).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                # Score only the new tokens (same as main scoring below)
                probe_loss = 0.0
                probe_count = 0
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    probe_loss += probe_nll[i, s:wlen].sum().item()
                    probe_count += wlen - s
                probe_avg = probe_loss / max(probe_count, 1)

                # Compare with current depth
                model.n_recurrent_iters = current_depth
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    curr_logits = model.forward_logits(x_batch)
                curr_nll = F.cross_entropy(
                    curr_logits.reshape(-1, vocab_size).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                curr_loss = 0.0
                curr_count = 0
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    curr_loss += curr_nll[i, s:wlen].sum().item()
                    curr_count += wlen - s
                curr_avg = curr_loss / max(curr_count, 1)

            if probe_avg < curr_avg:
                log0(f"DEPTH UPGRADE: {current_depth} → {probe_depth} "
                     f"(probe_loss={probe_avg:.4f} < curr_loss={curr_avg:.4f})")
                current_depth = probe_depth
            else:
                log0(f"depth_probe: staying at {current_depth} "
                     f"(probe={probe_avg:.4f} >= curr={curr_avg:.4f})")

        # --- Set depth ---
        model.n_recurrent_iters = current_depth

        # --- Forward pass (with gradients for training) ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x_batch)

        nll_flat = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            y_batch.reshape(-1), reduction="none",
        ).reshape(bsz, seq_len)

        # --- Score only new tokens (sliding window scoring) ---
        batch_nll_sum = 0.0
        batch_bytes_sum = 0.0
        batch_token_count = 0
        train_nll_list = []

        with torch.no_grad():
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll_flat[i, s:wlen]
                batch_nll_sum += scored_nll.to(torch.float64).sum().item()
                batch_token_count += wlen - s

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                batch_bytes_sum += tb.sum().item()

            nll_accum += batch_nll_sum
            bytes_accum += batch_bytes_sum
            tokens_accum += batch_token_count
            depth_nll[current_depth] += batch_nll_sum
            depth_bytes[current_depth] += batch_bytes_sum
            depth_tokens[current_depth] += batch_token_count

        # --- Train on the full window (all tokens, not just scored ones) ---
        # Use mean NLL over all valid tokens for training signal
        train_mask = torch.zeros(bsz, seq_len, device=device)
        for i in range(bsz):
            train_mask[i, :wlens[i]] = 1.0
        masked_nll = nll_flat * train_mask
        loss = masked_nll.sum() / train_mask.sum()

        loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.perf_counter() - t_step

        # --- Log ---
        if step % cfg.ttt_log_every == 0:
            running_bpb = (nll_accum.item() / math.log(2.0)) / max(bytes_accum.item(), 1)
            batch_bpb = (batch_nll_sum / math.log(2.0)) / max(batch_bytes_sum, 1)
            log0(f"ttt step:{step} depth:{current_depth} "
                 f"batch_bpb:{batch_bpb:.4f} running_bpb:{running_bpb:.4f} "
                 f"tokens:{int(tokens_accum.item())} step_ms:{1000*step_time:.0f}")

        step += 1

    total_time = time.perf_counter() - t_start

    # --- Distributed reduction ---
    if distributed:
        dist.all_reduce(nll_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_accum, op=dist.ReduceOp.SUM)

    final_bpb = (nll_accum.item() / math.log(2.0)) / bytes_accum.item() if bytes_accum.item() > 0 else float("inf")

    log0(f"\n{'='*80}")
    log0(f"FUSED TTT COMPLETE: bpb={final_bpb:.6f} tokens={int(tokens_accum.item())} "
         f"time={total_time:.1f}s steps={step}")
    log0(f"\nPER-DEPTH BREAKDOWN:")
    for d in sorted(depth_tokens.keys()):
        d_bpb = (depth_nll[d] / math.log(2.0)) / depth_bytes[d] if depth_bytes[d] > 0 else float("inf")
        log0(f"  depth={d}: bpb={d_bpb:.6f} tokens={depth_tokens[d]}")
    log0(f"{'='*80}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
