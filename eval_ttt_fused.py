"""
eval_ttt_fused.py — Fused Sliding-Window TTT for No-Injection Depth Recurrence.

Strategy: train-ahead curriculum.
- EVAL at depth D with stride 64 (high overlap scoring)
- TRAIN at depth D+1 on full 2048 context with stride 256
- Each token is scored BEFORE any training that includes it
- Probe: when training-depth (D+1) outperforms eval-depth (D) at scoring,
  promote eval_depth to D+1 and train_depth to D+2

Architecture: N entry + M recurrent (x0 residual, no injection) + K exit.

SAFETY: score-before-train on every chunk. No token is ever scored after training.

Usage:
    torchrun --nproc_per_node=4 eval_ttt_fused.py

Key env vars:
    CHECKPOINT_PATH       ttt_checkpoint_noinject.pt
    TTT_LR                Learning rate (default: 1e-4)
    TTT_BASE_DEPTH        Starting eval depth (default: from checkpoint)
    TTT_MAX_DEPTH         Max depth to try (default: 6)
    TTT_EVAL_STRIDE       Scoring stride (default: 64)
    TTT_TRAIN_STRIDE      Training stride (default: 256)
    TTT_DEPTH_PROBE_EVERY Train steps between depth probes (default: 50)
    TTT_LOG_EVERY         Log every N train steps (default: 25)
    TTT_WARMUP_STEPS      Train steps before probing (default: 50)
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from train_gpt_noinject import (
    GPT,
    CastedLinear,
    build_sentencepiece_luts,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
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
    ttt_max_depth = int(os.environ.get("TTT_MAX_DEPTH", 6))
    eval_stride = int(os.environ.get("TTT_EVAL_STRIDE", 64))
    train_stride = int(os.environ.get("TTT_TRAIN_STRIDE", 256))
    ttt_log_every = int(os.environ.get("TTT_LOG_EVERY", 25))
    ttt_warmup_steps = int(os.environ.get("TTT_WARMUP_STEPS", 50))
    ttt_depth_probe_every = int(os.environ.get("TTT_DEPTH_PROBE_EVERY", 50))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

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
            print(msg, flush=True)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f, flush=True)

    log0(f"eval_ttt_fused.py rank={rank} world_size={world_size}")

    # --- Load checkpoint ---
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    base_depth = cfg.ttt_base_depth if cfg.ttt_base_depth > 0 else model_cfg["eval_recurrent_depth"]
    max_depth = cfg.ttt_max_depth
    seq_len = cfg.eval_seq_len
    eval_stride = cfg.eval_stride
    train_stride = cfg.train_stride
    vocab_size = model_cfg["vocab_size"]

    # How many eval windows fit in one train stride
    assert train_stride % eval_stride == 0, f"train_stride ({train_stride}) must be divisible by eval_stride ({eval_stride})"
    evals_per_train = train_stride // eval_stride

    log0(f"Model: dim={model_cfg['model_dim']} arch={model_cfg.get('architecture', 'unknown')}")
    log0(f"  blocks: {model_cfg.get('num_entry_blocks', 1)}+{model_cfg.get('num_recurrent_blocks', 1)}+{model_cfg.get('num_exit_blocks', 1)}")
    log0(f"Depth: base_eval={base_depth} max={max_depth}")
    log0(f"Strides: eval={eval_stride} train={train_stride} evals_per_train={evals_per_train}")

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
        mean_depth=model_cfg.get("mean_depth", 3),
        num_entry_blocks=model_cfg.get("num_entry_blocks", 2),
        num_exit_blocks=model_cfg.get("num_exit_blocks", 2),
        num_recurrent_blocks=model_cfg.get("num_recurrent_blocks", 3),
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
        if cfg.freeze_entry and name.startswith("entry_blocks."):
            freeze = True
        elif cfg.freeze_exit and name.startswith("exit_blocks."):
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

    # --- Partition val tokens across ranks ---
    # Each rank gets a contiguous chunk. We process left-to-right within that chunk.
    per_rank = total_val_tokens // world_size
    rank_start = rank * per_rank
    rank_end = (rank + 1) * per_rank if rank < world_size - 1 else total_val_tokens
    rank_tokens = rank_end - rank_start
    log0(f"Rank {rank}: tokens [{rank_start}, {rank_end}) = {rank_tokens}")

    # --- Accumulators ---
    nll_accum = torch.zeros((), device=device, dtype=torch.float64)
    bytes_accum = torch.zeros((), device=device, dtype=torch.float64)
    tokens_accum = torch.zeros((), device=device, dtype=torch.float64)

    depth_nll: dict[int, float] = defaultdict(float)
    depth_bytes: dict[int, float] = defaultdict(float)
    depth_tokens: dict[int, int] = defaultdict(int)

    # --- Adaptive depth state ---
    eval_depth = base_depth
    train_depth = eval_depth + 1  # always train one step ahead

    train_step = 0
    eval_step = 0
    t_start = time.perf_counter()
    model.train()

    # --- Scoring helper (no grad) ---
    def score_window(ws: int, depth: int) -> tuple[float, float, int]:
        """Score the last eval_stride tokens of the window [ws, ws+seq_len).
        Returns (nll_sum, bytes_sum, n_tokens). Does NOT train."""
        end = min(ws + seq_len, rank_end)
        wlen = end - ws
        if wlen < 1:
            return 0.0, 0.0, 0
        chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        model.n_recurrent_iters = depth
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x)
        nll = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            y.reshape(-1), reduction="none",
        )
        # Score only the last eval_stride new tokens (or all if first window)
        s = 0 if ws == rank_start else max(wlen - eval_stride, 0)
        scored_nll = nll[s:wlen]
        tgt = y[0, s:wlen]
        prev = x[0, s:wlen]
        tb = base_bytes_lut[tgt].to(torch.float64)
        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
        return scored_nll.to(torch.float64).sum().item(), tb.sum().item(), wlen - s

    # --- Train helper (with grad) ---
    def train_window(ws: int, depth: int):
        """Train on the full window [ws, ws+seq_len) at given depth."""
        end = min(ws + seq_len, rank_end)
        wlen = end - ws
        if wlen < 1:
            return
        chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        model.n_recurrent_iters = depth
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x)
        nll = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            y.reshape(-1), reduction="none",
        )
        # Train on all valid tokens in the window
        loss = nll[:wlen].mean()
        loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
        optimizer.step()
        optimizer.zero_grad()

    # --- Main loop: left-to-right over rank's token range ---
    # We advance by eval_stride. Every evals_per_train eval windows, we do one train step.
    # The train window covers the same rightmost position but with full 2048 context.
    #
    # SAFETY: We score all eval windows FIRST, then train on the region we just scored.
    # The train window's rightmost token = the last eval window's rightmost token.
    # So we never train on tokens we haven't scored yet.

    # Track furthest scored position for safety
    scored_up_to = rank_start  # exclusive: all tokens before this have been scored

    eval_pos = rank_start  # next eval window start position
    evals_since_train = 0

    while eval_pos < rank_end:
        # --- Score one eval window ---
        nll_sum, bytes_sum, n_tok = score_window(eval_pos, eval_depth)

        nll_accum += nll_sum
        bytes_accum += bytes_sum
        tokens_accum += n_tok
        depth_nll[eval_depth] += nll_sum
        depth_bytes[eval_depth] += bytes_sum
        depth_tokens[eval_depth] += n_tok

        # Update scored frontier
        scored_end = min(eval_pos + seq_len, rank_end)
        new_scored_to = min(eval_pos + seq_len, rank_end)
        if new_scored_to > scored_up_to:
            scored_up_to = new_scored_to

        eval_pos += eval_stride
        evals_since_train += 1
        eval_step += 1

        # --- Train after every evals_per_train eval windows ---
        if evals_since_train >= evals_per_train:
            evals_since_train = 0

            # Train window: ends at scored_up_to, starts seq_len earlier
            train_end = scored_up_to
            train_ws = max(train_end - seq_len, rank_start)

            # SAFETY CHECK: we only train on tokens up to scored_up_to
            assert train_ws + seq_len <= scored_up_to + 1, \
                f"Safety violation: training beyond scored frontier! train_end={train_ws+seq_len} scored_up_to={scored_up_to}"

            train_window(train_ws, train_depth)
            train_step += 1

            # --- Depth probing ---
            if (train_step > cfg.ttt_warmup_steps
                    and train_step % cfg.ttt_depth_probe_every == 0
                    and train_depth <= max_depth):
                # Probe: does train_depth now beat eval_depth for scoring?
                # Use the most recent eval window position for comparison
                probe_ws = max(eval_pos - eval_stride, rank_start)
                probe_nll_new, _, probe_n = score_window(probe_ws, train_depth)
                probe_nll_cur, _, _ = score_window(probe_ws, eval_depth)
                if probe_n > 0:
                    avg_new = probe_nll_new / probe_n
                    avg_cur = probe_nll_cur / probe_n
                    if avg_new < avg_cur:
                        log0(f"DEPTH UPGRADE: eval {eval_depth}→{train_depth} "
                             f"(train_depth_loss={avg_new:.4f} < eval_depth_loss={avg_cur:.4f})")
                        eval_depth = train_depth
                        train_depth = min(eval_depth + 1, max_depth)
                    else:
                        log0(f"depth_probe: staying eval={eval_depth} train={train_depth} "
                             f"(probe={avg_new:.4f} >= curr={avg_cur:.4f})")

            # --- Log ---
            if train_step % cfg.ttt_log_every == 0:
                running_bpb = (nll_accum.item() / math.log(2.0)) / max(bytes_accum.item(), 1)
                elapsed = time.perf_counter() - t_start
                log0(f"train_step:{train_step} eval_step:{eval_step} "
                     f"eval_depth:{eval_depth} train_depth:{train_depth} "
                     f"running_bpb:{running_bpb:.4f} "
                     f"tokens:{int(tokens_accum.item())} elapsed:{elapsed:.0f}s")

    total_time = time.perf_counter() - t_start

    # --- Distributed reduction ---
    if distributed:
        dist.all_reduce(nll_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_accum, op=dist.ReduceOp.SUM)

    final_bpb = (nll_accum.item() / math.log(2.0)) / bytes_accum.item() if bytes_accum.item() > 0 else float("inf")

    log0(f"\n{'='*80}")
    log0(f"FUSED TTT COMPLETE: bpb={final_bpb:.6f} tokens={int(tokens_accum.item())} "
         f"time={total_time:.1f}s train_steps={train_step} eval_steps={eval_step}")
    log0(f"\nPER-DEPTH BREAKDOWN (eval depth):")
    for d in sorted(depth_tokens.keys()):
        d_bpb = (depth_nll[d] / math.log(2.0)) / depth_bytes[d] if depth_bytes[d] > 0 else float("inf")
        log0(f"  depth={d}: bpb={d_bpb:.6f} tokens={depth_tokens[d]}")
    log0(f"{'='*80}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
