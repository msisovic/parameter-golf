"""
eval_ttt.py — Test-Time Training with Decayed Input Injection for Depth Extension.

Path B strategy: Start from a model trained WITH input injection. At eval time,
decay the injection strength at iterations beyond the training mean to break the
fixed point, then use gradient updates to teach the recurrent block to do
productive work at higher depths.

SAFETY INVARIANT: Every token's score uses model weights that were NOT trained
on that token or any token after it. Enforced by score-then-train chunk ordering:
  1. Forward pass on chunk → record per-token NLL (these are the FINAL scores)
  2. Backward + optimizer step on the SAME chunk
  3. Move to next chunk

Usage:
    torchrun --nproc_per_node=4 eval_ttt.py

Key environment variables:
    CHECKPOINT_PATH       Path to ttt_checkpoint.pt (default: ttt_checkpoint.pt)
    TTT_LR                Learning rate (default: 1e-4)
    TTT_BASE_DEPTH        Starting depth = training mean (default: from checkpoint)
    TTT_TARGET_DEPTH      Depth to ramp to (default: 10)
    TTT_WARMUP_SEQS       Sequences at base depth before ramping (default: 200)
    TTT_RAMP_SEQS         Sequences over which to ramp depth (default: 1000)
    TTT_DECAY_K           Iterations beyond base where injection reaches 0 (default: 5)
    TTT_BATCH_SEQS        Sequences per gradient step (default: 4)
    TTT_FREEZE_EXIT       Freeze exit blocks (default: 1)
    TTT_FREEZE_ENTRY      Freeze entry blocks (default: 1)
    TTT_LOG_EVERY         Log every N steps (default: 50)
    DEPTH_ENSEMBLE        Comma-separated depths for final ensemble eval (default: "")
    FINAL_SLIDING_EVAL    Run sliding window eval after TTT (default: 1)
"""

from __future__ import annotations

import io
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

from train_gpt import (
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
# TTT CONFIGURATION
# ---------------------------------------------------------------------------

class TTTConfig:
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "ttt_checkpoint.pt")
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

    # TTT hyperparameters
    ttt_lr = float(os.environ.get("TTT_LR", 1e-4))
    ttt_wd = float(os.environ.get("TTT_WD", 0.0))
    ttt_base_depth = int(os.environ.get("TTT_BASE_DEPTH", 0))  # 0 = from checkpoint
    ttt_target_depth = int(os.environ.get("TTT_TARGET_DEPTH", 10))
    ttt_warmup_seqs = int(os.environ.get("TTT_WARMUP_SEQS", 200))
    ttt_ramp_seqs = int(os.environ.get("TTT_RAMP_SEQS", 1000))
    ttt_decay_k = int(os.environ.get("TTT_DECAY_K", 5))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 4))
    ttt_log_every = int(os.environ.get("TTT_LOG_EVERY", 50))

    # What to freeze
    freeze_entry = bool(int(os.environ.get("TTT_FREEZE_ENTRY", "1")))
    freeze_exit = bool(int(os.environ.get("TTT_FREEZE_EXIT", "1")))
    freeze_embeddings = bool(int(os.environ.get("TTT_FREEZE_EMBEDDINGS", "1")))

    # Post-TTT evaluation
    depth_ensemble = os.environ.get("DEPTH_ENSEMBLE", "")
    final_sliding_eval = bool(int(os.environ.get("FINAL_SLIDING_EVAL", "1")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Grad clipping
    grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP", 1.0))


# ---------------------------------------------------------------------------
# INJECTION DECAY SCHEDULE
# ---------------------------------------------------------------------------

def compute_inject_scales(depth: int, base_depth: int, decay_k: int) -> list[float]:
    """Compute per-iteration injection scale.

    Iterations 0..base_depth-1: alpha = 1.0 (full injection, as during training)
    Iterations base_depth..base_depth+decay_k-1: linear decay from 1.0 to 0.0
    Iterations beyond: alpha = 0.0 (no injection, pure recurrence)
    """
    scales = []
    for i in range(depth):
        if i < base_depth:
            scales.append(1.0)
        elif decay_k <= 0:
            scales.append(0.0)
        else:
            progress = (i - base_depth) / decay_k
            scales.append(max(0.0, 1.0 - progress))
    return scales


def compute_depth_for_seq(seq_idx: int, warmup_seqs: int, ramp_seqs: int,
                          base_depth: int, target_depth: int) -> int:
    """Depth curriculum: warmup at base, then linear ramp to target."""
    if seq_idx < warmup_seqs:
        return base_depth
    if ramp_seqs <= 0:
        return target_depth
    progress = min(1.0, (seq_idx - warmup_seqs) / ramp_seqs)
    return base_depth + int(round(progress * (target_depth - base_depth)))


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

class TTTLogger:
    """Tracks per-token NLL and bytes for BPB computation, with phase breakdowns."""

    def __init__(self, log_fn):
        self.log_fn = log_fn
        self.total_nll = 0.0
        self.total_bytes = 0.0
        self.total_tokens = 0
        # Per-depth tracking
        self.depth_nll: dict[int, float] = defaultdict(float)
        self.depth_bytes: dict[int, float] = defaultdict(float)
        self.depth_tokens: dict[int, int] = defaultdict(int)
        # Phase tracking (warmup / ramp / steady)
        self.phase_nll: dict[str, float] = defaultdict(float)
        self.phase_bytes: dict[str, float] = defaultdict(float)
        self.phase_tokens: dict[str, int] = defaultdict(int)
        # Timing
        self.step_times: list[float] = []

    def record(self, nll_sum: float, byte_sum: float, token_count: int,
               depth: int, phase: str, step_time: float):
        self.total_nll += nll_sum
        self.total_bytes += byte_sum
        self.total_tokens += token_count
        self.depth_nll[depth] += nll_sum
        self.depth_bytes[depth] += byte_sum
        self.depth_tokens[depth] += token_count
        self.phase_nll[phase] += nll_sum
        self.phase_bytes[phase] += byte_sum
        self.phase_tokens[phase] += token_count
        self.step_times.append(step_time)

    def bpb(self) -> float:
        if self.total_bytes == 0:
            return float("inf")
        return (self.total_nll / math.log(2.0)) / self.total_bytes

    def bpb_for_depth(self, depth: int) -> float:
        b = self.depth_bytes.get(depth, 0.0)
        if b == 0:
            return float("inf")
        return (self.depth_nll[depth] / math.log(2.0)) / b

    def bpb_for_phase(self, phase: str) -> float:
        b = self.phase_bytes.get(phase, 0.0)
        if b == 0:
            return float("inf")
        return (self.phase_nll[phase] / math.log(2.0)) / b

    def log_step(self, step: int, depth: int, phase: str, batch_nll: float,
                 batch_bytes: float, batch_tokens: int, lr: float):
        batch_bpb = (batch_nll / math.log(2.0)) / batch_bytes if batch_bytes > 0 else float("inf")
        avg_step_ms = 1000.0 * self.step_times[-1] if self.step_times else 0
        self.log_fn(
            f"ttt step:{step} depth:{depth} phase:{phase} "
            f"batch_bpb:{batch_bpb:.4f} running_bpb:{self.bpb():.4f} "
            f"tokens:{self.total_tokens} step_ms:{avg_step_ms:.0f} lr:{lr:.2e}"
        )

    def log_summary(self):
        self.log_fn("=" * 80)
        self.log_fn(f"TTT SUMMARY: total_bpb={self.bpb():.6f} "
                     f"total_tokens={self.total_tokens} total_bytes={self.total_bytes:.0f}")
        self.log_fn("-" * 40 + " PER-DEPTH BREAKDOWN " + "-" * 40)
        for depth in sorted(self.depth_tokens.keys()):
            self.log_fn(
                f"  depth={depth}: bpb={self.bpb_for_depth(depth):.6f} "
                f"tokens={self.depth_tokens[depth]} bytes={self.depth_bytes[depth]:.0f}"
            )
        self.log_fn("-" * 40 + " PER-PHASE BREAKDOWN " + "-" * 40)
        for phase in sorted(self.phase_tokens.keys()):
            self.log_fn(
                f"  phase={phase}: bpb={self.bpb_for_phase(phase):.6f} "
                f"tokens={self.phase_tokens[phase]} bytes={self.phase_bytes[phase]:.0f}"
            )
        if self.step_times:
            avg_ms = 1000.0 * sum(self.step_times) / len(self.step_times)
            total_s = sum(self.step_times)
            self.log_fn(f"  timing: avg_step={avg_ms:.0f}ms total={total_s:.1f}s steps={len(self.step_times)}")
        self.log_fn("=" * 80)


# ---------------------------------------------------------------------------
# SLIDING WINDOW EVAL (non-compiled, supports inject_scales)
# ---------------------------------------------------------------------------

def eval_sliding_nocompile(
    model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    seq_len: int,
    batch_seqs: int = 16,
) -> tuple[float, float]:
    """Sliding window eval WITHOUT torch.compile (needed for inject_scales)."""
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    return val_loss, bits_per_token * tokens_per_byte


# ---------------------------------------------------------------------------
# MAIN TTT EVAL
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = TTTConfig()

    # --- Distributed + CUDA setup ---
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

    # --- Logging ---
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/ttt_{int(time.time())}.txt" if master_process else None

    def log0(msg: str, console: bool = True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(f"eval_ttt.py starting: rank={rank} world_size={world_size}")
    log0(f"config: {json.dumps({k: v for k, v in vars(TTTConfig).items() if not k.startswith('_')}, default=str)}")

    # --- Load checkpoint ---
    log0(f"Loading checkpoint: {cfg.checkpoint_path}")
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    base_depth = cfg.ttt_base_depth if cfg.ttt_base_depth > 0 else model_cfg["eval_recurrent_depth"]
    target_depth = cfg.ttt_target_depth
    eval_seq_len = model_cfg.get("eval_seq_len", 2048)
    eval_stride = cfg.eval_stride

    log0(f"Model config: dim={model_cfg['model_dim']} heads={model_cfg['num_heads']} "
         f"kv_heads={model_cfg['num_kv_heads']} entry={model_cfg['num_entry_blocks']} "
         f"recurrent={model_cfg['num_recurrent_blocks']} exit={model_cfg['num_exit_blocks']}")
    log0(f"Depth: base={base_depth} target={target_depth} decay_k={cfg.ttt_decay_k}")
    log0(f"Schedule: warmup={cfg.ttt_warmup_seqs} ramp={cfg.ttt_ramp_seqs}")

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
        mean_depth=model_cfg["mean_depth"],
        num_entry_blocks=model_cfg["num_entry_blocks"],
        num_exit_blocks=model_cfg["num_exit_blocks"],
        num_recurrent_blocks=model_cfg["num_recurrent_blocks"],
        entry_mlp_mult=model_cfg.get("entry_mlp_mult", 4.0),
        exit_mlp_mult=model_cfg.get("exit_mlp_mult", 4.0),
    ).to(device).bfloat16()

    for mod in model.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)

    n_params = sum(p.numel() for p in model.parameters())
    log0(f"Model loaded: {n_params} params")

    # --- Freeze parameters ---
    trainable_names = []
    frozen_names = []
    for name, param in model.named_parameters():
        freeze = False
        if cfg.freeze_entry and name.startswith("entry_blocks."):
            freeze = True
        elif cfg.freeze_exit and name.startswith("exit_blocks."):
            freeze = True
        elif cfg.freeze_embeddings and (name.startswith("tok_emb.") or name.startswith("bigram.")):
            freeze = True
        elif cfg.freeze_embeddings and name.startswith("smear."):
            freeze = True
        elif name == "final_norm.weight" or name == "final_norm.bias":
            freeze = True  # keep output norm frozen
        # lm_head is tied to tok_emb, freezing tok_emb is enough

        if freeze:
            param.requires_grad_(False)
            frozen_names.append(name)
        else:
            trainable_names.append(name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log0(f"Trainable params: {n_trainable} ({100*n_trainable/n_params:.1f}%)")
    log0(f"Trainable: {trainable_names[:10]}{'...' if len(trainable_names) > 10 else ''}")
    log0(f"Frozen: {len(frozen_names)} parameter groups")

    # --- Optimizer (AdamW on trainable params) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.ttt_lr,
        weight_decay=cfg.ttt_wd,
        betas=(0.9, 0.999),
        fused=True,
    )

    # --- Load validation data ---
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    val_tokens = load_validation_tokens(cfg.val_files, eval_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, model_cfg["vocab_size"], device
    )
    total_val_tokens = val_tokens.numel() - 1
    log0(f"Validation tokens: {total_val_tokens}")

    # --- Partition sequences across ranks ---
    seq_len = eval_seq_len
    total_seqs = total_val_tokens // seq_len
    my_seq_start = (total_seqs * rank) // world_size
    my_seq_end = (total_seqs * (rank + 1)) // world_size
    my_num_seqs = my_seq_end - my_seq_start
    log0(f"Rank {rank}: sequences [{my_seq_start}, {my_seq_end}) = {my_num_seqs} seqs")

    # --- TTT Logger ---
    logger = TTTLogger(log0)

    # --- Baseline eval (before TTT, at base depth) ---
    log0("\n--- BASELINE EVAL (before TTT) ---")
    model.eval()
    model.n_recurrent_iters = base_depth
    model.inject_scales = None  # full injection, same as training

    # Use a quick non-overlapping eval as baseline
    # We need grad_accum_steps for eval_val; set to match training
    grad_accum_steps = max(1, 8 // world_size)
    baseline_args = Hyperparameters()
    baseline_args.val_batch_size = baseline_args.val_batch_size
    baseline_loss, baseline_bpb = eval_val(
        baseline_args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=eval_seq_len,
    )
    log0(f"BASELINE depth={base_depth} val_loss={baseline_loss:.4f} val_bpb={baseline_bpb:.6f}")

    # --- Save base model state for potential reset ---
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # --- TTT LOOP ---
    log0("\n--- TTT DEPTH EXTENSION ---")
    log0(f"Processing {my_num_seqs} sequences, batch_size={cfg.ttt_batch_seqs}")

    vocab_size = model_cfg["vocab_size"]
    model.train()
    global_seq_idx = my_seq_start  # track global position for depth curriculum

    # Accumulators for distributed reduction
    nll_accum = torch.zeros((), device=device, dtype=torch.float64)
    bytes_accum = torch.zeros((), device=device, dtype=torch.float64)
    tokens_accum = torch.zeros((), device=device, dtype=torch.float64)

    step = 0
    t_ttt_start = time.perf_counter()

    for batch_start in range(my_seq_start, my_seq_end, cfg.ttt_batch_seqs):
        batch_end = min(batch_start + cfg.ttt_batch_seqs, my_seq_end)
        bsz = batch_end - batch_start
        t_step = time.perf_counter()

        # --- Determine depth and phase for this batch ---
        # Use the middle sequence's global index for curriculum
        mid_global_seq = batch_start + bsz // 2
        current_depth = compute_depth_for_seq(
            mid_global_seq, cfg.ttt_warmup_seqs, cfg.ttt_ramp_seqs,
            base_depth, target_depth,
        )
        if mid_global_seq < cfg.ttt_warmup_seqs:
            phase = "warmup"
        elif mid_global_seq < cfg.ttt_warmup_seqs + cfg.ttt_ramp_seqs:
            phase = "ramp"
        else:
            phase = "steady"

        # --- Set depth and injection decay ---
        model.n_recurrent_iters = current_depth
        if current_depth > base_depth:
            model.inject_scales = compute_inject_scales(current_depth, base_depth, cfg.ttt_decay_k)
        else:
            model.inject_scales = None  # full injection at base depth

        # --- Build batch tensors ---
        raw_start = batch_start * seq_len
        raw_end = batch_end * seq_len + 1
        local_tokens = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
        x_batch = local_tokens[:-1].reshape(bsz, seq_len)
        y_batch = local_tokens[1:].reshape(bsz, seq_len)

        # --- Forward pass (with gradients for training) ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x_batch)

        # --- Score tokens: per-token NLL (BEFORE weight update = safe) ---
        nll_flat = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            y_batch.reshape(-1),
            reduction="none",
        )
        nll_2d = nll_flat.reshape(bsz, seq_len)

        with torch.no_grad():
            batch_nll = nll_2d.to(torch.float64).sum().item()
            # Byte counting (same as eval_val)
            prev_ids = x_batch.reshape(-1)
            tgt_ids = y_batch.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            batch_bytes = token_bytes.sum().item()
            batch_tokens = int(y_batch.numel())

            # Accumulate for distributed reduction
            nll_accum += batch_nll
            bytes_accum += batch_bytes
            tokens_accum += batch_tokens

        # --- Train: backward + step ---
        loss = nll_flat.mean()
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
        logger.record(batch_nll, batch_bytes, batch_tokens, current_depth, phase, step_time)

        if step % cfg.ttt_log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.log_step(step, current_depth, phase, batch_nll, batch_bytes, batch_tokens, current_lr)

        step += 1
        global_seq_idx = batch_end

    ttt_time = time.perf_counter() - t_ttt_start

    # --- Distributed reduction for final BPB ---
    if distributed:
        dist.all_reduce(nll_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_accum, op=dist.ReduceOp.SUM)

    final_ttt_bpb = (nll_accum.item() / math.log(2.0)) / bytes_accum.item() if bytes_accum.item() > 0 else float("inf")
    log0(f"\nTTT COMPLETE: bpb={final_ttt_bpb:.6f} tokens={int(tokens_accum.item())} "
         f"time={ttt_time:.1f}s steps={step}")
    logger.log_summary()

    # --- POST-TTT EVALUATION ---
    model.eval()

    # Standard eval at target depth with injection decay
    log0("\n--- POST-TTT EVAL (standard, target depth) ---")
    model.n_recurrent_iters = target_depth
    model.inject_scales = compute_inject_scales(target_depth, base_depth, cfg.ttt_decay_k)
    post_loss, post_bpb = eval_val(
        baseline_args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=eval_seq_len,
    )
    log0(f"POST-TTT standard depth={target_depth} val_loss={post_loss:.4f} val_bpb={post_bpb:.6f}")
    log0(f"  vs BASELINE: delta={post_bpb - baseline_bpb:+.6f}")

    # Also eval at base depth (check for regression)
    log0("\n--- POST-TTT EVAL (standard, base depth) ---")
    model.n_recurrent_iters = base_depth
    model.inject_scales = None
    post_base_loss, post_base_bpb = eval_val(
        baseline_args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=eval_seq_len,
    )
    log0(f"POST-TTT standard depth={base_depth} val_loss={post_base_loss:.4f} val_bpb={post_base_bpb:.6f}")
    log0(f"  vs BASELINE: delta={post_base_bpb - baseline_bpb:+.6f}")

    # Depth sweep: evaluate at multiple depths
    log0("\n--- POST-TTT DEPTH SWEEP ---")
    sweep_depths = sorted(set([base_depth, base_depth + 2, target_depth,
                                min(target_depth + 2, target_depth + cfg.ttt_decay_k),
                                max(2, base_depth - 2)]))
    for d in sweep_depths:
        model.n_recurrent_iters = d
        model.inject_scales = compute_inject_scales(d, base_depth, cfg.ttt_decay_k) if d > base_depth else None
        d_loss, d_bpb = eval_val(
            baseline_args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            eval_seq_len=eval_seq_len,
        )
        log0(f"  depth={d} val_bpb={d_bpb:.6f} delta_vs_baseline={d_bpb - baseline_bpb:+.6f}")

    # --- Sliding window eval (non-compiled to support inject_scales) ---
    if cfg.final_sliding_eval and eval_stride > 0:
        log0("\n--- POST-TTT SLIDING WINDOW EVAL ---")
        for d in [base_depth, target_depth]:
            model.n_recurrent_iters = d
            model.inject_scales = compute_inject_scales(d, base_depth, cfg.ttt_decay_k) if d > base_depth else None
            sw_loss, sw_bpb = eval_sliding_nocompile(
                model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=eval_stride, seq_len=eval_seq_len,
            )
            log0(f"  sliding depth={d} stride={eval_stride} val_bpb={sw_bpb:.6f}")

    # --- Depth ensembling ---
    ensemble_depths_str = cfg.depth_ensemble.strip()
    if ensemble_depths_str:
        ensemble_depths = [int(d.strip()) for d in ensemble_depths_str.split(",") if d.strip()]
        if len(ensemble_depths) >= 2:
            log0(f"\n--- DEPTH ENSEMBLE EVAL (depths={ensemble_depths}) ---")
            ensemble_bpb = eval_depth_ensemble(
                model, ensemble_depths, base_depth, cfg.ttt_decay_k,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                device, rank, world_size, eval_seq_len, vocab_size,
            )
            log0(f"  ensemble bpb={ensemble_bpb:.6f} delta_vs_baseline={ensemble_bpb - baseline_bpb:+.6f}")

    log0(f"\n{'='*80}")
    log0(f"FINAL RESULTS:")
    log0(f"  Baseline BPB (depth={base_depth}): {baseline_bpb:.6f}")
    log0(f"  TTT inline BPB: {final_ttt_bpb:.6f}")
    log0(f"  Post-TTT BPB (depth={target_depth}): {post_bpb:.6f}")
    log0(f"  Post-TTT BPB (depth={base_depth}): {post_base_bpb:.6f}")
    log0(f"{'='*80}")

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# DEPTH ENSEMBLE EVALUATION
# ---------------------------------------------------------------------------

def eval_depth_ensemble(
    model: GPT,
    depths: list[int],
    base_depth: int,
    decay_k: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    seq_len: int,
    vocab_size: int,
    batch_seqs: int = 8,
) -> float:
    """Evaluate by averaging logits across multiple depths."""
    total_val_tokens = val_tokens.numel() - 1
    total_seqs = total_val_tokens // seq_len
    my_seq_start = (total_seqs * rank) // world_size
    my_seq_end = (total_seqs * (rank + 1)) // world_size

    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_sum = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(my_seq_start, my_seq_end, batch_seqs):
            batch_end = min(batch_start + batch_seqs, my_seq_end)
            bsz = batch_end - batch_start

            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local_tokens = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x_batch = local_tokens[:-1].reshape(bsz, seq_len)
            y_batch = local_tokens[1:].reshape(bsz, seq_len)

            # Accumulate logits across depths
            all_logits = []
            for d in depths:
                model.n_recurrent_iters = d
                model.inject_scales = compute_inject_scales(d, base_depth, decay_k) if d > base_depth else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.forward_logits(x_batch)
                all_logits.append(logits)

            # Average logits
            avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)

            # Score
            nll = F.cross_entropy(
                avg_logits.reshape(-1, vocab_size).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            nll_sum += nll.to(torch.float64).sum()
            token_sum += float(y_batch.numel())
            prev_ids = x_batch.reshape(-1)
            tgt_ids = y_batch.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            byte_sum += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_sum, op=dist.ReduceOp.SUM)

    bpb = (nll_sum.item() / math.log(2.0)) / byte_sum.item()
    return bpb


if __name__ == "__main__":
    main()
