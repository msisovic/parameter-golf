"""
eval_ttt_lora.py — Per-depth LoRA TTT for depth-recurrent transformers.

Strategy: freeze base weights, add zero-initialized LoRA at each depth for each
recurrent block. TTT-train only the LoRAs with a 3-phase curriculum:
  Phase 1 (stabilize): train at base depth, LoRAs learn near-identity
  Phase 2 (ramp):      linearly increase depth from base to target
  Phase 3 (consolidate): train at target depth

Eval depth starts at base and is promoted when a deeper depth proves better.
Scoring uses stride 64 for maximum context overlap.

Architecture: N entry + M recurrent (x0 residual, no injection) + K exit.

SAFETY: score-before-train on every chunk.

Usage:
    torchrun --nproc_per_node=4 eval_ttt_lora.py
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
    Block,
    apply_rotary_emb,
    build_sentencepiece_luts,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
    CONTROL_TENSOR_NAME_PATTERNS,
)


# ---------------------------------------------------------------------------
# PER-DEPTH LORA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter: output += (x @ A) @ B, zero-initialized."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(in_features)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: Tensor, base_out: Tensor) -> Tensor:
        return base_out + (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype)


class LoRABlock(nn.Module):
    """Wraps a frozen Block with per-depth LoRA adapters on its linear layers."""
    def __init__(self, base_block: Block, rank: int = 8):
        super().__init__()
        self.base_block = base_block  # frozen
        dim = base_block.attn.c_q.in_features
        kv_dim = base_block.attn.c_k.out_features
        mlp_hidden = base_block.mlp.fc.out_features

        self.lora_c_q = LoRALinear(dim, dim, rank)
        self.lora_c_k = LoRALinear(dim, kv_dim, rank)
        self.lora_c_v = LoRALinear(dim, kv_dim, rank)
        self.lora_proj = LoRALinear(dim, dim, rank)
        self.lora_fc = LoRALinear(dim, mlp_hidden, rank)
        self.lora_mlp_proj = LoRALinear(mlp_hidden, dim, rank)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        # Replicate Block.forward but inject LoRA into each linear layer
        block = self.base_block
        mix = block.resid_mix.to(dtype=x.dtype)
        h = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = block.ln_scale_factor

        # Attention with LoRA (norm is on block, not attn)
        attn_in = block.attn_norm(h) * s
        attn_out = self._attn_with_lora(block.attn, attn_in)
        h = h + block.attn_scale.to(dtype=h.dtype)[None, None, :] * attn_out

        # MLP with LoRA
        mlp_in = block.mlp_norm(h) * s
        mlp_out = self._mlp_with_lora(block.mlp, mlp_in)
        h = h + block.mlp_scale.to(dtype=h.dtype)[None, None, :] * mlp_out
        return h

    def _attn_with_lora(self, attn, x: Tensor) -> Tensor:
        """Run attention on pre-normed input, adding LoRA to q/k/v/proj."""
        from train_gpt_noinject import flash_attn_3_func

        # Q, K, V with LoRA
        q = self.lora_c_q(x, attn.c_q(x))
        k = self.lora_c_k(x, attn.c_k(x))
        v = self.lora_c_v(x, attn.c_v(x))

        bsz, seq_len = x.shape[:2]
        head_dim = attn.head_dim

        q = q.view(bsz, seq_len, attn.num_heads, head_dim)
        k = k.view(bsz, seq_len, attn.num_kv_heads, head_dim)
        v = v.view(bsz, seq_len, attn.num_kv_heads, head_dim)

        # RMSNorm first, then RoPE, then q_gain (matching original order)
        q = F.rms_norm(q, (head_dim,))
        k = F.rms_norm(k, (head_dim,))
        cos, sin = attn.rotary(seq_len, x.device, x.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # Flash attention
        o = flash_attn_3_func(q, k, v, causal=True)
        o = o.reshape(bsz, seq_len, -1)

        # Output projection with LoRA
        out = self.lora_proj(o, attn.proj(o))
        return out

    def _mlp_with_lora(self, mlp, x: Tensor) -> Tensor:
        """Run MLP but add LoRA to fc and proj."""
        fc_out = self.lora_fc(x, mlp.fc(x))
        # relu^2 activation + gate
        hidden = F.relu(fc_out).square()
        out = self.lora_mlp_proj(hidden, mlp.proj(hidden))
        return out


class DepthLoRAWrapper(nn.Module):
    """Wraps a GPT model with per-depth LoRA blocks for the recurrent stage."""
    def __init__(self, base_model: GPT, max_depth: int, lora_rank: int = 8):
        super().__init__()
        self.base_model = base_model
        self.max_depth = max_depth

        # Create per-depth LoRA wrappers for each recurrent block
        # depth_loras[d][b] = LoRABlock for depth d, recurrent block b
        self.depth_loras = nn.ModuleList()
        for d in range(max_depth):
            block_loras = nn.ModuleList()
            for block in base_model.recurrent_blocks:
                block_loras.append(LoRABlock(block, rank=lora_rank))
            self.depth_loras.append(block_loras)

    def forward_logits(self, input_ids: Tensor, n_iters: int) -> Tensor:
        m = self.base_model
        x = m.tok_emb(input_ids)
        if m.bigram is not None:
            x = x + m.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = m.smear(x)
        x0 = x

        # Entry blocks (frozen, no LoRA)
        for block in m.entry_blocks:
            x = block(x, x0)

        # Recurrent blocks x N with per-depth LoRA
        for depth_idx in range(n_iters):
            if depth_idx < self.max_depth:
                # Use LoRA-wrapped blocks
                for b_idx, lora_block in enumerate(self.depth_loras[depth_idx]):
                    x = lora_block(x, x0)
            else:
                # Beyond max LoRA depth, use base blocks
                for block in m.recurrent_blocks:
                    x = block(x, x0)

        # Exit blocks (frozen, no LoRA)
        for block in m.exit_blocks:
            x = block(x, x0)

        x = m.final_norm(x)
        if m.tie_embeddings:
            logits_proj = F.linear(x, m.tok_emb.weight)
        else:
            logits_proj = m.lm_head(x)
        return m.logit_softcap * torch.tanh(logits_proj / m.logit_softcap)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

class TTTConfig:
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "ttt_checkpoint_noinject.pt")
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

    ttt_lr = float(os.environ.get("TTT_LR", 3e-4))
    ttt_wd = float(os.environ.get("TTT_WD", 0.0))
    lora_rank = int(os.environ.get("LORA_RANK", 8))
    base_depth = int(os.environ.get("TTT_BASE_DEPTH", 0))  # 0 = from checkpoint
    target_depth = int(os.environ.get("TTT_TARGET_DEPTH", 6))
    eval_stride = int(os.environ.get("TTT_EVAL_STRIDE", 64))
    train_stride = int(os.environ.get("TTT_TRAIN_STRIDE", 256))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

    # Curriculum phase fractions (of total train steps)
    phase1_frac = float(os.environ.get("TTT_PHASE1_FRAC", 0.15))
    phase2_frac = float(os.environ.get("TTT_PHASE2_FRAC", 0.50))
    # phase3 = 1 - phase1 - phase2

    # Depth probe
    depth_probe_every = int(os.environ.get("TTT_DEPTH_PROBE_EVERY", 100))
    depth_probe_margin = float(os.environ.get("TTT_DEPTH_PROBE_MARGIN", 0.0))  # require this much improvement

    log_every = int(os.environ.get("TTT_LOG_EVERY", 50))


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
    logfile = f"logs/ttt_lora_{int(time.time())}.txt" if master_process else None

    def log0(msg: str):
        if not master_process:
            return
        print(msg, flush=True)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f, flush=True)

    log0(f"eval_ttt_lora.py rank={rank} world_size={world_size}")

    # --- Load checkpoint ---
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    base_depth = cfg.base_depth if cfg.base_depth > 0 else model_cfg["eval_recurrent_depth"]
    target_depth = cfg.target_depth
    seq_len = cfg.eval_seq_len
    eval_stride = cfg.eval_stride
    train_stride = cfg.train_stride
    vocab_size = model_cfg["vocab_size"]

    assert train_stride % eval_stride == 0
    evals_per_train = train_stride // eval_stride

    log0(f"Model: dim={model_cfg['model_dim']} arch={model_cfg.get('architecture', 'unknown')}")
    log0(f"  blocks: {model_cfg.get('num_entry_blocks', 1)}+{model_cfg.get('num_recurrent_blocks', 1)}+{model_cfg.get('num_exit_blocks', 1)}")
    log0(f"Depth: base={base_depth} target={target_depth} lora_rank={cfg.lora_rank}")
    log0(f"Strides: eval={eval_stride} train={train_stride}")

    # --- Build base model (frozen) ---
    base_model = GPT(
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

    for mod in base_model.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    restore_low_dim_params_to_fp32(base_model)
    base_model.load_state_dict(state_dict, strict=True)

    # Freeze ALL base model params
    for p in base_model.parameters():
        p.requires_grad_(False)

    n_base_params = sum(p.numel() for p in base_model.parameters())
    log0(f"Base model: {n_base_params} params (all frozen)")

    # --- Build LoRA wrapper ---
    wrapper = DepthLoRAWrapper(base_model, max_depth=target_depth, lora_rank=cfg.lora_rank).to(device)

    n_lora_params = sum(p.numel() for p in wrapper.depth_loras.parameters())
    log0(f"LoRA params: {n_lora_params} ({100*n_lora_params/n_base_params:.2f}% of base)")

    # --- Optimizer (only LoRA params) ---
    lora_params = list(wrapper.depth_loras.parameters())
    optimizer = torch.optim.AdamW(lora_params, lr=cfg.ttt_lr, weight_decay=cfg.ttt_wd,
                                   betas=(0.9, 0.999), fused=True)

    # --- Load val data ---
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    val_tokens = load_validation_tokens(cfg.val_files, seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, vocab_size, device
    )
    total_val_tokens = val_tokens.numel() - 1
    log0(f"Val tokens: {total_val_tokens}")

    # --- Partition across ranks ---
    per_rank = total_val_tokens // world_size
    rank_start = rank * per_rank
    rank_end = (rank + 1) * per_rank if rank < world_size - 1 else total_val_tokens

    # Estimate total train steps for curriculum scheduling
    total_eval_windows = (rank_end - rank_start) // eval_stride
    total_train_steps = total_eval_windows // evals_per_train
    phase1_end = int(total_train_steps * cfg.phase1_frac)
    phase2_end = int(total_train_steps * (cfg.phase1_frac + cfg.phase2_frac))

    log0(f"Rank {rank}: tokens [{rank_start}, {rank_end})")
    log0(f"Total train steps: ~{total_train_steps}, phase1 end: {phase1_end}, phase2 end: {phase2_end}")

    # --- Accumulators ---
    nll_accum = torch.zeros((), device=device, dtype=torch.float64)
    bytes_accum = torch.zeros((), device=device, dtype=torch.float64)
    tokens_accum = torch.zeros((), device=device, dtype=torch.float64)
    depth_nll: dict[int, float] = defaultdict(float)
    depth_bytes: dict[int, float] = defaultdict(float)
    depth_tokens: dict[int, int] = defaultdict(int)

    # --- Depth state ---
    eval_depth = base_depth
    train_step = 0
    eval_step_count = 0
    evals_since_train = 0
    scored_up_to = rank_start
    t_start = time.perf_counter()

    def get_train_depth(step: int) -> int:
        """Curriculum: base during phase 1, ramp during phase 2, target during phase 3."""
        if step < phase1_end:
            return base_depth
        elif step < phase2_end:
            frac = (step - phase1_end) / max(phase2_end - phase1_end, 1)
            return base_depth + int(round(frac * (target_depth - base_depth)))
        else:
            return target_depth

    def score_window(ws: int, depth: int) -> tuple[float, float, int]:
        end = min(ws + seq_len, rank_end)
        wlen = end - ws
        if wlen < 1:
            return 0.0, 0.0, 0
        chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = wrapper.forward_logits(x, n_iters=depth)
        nll = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            y.reshape(-1), reduction="none",
        )
        s = 0 if ws == rank_start else max(wlen - eval_stride, 0)
        scored_nll = nll[s:wlen]
        tgt = y[0, s:wlen]
        prev = x[0, s:wlen]
        tb = base_bytes_lut[tgt].to(torch.float64)
        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
        return scored_nll.to(torch.float64).sum().item(), tb.sum().item(), wlen - s

    def train_window(ws: int, depth: int):
        end = min(ws + seq_len, rank_end)
        wlen = end - ws
        if wlen < 1:
            return
        chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = wrapper.forward_logits(x, n_iters=depth)
        nll = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            y.reshape(-1), reduction="none",
        )
        loss = nll[:wlen].mean()
        loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    # --- Main loop ---
    eval_pos = rank_start

    while eval_pos < rank_end:
        # Score one eval window
        nll_sum, bytes_sum, n_tok = score_window(eval_pos, eval_depth)
        nll_accum += nll_sum
        bytes_accum += bytes_sum
        tokens_accum += n_tok
        depth_nll[eval_depth] += nll_sum
        depth_bytes[eval_depth] += bytes_sum
        depth_tokens[eval_depth] += n_tok

        scored_up_to = max(scored_up_to, min(eval_pos + seq_len, rank_end))
        eval_pos += eval_stride
        evals_since_train += 1
        eval_step_count += 1

        # Train after evals_per_train eval windows
        if evals_since_train >= evals_per_train:
            evals_since_train = 0
            curr_train_depth = get_train_depth(train_step)

            train_end = scored_up_to
            train_ws = max(train_end - seq_len, rank_start)
            train_window(train_ws, curr_train_depth)
            train_step += 1

            # Depth probing
            if (train_step > 0
                    and train_step % cfg.depth_probe_every == 0
                    and eval_depth < target_depth
                    and curr_train_depth > eval_depth):
                probe_ws = max(eval_pos - eval_stride, rank_start)
                probe_nll, _, probe_n = score_window(probe_ws, curr_train_depth)
                curr_nll, _, _ = score_window(probe_ws, eval_depth)
                if probe_n > 0:
                    avg_new = probe_nll / probe_n
                    avg_cur = curr_nll / probe_n
                    if avg_new < avg_cur - cfg.depth_probe_margin:
                        log0(f"EVAL DEPTH UPGRADE: {eval_depth}→{curr_train_depth} "
                             f"(deeper={avg_new:.4f} < current={avg_cur:.4f})")
                        eval_depth = curr_train_depth
                    else:
                        log0(f"depth_probe: eval stays at {eval_depth} "
                             f"(deeper={avg_new:.4f} vs current={avg_cur:.4f})")

            # Log
            if train_step % cfg.log_every == 0:
                running_bpb = (nll_accum.item() / math.log(2.0)) / max(bytes_accum.item(), 1)
                elapsed = time.perf_counter() - t_start
                phase = "P1-stabilize" if train_step < phase1_end else ("P2-ramp" if train_step < phase2_end else "P3-consolidate")
                log0(f"step:{train_step} eval:{eval_step_count} {phase} "
                     f"eval_d:{eval_depth} train_d:{curr_train_depth} "
                     f"bpb:{running_bpb:.4f} tokens:{int(tokens_accum.item())} "
                     f"elapsed:{elapsed:.0f}s")

    total_time = time.perf_counter() - t_start

    # --- Distributed reduction ---
    if distributed:
        dist.all_reduce(nll_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_accum, op=dist.ReduceOp.SUM)

    final_bpb = (nll_accum.item() / math.log(2.0)) / bytes_accum.item() if bytes_accum.item() > 0 else float("inf")

    log0(f"\n{'='*80}")
    log0(f"TTT LORA COMPLETE: bpb={final_bpb:.6f} tokens={int(tokens_accum.item())} "
         f"time={total_time:.1f}s train_steps={train_step}")
    log0(f"\nPER-DEPTH BREAKDOWN:")
    for d in sorted(depth_tokens.keys()):
        d_bpb = (depth_nll[d] / math.log(2.0)) / depth_bytes[d] if depth_bytes[d] > 0 else float("inf")
        log0(f"  depth={d}: bpb={d_bpb:.6f} tokens={depth_tokens[d]}")
    log0(f"\nLoRA params: {n_lora_params}")
    log0(f"{'='*80}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
