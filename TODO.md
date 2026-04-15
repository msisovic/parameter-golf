# LM-head fused FP8 softcap+CE — TODO

## Current state (branch `pr-1530-lm-head-kernels`)

### Forward: done
Slim 2D persistent forward (`fused_fp8_softcap_ce_slim_stats_kernel` + targets + finalize) is wired into `NoLogitFusedFP8SoftcapCrossEntropyFn.forward`. ~1.8ms on M=98304, K=512, V=8192.

A writeback variant (`fused_fp8_softcap_ce_slim_stats_writeback_kernel`) stores bf16 logits to HBM during the forward epilogue for +0.27ms overhead. This is needed for the cuBLAS backward path.

### Backward: needs rewiring

**The one thing left to do:**

Wire `NoLogitFusedFP8SoftcapCrossEntropyFn.backward` (currently at ~line 2320 in `train_gpt.py`) to use the **writeback forward + cuBLAS backward** path instead of the current slow Triton recompute kernels.

Current backward (`NoLogitFusedFP8SoftcapCrossEntropyFn.backward`):
- Uses `fused_fp8_softcap_ce_dx_recompute_kernel` + `fused_fp8_softcap_ce_dw_recompute_kernel`
- These are Triton recompute kernels: ~15ms combined (tuned) or ~125ms (original tile shapes)
- Cannot compete with cuBLAS — structural limitation (tile-based bf16 reduction is 5× slower than cuBLAS's single large GEMM)

Target backward (same as `FusedFP8SoftcapCrossEntropyFn.backward` at ~line 730):
- `fused_softcap_ce_bwd_kernel` — row-wise CE gradient from stored logits → grad_logits buffer
- `grad_logits @ weight` — cuBLAS bf16 GEMM for dx
- `grad_logits.t() @ x` — cuBLAS bf16 GEMM for dw
- Total: **3.2ms**

### Concrete steps

1. **Switch forward** to the writeback variant:
   - Replace `fused_fp8_softcap_ce_slim_stats_kernel` call in `NoLogitFusedFP8SoftcapCrossEntropyFn.forward` with `fused_fp8_softcap_ce_slim_stats_writeback_kernel`
   - Allocate `logits_buf = torch.empty(n_rows, n_cols, dtype=torch.bfloat16, device=x.device)`
   - Save `logits_buf` in `ctx` for backward

2. **Rewrite backward** to use stored logits + cuBLAS:
   - Call `fused_softcap_ce_bwd_kernel` to compute `grad_logits` from `logits_buf` + `lse` + `targets` + `grad_output`
   - `grad_input = grad_logits @ weight` (cuBLAS bf16)
   - `grad_weight = grad_logits.t() @ x` (cuBLAS bf16)
   - This is the same pattern as `FusedFP8SoftcapCrossEntropyFn.backward` (~line 730)

3. **Run training comparison** (500 steps):
   ```bash
   NUM_ITERATIONS=500 TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
       FP8_LM_HEAD=1 NOLOGIT_FUSED_CE=1 \
       torchrun --standalone --nproc_per_node=8 train_gpt.py
   ```
   Compare tok/s and val_loss against split-path baseline (8.4M tok/s).

### Expected performance

- Writeback fwd: ~2.1ms (vs ~1.8ms no-writeback, +0.27ms overhead)
- cuBLAS bwd: ~3.2ms
- Total fwd+bwd: ~5.3ms
- Split-path reference: ~5.0ms total
- Memory: needs 1.6GB logits buffer (same as split-path)

The nologit path's advantage is that it fuses the FP8 GEMM + softcap + CE loss into a single forward kernel (no separate `_scaled_mm` + CE fwd). With the writeback, it's ~0.3ms slower total than split-path but shares the same memory footprint.

## What was tried and ruled out for backward

See `experiment_log.md` "Backward optimization (2026-04-15)" section for full details.

- **Triton recompute dx+dw separate**: 11ms combined. Tile-based bf16 reduction 5× slower than cuBLAS.
- **Fused dx+dw kernel (shared FP8 recompute)**: 33-64ms. atomicAdd contention for dw + register pressure.
- **FP8 backward GEMMs**: Raw GEMMs 2× faster (1.15ms vs 2.25ms), but quantizing+transposing the 1.6GB grad_logits tensor costs 17ms.
- **cuBLAS bf16 GEMM breakdown**: dx 1.05ms + dw 1.04ms = 2.37ms. This is the floor. CE bwd adds ~0.8ms → 3.2ms total.

## Key files

- `train_gpt.py` — all kernels and autograd functions
- `experiment_log.md` — full optimization history
- `bench_writeback.py` — writeback fwd + cuBLAS bwd comparison
- `bench_dxdw_fused.py` — fused kernel attempt (for reference)
- `bench_bwd_components.py` — component decomposition showing cuBLAS dominance
