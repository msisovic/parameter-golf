# LM-head fused FP8 softcap+CE — next steps

## Current state (branch `pr-1530-lm-head-kernels`)

**Best microbench result:** slim 2D persistent stats + targets + finalize kernels, `flatten=True`: **~1.17 ms** forward on the real training shape (M=98304, K=512, V=8192, softcap=30, tensorwise FP8). All microbench numbers in this doc use the same "no-quant" protocol: `x_fp8` precomputed once, timed region is GPU kernel launches only.

Baseline references (same protocol):
- `_scaled_mm` alone: ~0.95 ms
- Wrapper (`_scaled_mm + fused softcap+CE over bf16 logits`): ~1.68 ms
- Matmul-only ceiling (no epilogue, no write): ~0.66 ms
- Codex-era row-owner sigmoid block_ptr: ~2.08 ms

Public-forward (includes FP8 quant + autograd.Function overhead, what Codex reported as ~2.03–2.29 ms) is roughly microbench + ~0.05–0.10 ms. Projected public fwd for the slim kernel: ~1.22 ms, vs Codex's 2.29 ms baseline → ~46% faster if the microbench win transfers.

See `experiment_log.md` iterations 10–14 for the full trail. `bench_rowreduce.py` and `bench_matmul_only.py` are the working bench scripts.

## Levers already landed (fwd)

1. **PTX `tanh.approx.f32` softcap** (via `_ptx_tanh` helper, `train_gpt.py:20`). Replaces `A*sigmoid(x/C)` with `s*tanh(x/s)` — CE-equivalent up to a row-wise constant. Hardware MUFU.TANH on sm_90, not the libdevice software polynomial.
2. **TMA descriptors** (`tl.make_tensor_descriptor`) for x and w loads. Requires `weight_fp8_cm` (row-major `[V, K]`), i.e. `weight_fp8_t.contiguous().t()`.
3. **Slim per-tile epilogue**: target-logit `tl.where` pulled *out* of the per-tile hot loop into a separate `fused_fp8_softcap_ce_targets_kernel`. This was the unlock — it freed enough register budget and compute that `flatten=True` can actually overlap next-tile WGMMA with the current tile's epilogue.
4. **2D persistent + `flatten=True`** over tile_ids across NUM_SMS with grouped PID. Autotune lands at `BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=3`. Note: `BLOCK_M=256` did *not* win even with the slim epilogue — worth a direct probe to confirm.

Relevant new code (all in `train_gpt.py`):
- `_ptx_tanh` helper (~line 20)
- `fused_fp8_softcap_ce_slim_stats_kernel` + `_fused_fp8_softcap_ce_slim_configs` + `_grouped_pid_gm`
- `fused_fp8_softcap_ce_targets_kernel` (tiny dot-per-row kernel for target_logit)
- `fused_fp8_softcap_ce_rowreduce_desc_persistent_kernel` (persistent row-owner variant, not best)
- `fused_fp8_softcap_ce_rowreduce_desc_ws_kernel` (TMA row-owner, Iteration 11's best, currently used by `NoLogitFusedFP8SoftcapCrossEntropyFn`)
- Swapped sigmoid → PTX tanh in: `fused_fp8_softcap_ce_rowreduce_nologits_kernel`, `fused_fp8_softcap_ce_stats_nologits_desc_persistent_kernel`, `fused_fp8_softcap_ce_finalize_nologits_kernel`

## Critical gotchas discovered

- **`libdevice.tanh` is software**, not hardware — regressed the kernel. Must use PTX `tanh.approx.f32` via inline asm. See `_ptx_tanh`.
- **Accuracy of `tanh.approx.f32`**: max |loss - IEEE| ≈ 0.0087, mean ≈ 8e-4. Well below FP8 noise. Safe.
- **Warp-specialize crashes** on the row-owner's inner N-loop (`WSDataPartition.cpp:1196: "reduce should not happen on the partitioned dimension"`). WS's auto-partitioner picks the vocab (N) dim because it is the tile width, but we reduce over N. Structural conflict; WS is off the table for row-owner-shaped kernels.
- **`flatten=True` only helps if the per-tile epilogue is slim.** A heavy epilogue (esp. the `tl.where` target gather with full `BLOCK_M × BLOCK_N` mask) steals compute from the overlapped next tile's WGMMA and makes flatten actively hurt. This was the iteration-13 surprise.
- **Matmul-only ceiling of 0.66 ms does NOT directly transfer.** It's specific to epilogue-free kernels with `BLOCK_M=256`. Fused kernels can't run that shape because of epilogue register pressure. Realistic fused floor is probably ~0.9 ms, not 0.66 ms.
- **`flatten=True` gives only ~2% in a kernel with few outer iterations per SM** (persistent row-owner has only ~6 row-blocks/SM). The 10–15% flatten wins require many tiles per SM (2D tile iteration = 186/SM here).

## Plan from here

**Step 1 — real-training validation (load-bearing). ~1h work.**

De-risk everything below by confirming the forward microbench win shows up in tok/s.

1. Wire the slim forward into `NoLogitFusedFP8SoftcapCrossEntropyFn.forward` (it currently calls `fused_fp8_softcap_ce_rowreduce_nologits_kernel`, the block-ptr row-owner). Launch slim_stats + targets + finalize instead. Same `lse` + `losses` output contract.
2. Port the two existing backward kernels (`fused_fp8_softcap_ce_dx_recompute_kernel`, `fused_fp8_softcap_ce_dw_recompute_kernel`) to the **tanh gradient form** for correctness only (no perf work yet):
   - Current: `grad_logits = grad_z * (inv_C_A * sigmoid_u * (1.0 - sigmoid_u))` where `sigmoid_u = tl.sigmoid(logits * inv_C)`.
   - New: compute `u = tl.tanh(logits / softcap)` (use `_ptx_tanh`), then `grad_logits = grad_z * (1.0 - u * u)`. Here `z = softcap * u` (the tanh-form softcap matches the new fwd).
   - Also: `p = tl.exp(z - lse)` stays correct because `z` and `lse` are both in the tanh form now (the constant-shift CE-equivalence guarantees `p` is the same either way).
3. Validate gradients vs a pytorch reference path (bf16 logits + `torch.tanh` softcap + `F.cross_entropy`) — use `torch.autograd.gradcheck`-style comparison or just `(our_grad - ref_grad).abs().max()` at multiple logit scales.
4. Short training run (a few hundred steps). Compare tok/s to main. If the forward win shows up, proceed to step 2. If not, diagnose (is the kernel actually being called? is there CPU-side Python overhead eating the GPU win? is the backward dominating?).

**Step 2 — optimize backward. ~4–8h work.**

Apply the same levers to `dx` and `dw` kernels. Same playbook as fwd:

1. PTX tanh + tanh-form gradient `1 - (z/s)²` → SFU savings + simpler formula.
2. TMA descriptors on load paths.
3. Persistent + flatten scheduling.
4. Slim per-tile epilogue: pull out anything that materializes a large mask (any `tl.where(cols == targets, ..., ...)` pattern) into a separate tiny kernel.
5. Autotune probe `BLOCK_M ∈ {64, 128, 256}`, `BLOCK_N ∈ {64, 128, 256}`.

Backward structural notes:
- `dx` reduces over V (same as fwd row-owner). Same slim-epilogue design should work.
- `dw` reduces over M (perpendicular). Probably needs split-K-like partials + finalize, or atomic accumulation. Not a clean row-owner pattern.
- Backward holds more per-thread state (lse, targets, recomputed intermediates) — register budget tighter than fwd. Don't assume the same tile shapes win.

Estimated win: backward currently ~4–5 ms unvalidated; 30–40% seems plausible given the gradient formula simplification gives more SFU savings than fwd got. Absolute savings on bwd > fwd because bwd is ~2x the work.

**Step 3 — end-to-end training validation.**

Full training run on the branch. Compare tok/s, val_loss, val_bpb against the `4838 steps @ 6.48M tok/s, val_loss 2.7725, val_bpb 1.0733` baseline documented at the top of `experiment_log.md`. Watch for any accuracy degradation from the `tanh.approx` (we expect none, but confirm).

## Deferred / probably-dead-ends

- **Warp specialization** on any kernel with a row-wise reduction. Structural conflict with Triton's WS auto-partitioner on this kernel shape.
- **`BLOCK_M=256` in fused kernels**. Has been tried in multiple autotune spaces and never wins once the epilogue is present. Worth one more direct probe after any slimming but expect no.
- **exp → exp2 in online-LSE rescale.** Still potentially 3–5%, but hasn't been tried yet because slim + flatten was the bigger win. Good follow-up after step 1 lands.
- **CUTLASS-level rewrite.** Reserved for "we really need sub-1 ms and Triton can't deliver." Not on the current path.

## Key files / commands for a fresh machine

- Branch: `pr-1530-lm-head-kernels` off `main`.
- Main file: `/root/parameter-golf/train_gpt.py`.
- Log: `/root/parameter-golf/experiment_log.md` (read iterations 10–14 for recent history).
- Microbench: `python bench_rowreduce.py` — runs all fused variants + wrapper + scaled_mm + slim path, reports CUDA-event times + correctness deltas vs IEEE-tanh reference.
- GEMM ceiling bench: `python bench_matmul_only.py`.
- Training baseline command (from log):
  ```bash
  TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
      torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```
- Triton version: 3.5.1 on CUDA 12.x, H100 SXM (1979 TFLOP/s FP8 peak, 3.35 TB/s HBM). Assumed sm_90.
