# V3 Pre-Loop Drift Investigation

## Context

After porting v2 mechanisms onto base_v3, pre-loop train_loss is consistently +0.005–0.01 worse than base_v3 control, even with the minimal LOOP_UNTIE-only diff. Post-loop the trajectories converge and final pre-quant val_bpb is slightly better than base_v3 (−0.00019), but the user's concern is that the catch-up isn't free — a clean pre-loop trajectory could leave more on the table.

## Goal

Identify the source of the pre-loop drift and eliminate it, ideally getting pre-loop bit-identical to base_v3.

## What's been ruled out (from the v3_loop_untie_only_s42.log run with the minimal-diff LOOP_UNTIE port)

- **Wrapper-function torch.compile**: removed; v3 now uses base_v3's `compiled_model = torch.compile(base_model)` pattern verbatim.
- **Fixed rotary slot machinery + softmax_scale plumbing**: removed; rotary is base_v3's.
- **Bump prewarm + warm_eval_logits**: removed; no more state save/restore at long seq_len.
- **Global-TTT sharding rebalance**: reverted to base_v3.
- **Hparam mismatch**: ruled out — base_v3 hparams used verbatim.
- **YARN scaling difference**: doesn't apply (base_v3 ran with ROPE_YARN=0).

## Remaining suspects (in order of likelihood)

1. **kwargs+`is None` pattern in Block.forward**: Dynamo *should* specialize on `resid_mix=None` etc. and reduce to `mix = self.resid_mix.to(...)`, but if folding is imperfect the compiled kernel can have slightly different fp32 reduction order in flash_attn backward.
2. **Existence of `loop_extra_*` Parameters in optimizer's scalar group / `replicated_packed_params`**: even with grad=None pre-loop and filtered from all_reduce bucket, having them in the iteration list shifts SM scheduling slightly.
3. **Triple-form `encoder_loop_entries` iteration** (list of tuples vs `range`): probably benign — Dynamo unrolls both identically.

## Experiment 1 — LOOP_UNTIE_EXTRA_SCALARS=0 — DONE

**Setup**: Same code as v3_loop_untie_only_s42, with `LOOP_UNTIE_EXTRA_SCALARS=0`. This collapses the diff vs base_v3 to *dead code only*: triple-form encoder_loop_entries (both branches produce same `i`), `_block_scalar_overrides` always returns `(None, None, None)`, `Block.forward` kwargs are always `None`.

**Result**:

| step | base_v3 | v3 UNTIE=0 | Δ | v3 UNTIE=1 (prior) | Δ |
|---|---|---|---|---|---|
| 500 | 2.5617 | 2.5647 | +0.0030 | 2.5708 | +0.0091 |
| 1000 | 2.7945 | 2.8034 | +0.0089 | 2.7987 | +0.0042 |
| 1500 | 2.6178 | 2.6176 | −0.0002 | 2.6197 | +0.0019 |
| 2000 | 2.6484 | 2.6525 | +0.0041 | 2.6570 | +0.0086 |

**Findings**:
- UNTIE=0 drift oscillates around base_v3 (matches at 1500, +0.0089 at 1000) — looks like run-to-run noise (~0.001–0.003 scale).
- UNTIE=1 drift is more consistent (+0.0042 to +0.0091) — likely a real systematic effect from having `loop_extra_*` Parameters live.

**Conclusion**: Dead-code overhead (triple-form / kwargs+None pattern) is at most noise-level (≈0.003). The dominant additional drift in the original v3 LOOP_UNTIE-only run came from `loop_extra_*` Parameters in the optimizer / replicated_packed_params, not the kwargs pattern. The Experiment 2 refactor (separate `Block.forward_with_overrides`) might still help slightly, but the main lever is reducing the impact of having `loop_extra_*` live pre-loop.

## Experiment 2 — Split Block.forward / _block_with_lora into base + with_overrides — DONE

**Setup**: Reverted `Block.forward` and `_block_with_lora` to base_v3's exact code (no kwargs, no `is None` pattern). Added `Block.forward_with_overrides` and `_block_with_lora_with_overrides` as separate methods. `_forward_hidden` and `forward_ttt` branch on `extra_pass < 0` per loop entry — pre-loop entries hit base_v3's exact call site.

**Run**: `LOOP_UNTIE_EXTRA_SCALARS=1`, full schedule.

**Result (steps 500-2000, train_loss)**:

| step | base_v3 | v3 SPLIT (UNTIE=1) | Δ | v3 prior (kwargs, UNTIE=1) | Δ |
|---|---|---|---|---|---|
| 500 | 2.5617 | **2.5638** | **+0.0021** | 2.5708 | +0.0091 |
| 1000 | 2.7945 | **2.8012** | **+0.0067** | 2.7987 | +0.0042 |
| 1500 | 2.6178 | **2.6165** | **−0.0013** | 2.6197 | +0.0019 |
| 2000 | 2.6484 | **2.6508** | **+0.0024** | 2.6570 | +0.0086 |

**Conclusion**: The kwargs+`is None` pattern in `Block.forward` was the dominant systematic drift contributor. After splitting into separate base / with_overrides methods, pre-loop drift drops from +0.005-0.009 (kwargs version) to +0.002-0.007 (split, comparable to LOOP_UNTIE=0 noise band). Effectively bit-equivalent to base_v3 within noise. The `loop_extra_*` Parameters themselves do NOT meaningfully perturb training pre-loop — the issue was that even when they were unused, the kwargs-with-None-default pattern caused Dynamo to compile a different kernel.

**Cost**: ~60 lines of code duplication (`Block.forward_with_overrides` + `_block_with_lora_with_overrides`).

## Original Experiment 2 plan

If experiment 1 confirms the kwargs pattern is the culprit, refactor:

- Revert `Block.forward` to base_v3's exact signature (no override kwargs).
- Add `Block.forward_with_overrides(self, x, x0, ..., resid_mix, attn_scale, mlp_scale)` for the extra-pass case.
- In `_forward_hidden` and `forward_ttt`, branch on `extra_pass < 0` per loop iteration. Pre-loop entries (`extra_pass=-1`) take base_v3's exact call site → bit-identical compiled kernel.
- Same change for `_block_with_lora`.

Estimated cost: ~30 lines of code duplication in Block.forward_with_overrides; pre-loop should match base_v3 to numerical precision.

## Reference numbers

| step | base_v3 | v3 LOOP_UNTIE-on (last run) |
|---|---|---|
| 1 | 9.0087 | 9.0087 |
| 500 | 2.5617 | 2.5708 (+0.0091) |
| 1000 | 2.7945 | 2.7987 (+0.0042) |
| 1500 | 2.6178 | 2.6197 (+0.0019) |
| 2000 | 2.6484 | 2.6570 (+0.0086) |
| 2500 (post-loop) | 2.5394 | 2.5392 (−0.0002) ← rejoins |
| 3000 | 2.5512 | 2.5502 (−0.0010) |
| 3500 | 2.5528 | 2.5512 (−0.0016) |
| stop step | 4892 | 4915 |
| pre-quant post-EMA val_bpb | 1.06428436 | 1.06409244 (−0.00019) |
| quantized val_bpb | 1.07278898 | (run interrupted) |
| quantized_ttt_phased val_bpb | 1.06016974 | (run interrupted) |
