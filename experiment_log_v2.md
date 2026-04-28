# Experiment Log V2

## 2026-04-21 - Seq-Len Curriculum Train Recompile Fix

### Problem

- `train_gpt.py` still hit live recompiles after adding the `2048 -> 8192` seq-len bump prewarm.
- Recompile logs showed the original bad case was on the train path, not just eval:
  - train-frame `cu_seqlens` bucket changes
  - rotary cache guard mismatches after bumped prewarm / restore

### What Fixed It

- Split the train path into separate compiled train variants for the short and long seq-len regimes, while keeping one shared training loop.
- Concretely:
  - added a compiled short train wrapper for `max_seqlen=2048`
  - added a compiled long train wrapper for `max_seqlen=8192`
  - kept one shared `step_fn(...)` that takes the active compiled train callable
  - switched runtime training from short to long by swapping the compiled callable at the curriculum bump
- Also changed rotary priming to use the live microstep token length instead of the larger per-rank token count.
- Removed the post-bump-restore rotary cache reset that had been invalidating the already-correct pre-bump short regime.

### Result

- The original seq-len-curriculum train recompile problem appears fixed.
- After the change, the remaining recompiles observed in probing were eval-side recompiles, and baseline showed the same eval-side behavior.
- So the seq-len work no longer introduces a unique timed-train-path compiler regression.

### Landed Run Command

```bash
RUN_ID=seq8192_binary_bump09_recomp_probe \
TORCH_LOGS=recompiles \
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=8192 \
SEQ_LEN_BUMP_FRAC=0.90 \
EVAL_SEQ_LEN=8192 TTT_EVAL_SEQ_LEN=8192 \
TTT_BATCH_SIZE=16 \
MIN_LR=0.05 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
SKIP_GPTQ=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 2026-04-22 - Fixed Rotary Cache Regimes, GPTQ Follow-Ups, and 2xH100 Split Run

### Context

- This continues from the seq-len curriculum work above.
- Main changes from here onward started with moving away from ad hoc rotary cache resets and toward explicit fixed rotary cache regimes.
- Because post-train GPTQ / eval / TTT still had some integration bugs during this pass, the final result below was obtained with a split workflow:
  - first a normal train run to produce `final_model.pt`
  - then an eval-only rerun to iterate on GPTQ / post-quant eval / TTT without retraining
- So the full one-shot run is still TBD.

### What Changed

- Simplified rotary handling to fixed short and long regimes.
  - Instead of trying to reuse mutable rotary cache state across the curriculum handoff, the code now treats short-train and long-eval / long-train as separate fixed regimes.
  - In practice this meant:
    - explicit rotary slot `0` for short train
    - explicit rotary slot `1` for long train / long eval / long TTT
    - priming the relevant fixed cache slots up front
  - This removed a lot of cache-state drift and made compile behavior easier to reason about.

- Added a seq-len handoff / baseline comparison pass.
  - This established that the intended curriculum was:
    - train at `2048`
    - hand off to `8192` late
    - evaluate and run TTT at `8192`
  - It also gave a clearer comparison point versus the baseline script and made it easier to isolate whether regressions were from compile behavior, rope behavior, or GPTQ.

- GPTQ calibration was moved to eval seq length.
  - Calibration loader was switched to `seq_len=h.eval_seq_len`.
  - The point of this change was to let GPTQ see long-context activations rather than only the original short-train regime.
  - This turned out to be directionally useful, but it also exposed that the post-curriculum path was more sensitive to calibration-path mismatches than the baseline.

- Fixed the post-GPTQ / eval / TTT handoff memory leak.
  - After quantized eval, the compiled quantized forward wrapper still held a live reference to the eval model.
  - Then TTT deserialized a second copy of the quantized model on top of that.
  - On 2xH100 this caused a post-GPTQ TTT compile-warmup OOM.
  - Fix:
    - delete `compiled_forward_logits` as well as `eval_model` / `compiled_model`
    - reset Dynamo
    - run `gc.collect()`
    - then `torch.cuda.empty_cache()`
  - This let the TTT stage start from one live quantized model instead of two.

- Restored GPTQ calibration to isolated long rows.
  - Baseline GPTQ calibration uses isolated batch rows rather than packed attention.
  - To stay close to baseline while still targeting long context, GPTQ calibration now uses:
    - `ShuffledSequenceLoader(..., seq_len=h.eval_seq_len)`
    - plain isolated row forward
    - but still with explicit long-regime rotary arguments required by the new fixed-cache path
  - In other words:
    - same semantics as baseline for isolation
    - only intentional difference is the row length being long (`8192`)

- Added better eval-only iteration support.
  - `EVAL_ONLY=1` now defaults to loading `final_model.pt` or `ARTIFACT_DIR/final_model.pt`.
  - `FORCE_SERIALIZE_IN_EVAL_ONLY=1` forces rebuild of the quantized artifact from that checkpoint.
  - This was necessary because otherwise `EVAL_ONLY` would silently reuse an older `final_model.int6.ptz` if one already existed.
  - That made it much faster to iterate on GPTQ / eval / TTT fixes after a training run without paying the full retrain cost again.

### Important Bugs Hit Along The Way

- Phased TTT global-SGD could deadlock on 8 GPUs with the long-context eval setup.
  - Cause:
    - `GLOBAL_TTT_CHUNK_TOKENS=32768` with `EVAL_SEQ_LEN=8192` yields 4 eval-length sequences per global-TTT chunk.
    - The old distributed global-TTT sharding assumed each rank would get local work.
    - On 8 GPUs, some ranks got zero sequences for a chunk and skipped the grad `all_reduce`, while others still entered it.
    - That produced an NCCL timeout during the post-quant phased-TTT stage even though the same config could complete on 4 GPUs.
  - Fix:
    - changed global-TTT SGD so every rank executes the same sync cadence each step
    - ranks with no local slice now contribute zero grads instead of skipping collectives
  - Practical note:
    - `GLOBAL_TTT_CHUNK_TOKENS=65536` is a better fit for 8 GPUs because it usually gives 8 eval-length sequences per chunk, so all ranks do useful work
    - this improves utilization, but the code fix is what makes the run correct

- GPTQ calibration briefly crashed after reverting to baseline-like isolated rows.
  - Cause:
    - the new fixed rotary cache path requires explicit `rotary_slot`
    - calibration had been switched back to `model.forward_logits(x)` without that argument
  - Fix:
    - keep isolated long rows
    - but call the long fixed-cache path explicitly with `max_seqlen=h.eval_seq_len, rotary_slot=1`

- Post-GPTQ TTT OOM was not caused by GPTQ quality itself.
  - It was mostly a lifetime / ownership bug in the post-quant eval model handoff.

- Full integrated run still needs to be rerun cleanly.
  - The best result below used the split train + eval-only workflow because that was the fastest reliable way to debug the post-train stages.

### 2xH100 Result

- On a 2xH100 machine, with approximately 4x wallclock time to compensate relative to the original setup, we got a promising result.
- Caveats:
  - that wallclock normalization is approximate
  - the run used split train and eval scripts due to the bugs above
  - full one-shot confirmation is still TBD

### Training Command Used

```bash
RUN_ID=rotary_regular_20260421 \
TTT_ENABLED=1 \
TORCH_LOGS=recompiles \
SEED=0 \
GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 \
TRAIN_SEQ_LEN_END=8192 \
SEQ_LEN_BUMP_FRAC=0.90 \
EVAL_SEQ_LEN=8192 \
TTT_EVAL_SEQ_LEN=8192 \
MIN_LR=0.05 \
ROPE_YARN=1 \
ROPE_TRAIN_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=2400 \
TTT_BATCH_SIZE=16 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

### Final Reported Numbers

```text
5188/20000 val_loss: 2.7476 val_bpb: 1.0636
diagnostic pre-quantization post-ema val_loss:2.74332379 val_bpb:1.06199533 eval_time:13963ms
diagnostic quantized val_loss:2.87890586 val_bpb:1.11448185 eval_time:36642ms
quantized_ttt_lora val_loss:2.76373377 val_bpb:1.06992011 eval_time:2591769ms
```

### Takeaways

- The fixed short / long rotary cache regime was a useful simplification.
  - It made the seq-len curriculum and long-eval path much easier to stabilize than the previous mutable-cache approach.

- The largest remaining quality issue in this branch is GPTQ.
  - Pre-quant long-context eval is strong.
  - TTT recovers most of the post-quant drop.
  - But the raw quantized penalty is still too large relative to the baseline and still needs another pass.

- The new eval-only flow is now good enough for fast iteration on that remaining GPTQ / post-quant gap without retraining every time.

### Follow-Up Root Cause

- We found a specific reason full train+eval and eval-only were disagreeing on GPTQ quality even for the same checkpoint.
- During the runtime seq-len curriculum bump, the training path mutated `h.train_seq_len` in place from `2048` to `8192`.
- That mutation leaked past training and affected later post-training code.
- In particular, `deserialize(h, device)` builds a fresh `GPT(h)` before loading dequantized weights.
- So:
  - full-train post-GPTQ eval was deserializing into a model configured with `train_seq_len=8192`
  - eval-only post-GPTQ eval was deserializing into a fresh model configured with the correct env value `train_seq_len=2048`
- With YaRN enabled, that difference changes the rotary behavior enough to materially change quantized eval quality.
- Fix:
  - stop mutating `h.train_seq_len` during the live curriculum bump
  - only update `train_loader.max_seq_len` at runtime
- This should make full-train post-GPTQ behavior match eval-only for the same `final_model.pt`.

### TTT Recompile Follow-Up

- The 8192-token TTT path was paying avoidable timed recompiles after warmup.
- Cause:
  - warmup compiled `_fwd_ttt_inner` while the TTT model was still in train mode
  - real TTT switches the frozen base model to eval mode
  - the final partial doc batch can also introduce a smaller batch shape
- Fix:
  - call `ttt_model.eval()` before TTT compile warmup
  - warm both the normal `TTT_BATCH_SIZE` shape and the tail batch shape
- Result from eval-only rerun with `TTT_CHUNK_SIZE=48`, `TTT_BATCH_SIZE=16`:
  - original timed TTT: `958.5s`, `val_bpb=1.06346814`
  - patched timed TTT: `765.7s`, `val_bpb=1.06346775`
- Score stayed effectively identical while removing about 193s of timed TTT overhead.

### TTT Chunk-64 LR Sweep

- After the recompile fix, `TTT_CHUNK_SIZE=48` preserved the known score but remained too slow.
- `TTT_CHUNK_SIZE=64`, `TTT_BATCH_SIZE=16` was tested as the next speed/quality tradeoff.
- The timed TTT speed was near the 600s boundary and largely invariant to `TTT_LORA_LR`.
- LR sweep results from eval-only runs on the same seed-1337 checkpoint:

| TTT chunk | TTT batch | TTT LR | val_loss | val_bpb | timed TTT |
|---:|---:|---:|---:|---:|---:|
| 64 | 16 | `0.0001000` | `2.32793152` | `1.06376711` | `606.4s` |
| 64 | 16 | `0.0001333333` | `2.32860927` | `1.06407682` | `605.1s` |
| 64 | 16 | `0.0000750` | `2.32784477` | `1.06372748` | `600.6s` |

- Takeaways:
  - Linear LR scaling from chunk 48 to 64 (`1e-4 * 64/48 = 1.333e-4`) was worse.
  - Slightly lower LR (`7.5e-5`) was the best chunk-64 point tried, but only marginally better than `1e-4`.
  - Chunk 64 is close to the 600s target but does not recover chunk-48 quality.
  - The quality loss appears to be mostly from coarser online adaptation rather than a simple LR-scale mismatch.

### Current Full-Run Command

```bash
SEED=1337 \
TORCH_LOGS="recompiles" \
TORCHDYNAMO_VERBOSE=1 \
NCCL_NET=Socket \
DATA_DIR=. \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
GLOBAL_TTT_CHUNK_TOKENS=65536 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=8192 SEQ_LEN_BUMP_FRAC=0.85 \
EVAL_SEQ_LEN=8192 TTT_EVAL_SEQ_LEN=8192 \
TTT_BATCH_SIZE=16 TTT_CHUNK_SIZE=64 TTT_LORA_LR=0.000075 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
torchrun --standalone --nproc_per_node=8 \
train_gpt_v2.py 2>&1 | tee train_1024_seed${SEED}_seq8192_ttt_chunk64_bsz16_lr7p5e-5.log
```

## 2026-04-27 - Port Loop Untie + Zero-Contrib Init To `train_gpt_v2`

### Context

- The recurrence (`NUM_LOOPS>0`) introduces a discontinuity at the loop-on step: the model goes from one pass over the looped layers (`loop_start..loop_end`) to `1 + num_loops` passes overnight. With per-block `attn_scale` / `mlp_scale` / `resid_mix` shared across all passes, the extra passes contribute full-strength immediately and the train loss spikes hard.
- The earlier `loop_untie_fast` study script (`38d5531`) carried a fix that smoothed this transition: it introduced per-(extra_pass, looped_layer) copies of `attn_scale`, `mlp_scale`, `resid_mix` ("untie"), zero-initialized the attn/mlp scales so the extra passes contribute zero at the moment looping turns on, and let SGD ramp them up.
- That mechanism never made it into `train_gpt_v2.py`. This entry covers the port and validation on the same seed-1337 short bump-observation setup.

### What Changed

- Resurrected `train_gpt_v2_loop_bump_analysis_loop_untie_fast_38d5531.py` and reproduced the canonical `loop_untie_fast_s1337.log` on the current 8xH100 host.
  - Pre-loop-on per-step train_loss matched the canonical to within ~0.001 noise.
  - The new machine clocked about 1.1% slower in pre-loop-on tok/s than the canonical's machine. This was consistent across three independent repros (zero-init, hot-copy, v2 + untie) so it reads as machine-class variance, not run noise. In a 600s wallclock cap it costs ~25–50 base-rate steps relative to the canonical.

- Briefly tried a `LOOP_HOT_COPY_ON_ENABLE` variant inside the study script.
  - Behaviour: keep the untied loop-extra params, but at the loop-on transition copy each base block's current `attn_scale` / `mlp_scale` / `resid_mix` into the corresponding `loop_extra_*[ep, idx]` slot, instead of leaving the zero-init.
  - The bump-obs probe overlay showed the immediate post-loop-on spike roughly doubling: peak per-step loss 2.94 with hot-copy vs 2.74 with zero-init, and the post-loop-on window mean ~+0.035 worse with hot-copy.
  - Interestingly, by the post-late probe window (steps ~2316–2319) the train probe with hot-copy was *slightly better* than zero-init (2.557–2.560 vs 2.565–2.568), but val_loss at the next eval was worse, so the brief train-probe lead did not propagate to held-out loss.
  - Conclusion: hot-copy trades a worse transient for no real downstream benefit on this setup. Reverted.

- Ported the untie + zero-init mechanism into `train_gpt_v2.py` behind a new env flag `LOOP_UNTIE_EXTRA_SCALARS` (default `0`).
  - `encoder_loop_entries` / `decoder_loop_entries` are now `(layer_idx, extra_pass, loop_local_idx)` triples. Base passes carry `(-1, -1)` and extra passes carry the per-pass index. The flat `encoder_indices` / `decoder_indices` are derived from these so the rest of the codebase keeps its existing view.
  - Three new params on the GPT root, only when the flag is on:
    - `loop_extra_attn_scale` shape `[num_loops, loop_width, dim]`, init `zeros` (this is the spike suppressor)
    - `loop_extra_mlp_scale` same shape, init `zeros`
    - `loop_extra_resid_mix` shape `[num_loops, loop_width, 2, dim]`, init `[1, 0]` per (pass, layer) — pass-through
  - `Block.forward` and `_block_with_lora` accept optional `resid_mix` / `attn_scale` / `mlp_scale` overrides; when not provided they fall back to the base block params, so the base pass behaviour is unchanged.
  - `forward_logits` and `forward_ttt` resolve overrides via a small `_block_scalar_overrides(extra_pass, loop_local_idx)` helper for each loop entry. The helper returns `(None, None, None)` for base passes.
  - Asserts `loop_end < parallel_start_layer` when `LOOP_UNTIE_EXTRA_SCALARS` is on alongside the parallel block path, since `_parallel_block` and `_parallel_block_with_lora` are not threaded with overrides (matches the source script's assumption that recurrent layers are pre-parallel).
  - Optimizer: the three new tensors are appended to the scalar AdamW group explicitly because they live on the GPT root, not in `.blocks.named_parameters()` where the pattern-based router runs.
  - `CONTROL_TENSOR_NAME_PATTERNS` extended with the three names so `restore_fp32_params` keeps them in fp32 (their `ndim >= 2` would otherwise miss the default rule).
  - Off-state is byte-identical to the prior v2: triples collapse to `(i, -1, -1)` and the override helper returns `None` everywhere.

### Results

Same wallclock cap (`MAX_WALLCLOCK_SECONDS=600`), same bump-observation env. Step-aligned comparison:

| run | step 4000 val_loss | step 4000 val_bpb | quantized_ttt_phased loss / bpb | stop step |
|---|---:|---:|---:|---:|
| canonical (`loop_untie_fast_s1337`, old machine) | 2.4254 | 1.1083 | 2.32657 / 1.06315 | 4751 |
| zero-init repro (study script, new machine) | 2.4240 | 1.1077 | (run was killed early) | 4719 |
| hot-copy variant (study script, new machine) | 2.4243 | 1.1078 | (variant abandoned) | 4718 |
| `train_gpt_v2.py` + untie (new machine) | **2.4197** | **1.1057** | 2.32840 / 1.06398 | 4694 |

Pre-bump v2 + untie was the best of the four at step 4000 val_loss by `-0.0057` over the canonical and by `-0.004` over the same-machine zero-init repro — a submission-significant pre-bump gap. After the seq-len bump that lead narrowed, and on `quantized_ttt_phased` v2 + untie ended `+0.002` behind the canonical.

That gap is fully accounted for by three confounds stacked against the v2 run:
- Machine speed (~1.1% slower → ~25–50 fewer base-rate steps in 600s).
- Loop-on trigger: v2 has only `ENABLE_LOOPING_AT` (frac), no `ENABLE_LOOPING_STEP`. The frac trigger fired at step 2109 here vs the canonical's step 2200, so v2 spent 91 extra steps in the slower looping regime → ~15 lost wallclock-equivalent steps.
- Final stop step: 4694 vs 4751.

Same-machine vs same-machine (zero-init repro at step 4719 vs v2 + untie at step 4694), the diagnostic-stage gap is `+0.0004` on quantized — within the step-shortfall budget.

### Takeaways

- The untie + zero-init mechanism in v2 is at-or-tied with the same-machine zero-init repro after correcting for the loop-on trigger mismatch. It has the same spike-suppression effect.
- The bump-obs overlay quantified the suppression directly: zero-init holds the instantaneous post-loop-on spike to roughly half the magnitude of the warm-start variant, while still recovering to the pre-loop-on probe level within ~100 steps.
- For a clean head-to-head on the new hardware, the simplest follow-up is to run v2 + untie with `ENABLE_LOOPING_AT≈0.367` so its loop-on lands near step 2200; either that or adding `ENABLE_LOOPING_STEP` support to v2 closes the trigger confound.
- Hot-copy was a dead end on this objective: the transient cost was real and the probe-level win did not survive to val_loss.

### Run Command (`train_gpt_v2.py` + untie)

```bash
SEED=1337 \
RUN_ID=loop_untie_v2_s1337 \
ARTIFACT_DIR=artifacts/loop_untie_v2_s1337 \
NCCL_NET=Socket \
DATA_DIR=. \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
GLOBAL_TTT_CHUNK_TOKENS=65536 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=8192 SEQ_LEN_BUMP_FRAC=0.85 \
EVAL_SEQ_LEN=8192 TTT_EVAL_SEQ_LEN=8192 \
TTT_BATCH_SIZE=16 TTT_CHUNK_SIZE=64 TTT_LORA_LR=0.000075 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
LOOP_UNTIE_EXTRA_SCALARS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt_v2.py 2>&1 | tee loop_untie_v2_s1337.log
```
