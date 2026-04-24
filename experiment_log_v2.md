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
