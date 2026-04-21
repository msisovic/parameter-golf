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
MIN_LR=0.05 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
SKIP_GPTQ=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
