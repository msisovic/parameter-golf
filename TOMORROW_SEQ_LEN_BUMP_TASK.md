# Seq-Len Bump Follow-Up

## Goal

Keep exact baseline-equivalent behavior before the seq-len bump, while still getting a prewarmed bumped train regime for a single `2048 -> 8192` binary bump.

## Current State

- `train_gpt.py` was reset to the PR-1530 baseline path, then modified minimally for:
  - `TRAIN_SEQ_LEN_END`
  - `SEQ_LEN_BUMP_FRAC`
  - one-time binary bump
  - compiled short/long eval logits variants
  - in-place bumped train prewarm
- `train_gpt_base.py` is a local copy of the exact baseline control script.
- Current commit before the latest bumped-train-prewarm experiments:
  - `b5cc7cc` `Add binary seq-len bump baseline variant`

## What We Learned

### Confirmed

- Earlier bad results were not just "cache pollution from a second model."
- The active pre-bump path had drifted from baseline in several earlier attempts:
  - loader behavior
  - rotary/attention plumbing
  - train loop structure
- Resetting to the baseline file and adding only a minimal binary bump was the right direction.

### Current Problem

- Adding train-side prewarm for the bumped `8192` regime causes the already-correct pre-bump `2048` warmup to stop being sufficient.
- Symptom:
  - step `0/1` training compile cost becomes huge again
  - previously baseline-equivalent pre-bump startup no longer holds
- This still happens even after raising:
  - `torch._dynamo.config.cache_size_limit = 512`

## Important Conclusion

The current issue is not explained just by "cache too small."

More likely causes:

- the same compiled train frame is being specialized across too many guard combinations:
  - `2048` non-loop
  - `2048` loop
  - `8192` non-loop
  - `8192` loop
- nested frames may be recompiling independently
- re-entering the pre-bump `2048` regime after warming `8192` may not match an already-warmed variant exactly

## Next Debugging Step

Do a very short probe on the current code with recompilation logging enabled, specifically to learn what recompiles at the first live train step after bumped prewarm.

### Probe Command

```bash
RUN_ID=seq8192_binary_bump09_recomp_probe \
TORCH_LOGS=recompiles \
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=8192 \
SEQ_LEN_BUMP_FRAC=0.90 \
EVAL_SEQ_LEN=8192 TTT_EVAL_SEQ_LEN=8192 \
MIN_LR=0.05 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
TTT_ENABLED=0 SKIP_GPTQ=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Questions To Answer

1. Which frame recompiles at the first live train step?
2. Is it the top-level compiled train forward, or a nested frame?
3. Are the misses caused by:
   - `max_seqlen`
   - `cu_seqlens` size
   - `looping_active`
   - something else
4. Does the recompilation happen even if bumped prewarm is reduced to the absolute minimum?

## Constraints

- Do not hand-wave "cache pollution" without logs.
- Keep baseline equivalence as the main criterion.
- Avoid changing eval/train behavior unnecessarily just to silence compiler issues.
- If a fix changes baseline semantics before the bump, treat it as suspect.
