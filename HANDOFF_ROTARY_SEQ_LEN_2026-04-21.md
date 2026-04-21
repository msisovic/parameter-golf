## Context

This handoff captures the current state of the `train_gpt.py` seq-len bump / RoPE / YaRN / recompile investigation as of 2026-04-21.

Current branch:

- `pr1530-exact-record`

Current relevant files:

- `train_gpt.py`
- `train_gpt_older.py`
- `modded_nanogpt.py`

`train_gpt.py` has been restored to the earlier experimental seq-len-bump version that uses fixed short/long rotary slots and separate compiled short/long wrappers. `train_gpt_old.py` was deleted after restoring from it.

## What Was Done

1. `train_gpt_older.py` was fetched from commit:

- `035bf9ed89fcc79c2fd6ef957d62c39cbab19244`

Command used:

```bash
git show 035bf9ed89fcc79c2fd6ef957d62c39cbab19244:train_gpt.py > train_gpt_older.py
```

2. `modded_nanogpt.py` was added as an external baseline reference.

3. `train_gpt.py` was restored to the earlier experimental version discussed in this thread, and `train_gpt_old.py` was deleted.

## Current `train_gpt.py` Shape

The current `train_gpt.py` is **not** the older known-good file. It is the experimental bump version with:

- `Rotary` fixed-slot support:
  - `_fixed_short_cos/_sin`
  - `_fixed_long_cos/_sin`
  - `prime_fixed_slot(...)`
  - `forward(..., yarn_seq_len=None, cache_slot=None)`
- attention plumbing for `rotary_slot`
- separate compiled short/long train wrappers
- separate compiled short/long eval-logits wrappers
- no YaRN-aware attention softmax correction

Important locations:

- `Rotary`: `train_gpt.py:562`
- attention callsite: `train_gpt.py:691`
- fixed-slot priming: `train_gpt.py:2478`
- short/long compiled train wrappers: `train_gpt.py:2513`
- eval path with fixed short/long wrappers: `train_gpt.py:2718` onward
- current TTT priming: `train_gpt.py:2954`

## Comparison: `train_gpt_older.py` vs Current `train_gpt.py`

### 1. Rotary / Attention

Older file:

- one rotary cache
- keyed by YaRN scale
- explicit forcing via `_force_yarn_seq_len`
- explicit helper `_set_rotary_state(...)`
- YaRN-aware attention softmax scale

Relevant lines:

- `train_gpt_older.py:570`
- `train_gpt_older.py:587`
- `train_gpt_older.py:600`
- `train_gpt_older.py:673`
- `train_gpt_older.py:691`

Current file:

- fixed short/long rotary slots
- wrapper passes `rotary_slot`
- no YaRN-aware attention softmax scale

Relevant lines:

- `train_gpt.py:562`
- `train_gpt.py:600`
- `train_gpt.py:613`
- `train_gpt.py:691`

### 2. Train Compile / Warmup Flow

Older file:

- one compiled train path
- one optional compiled looping path
- `_set_rotary_state(...)` called at explicit regime boundaries:
  - startup priming
  - warmup
  - validation
  - layer-loop activation
  - seq-len bump

Relevant lines:

- `train_gpt_older.py:2444`
- `train_gpt_older.py:2535`
- `train_gpt_older.py:2642`
- `train_gpt_older.py:2706`
- `train_gpt_older.py:2856`
- `train_gpt_older.py:2893`
- `train_gpt_older.py:2918`

Current file:

- separate compiled short/long train wrappers
- fixed slots primed once
- no `_set_rotary_state(...)`
- seq-len bump selects wrapper rather than mutating rotary regime

Relevant lines:

- `train_gpt.py:2478`
- `train_gpt.py:2513`
- `train_gpt.py:2689`
- `train_gpt.py:2802`

### 3. Eval Path

Older file:

- only compiles eval logits when shape-stable enough
- otherwise leaves long-context eval logits eager
- explicitly switches rotary state into eval regime and then back into train regime

Relevant lines:

- `train_gpt_older.py:2456`
- `train_gpt_older.py:2543`
- `train_gpt_older.py:2856`
- `train_gpt_older.py:2869`
- `train_gpt_older.py:3025`
- `train_gpt_older.py:3056`

Current file:

- always compiles separate short and long eval-logits wrappers
- `eval_val(..., fixed_max_seqlen=None)` can force logical regime from wrapper
- relies on slot selection, not state flipping

Relevant lines:

- `train_gpt.py:1977`
- `train_gpt.py:2539`
- `train_gpt.py:2718`
- `train_gpt.py:2857`
- `train_gpt.py:2910`

### 4. TTT Setup

Older file:

- forces YaRN regime to `TTT_EVAL_SEQ_LEN`
- uses `_set_rotary_state(...)` before TTT compile

Relevant lines:

- `train_gpt_older.py:3094`
- `train_gpt_older.py:3100`
- `train_gpt_older.py:3102`

Current file:

- when `ROPE_YARN=1`, primes TTT using:
  - `train_batch_tokens // grad_accum_steps`
- this differs semantically from the older file

Relevant lines:

- `train_gpt.py:2954`
- `train_gpt.py:2960`

This TTT drift is real and should be treated as a bug relative to the older file.

## What The Recompile Investigation Already Established

### A. The rotary-specific recompiles were real

When rotary cache identity depended on mutable in-forward module state like:

- `_yarn_scale_cached`
- `_seq_len_cached`
- `_cos_cached`

`torch.compile` guarded on those values and recompiled when they changed.

### B. Fixed-slot rotary removed the rotary guard failures, but not all recompiles

With the fixed-slot version now in `train_gpt.py`, later runs no longer showed the earlier rotary-specific guard failures.

The remaining recompiles in logs were instead from:

- `cu_seqlens` size mismatches
- `base_model.looping_active`
- optimizer helper shape changes

So the fixed-slot approach improved the rotary-specific part, but did not eliminate all recompiles.

### C. The older file almost certainly stayed stable by avoiding lazy rotary-state churn in the hot path

The older strategy is:

- mutate rotary state only at explicit boundaries
- keep one current active regime
- do not try to multiplex multiple rotary regimes through one compiled hot path using in-forward cache logic

## RoPE / Varlen / Packed Sequences Conclusions

These were clarified during investigation:

1. RoPE cache is typically a monotonic table over flat positions `0..L-1`.

2. Packed varlen batches **do not** require a separate rotary cache per packed sequence.

3. A single monotonic cache works across different packs because:

- rotary only depends on flat positions
- `cu_seqlens` / segment boundaries are enforced by attention, not by rotary
- within a segment, shifting all positions by a constant offset preserves RoPE relative geometry

4. Therefore the cache does **not** depend on exact boundaries.

It depends on:

- required packed length capacity
- active RoPE / YaRN regime

5. In the current implementation family, cache length must be large enough for the **packed flattened token count**, not merely `TRAIN_SEQ_LEN`.

6. A tiny `2048` cache would only work if the code were rewritten to use per-sequence local positions rather than flat packed positions.

## What `modded_nanogpt.py` Does

`modded_nanogpt.py` is useful because it cleanly separates YaRN from lazy rotary caching.

Relevant lines:

- `Yarn` class: `modded_nanogpt.py:981`
- precompute/reset: `modded_nanogpt.py:998`
- schedule transition update: `modded_nanogpt.py:1030`
- attention reads `yarn.rotary(...)` and `yarn.attn_scale`: `modded_nanogpt.py:1075`
- scheduler updates YaRN only at explicit boundaries: `modded_nanogpt.py:1766`
- whole model compiled once: `modded_nanogpt.py:1907`

Important observations:

1. It sizes YaRN tables to the **packed local token budget**, not to per-sequence max length.

Model construction uses:

- `max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)`

at:

- `modded_nanogpt.py:1894`

2. `Yarn.reset()` precomputes the tables up front.

3. `Yarn.apply(...)` updates the tables only at explicit schedule boundaries.

4. Attention never lazily rebuilds rotary state inside the compiled hot path.

That is the main reason it avoids our cache headaches.

## Best Current Understanding Of Minimal Correct Port From `train_gpt_older.py`

If the goal is:

- known-good semantics
- known-good compile behavior
- minimum conceptual complexity

then the smallest defensible port is:

1. Restore the older `Rotary` implementation from `train_gpt_older.py`.
2. Restore YaRN-aware attention softmax scaling.
3. Restore `_set_rotary_state(...)`.
4. Call `_set_rotary_state(...)` only at regime boundaries:
   - before validation
   - after validation, back to train regime
   - when looping turns on
   - when seq-len bump happens
   - before TTT compile
5. Restore older TTT priming semantics at `TTT_EVAL_SEQ_LEN`.
6. Restore `_should_compile_eval_logits(...)` behavior so long-context eval logits are not strict-compiled when unstable.

What not to keep if taking that route:

- fixed rotary slots
- extra slot abstractions
- approximations of older behavior

## Alternative Clean Direction Inspired By `modded_nanogpt.py`

The other clean direction is:

- make YaRN a first-class precomputed object
- keep precomputed RoPE buffers
- update those buffers only at explicit schedule boundaries
- keep attention forward a pure consumer of already-prepared tensors

This is probably the cleanest long-term design, but it is a more structural refactor than simply porting the older proven behavior.

## Recommended Next Step

If resuming with the least risk:

1. Revert current fixed-slot rotary strategy.
2. Port the older `Rotary` + `_set_rotary_state(...)` + attention softmax scale + TTT setup.
3. Keep the rest of the current seq-len curriculum logic only where it does not fight that model.
4. Then rerun the recompilation probe and inspect whether remaining recompiles are only from:
   - `cu_seqlens`
   - `looping_active`
   - other known non-rotary sources

## Git State At Time Of Handoff

Expected relevant changes to commit:

- `train_gpt.py`
- `train_gpt_older.py`
- `modded_nanogpt.py`
- this handoff file

Do not include:

- `run_logs/`
