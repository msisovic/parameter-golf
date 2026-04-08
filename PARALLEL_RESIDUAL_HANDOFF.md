# Parallel Residual Handoff

This file records what we know so far about the parallel residual implementation in `train_gpt.py`, what has been tried, what actually worked in representative benchmarks, and what the next exact-semantic optimization path likely is.

## Current Best State

- Branch: `pr-1394-baseline-parallel-residuals`
- Current pushed commit with the only proven speedup: `5f80168`
- Commit message: `Overlap parallel lane0 attn write`

That change is semantics-preserving and is behind:

```bash
PARALLEL_OVERLAP_ATTN_LANE0=1
```

Recommended current run:

```bash
SEED=1337 PARALLEL_RESIDUAL=1 PARALLEL_START_LAYER=4 PARALLEL_OVERLAP_ATTN_LANE0=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Representative Benchmark Harness

Do not trust microbenches for final decisions here. They were misleading.

Use the real harness:

```bash
timeout 170s env \
  SEED=1337 \
  PARALLEL_RESIDUAL=1 \
  PARALLEL_START_LAYER=4 \
  TRAIN_LOG_EVERY=100 \
  VAL_LOSS_EVERY=0 \
  MAX_WALLCLOCK_SECONDS=120 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Baseline comparison:

```bash
timeout 170s env \
  SEED=1337 \
  TRAIN_LOG_EVERY=100 \
  VAL_LOSS_EVERY=0 \
  MAX_WALLCLOCK_SECONDS=120 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Global train batch size is:

- `TRAIN_BATCH_TOKENS = 786432`

Step time formula:

- `step_time_ms = 786432 / tok_per_sec * 1000`

## Current Measured Gap To Baseline

On the reverted fast codepath before any new experiments:

- Baseline, no parallel residual:
  - step 200: about `7.68M tok/s`
  - step 600: about `7.37M tok/s`
- Parallel residual, `PARALLEL_START_LAYER=4`, no overlap:
  - step 200: about `7.11M tok/s`
  - step 600: about `6.53M tok/s`

Equivalent step times:

- Baseline at step 600: `106.65 ms`
- Parallel at step 600: `120.35 ms`

So the post-loop gap to baseline was about:

- `13.70 ms/step`

## What Worked

### 1. Overlap `lane0` post-attn write under MLP

This is the change in commit `5f80168`.

The logic:

- After attention, `lane1_mid` is immediately needed because MLP reads `lane1`.
- `lane0_mid` is not needed until the MLP post-write.
- So:
  - compute `lane1` post-attn update on the main stream
  - launch `lane0` post-attn update on a side stream
  - run MLP
  - join before the MLP post-write

This is implemented in `_parallel_block`.

It was checked for exactness:

- same weights
- same inputs
- max diff `0.0`

Representative 8-GPU A/B:

- No overlap:
  - step 200: `7113885 tok/s`
  - step 300: `7097582 tok/s`
  - step 400: `7099055 tok/s`
  - step 500: `6981769 tok/s`
  - step 600: `6534448 tok/s`

- With `PARALLEL_OVERLAP_ATTN_LANE0=1`:
  - step 200: `7164647 tok/s`
  - step 300: `7160793 tok/s`
  - step 400: `7155403 tok/s`
  - step 500: `7108384 tok/s`
  - step 600: `6742359 tok/s`

Recovered step time:

- step 600: `120.35 ms -> 116.64 ms`
- recovered `3.71 ms`
- recovered about `27%` of the post-loop gap to baseline

## What Did Not Work

### 1. Packed-lane representation

Commit `b94fbc4` packed the two lanes into one stacked tensor.

It looked faster in local block-level timing but was much slower end-to-end.

Representative 8-GPU result:

- packed-lane head dropped to roughly `4.2M-4.6M tok/s`
- pre-pack parallel code was around `7.1M tok/s`

This was reverted in:

- `6a0e4ff` `Revert packed parallel residual lanes`

Important lesson:

- local helper microbench can be directionally wrong for this model
- only trust representative 8-GPU `torchrun`

### 2. Removing the second fresh `x0` mix

Experiment:

- `PARALLEL_MLP_MIX_X0=0`

This changed semantics and hurt quality badly in a real run. It was abandoned.

### 3. Local packing/fusion attempts that kept the whole graph similar

Tried:

- local block packing only
- local skip packing
- small Triton fused update kernels
- `foreach` multi-tensor updates

None produced a real 8-GPU win.

### 4. Restricting where parallel residual is active

Tried ideas like:

- only while looping
- only on loop segment

These are algorithmic tradeoffs, not real per-layer savings.

User explicitly did not want this counted as a win.

Also `parallel_only_when_looping` hit DDP unused-parameter issues.

### 5. `skip1` overlap under attention

Idea:

- in decoder parallel blocks, apply `skip0` immediately
- compute `skip1` on a side stream while attention runs
- wait only when `lane1` is needed

This was made exact after fixing a scheduling bug, but still did not beat the current best.

Representative result with `PARALLEL_OVERLAP_ATTN_LANE0=1 PARALLEL_OVERLAP_SKIP1=1`:

- step 200: `7135910 tok/s`
- step 300: `7142329 tok/s`
- step 400: `7140784 tok/s`
- step 500: `7096146 tok/s`
- step 600: `6731333 tok/s`

That is slightly worse than `PARALLEL_OVERLAP_ATTN_LANE0=1` alone.

### 6. Encoder-side `lane1` MLP-write pipeline into next attention

Idea:

- after MLP, `lane0_out` is needed immediately by next attention
- `lane1_out` is not needed until next block’s MLP read
- so defer `lane1_out` and let next attention start from `lane0_out`

This was implemented with a side stream and exactness checks, but did not beat the current best.

Representative result with `PARALLEL_OVERLAP_ATTN_LANE0=1 PARALLEL_OVERLAP_MLP_LANE1=1`:

- step 200: `7135151 tok/s`
- step 300: `7138666 tok/s`
- step 400: `7137309 tok/s`
- step 500: `7087846 tok/s`
- step 600: `6563570 tok/s`

Again, slightly worse than `PARALLEL_OVERLAP_ATTN_LANE0=1` alone.

### 7. Fusing `skip1` into `lane1_mid`

Idea:

- avoid materializing `lane1_skip`
- fold `skip1` directly into the `lane1` post-attn update

This is algebraically plausible, but the direct implementation tested was not bitwise-equivalent:

- max diff observed: `0.015625`

It was dropped.

## Why The Dependency Graph Is Asymmetric In Practice

At a high level there is a symmetry:

- after attention:
  - `lane1_mid` is immediately needed by MLP
  - `lane0_mid` is slack until MLP post-write
- after MLP:
  - `lane0_out` is immediately needed by next attention
  - `lane1_out` is slack until next block’s MLP read

So on paper both directions offer a delayed-lane opportunity.

But in practice they are not equally favorable:

1. `lane0_mid` is hidden under a relatively large MLP region.
2. `lane1_out` only has the next block’s attention as its slack window, which is shorter and less forgiving.
3. Decoder skip handling tends to require both lanes sooner than the nice “slack lane” story would suggest.
4. Stream/event overhead is not free. The smaller the hidden work and the shorter the window, the easier it is for overlap overhead to wipe out the gain.

So:

- the `lane0` overlap was a real win
- the `lane1` candidates did not pay for themselves

## Important Bug Found During Experiments

During the later overlap experiments, a scheduling bug was found:

- I had been waiting on deferred `lane1` too early in `_parallel_block`
- that reduced or destroyed the intended overlap

That wait was moved to the actual dependency point during testing.

Even after that fix:

- `skip1` overlap still did not win
- encoder `lane1` pipeline still did not win

So the negative result is more trustworthy now.

## What The Profiler Suggested

Single-GPU compiled profiling showed:

- the extra cost is not from doubled attention/MLP large kernels
- the cost is mostly in the surrounding compiled graph shell and Triton pointwise/reduction work
- flash attention count did not blow up in proportion to the slowdown

This supports the view that the remaining tax is mostly:

- lane-state traffic
- wrapper kernel overhead
- extra full-tensor reads/writes

## Current Exact-Semantics Conclusions

1. The easy stream-overlap win was `lane0` post-attn under MLP.
2. The obvious “symmetric” next overlaps do not seem to pay off.
3. We likely need a representation or kernel change, not another simple scheduling trick.

## Best Next Direction

The next serious exact-semantic path is probably:

- custom fused kernel or custom op for the parallel shell

More specifically:

1. Keep the current semantics exactly.
2. Stop trying to materialize every intermediate lane eagerly.
3. Build a kernel or opaque op that fuses more of:
   - post-attn lane updates
   - any deferred lane state
   - post-MLP lane updates

The main goal is not “new math.”
The main goal is:

- reduce intermediate lane materialization
- reduce wrapper kernel count
- lower memory traffic

## If Continuing Later

Start from:

- commit `5f80168`

Run:

```bash
SEED=1337 PARALLEL_RESIDUAL=1 PARALLEL_START_LAYER=4 PARALLEL_OVERLAP_ATTN_LANE0=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Compare against:

```bash
SEED=1337 PARALLEL_RESIDUAL=1 PARALLEL_START_LAYER=4 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

And baseline:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Remember:

- do not trust block microbenchmarks as decision-makers
- do not count “parallel active on fewer layers” as a win
- only exact-semantic changes count
- keep benchmarking with the real 8-GPU harness

## Current Status At Time Of Writing

- Pushed win: `5f80168`
- Worktree should be restored to match `HEAD`
- Only untracked generated files are expected:
  - `final_model.pt`
  - `final_model.int6.ptz`
