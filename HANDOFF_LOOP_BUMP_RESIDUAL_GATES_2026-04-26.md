# Loop Bump Analysis And Residual Write-Gate Proposal

Date: 2026-04-26

## Context

We investigated the recurrence/loop activation in `train_gpt_v2.py`.

Current default loop configuration:

```text
NUM_LAYERS=11
NUM_LOOPS=2
LOOP_START=3
LOOP_END=5
ENABLE_LOOPING_AT=0.35
```

Loop-off effective order:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

Loop-on effective order:

```text
0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10
```

So physical layers `3,4,5` are reused 3x total once recurrence is active.

## Evidence Captured

Committed artifact branch:

```text
pr1530-exact-record
commit 6d20ec0 Add loop bump analysis artifacts
```

Important committed files:

```text
train_gpt_v2_loop_bump_analysis.py
analysis/analyze_bump_obs.py
analysis/overlay_bump_obs.py
analysis/bump_obs_efa07104/*.svg
analysis/bump_obs_overlay_hard_vs_ramp/*.svg
```

The first diagnostic showed a real hard-switch validation-probe bump:

```text
pre current loss:        2.5820
post_early current loss: 2.6740
post_late current loss:  2.5638
```

Counterfactual loop-on/off comparison:

```text
pre loop_on - loop_off:        +0.5067
post_early loop_on - loop_off: +0.0020
post_late loop_on - loop_off:  -0.1073
```

Interpretation: the looped graph is very incompatible with pre-bump weights, but the optimizer adapts quickly after the hard switch.

We then compared:

```text
hard switch at step 2200
stochastic loop probability ramp over steps 2000..2400
```

Wide-probe overlay summary:

```text
hard:    policy_min=2.528622 policy_max=2.923914
ramp400: policy_min=2.530471 policy_max=2.643447
```

The stochastic ramp substantially reduced the transient policy-loss spike, but it did not produce a meaningful longer-term gain. From `2100..2600`, expected policy loss was essentially equal:

```text
hard:    2.565395
ramp400: 2.564870
diff:   -0.000525
```

After `2400`, hard switch was slightly better:

```text
2400..2600 policy_loss:
hard:    2.541567
ramp400: 2.543665
```

Conclusion: stochastic ramp likely hides/amortizes the shock but does not address the underlying architecture mismatch enough to matter for this config.

## Proposed Next Direction: Residual Write Gates For Extra Loop Calls

Instead of stochastic scheduling, make the added repeated calls initially function-preserving.

Current loop-on repeats layers `3,4,5` two extra times. The proposal is to gate only the **extra repeated applications**, not the base call.

Conceptually:

```python
y = block(x, ...)
x = x + gate * (y - x)
```

For extra loop calls:

```text
gate = 0 -> identity, no shock
gate = 1 -> current behavior
```

This directly targets the pre-bump counterfactual gap because loop-on with zero extra gates is initially close to loop-off.

Recommended first variant:

```text
LOOP_EXTRA_WRITE_GATES=1
LOOP_EXTRA_WRITE_INIT=0.0
```

Use per-extra-pass/per-layer scalar gates:

```text
shape: [NUM_LOOPS, LOOP_END - LOOP_START + 1]
for current defaults: [2, 3]
```

This gives six tiny scalar parameters:

```text
extra pass 1: layer 3, 4, 5
extra pass 2: layer 3, 4, 5
```

Implementation sketch:

1. Construct looped effective indices with metadata marking whether a block call is base or extra.
2. In `forward_logits` and `forward_ttt`, after an extra loop block call:

```python
x_next = block(...)
gate = loop_extra_write[extra_pass_idx, loop_local_idx]
x = x + gate * (x_next - x)
```

3. Apply the same logic in parallel-lane path where the repeated layer may return `(lane0, lane1)`:

```python
new_lane0, new_lane1 = ...
lane0 = lane0 + gate * (new_lane0 - lane0)
lane1 = lane1 + gate * (new_lane1 - lane1)
```

4. Route gates to scalar AdamW and fp32 restore path.
5. Log learned gate values at train logs or final serialization.

Open design choice:

```text
unconstrained scalar gate initialized at 0
```

is simplest. If it grows above 1 or negative, that may be useful but risky. A constrained alternative is:

```text
gate = sigmoid(raw_gate)
```

but that cannot initialize exactly at 0 without saturation. For the first experiment, prefer direct scalar gates and monitor values.

## Evaluation Plan

Reuse the committed bump-analysis script or port the probe back temporarily.

Primary metrics:

```text
pre loop_on - loop_off
post_early policy/current loss
post_late loop_on loss
final pre-quant val_bpb
final quantized val_bpb
final TTT val_bpb
```

Expected success pattern:

```text
pre loop_on - loop_off shrinks materially
post_early spike shrinks
post_late/final does not regress
learned gates grow above zero if extra depth is useful
```

If gates stay near zero and final quality is unchanged or better, the extra loop depth may not be earning its compute. If gates grow and pre-gap shrinks, this is a stronger direction than stochastic scheduling.
