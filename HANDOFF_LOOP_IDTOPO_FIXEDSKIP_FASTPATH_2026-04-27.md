# Loop Identity Topology Fixedskip Fastpath Handoff - 2026-04-27

## Current run

Run id:

```bash
loop_idtopo_fixedskip_fastpath_s1337
```

Monitor these files:

```bash
loop_idtopo_fixedskip_fastpath_s1337.log
artifacts/loop_idtopo_fixedskip_fastpath_s1337/loop_idtopo_fixedskip_fastpath_s1337.txt
```

The run was launched detached with the same apples-to-apples bump observation settings used for the earlier idtopo comparisons:

```bash
SEED=1337
ENABLE_LOOPING_STEP=2200
ENABLE_LOOPING_AT=999
LOOP_RAMP_STEPS=0
LOOP_UNTIE_EXTRA_SCALARS=1
LOOP_IDENTITY_TOPOLOGY=1
LOOP_SCALAR_LOG_ENABLED=0
BUMP_OBS_ENABLED=1
BUMP_OBS_N=20
BUMP_OBS_EVERY=1
BUMP_OBS_PROBE_BATCHES=4
BUMP_OBS_COUNTERFACTUAL=1
BUMP_OBS_LAYER_STATS=1
```

## Goal

We want identity-preserving loop introduction without changing baseline U-net behavior before the loop is active.

Hard requirements:

- Pre-loop activation should be identical to the baseline path in score and per-step runtime.
- When recurrence/loop activates, old skip connections stay semantically owned by the old base layers.
- New looped layers get their own intra-loop skips.
- The loop introduction should stay smooth/identity-preserving.
- Keep logging/probes enabled for apples-to-apples overlays.

It is acceptable for this handling to be special-cased to this topology instead of generalized.

## Reference baselines

Main comparison logs:

- `loop_untie_fast_s1337.log`
- `loop_idtopo_fixedskip_s1337.log`
- current/new: `loop_idtopo_fixedskip_fastpath_s1337.log`

Recovered source snapshots:

- `train_gpt_v2_loop_bump_analysis_38d5531_baseline.py`
  - Scalar-untie / `loop_untie_fast` implementation.
- `train_gpt_v2_loop_bump_analysis_fixedskip_run.py`
  - Exact source extracted from `artifacts/loop_idtopo_fixedskip_s1337/loop_idtopo_fixedskip_s1337.txt`.
- `train_gpt_v2_loop_bump_analysis_5041128_fixedskip_commit.py`
  - `git show 5041128:train_gpt_v2_loop_bump_analysis.py`.
  - Verified byte-for-byte equal to `train_gpt_v2_loop_bump_analysis_fixedskip_run.py`.

## Important log facts

`loop_untie_fast_s1337.log`:

- Pre-loop speed around 8.03M tok/s at step 2000.
- Train losses:
  - 2500: 2.5566
  - 3000: 2.5670
  - 3500: 2.5678
  - 4000: 2.4090
  - 4500: 2.1936
- Final step: 4751.
- Post-EMA: 2.31651199.
- Quant: 2.33645108.
- Bump at 2200: loop_on 2.7736, delta about +0.191.

`loop_idtopo_fixedskip_s1337.log`:

- Pre-loop speed around 7.97M tok/s at step 2000.
- Train losses:
  - 2500: 2.5545
  - 3000: 2.5617
  - 3500: 2.5633
  - 4000: 2.4007
  - 4500: 2.1798
- Final step: 4609.
- Post-EMA: 2.32223941.
- Quant: 2.34173671.
- Bump at 2200 was essentially zero:
  - loop_off 2.57906929
  - loop_on 2.57907360
  - delta +0.00000431

## Fixedskip semantics

This is the target semantic behavior.

Base encoder:

```text
0 1 2 3 4
```

Loop encoder:

```text
0 1 2 3 4 5a 3a 4a
```

Loop decoder:

```text
5b 3b 4b 5 6 7 8 9 10
```

Base decoder:

```text
5 6 7 8 9 10
```

Absolute skip slot ownership:

```text
slot 0: base 5 <- base 4
slot 1: base 6 <- base 3
slot 2: base 7 <- base 2
slot 3: base 8 <- base 1
slot 4: base 9 <- base 0
slot 5: aux  5b <- 4a
slot 6: aux  3b <- 3a
slot 7: aux  4b <- 5a
```

The original fixedskip implementation used helper-based absolute routing:

- encoder stores into `identity_skips[slot]`
- decoder reads by `_identity_decoder_skip_slot(...)`
- `_apply_skip(...)` applies the per-slot skip weight/gate

## Failed/intermediate attempts

Several faster-looking logical-stack variants were tried but did not reproduce fixedskip per-step loss:

- `loop_idtopo_logicalslot_s1337.log`
  - Zero bump but weak post-loop loss.
  - Step 2500 around 2.5621.
- `loop_idtopo_logicalslot_clamp_s1337.log`
  - Restored fixedskip-style aux clamp and bump behavior.
  - Post-loop tracked `loop_untie_fast` more than fixedskip.
  - Step 2500 around 2.5565, step 3000 around 2.5665.

Conclusion: the semantic target is fixed absolute skip ownership, not logical stack remapping.

## Current implementation

Current file:

```bash
train_gpt_v2_loop_bump_analysis.py
```

It was restored to the exact fixedskip source and then changed narrowly:

1. `forward_logits` has a specialized fast path for:

```python
self.looping_active and self.loop_identity_topology
```

2. `forward_ttt` has the same specialized fast path.

3. The identity fast path keeps fixedskip semantics but removes Python helper routing in the hot loop:

- literal encoder entries with skip slots
- literal decoder entries with skip slots
- direct skip application inline
- no `_identity_encoder_skip_slot(...)` calls in the active identity hot loop
- no `_identity_decoder_skip_slot(...)` calls in the active identity hot loop
- no `_apply_skip(...)` calls in the active identity hot loop

4. The non-identity and pre-loop fallback path was restored to the `untie_fast` style:

- regular `skips.append(x)`
- decoder `skips.pop()`
- direct skip weight/gate math
- no fixedskip helper routing when identity topology is inactive

5. Aux skip clamp/reset changed for speed:

Old fixedskip behavior:

- Clamp aux skip slots every inactive optimizer step.

Current behavior:

- Clamp aux skip slots immediately before counterfactual loop-on probe.
- Clamp aux skip slots immediately when training transitions from loop-off to loop-on.

Reasoning:

- Aux skip slots are not read pre-loop.
- Resetting them right before any loop-on use preserves the loop-on initialized state.
- Avoids the small GPU writes every inactive optimizer step, which likely explains part of fixedskip's pre-loop speed loss.

Validation already run:

```bash
python -m py_compile train_gpt_v2_loop_bump_analysis.py
git diff --check
```

## What to compare next

Once `loop_idtopo_fixedskip_fastpath_s1337` gets past the bump:

1. Compare pre-loop speed against `loop_untie_fast_s1337.log`.
2. Compare bump probe at step 2200 against `loop_idtopo_fixedskip_s1337.log`.
3. Compare post-loop train losses against both:
   - `loop_untie_fast_s1337.log`
   - `loop_idtopo_fixedskip_s1337.log`
4. Generate a new 3-line overlay replacing old idtopo/fixedskip with the newest run.

Expected outcome if the change worked:

- Pre-loop speed closer to `loop_untie_fast`.
- Bump still near fixedskip's near-zero delta.
- Post-loop per-step loss closer to `loop_idtopo_fixedskip_s1337.log` than to `loop_untie_fast_s1337.log`.

If the new run tracks `loop_untie_fast` loss instead, inspect for semantic drift in the literal fixed slots or aux clamp timing.

## Commit contents intended

The commit should include:

- `train_gpt_v2_loop_bump_analysis.py`
- recovered training/reference files:
  - `train_gpt_v2_loop_bump_analysis_38d5531_baseline.py`
  - `train_gpt_v2_loop_bump_analysis_5041128_fixedskip_commit.py`
  - `train_gpt_v2_loop_bump_analysis_fixedskip_run.py`
- `loop_idtopo_fixedskip_s1337.log`
- overlay analysis artifacts that were generated during this investigation:
  - `analysis/bump_obs_overlay_balanced_vs_tied_untied_fast/probe_loss_overlay.svg`
  - `analysis/bump_obs_overlay_fixedskip_vs_tied_untied_fast/probe_loss_overlay.svg`
  - `analysis/bump_obs_overlay_gatedskip_vs_tied_untied_fast/probe_loss_overlay.svg`
- this handoff file

Do not commit:

- datasets
- tokenizers
- full `artifacts/`
- the currently growing `loop_idtopo_fixedskip_fastpath_s1337.log`
