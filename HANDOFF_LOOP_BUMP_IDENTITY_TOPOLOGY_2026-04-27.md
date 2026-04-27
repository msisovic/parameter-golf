# Handoff: Loop Bump Scalar Untie and Identity Topology

Date: 2026-04-27
Repo: `/root/parameter-golf`
Branch: `pr1530-exact-record`

## Goal

Eliminate or materially reduce the loss bump when the recurrent loop is hard-enabled around step 2200.

The original issue was that the hard switch from the base path to the loop-active path caused a large policy/probe loss spike. A ramp can hide this, but the goal here was to make the hard switch itself close to identity, then let the loop learn useful extra compute.

## Current State

Committed work:

- Commit `38d5531 Add loop bump scalar untie analysis`
- Added scalar untying for recurrent extra calls in `train_gpt_v2_loop_bump_analysis.py`
- Added bump analysis and overlay tooling in `analysis/`
- Added logs for:
  - `loop_tied_s1337.log`
  - `loop_untie_s1337.log`
  - `loop_untie_fast_s1337.log`

Uncommitted work after that commit:

- `train_gpt_v2_loop_bump_analysis.py`
  - Added `NUM_ENCODER_LAYERS`
  - Added `LOOP_IDENTITY_TOPOLOGY`
  - Added identity-preserving skip handling for that topology
- `analysis/analyze_bump_obs.py`
  - Simplified `--plot` to emit only `bump_probe_loss.svg`
- New untracked files:
  - `loop_idtopo_s1337.log`
  - `loop_idtopo_s1337_analysis.txt`
  - artifacts under `artifacts/loop_idtopo_s1337/`

Do not assume the uncommitted changes are final. They are experimental but useful.

## What We Did Today

### 1. Untied Only the Cheap Recurrent Scalars

The original loop path repeated physical layers `3,4,5`, but the repeated calls reused the same scalar controls as the base physical layers. We changed this so only the recurrent extra calls get their own cheap per-channel scalar parameters, while the big matrices remain shared.

With default `NUM_LOOPS=2`, segment `3,4,5` appears three total times:

```text
base/original pass: 3 4 5
extra pass 0:       3 4 5
extra pass 1:       3 4 5
```

The added untied tensors are:

```text
loop_extra_attn_scale: shape [num_loops, loop_width, model_dim]
loop_extra_mlp_scale:  shape [num_loops, loop_width, model_dim]
loop_extra_resid_mix:  shape [num_loops, loop_width, 2, model_dim]
```

For the current config this is:

```text
num_loops=2
loop_width=3
model_dim=512
numel=12288
raw fp32 bytes=49152
```

So the untie is tiny. The weight matrices remain tied/shared.

### 2. Kept the Existing Gating/Block Formula

We did not change the block formula. The extra-call override still uses the same structure:

```text
x_in  = resid_mix[0] * x + resid_mix[1] * x0
x_out = x_in + attn_scale * attn_out
x_out = x_out + mlp_scale * mlp_out
```

This applies both in the normal forward path and TTT forward path.

The implementation assumes recurrent layers are pre-parallel when scalar untying is active. The code raises if `loop_end >= parallel_start_layer`.

### 3. Initialized Recurrent Extra Calls to Identity

For extra recurrent calls only:

```text
attn_scale = 0
mlp_scale = 0
resid_mix = [1, 0]
```

This means each extra repeated block should behave like:

```text
x_out = x
```

assuming the surrounding graph topology is unchanged.

Base physical layer scalar parameters remain as before.

### 4. Added Logging for Scalars and Bump Probes

Added controls:

```text
LOOP_SCALAR_LOG_ENABLED
LOOP_SCALAR_LOG_START_STEP
LOOP_SCALAR_LOG_END_STEP
LOOP_SCALAR_LOG_PRE_STEPS
LOOP_SCALAR_LOG_POST_STEPS
LOOP_SCALAR_LOG_EVERY
```

Important lesson: per-step full scalar JSONL is expensive. The first instrumented untied run wrote about 101 MB and cost enough wallclock to reduce the number of training steps. For fair runs, use:

```text
LOOP_SCALAR_LOG_ENABLED=0
```

Keep `BUMP_OBS_ENABLED=1` for transition comparison.

The final-control scalar dumps can still be useful, but the per-step full dump should not be on in fair wallclock comparisons.

### 5. Ran Tied Baseline vs Untied Identity Init

Fair tied baseline:

```text
RUN_ID=loop_tied_s1337
LOOP_UNTIE_EXTRA_SCALARS=0
```

Fair untied identity run:

```text
RUN_ID=loop_untie_fast_s1337
LOOP_UNTIE_EXTRA_SCALARS=1
LOOP_SCALAR_LOG_ENABLED=0
```

Both used hard switch:

```text
ENABLE_LOOPING_STEP=2200
ENABLE_LOOPING_AT=999
LOOP_RAMP_STEPS=0
```

Probe-window means:

```text
tied:
  pre         2.58239026
  post_early 2.65782949
  post_late  2.56274508

untied_fast:
  pre         2.57987920
  post_early 2.62032003
  post_late  2.56972763
```

Hard-switch spike:

```text
tied:        +0.41811 at step 2200
untied_fast: +0.19372 at step 2200
```

Final wallclock-capped eval:

```text
tied:
  stop step     4756
  post-EMA val  2.31906923
  quant val     2.33917474

untied_fast:
  stop step     4751
  post-EMA val  2.31651199
  quant val     2.33645108
```

Interpretation:

- Untied identity init cuts the immediate spike by more than half.
- The untied run also wins final val despite getting 5 fewer optimizer steps.
- The final win is likely from scalar untying itself: repeated calls can learn different write/routing policies from the base physical layers.

### 6. Diagnosed Why Identity Init Did Not Produce Zero Bump

With the original topology builder, loop activation expands the full layer list and then splits it in half.

Base split:

```text
encoder: 0 1 2 3 4
decoder: 5 6 7 8 9 10
```

Base skip pairings:

```text
5  <- 4
6  <- 3
7  <- 2
8  <- 1
9  <- 0
10 <- none
```

Loop-active expanded sequence with `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5`:

```text
0 1 2 3 4 5 3a 4a 5a 3b 4b 5b 6 7 8 9 10
```

Current old splitting makes:

```text
encoder: 0 1 2 3 4 5 3a 4a
decoder: 5a 3b 4b 5b 6 7 8 9 10
```

Resulting skip pairings:

```text
5a <- 4a
3b <- 3a
4b <- 5
5b <- 4
6  <- 3
7  <- 2
8  <- 1
9  <- 0
10 <- none
```

So even if all extra repeated blocks are exact identity, the original U-Net skip topology is not preserved. This explains the remaining bump.

### 7. Tried a Simpler Identity-Preserving Topology With a 6/5 Split

We added experimental knobs:

```text
NUM_ENCODER_LAYERS=6
LOOP_IDENTITY_TOPOLOGY=1
```

This changes base split to:

```text
base encoder: 0 1 2 3 4 5
base decoder: 6 7 8 9 10
```

Base skip pairings become:

```text
6  <- 5
7  <- 4
8  <- 3
9  <- 2
10 <- 1
0  unused
```

Loop-active topology logged by the run:

```text
encoder:[0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]
decoder:[6, 7, 8, 9, 10]
```

Important implementation detail:

- Only base encoder entries push to the original skip stack.
- Extra recurrent entries do not push to the original skip stack.
- Decoder consumes the same base skip schedule as pre-bump.

The code does this in both `forward_logits` and `forward_ttt`:

```python
if not self.loop_identity_topology or extra_pass < 0:
    skips.append(x)
```

This was a quick proof-of-concept because the recurrent segment `3,4,5` lies fully inside the base encoder when `NUM_ENCODER_LAYERS=6`.

### 8. Identity Topology Run Result

Run:

```text
RUN_ID=loop_idtopo_s1337
NUM_ENCODER_LAYERS=6
LOOP_IDENTITY_TOPOLOGY=1
LOOP_UNTIE_EXTRA_SCALARS=1
LOOP_SCALAR_LOG_ENABLED=0
```

Single-run analysis:

```text
loop_idtopo_s1337_analysis.txt
artifacts/loop_idtopo_s1337/loop_idtopo_s1337/bump_probe_loss.svg
```

Probe means:

```text
idtopo:
  pre         2.58856424
  post_early 2.58879236
  post_late  2.57858878
```

Hard-switch counterfactual:

```text
step 2200:
  loop_off 2.58700996
  loop_on  2.58698277
  delta   -0.00002719
```

Final wallclock-capped eval:

```text
stop step     4764
post-EMA val  2.32283058
quant val     2.34399248
```

Interpretation:

- The bump is essentially gone.
- The 6/5 base split is worse pre-bump and worse final-val than the prior 5/6 split runs.
- This strongly supports the diagnosis that the bump was skip-topology induced, not a failure of scalar identity init.

## Analysis Scripts

`analysis/overlay_bump_obs.py`

- Simplified to only generate:

```text
probe_loss_overlay.svg
```

Useful command:

```bash
python analysis/overlay_bump_obs.py \
  --out-dir analysis/bump_obs_overlay_tied_vs_untied \
  tied=artifacts/loop_tied_s1337/loop_tied_s1337.txt \
  untied_fast=artifacts/loop_untie_fast_s1337/loop_untie_fast_s1337.txt
```

`analysis/analyze_bump_obs.py`

- Still prints textual summaries for probes, counterfactuals, layer stats, and scalar records if provided.
- `--plot` now only writes `bump_probe_loss.svg`.

Useful command:

```bash
python analysis/analyze_bump_obs.py --plot \
  artifacts/loop_idtopo_s1337/loop_idtopo_s1337.txt \
  | tee loop_idtopo_s1337_analysis.txt
```

## Commands Used

Fast untied fair run:

```bash
SEED=1337 \
RUN_ID=loop_untie_fast_s1337 \
ARTIFACT_DIR=artifacts/loop_untie_fast_s1337 \
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
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 FP8_LM_HEAD=1 \
ENABLE_LOOPING_STEP=2200 ENABLE_LOOPING_AT=999 LOOP_RAMP_STEPS=0 \
LOOP_UNTIE_EXTRA_SCALARS=1 LOOP_SCALAR_LOG_ENABLED=0 \
BUMP_OBS_ENABLED=1 BUMP_OBS_N=20 BUMP_OBS_EVERY=1 \
BUMP_OBS_PROBE_BATCHES=4 BUMP_OBS_COUNTERFACTUAL=1 BUMP_OBS_LAYER_STATS=1 \
torchrun --standalone --nproc_per_node=8 \
train_gpt_v2_loop_bump_analysis.py 2>&1 | tee loop_untie_fast_s1337.log
```

Identity-topology 6/5 split run:

```bash
SEED=1337 \
RUN_ID=loop_idtopo_s1337 \
ARTIFACT_DIR=artifacts/loop_idtopo_s1337 \
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
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 FP8_LM_HEAD=1 \
NUM_ENCODER_LAYERS=6 LOOP_IDENTITY_TOPOLOGY=1 \
ENABLE_LOOPING_STEP=2200 ENABLE_LOOPING_AT=999 LOOP_RAMP_STEPS=0 \
LOOP_UNTIE_EXTRA_SCALARS=1 LOOP_SCALAR_LOG_ENABLED=0 \
BUMP_OBS_ENABLED=1 BUMP_OBS_N=20 BUMP_OBS_EVERY=1 \
BUMP_OBS_PROBE_BATCHES=4 BUMP_OBS_COUNTERFACTUAL=1 BUMP_OBS_LAYER_STATS=1 \
torchrun --standalone --nproc_per_node=8 \
train_gpt_v2_loop_bump_analysis.py 2>&1 | tee loop_idtopo_s1337.log
```

## Suggested Next Direction

The 6/5 split proved that identity-preserving skip topology can remove the hard-switch bump, but the changed base U-Net split hurts quality.

The next experiment should preserve the old base split:

```text
base encoder: 0 1 2 3 4
base decoder: 5 6 7 8 9 10
```

while still making loop activation identity-preserving. That means the original skip pairings must remain:

```text
5  <- 4
6  <- 3
7  <- 2
8  <- 1
9  <- 0
10 <- none
```

One possible loop-active schedule:

```text
encoder-ish side:
0 1 2 3 4 5a 3a 4a 5b

decoder-ish side:
3b 4b 5 6 7 8 9 10
```

But the important invariant is not where the entries sit visually. The invariant is:

```text
Only base 0..4 push the original U-Net skip stack.
Only base decoder 5..9 consume the original U-Net skip stack.
Extra loop calls do not touch the original skip stack.
```

Then with:

```text
extra attn_scale = 0
extra mlp_scale = 0
extra resid_mix = [1, 0]
```

the loop-active forward should match the base forward exactly.

If auxiliary recurrent skips are desired, add a separate auxiliary skip stack with zero-initialized lambdas. Do not reuse or consume the base skip stack.

## Verification Checklist for Next Implementation

1. Print/log base and loop-active logical execution entries.
2. Print/log original skip producer/consumer mapping.
3. Add a small topology smoke check:

```text
base:      5<-4, 6<-3, 7<-2, 8<-1, 9<-0
loop-on:   same for base decoder entries
```

4. Run a small model/logits identity test if feasible:

```text
looping_active=False logits
looping_active=True logits
max_abs_diff should be near numerical noise
```

5. Run full hard-switch experiment with:

```text
LOOP_UNTIE_EXTRA_SCALARS=1
LOOP_SCALAR_LOG_ENABLED=0
BUMP_OBS_ENABLED=1
LOOP_RAMP_STEPS=0
ENABLE_LOOPING_STEP=2200
```

6. Overlay `probe_loss_overlay.svg` against:

```text
tied
untied_fast
idtopo_6_5
oldsplit_identity_topology
```

## Main Takeaways

- Scalar untying helps both the bump and final quality.
- Identity init of recurrent extra calls is not sufficient if the U-Net skip topology changes.
- The original remaining bump was mainly caused by reshuffled skip producer/consumer pairs.
- A topology that preserves base skip pairings can make the hard-switch counterfactual delta essentially zero.
- The quick 6/5 split achieves zero bump but hurts base quality; the likely best version is old 5/6 split plus explicit skip-stack ownership.
