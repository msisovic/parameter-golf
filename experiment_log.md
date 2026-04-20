# Experiment Log

## 2026-04-19 - Seq-Len Curriculum + Recompile Investigation

### Goal

Evaluate `TRAIN_SEQ_LEN=2048 -> 4096` curriculum, understand the suspicious slowdown near the bump, and compare post-training eval / quantized eval / TTT against the non-curriculum baseline.

### Key Training Command

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Recompile Finding

- The large slowdown near the seq-len bump was real and was caused by a live recompile around the transition to `4096`.
- Recompile logs showed `_train_forward_bump_loop` recompiling when `cu_seqlens` changed from length `192 -> 256`.
- Simply adding `256` to the synthetic warmup bucket list was not sufficient by itself.
- The working fix was to keep the existing synthetic startup warmup, and additionally prime the live post-bump regime with real loader-produced batches right after the seq-len switch, excluding that one-time cost from the training timer.

### Result of Hybrid Warmup Fix

- After the fix, there were no recompiles after actual training started and no recompiles after the live seq-len bump.
- The fixed curriculum run reached `4770` steps before the wallclock cap, versus `4533` in the original suspicious run.

500-step training logs from the fixed curriculum run:

- `500`: `train_loss 3.2561`, `train_time 0.8m`, `tok/s 8220365`
- `1000`: `train_loss 3.0031`, `train_time 1.6m`, `tok/s 8181105`
- `1500`: `train_loss 3.0132`, `train_time 2.4m`, `tok/s 8177328`
- `2000`: `train_loss 2.9752`, `train_time 3.2m`, `tok/s 8178470`
- `2500`: `train_loss 3.0573`, `train_time 4.3m`, `tok/s 7620238`
- `3000`: `train_loss 2.8952`, `train_time 5.6m`, `tok/s 6998953`
- `3500`: `train_loss 2.9624`, `train_time 6.8m`, `tok/s 6758070`
- `4000`: `train_loss 2.8759`, `train_time 8.0m`, `tok/s 6586898`
- `4500`: `train_loss 2.8528`, `train_time 9.1m`, `tok/s 6449402`

Other key events:

- Looping enabled at step `2137`
- Seq-len bump to `4096` at step `3943`
- Stopped early at step `4770` due to wallclock cap

### Validation / Quantized / TTT Results

#### Curriculum Run, Validation At 2048

- End-of-training val: `2.7727` bpb `1.0734`
- Diagnostic pre-quantization post-EMA val: `2.77139888` bpb `1.07286029`
- Diagnostic quantized val: `2.80068618` bpb `1.08419796`
- Quantized TTT LoRA val: `2.77292349` bpb `1.07348509`

This remains the best TTT result observed so far.

#### Eval-Only From Saved Artifact, Validation At 4096

Command pattern:

```bash
EVAL_ONLY_PATH=final_model.pt \
EVAL_SEQ_LEN=4096 \
TTT_EVAL_SEQ_LEN=4096 \
...
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Pre/post quantized eval at `4096`:

- Diagnostic pre-quantization post-EMA val: `2.76580509` bpb `1.07069019`
- Diagnostic quantized val: `2.79578644` bpb `1.08229648`

This confirms that the quantized `4096` eval baseline is better than the quantized `2048` eval baseline.

### 4096 TTT Investigation

- `TTT_EVAL_SEQ_LEN=4096` with default `TTT_BATCH_SIZE=64` OOMed during TTT compile warmup.
- Reducing to `TTT_BATCH_SIZE=16` allowed `4096` TTT to complete.

Results at `TTT_BATCH_SIZE=16`:

- `TTT_LORA_LR=1e-4`: `2.77790876` bpb `1.07541132`
- `TTT_LORA_LR=5e-5`: `2.78081074` bpb `1.07653476`
- `TTT_LORA_LR=2.5e-5`: `2.78751265` bpb `1.07912927`
- `TTT_LORA_LR=1.25e-4`: `2.77826573` bpb `1.07554951`

Current conclusion:

- `TTT_LORA_LR=1e-4` is the best `4096` TTT setting tried so far.
- `4096` wins before TTT, but does not retain that lead after TTT.
- Best `2048` TTT result (`2.77292349`) is still better than best `4096` TTT result (`2.77790876`).
- The likely interpretation is that long-context TTT currently gives a smaller gain than 2048 TTT in this regime, even though the base and quantized `4096` evals are stronger.

### Working Conclusions

- The seq-len curriculum recompile issue appears fixed well enough to continue experimenting with this regime.
- The limiting issue is no longer the live seq-len bump recompile.
- The next priority should be improving the training baseline / quantized baseline under seq-len curriculum rather than continuing to chase TTT gains first.
- For now, it is reasonable to treat long-context TTT as a smaller win in this regime and focus on beating the non-curriculum training baseline.

## 2026-04-19 - YARN Investigation And Eval-State Fix

### What Was Wrong

- With `ROPE_YARN=1`, a real runtime recompile bug was found: training forwards were allowing runtime YARN scale to vary because the forced YARN length was not being pinned on the rotary modules.
- Fix: `_set_rotary_state(...)` now sets `rotary._force_yarn_seq_len = yarn_seq_len`.
- A second issue was found in post-training diagnostics:
  - `diagnostic pre-quantization post-ema` was evaluating the in-memory model without explicitly resetting rotary/YARN state to `eval_seq_len`
  - `diagnostic quantized` used a fresh deserialized model
  - this could make the two diagnostics measure different rotary/YARN regimes
- Fix: both pre-quantized and quantized diagnostics now explicitly call `_set_rotary_state(..., yarn_seq_len=h.eval_seq_len)` before eval.

### YARN Run Without LR Floor, Before Eval-State Fix

Command:

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.7 \
ROPE_YARN=1 \
CURRICULUM_MONITOR_STEPS=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Observed metrics before the eval-state fix:

- End-of-training val: `2.7846`
- Diagnostic pre-quantization post-EMA val: `2.77005797`
- Diagnostic quantized val: `3.02985493`

This quantized result was suspiciously bad and is no longer considered trustworthy due to the diagnostic rotary/YARN state mismatch.

### YARN Run Without LR Floor, After Eval-State Fix

Same command as above, but with the diagnostic eval-state fix applied.

500-step logs:

- `500`: `train_loss 3.2552`, `train_time 0.8m`, `tok/s 8199726`
- `1000`: `train_loss 3.0093`, `train_time 1.6m`, `tok/s 8161478`
- `1500`: `train_loss 3.0135`, `train_time 2.4m`, `tok/s 8152870`
- `2000`: `train_loss 2.9736`, `train_time 3.2m`, `tok/s 8153124`
- `2500`: `train_loss 3.0585`, `train_time 4.3m`, `tok/s 7634081`
- `3000`: `train_loss 2.9005`, `train_time 5.5m`, `tok/s 7195419`
- `3500`: `train_loss 2.9648`, `train_time 6.6m`, `tok/s 6912457`
- `4000`: `train_loss 2.8778`, `train_time 7.8m`, `tok/s 6701395`
- `4500`: `train_loss 2.8606`, `train_time 9.0m`, `tok/s 6545224`

Transition monitors:

- Loop monitor:
  - pre: `loss_avg 3.0726`, `step_ms_avg 36.3`
  - post: `loss_avg 3.1308`, `step_ms_avg 70.2`
- Seq-len bump monitor:
  - pre: `loss_avg 2.9331`, `step_ms_avg 70.4`
  - post: `loss_avg 2.9326`, `step_ms_avg 66.1`

Validation / quantization / TTT:

- `4000` val: `2.8826`
- End-of-training val: `2.7847` bpb `1.0780`
- Diagnostic pre-quantization post-EMA val: `2.78223137` bpb `1.07705376`
- Diagnostic quantized val: `2.81182195` bpb `1.08850882`
- Quantized TTT LoRA val: `2.78466097` bpb `1.07802904`

### Current Read

- The catastrophic YARN quantized regression was partly an eval-state bug.
- After fixing eval-state handling, YARN is no longer obviously broken, but this exact YARN config is still worse than the earlier no-YARN curriculum baseline:
  - no-YARN curriculum pre-quantized: `2.77139888`
  - no-YARN curriculum quantized: `2.80068618`
  - corrected YARN run pre-quantized: `2.78223137`
  - corrected YARN run quantized: `2.81182195`
- The loop transition at `2048` still looks like the main source of degradation before the `4096` phase.
- GPTQ calibration remains a likely source of measurement mismatch because calibration currently uses dense fixed-length `h.train_seq_len` sequences while evaluation is still being reported at `2048`.

## 2026-04-19 - Autonomous Search Run 1: Delay Loop To Bump, Eval At 4096

Command:

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.7 \
EVAL_SEQ_LEN=4096 ENABLE_LOOPING_AT=0.7 \
CURRICULUM_MONITOR_STEPS=64 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Purpose:

- Keep the healthy no-YARN baseline
- Remove the long early looped-2048 regime
- Judge the run directly on long-context eval (`4096`)
- Skip TTT during search

500-step logs:

- `500`: `train_loss 3.2505`, `train_time 0.8m`, `tok/s 8204386`
- `1000`: `train_loss 3.0061`, `train_time 1.6m`, `tok/s 8164274`
- `1500`: `train_loss 3.0146`, `train_time 2.4m`, `tok/s 8154352`
- `2000`: `train_loss 2.9743`, `train_time 3.2m`, `tok/s 8154449`
- `2500`: `train_loss 3.0914`, `train_time 4.0m`, `tok/s 8152546`
- `3000`: `train_loss 2.9452`, `train_time 4.8m`, `tok/s 8151092`
- `3500`: `train_loss 3.0280`, `train_time 5.6m`, `tok/s 8148671`
- `4000`: `train_loss 2.9795`, `train_time 6.4m`, `tok/s 8149499`
- `4500`: `train_loss 2.9552`, `train_time 7.4m`, `tok/s 7942955`
- `5000`: `train_loss 2.8357`, `train_time 8.6m`, `tok/s 7609230`

Transition monitors:

- Loop and seq-len bump were aligned at step `4258`
- Shared pre-window: `loss_avg 2.9647`, `step_ms_avg 37.7`
- Shared post-window: `loss_avg 3.0230`, `step_ms_avg 70.5`

Validation / diagnostics:

- `4000` val at `4096`: `2.9529`
- End-of-training val at `4096`: `2.7718` bpb `1.0730`
- Diagnostic pre-quantization post-EMA val at `4096`: `2.76971049` bpb `1.07220203`

Notes:

- This was the best autonomous search result so far and very close to the `< 2.76` target.
- The quantized phase did not complete cleanly before manual interruption, so no trustworthy quantized metric was recorded for this run.
- Delaying looping avoided the long early looped-2048 slowdown, but the combined loop+4096 transition still caused a visible loss/time shock.

## 2026-04-19 - YARN / Seq-Len Bug Diff Against `pr-1530-lm-head-kernels`

Findings from diffing the current branch against `origin/pr-1530-lm-head-kernels`:

- `DocumentPackingLoader` had drifted from the working branch and was using a fixed `self.max_seq_len = h.train_seq_len` captured at construction time.
- The working branch uses the live `self.h.train_seq_len` on each batch instead.
- `CausalSelfAttention.forward()` had also drifted: this branch stopped passing `max_seqlen` into rotary/YARN and relied on forced rotary cache state instead.
- The working branch still calls rotary with `yarn_seq_len=max_seqlen or seqlen`.

Applied fixes on this branch:

- Reverted `DocumentPackingLoader.next_batch()` to use the live current `self.h.train_seq_len`.
- Restored the working branch rotary call pattern so attention passes `yarn_seq_len=max_seqlen or seqlen`.
- Kept the current branch structure otherwise; this was a targeted bug fix, not a wholesale branch port.

Interpretation:

- The stale loader seq-len would make curriculum packing diverge from the active training regime.
- The rotary call drift changed YARN semantics in the actual attention path, not just compile behavior.
- These are both plausible root-cause bugs for the gap between this branch and the known-good branch.

## 2026-04-19 - Verification Run After Loader + Rotary Fix

Command:

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.7 \
EVAL_SEQ_LEN=4096 ROPE_YARN=1 \
CURRICULUM_MONITOR_STEPS=64 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

500-step logs:

- `500`: `train_loss 3.2552`, `train_time 0.8m`, `tok/s 8203478`
- `1000`: `train_loss 3.0043`, `train_time 1.6m`, `tok/s 8159803`
- `1500`: `train_loss 3.0192`, `train_time 2.4m`, `tok/s 8147944`
- `2000`: `train_loss 2.9735`, `train_time 3.2m`, `tok/s 8148639`
- `2500`: `train_loss 3.0558`, `train_time 4.3m`, `tok/s 7629644`
- `3000`: `train_loss 2.9010`, `train_time 5.5m`, `tok/s 7191450`
- `3500`: `train_loss 2.9655`, `train_time 6.6m`, `tok/s 6908994`
- `4000`: `train_loss 2.8659`, `train_time 7.9m`, `tok/s 6653963`
- `4500`: `train_loss 2.8114`, `train_time 9.1m`, `tok/s 6459437`

Transition monitors:

- Loop monitor:
  - pre: `loss_avg 3.0744`, `step_ms_avg 36.6`
  - post: `loss_avg 3.1228`, `step_ms_avg 70.1`
- Seq-len bump monitor:
  - pre: `loss_avg 2.9347`, `step_ms_avg 70.6`
  - post: `loss_avg 2.9189`, `step_ms_avg 68.0`

Validation / diagnostics:

- `4000` val at `4096`: `2.8526`
- End-of-training val at `4096`: `2.7563` bpb `1.0670`
- Diagnostic pre-quantization post-EMA val at `4096`: `2.75512949` bpb `1.06655748`

Quantized follow-up:

```bash
EVAL_ONLY_PATH=final_model.pt EVAL_SEQ_LEN=4096 ROPE_YARN=1 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Diagnostic pre-quantization post-EMA val at `4096`: `2.75512953` bpb `1.06655750`
- Diagnostic quantized val at `4096`: `2.78594677` bpb `1.07848737`

Current read:

- This fix closed the gap to the known-good branch behavior.
- The target was reached: pre-quantized `4096` val is now `2.7551`, below `< 2.76`.
- The improvement showed up mainly after the curriculum bump, which matches the bug hypothesis.

## 2026-04-19 - MIN_LR 0.05 On Top Of Fixed YARN/Curriculum Path

Command:

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.7 \
EVAL_SEQ_LEN=4096 ROPE_YARN=1 MIN_LR=0.05 \
CURRICULUM_MONITOR_STEPS=64 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

500-step logs:

- `500`: `train_loss 3.2447`, `train_time 0.8m`, `tok/s 8203515`
- `1000`: `train_loss 3.0070`, `train_time 1.6m`, `tok/s 8163595`
- `1500`: `train_loss 3.0144`, `train_time 2.4m`, `tok/s 8145194`
- `2000`: `train_loss 2.9728`, `train_time 3.2m`, `tok/s 8146272`
- `2500`: `train_loss 3.0602`, `train_time 4.3m`, `tok/s 7627433`
- `3000`: `train_loss 2.8969`, `train_time 5.5m`, `tok/s 7188901`
- `3500`: `train_loss 2.9669`, `train_time 6.6m`, `tok/s 6907008`
- `4000`: `train_loss 2.8662`, `train_time 7.9m`, `tok/s 6652386`
- `4500`: `train_loss 2.8061`, `train_time 9.1m`, `tok/s 6458378`

Transition monitors:

- Loop monitor:
  - pre: `loss_avg 3.0732`, `step_ms_avg 37.3`
  - post: `loss_avg 3.1232`, `step_ms_avg 70.1`
- Seq-len bump monitor:
  - pre: `loss_avg 2.9361`, `step_ms_avg 70.2`
  - post: `loss_avg 2.9216`, `step_ms_avg 67.9`

Validation / diagnostics:

- `4000` val at `4096`: `2.8511`
- End-of-training val at `4096`: `2.7591` bpb `1.0681`
- Diagnostic pre-quantization post-EMA val at `4096`: `2.75383580` bpb `1.06605667`

Eval-only quantized follow-up from saved checkpoint:

```bash
EVAL_ONLY_PATH=final_model.pt EVAL_SEQ_LEN=4096 ROPE_YARN=1 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Diagnostic pre-quantization post-EMA val at `4096`: `2.75384741` bpb `1.06606117`
- Diagnostic quantized val at `4096`: `2.78306027` bpb `1.07736996`

Current read:

- `MIN_LR=0.05` improved both metrics over the fixed baseline.
- Previous fixed baseline pre-quantized / quantized at `4096`: `2.75512953` / `2.78594677`
- New `MIN_LR=0.05` run: `2.75384741` / `2.78306027`

## 2026-04-19 - Post-GPTQ Hang Fix

Symptom:

- Full training runs were often getting stuck after GPTQ packaging, before the final `diagnostic quantized` line.
- The same saved checkpoint would quantize/evaluate fine in a later `eval_only` pass, so the quantized eval path itself was not the problem.

Root-cause hypothesis and fix:

- `serialize()` was redundantly running GPTQ on all 8 ranks and then synchronizing before quantized eval.
- That work is not rank-dependent; only one rank needs to write the quantized artifact.
- Changed `serialize()` so GPTQ/serialization runs only on the main process, while the other ranks wait for the artifact.

Verification command:

```bash
ARTIFACT_DIR=/tmp/pg_evalonly_serialize_check \
EVAL_ONLY_PATH=final_model.pt EVAL_SEQ_LEN=4096 ROPE_YARN=1 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Verification result:

- Fresh serialize completed without hanging.
- Fresh quantized diagnostic also completed in the same run:
  - `diagnostic quantized val_loss:2.78269538`
  - `val_bpb:1.07722871`

Current read:

- The post-GPTQ hang was in the full-rank serialize path, not in quantized evaluation itself.
- The rank-0-only serialize fix removes that stall and slightly simplifies post-training evaluation.

## 2026-04-20 - Seq-Len Curriculum Perturbance Instrumentation And Staged Follow-Up

Changes made:

- Added step-level transition logging for curriculum events:
  - `transition_monitor:` JSON lines now record every monitored pre/post step with `loss`, `step_ms`, `seq_len`, `looping`, `frac`, and deltas vs the pre-window average.
- Added staged seq-len support:
  - `TRAIN_SEQ_LEN_STAGES`
  - `SEQ_LEN_BUMP_FRACS`
- Added [analysis/plot_transition_monitor.py](analysis/plot_transition_monitor.py) to turn `transition_monitor:` log lines into CSV + SVG plots.

Initial direct-jump perturbance baseline:

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.7 \
EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 \
MIN_LR=0.05 CURRICULUM_MONITOR_STEPS=64 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
TTT_BATCH_SIZE=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Observed direct-jump monitor:

- bump at `step 3591`, `frac 0.700`, `2048 -> 4096`
- pre: `loss_avg 2.9324`, `step_ms_avg 70.1`
- post: `loss_avg 2.9169`, `step_ms_avg 67.9`
- end step: `4764`
- post-EMA `4096` diagnostic: `2.75405689`

Staged curriculum experiments:

- First staged attempt with evenly spaced post-`0.7` bumps produced overlapping transitions and poor observability.
- Added startup regime priming for all staged seq-lens.
- Removed live bump-time priming to match the `modded-nanogpt` pattern more closely: compile/warmup up front, cheap runtime bump path.

Compile/recompile debugging:

- Enabled `TORCH_LOGS=recompiles` and found the main live recompile sources were:
  - variable `cu_seqlens` tensor length (`128` vs `192`)
  - switching `looping_active` through the same compiled train graph
- Fixes:
  - added `TRAIN_CU_BUCKET_SIZE` and set the train path to use a fixed padded `cu_seqlens` length (`256`)
  - compiled separate train entry points for loop-off and loop-on
- After the fix, the bad live train recompiles disappeared.
- Remaining compile log noise was:
  - expected untimed startup specialization across different `max_seqlen`
  - optimizer-side recompiles in `zeropower_via_newtonschulz5(...)`

Final clean staged perturbance run:

```bash
RUN_ID=seq4096_gradual_seed0_recomp2 \
TTT_ENABLED=0 SKIP_GPTQ=1 TORCH_LOGS=recompiles \
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 \
TRAIN_SEQ_LEN_STAGES=2048,2560,3072,3584,4096 \
SEQ_LEN_BUMP_FRACS=0.75,0.82,0.89,0.95 \
EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 \
MIN_LR=0.05 CURRICULUM_MONITOR_STEPS=64 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
TTT_BATCH_SIZE=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key outcomes:

- reached `4733` steps inside the wallclock cap (`587124ms`)
- in-run val at `4733`: `2.7597`
- post-EMA diagnostic at `4096`: `2.75427937`
- all four staged bumps had clean 64-step isolated post windows

Per-bump perturbance summary:

- `2048 -> 2560` at `3751`
  - loss: `2.9203 -> 2.8842` (`-0.0361`)
  - step_ms: `65.60 -> 70.25` (`+4.65`)
- `2560 -> 3072` at `4034`
  - loss: `2.8528 -> 2.8596` (`+0.0069`)
  - step_ms: `66.53 -> 71.62` (`+5.09`)
- `3072 -> 3584` at `4311`
  - loss: `2.8138 -> 2.8229` (`+0.0091`)
  - step_ms: `67.39 -> 70.42` (`+3.03`)
- `3584 -> 4096` at `4543`
  - loss: `2.8035 -> 2.7814` (`-0.0221`)
  - step_ms: `67.47 -> 72.75` (`+5.29`)

Current read:

- The staged run is effectively on par with the direct `2048 -> 4096` jump in end quality:
  - direct-jump post-EMA: `2.75405689`
  - staged post-EMA: `2.75427937`
- Loss perturbance from staged bumps is mild; no stage shows a severe optimization shock.
- The main cost of staging is runtime (`+3ms` to `+5ms` per step after each bump), not model instability.
- Conclusion from the `4096` study: we likely do not need finer-grained bumps than one intermediate step.
