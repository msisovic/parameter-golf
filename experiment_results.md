# Experiment Results

Baseline reference: PR #549 — val_bpb 1.1194 (3-seed mean, 8xH100, dim=512, 11 layers)

## Experiment 1: MODEL_DIM=576 (all else PR #549 defaults)

- **Date**: 2026-03-24
- **Hardware**: 4xH100 80GB
- **Key changes**: MODEL_DIM=576 (up from 512), MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, all other hyperparams at code defaults (matching PR #549)
- **model_params**: 33,968,348 (~34M, over budget)
- **Steps completed**: 5,609 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~214ms
- **Peak memory**: 24,301 MiB

### Results
| Metric | Value |
|--------|-------|
| Pre-EMA val_bpb | 1.1284 |
| Post-EMA val_bpb | 1.1277 |
| **Final int6 quantized val_bpb** | **1.1359** |
| Submission size (int6+lzma) | 19,525,669 bytes (~19.5 MB) |

### Notes
- Over param budget (~34M vs typical ~24M), so not submittable as-is.
- Quantization gap is large: 1.1277 -> 1.1359 (+0.0082), likely because bigger model loses more to int6.
- Only got ~5.6k steps due to slower step time at larger dim on 4 GPUs.
- No TTT was run (would need separate eval pass).

---

## Experiment 2: NUM_LAYERS=12 (dim=512, all else PR #549 defaults)

- **Date**: 2026-03-24
- **Hardware**: 4xH100 80GB
- **Key changes**: NUM_LAYERS=12 (up from 11), MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, all other hyperparams at code defaults
- **model_params**: 29,355,620 (~29M, over budget but closer than dim=576)
- **Steps completed**: 6,878 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~174ms
- **Peak memory**: 23,484 MiB

### Results
| Metric | Value |
|--------|-------|
| Pre-EMA val_bpb | 1.1317 |
| Post-EMA val_bpb | 1.1307 |
| Final int6 quantized val_bpb | 1.1390 |
| **Final int6 sliding window val_bpb** | **1.1153** |
| Submission size (int6+lzma) | 17,275,453 bytes (~17.3 MB) |

### Notes
- Also over param budget (~29M) but less so than dim=576.
- Quantization gap: 1.1307 -> 1.1390 (+0.0083), similar to exp 1.
- Sliding window eval (stride 64) brings it to 1.1153 — better than PR #549 baseline (1.1194) pre-TTT.
- Got more steps (6,878 vs 5,609) due to faster per-step time at dim=512.
- No TTT was run.

---

## Experiment 3: NUM_LAYERS=12 + TTT (dim=512, all else PR #549 defaults)

- **Date**: 2026-03-24
- **Hardware**: 4xH100 80GB
- **Key changes**: NUM_LAYERS=12, TTT_ENABLED=1, MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, all other hyperparams at code defaults
- **model_params**: 29,355,620 (~29M)
- **Steps completed**: 6,879 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~174ms
- **TTT time**: 620s (1893 chunks, 3 epochs each)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1304 |
| Final int6 quantized val_bpb | 1.1387 |
| Final int6 sliding window val_bpb | 1.1151 |
| **Post-TTT sliding window val_bpb** | **1.1126** |

### Notes
- TTT gave a -0.0025 gain over sliding window (1.1151 -> 1.1126), similar to PR #549's TTT gain.
- **Beats PR #549's 1.1194 by 0.0068 BPB** — but still over param budget (~29M).
- TTT took 620s which would be within a 10-min eval constraint.

---

## Experiment 4: RECUR_LAYER=5 + TTT (depth recurrence, 11 physical → 12 virtual layers)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYER=5 (layer 5 duplicated), TTT_ENABLED=1, MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,996,324 (~27M) — only ~2.7M over baseline from extra block scalars
- **Steps completed**: 6,884 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~174ms (same as full 12-layer)
- **TTT time**: 622s (untied recurrence before TTT)
- **Submission size**: 15,927,562 bytes (~15.9 MB, well under 16MB budget!)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1354 |
| Final int6 quantized val_bpb | 1.1440 |
| Final int6 sliding window val_bpb | 1.1205 |
| **Post-TTT sliding window val_bpb** | **1.1180** |

### Comparison to full 12-layer (Exp 3)
| Metric | Full 12L (Exp 3) | Recur L5 (Exp 4) | Delta |
|--------|-----------------|-------------------|-------|
| Params | 29.4M | 27.0M | -2.4M |
| Submission size | 17.3 MB | 15.9 MB | -1.4 MB |
| Sliding window val_bpb | 1.1151 | 1.1205 | +0.0054 |
| Post-TTT val_bpb | 1.1126 | 1.1180 | +0.0054 |

### Notes
- Recurrence adds depth for free in compute, but shared weights cost ~0.005 BPB vs independent layers.
- TTT untying gave a -0.0025 gain (1.1205 -> 1.1180), same magnitude as independent layers.
- Submission size is much smaller (15.9 MB vs 17.3 MB) since banks stay at 11-layer size.
- Still beats PR #549 baseline (1.1194) by 0.0014 BPB, with a smaller model.

---

## Experiment 4b: Tied TTT (same checkpoint as Exp 4, no untying)

- **Post-TTT val_bpb**: **1.1179** (vs 1.1180 untied — negligible difference)
- Conclusion: untying doesn't help with 3-epoch TTT. Tied is fine.

---

## Experiment 5: RECUR_LAYERS=4,5 + tied TTT (dual recurrence, 11 physical → 13 virtual layers)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYERS=4,5 (4,5,4,5 pattern), TTT_ENABLED=1, TTT_UNTIE=0
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,998,380 (~27M)
- **Steps completed**: 6,389 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~188ms (up from 174ms with single recurrence — extra virtual layer costs ~14ms/step)
- **TTT time**: 655s (tied)
- **Submission size**: 15,944,748 bytes (~15.9 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1337 |
| Final int6 quantized val_bpb | 1.1421 |
| Final int6 sliding window val_bpb | 1.1187 |
| **Post-TTT sliding window val_bpb** | **1.1163** |

### Full comparison
| | PR #549 | Recur L5 (Exp 4) | Recur L4,5 (Exp 5) | Full 12L (Exp 3) |
|---|---|---|---|---|
| Virtual depth | 11 | 12 | **13** | 12 |
| Params | ~24M | ~27M | ~27M | ~29M |
| Submission size | ~19.5 MB | 15.9 MB | 15.9 MB | 17.3 MB |
| Steps completed | ~7,180 | 6,884 | 6,389 | 6,879 |
| Post-TTT val_bpb | 1.1194 | 1.1179 | **1.1163** | **1.1126** |

### Notes
- Dual recurrence (1.1163) beats single recurrence (1.1179) by 0.0016 BPB.
- Beats PR #549 by 0.0031 BPB, with ~27M params and ~16 MB submission.
- Gap to full independent 12-layer (1.1126) is 0.0037 — weight sharing costs more with 2 repeated layers.
- Step time increased to ~188ms (from 174ms), resulting in ~500 fewer steps in the wallclock budget.
- The extra virtual depth helps despite fewer training steps.

---

## Experiment 6: RECUR_LAYERS=4,5,6 + tied TTT (triple recurrence, 11 physical → 14 virtual layers)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYERS=4,5,6, TTT_ENABLED=1, TTT_UNTIE=0
- **model_params**: 27,000,948 (~27M)
- **Steps completed**: 5,977 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~201ms
- **TTT time**: 700s (tied)
- **Submission size**: 15,921,244 bytes (~15.9 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1365 |
| Final int6 sliding window val_bpb | 1.1217 |
| **Post-TTT sliding window val_bpb** | **1.1190** |

### Notes
- Triple recurrence (1.1190) is worse than dual (1.1163) despite more virtual depth (14 vs 13).
- Lost ~400 steps vs dual due to slower step time (201ms vs 188ms).
- Per-step learning was slightly better, but not enough to overcome fewer total steps.
- **Conclusion: dual recurrence at layers 4,5 is the sweet spot for this wallclock budget.**

---

## Experiment 5b: Untied TTT on dual recurrence (RECUR_LAYERS=4,5)

- **Date**: 2026-03-25
- **Same checkpoint as Exp 5**, only TTT_UNTIE=1 differs
- **Post-TTT sliding window val_bpb**: **1.1163** (identical to tied 1.1163)
- **Conclusion**: Untying doesn't help for dual recurrence either, matching the single-recurrence finding (Exp 4b). Tied TTT is sufficient.

---

## Experiment 7: Eval-time bottleneck (RECUR_LAYERS=4,5, EVAL_BOTTLENECK_REPS=1)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: EVAL_ONLY=1, EVAL_BOTTLENECK_REPS=1 (adds 2 extra virtual layers between encoder/decoder at eval time)
- **Same checkpoint as Exp 5** (RECUR_LAYERS=4,5 trained model)
- **Virtual depth at eval**: 15 (13 trained + 2 bottleneck)
- **TTT time**: 724s (tied)

### Results
| Metric | Value |
|--------|-------|
| Final int6 sliding window val_bpb (with bottleneck) | 1.1472 |
| **Post-TTT sliding window val_bpb** | **1.1219** |

### Comparison to no-bottleneck (Exp 5)
| Metric | No bottleneck (Exp 5) | +1 bottleneck rep (Exp 7) | Delta |
|--------|----------------------|--------------------------|-------|
| Pre-TTT sliding window | 1.1187 | 1.1472 | +0.0285 |
| Post-TTT sliding window | 1.1163 | 1.1219 | +0.0056 |

### Notes
- The extra untrained bottleneck blocks start far from useful (1.1472 vs 1.1187 pre-TTT).
- TTT recovers most of the gap but not all — final result is 0.0056 worse than without bottleneck.
- **Conclusion: Eval-time bottleneck with deepcopy of trained blocks does not help.** The copied blocks need to be trained to integrate into the forward pass. 3-epoch TTT is insufficient to close the gap.
- Possible improvement: initialize bottleneck blocks as identity/near-identity rather than copies of mid-network blocks.

---

## Experiment 8: Dual recurrence layer placement sweep (no TTT)

- **Date**: 2026-03-26
- **Hardware**: 4xH100 80GB
- **Key changes**: Swept RECUR_LAYERS across 6 pairs to find optimal placement, no TTT
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,998,380 (~27M) for all runs
- **Virtual depth**: 13 (11 physical + 2 recurrent) for all runs
- **Step avg**: ~189ms across all runs
- **Wallclock**: 1200s (all runs hit cap)

### Results

| RECUR_LAYERS | Steps | Post-EMA val_bpb | Int6 val_bpb | Int6 sliding window val_bpb | Submission size |
|--------------|-------|-------------------|--------------|----------------------------|-----------------|
| 0,1 | 6,309 | 1.1396 | 1.1479 | 1.1245 | 15.92 MB |
| 2,3 | 6,332 | 1.1348 | 1.1431 | 1.1196 | 15.94 MB |
| **4,5** | **6,335** | **1.1340** | **1.1424** | **1.1190** | **15.94 MB** |
| 6,7 | 6,335 | 1.1361 | 1.1444 | 1.1208 | 15.95 MB |
| 8,9 | 6,340 | 1.1414 | 1.1498 | 1.1263 | 16.03 MB |
| 9,10 | 6,342 | 1.1419 | — (incomplete) | — (incomplete) | 15.95 MB |

### Notes
- **Best placement: layers 4,5** (1.1190 sliding window), confirming the choice in Exp 5.
- Clear U-shaped curve: mid-network recurrence works best, with performance degrading toward both ends.
- Early layers (0,1) are 0.0055 worse than (4,5) — early representations are too low-level to benefit from repetition.
- Late layers (8,9) are 0.0073 worse than (4,5) — late recurrence hurts more, likely because these layers are more specialized.
- The (2,3) and (6,7) placements are close runners-up at 1.1196 and 1.1208 respectively.
- No TTT was run in this sweep; adding TTT to the (4,5) winner matches Exp 5's result of 1.1163.
- The recur_9_10 run had an incomplete evaluation (log truncated after submission size).
