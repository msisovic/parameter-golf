# Recurrent U-Net Findings

## Architecture
- 1 entry + 2 encoder_recurrent (×N) + 2 decoder_recurrent (×N) + 1 exit
- U-net skip connections: encoder iteration i → decoder iteration (N-1-i)
- Single learned per-dim skip_weight for all skip connections
- Encoder and decoder have fully separate weights (no sharing)
- Keeps from baseline: attn_scale, mlp_scale, resid_mix/x0, XSA on exit, EMA, late QAT, BigramHash, SmearGate
- Dropped from baseline: ln_scale (depth-dependent norm scaling incompatible with weight sharing)
- Log-normal Poisson depth sampling during training, truncated backprop

## Run 1: dim=640, mean_depth=3, max_depth=6

**Config:** 1+2+2+1, dim=640, 8 heads, 4 KV heads, 3x MLP, int6+zstd.
23.1M params. BIGRAM_VOCAB_SIZE=2048, XSA_LAST_N=1, EMA_DECAY=0.997.

**Training:**
| Step | Val BPB | Train Loss | Notes |
|------|---------|------------|-------|
| 1000 | 1.3248  | 2.2911     | |
| 2000 | 1.2588  | 2.0860     | |
| 3000 | 1.2131  | 2.1265     | |
| 4000 | 1.1748  | 2.1109     | |
| 4034 | 1.1745  | -          | Wallclock cap (1200s) |

**Final results:**
- Pre-quant val BPB: **1.1745** (standard eval, depth 3)
- Int6 roundtrip BPB: **1.1892** (quant penalty: 0.015)
- Int6 sliding window BPB: **1.1652** (stride 64)
- Model size: **12.76MB** int6+zstd (3.24MB under 16MB budget)
- Step avg: 297ms, 4034 steps in 1200s

**Observations:**
1. **Int6 quant penalty is 2x baseline** (0.015 vs ~0.007). Likely because recurrent blocks' weights are reused N=3 times, amplifying quantization error through each iteration.
2. **3.2MB headroom** — room for 1-2 more unique blocks or larger dim.
3. **Loss still dropping fast** at wallclock cutoff — more steps would help.
4. **OOM at max_depth > 7** with dim=640 — encoder stores N intermediate states for skip connections, plus backward pass through 2×N×2 block applications.
5. **Step avg 297ms** vs baseline ~226ms — slower due to variable-depth compilation overhead and more effective layers per step.

**Comparison with other architectures:**
| Architecture | Sliding BPB | Size | Steps |
|---|---|---|---|
| Baseline (11 unique layers) | 1.1248 | ~16MB | ~6000+ |
| Run 7b (injection 2+2+2 dim=640) | 1.1481 | over budget | 4085 |
| Run 15 (injection 2+3+2 dim=512) | 1.1619 | 16.5MB | 5308 |
| **U-Net recurrent (1+2+2+1 dim=640)** | **1.1652** | **12.8MB** | 4034 |

## Run 2: 2+2+2+2 dim=640, more QAT

**Config:** 2+2+2+2, dim=640, 8 heads, 4 KV heads, 3x MLP, int6+zstd.
30.5M params. BIGRAM_VOCAB_SIZE=2048, XSA_LAST_N=2, EMA_DECAY=0.997.
QAT_THRESHOLD=0.3 (kicked in at step 2637, scale=0.30 — ~900 steps of QAT).

**Training:**
| Step | Val BPB | Train Loss | Notes |
|------|---------|------------|-------|
| 1000 | 1.2983  | 2.2419     | |
| 2000 | 1.2308  | 2.0248     | |
| 2637 | -       | -          | QAT enabled (scale < 0.3) |
| 3000 | 1.1790  | 2.0569     | |
| 3524 | 1.1583  | -          | Wallclock cap (1200s) |

**Final results:**
- Pre-quant sliding window BPB: **1.1352** (stride 64)
- Int6 roundtrip BPB: **1.1719** (standard eval)
- Int6 sliding window BPB: **1.1481** (stride 64)
- **Quant penalty: 0.013** (1.1352 → 1.1481, improved from Run 1's 0.015 — more QAT helped)
- Model size: **16.44MB** int6+zstd (16.52MB with code — slightly over 16MB budget)
- Step avg: 340ms, 3524 steps in 1200s

**Observations:**
1. **Pre-quant BPB 1.1352 is very close to baseline's 1.1248** — only 0.010 behind with 500 fewer steps.
2. **QAT reduced quant penalty** from 0.015 to 0.013 (QAT_THRESHOLD=0.3 gave ~900 steps of QAT vs ~400 in Run 1).
3. **Slightly over budget** — 16.52MB with code. Need to trim ~0.5MB (reduce dim slightly or use int8 for recurrent blocks + int6 for entry/exit).
4. **Fewer steps** (3524 vs 4034 in Run 1) due to larger model — 340ms vs 297ms per step.
5. **Loss still dropping** at cutoff — the curve suggests we'd reach ~1.14-1.15 BPB pre-quant with more steps.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Size | Steps |
|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~16MB | ~6000+ |
| U-Net Run 1 (1+2+2+1) | ~1.14* | 1.1652 | 12.8MB | 4034 |
| **U-Net Run 2 (2+2+2+2)** | **1.1352** | **1.1481** | **16.4MB** | 3524 |

## Run 3: Fixed depth training (FIXED_TRAIN_DEPTH=3), 600s

**Config:** 2+2+2+2, dim=640, 8 heads, 4 KV heads, 3x MLP, int6+zstd.
30.5M params. Same as Run 2 except: FIXED_TRAIN_DEPTH=3 (no random Poisson sampling),
MAX_WALLCLOCK_SECONDS=600 (half of Run 2's 1200s — accidental default).
QAT_THRESHOLD=0.3 (kicked in at step 1505, scale=0.30 — ~360 steps of QAT).

**Training:**
| Step | Val BPB | Train Loss | Notes |
|------|---------|------------|-------|
| 1000 | 1.2801  | 2.2095     | Run 2 was 1.2983 here — 0.018 better |
| 1866 | 1.1970  | -          | Wallclock cap (600s) |

**Final results:**
- Pre-quant sliding window BPB: **1.1825** (stride 64)
- Int6 roundtrip BPB: **1.2170** (standard eval)
- Int6 sliding window BPB: **1.1947** (stride 64)
- **Quant penalty: 0.012** (1.1825 → 1.1947, slightly better than Run 2's 0.013)
- Model size: **17.25MB** int6+zstd (over budget)
- Step avg: 322ms, 1866 steps in 600s

**Observations:**
1. **Fixed depth converges faster per step.** At step 1000, val_bpb 1.2801 vs Run 2's 1.2983 (0.018 better). Every gradient update optimizes the actual eval configuration instead of wasting signal on unused depths.
2. **Only half the wallclock** (600s vs 1200s) — accidental default. Need to rerun at 1200s for fair comparison, where we'd expect ~3700 steps (vs Run 2's 3524).
3. **Compilation much faster**: only 1 depth graph compiled vs 12 in Run 2. Warmup is trivial.
4. **Step speed slightly improved**: 322ms vs 340ms (~5%) — less dynamo cache pressure from single graph.
5. **Loss still dropping fast** at step 1866 — strong indication a 1200s run would beat Run 2 significantly.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Size | Steps | Wallclock |
|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~16MB | ~6000+ | 600s |
| U-Net Run 1 (1+2+2+1) | ~1.14* | 1.1652 | 12.8MB | 4034 | 1200s |
| U-Net Run 2 (2+2+2+2, random depth) | 1.1352 | 1.1481 | 16.4MB | 3524 | 1200s |
| **U-Net Run 3 (2+2+2+2, fixed depth)** | **1.1825** | **1.1947** | **17.3MB** | 1866 | 600s |

*Run 3 had half the wallclock of Run 2. Per-step convergence is clearly better — needs 1200s rerun.*

## Run 4: Fixed depth=2 training, 1200s

**Config:** 2+2+2+2, dim=640, 8 heads, 4 KV heads, 3x MLP, int6+zstd.
30.5M params. FIXED_TRAIN_DEPTH=2, EVAL_RECURRENT_DEPTH=2. MAX_WALLCLOCK_SECONDS=1200.
QAT_THRESHOLD=0.3 (kicked in at step 4574, scale=0.30 — ~360 steps of QAT).

**Training:**
| Step | Val BPB | Train Loss | Notes |
|------|---------|------------|-------|
| 1000 | 1.2985  | 2.2397     | Worse than depth=3 per-step |
| 2000 | 1.2454  | 2.0051     | |
| 3000 | 1.2280  | 2.0888     | |
| 4000 | 1.2062  | 2.1045     | |
| 4935 | 1.1474  | -          | Wallclock cap (1200s) |

**Final results:**
- Pre-quant sliding window BPB: **1.1316** (stride 64)
- Int6 roundtrip BPB: **1.1630** (standard eval)
- Int6 sliding window BPB: **1.1406** (stride 64)
- **Quant penalty: 0.009** (1.1316 → 1.1406, best yet — fewer recurrence iters = less error amplification)
- Model size: **18.18MB** int6+zstd (over budget)
- Step avg: 243ms, 4935 steps in 1200s

**Observations:**
1. **Beats Run 2 on all metrics.** Pre-quant 1.1316 vs 1.1352, int6 1.1406 vs 1.1481. The extra 1400 steps from faster iteration (243ms vs 340ms) more than compensated for shallower effective depth (12 vs 16 layers).
2. **Quant penalty dramatically reduced** to 0.009 (from 0.013 at depth=3). Fewer recurrence iterations = less quantization error amplification. This is the closest to baseline's ~0.007.
3. **Model still over budget** at 18.18MB. Same issue as Run 2 — need to reduce dim or use mixed quantization.
4. **Memory usage much lower**: 26.7GB vs 34.9GB (depth=3). Could potentially increase dim or batch size.
5. **Per-step convergence worse than depth=3** (1.2985 vs 1.2801 at step 1000), but total wallclock convergence wins due to 33% more steps.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Quant Δ | Size | Steps | Wallclock |
|---|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~0.007 | ~16MB | ~6000+ | 600s |
| U-Net Run 1 (1+2+2+1, random) | ~1.14* | 1.1652 | 0.015 | 12.8MB | 4034 | 1200s |
| U-Net Run 2 (2+2+2+2, random d=3) | 1.1352 | 1.1481 | 0.013 | 16.4MB | 3524 | 1200s |
| U-Net Run 3 (2+2+2+2, fixed d=3) | 1.1825 | 1.1947 | 0.012 | 17.3MB | 1866 | 600s |
| **U-Net Run 4 (2+2+2+2, fixed d=2)** | **1.1316** | **1.1406** | **0.009** | **18.2MB** | 4935 | 1200s |

## Run 5: Entry→exit skip connections (killed early)

**Config:** Same as Run 4 + entry→exit U-net skip connections (mirror pattern matching baseline).
Killed at ~step 2600 — no measurable improvement.

- Step 1000: 1.2999 (Run 4: 1.2985)
- Step 2000: 1.2471 (Run 4: 1.2454)

**Conclusion:** Entry→exit skips don't help with only 2 entry and 2 exit blocks. The recurrent body dominates.

## Run 6: Per-iteration control params, 1200s

**Config:** 2+2+2+2, dim=640, fixed depth=2. Each recurrence iteration gets its own
attn_scale, mlp_scale, resid_mix, q_gain, skip_weight (~22K extra params).
Used find_unused_parameters=True in DDP (adds ~7% step overhead: 259ms vs 243ms).

**Training:**
| Step | Val BPB | Train Loss | Notes |
|------|---------|------------|-------|
| 1000 | 1.2987  | 2.2411     | Same as Run 4 (1.2985) |
| 2000 | 1.2477  | 2.0054     | Slightly worse than Run 4 (1.2454) |
| 3000 | 1.2295  | 2.0931     | Run 4: 1.2280 |
| 4000 | 1.1902  | 2.0779     | **Run 4: 1.2062 — 0.016 better!** |
| 4629 | 1.1509  | -          | Wallclock cap (1200s) |

**Final results:**
- Pre-quant sliding window BPB: **1.1321** (stride 64)
- Int6 roundtrip BPB: **1.1642** (standard eval)
- Int6 sliding window BPB: **1.1417** (stride 64)
- **Quant penalty: 0.010** (1.1321 → 1.1417)
- Model size: **18.07MB** int6+zstd (over budget)
- Step avg: 259ms, 4629 steps in 1200s

**Observations:**
1. **Per-iteration params help in later training.** At step 1000-2000 they show no benefit, but by step 4000 they're 0.016 better than Run 4. The model learns to use different control params per iteration as training progresses.
2. **Speed penalty from find_unused_parameters** (259ms vs 243ms) cost ~300 steps. Fix: delete unused block params before DDP wrapping to avoid the overhead.
3. **With the speed fix, this would likely beat Run 4.** At 243ms we'd get ~4935 steps, and the per-step advantage at step 4000+ would carry through to the final result.
4. **Final int6 SW 1.1417 vs Run 4's 1.1406** — slightly worse due to fewer steps, but per-step convergence is clearly better.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Quant Δ | Size | Steps | Wallclock |
|---|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~0.007 | ~16MB | ~6000+ | 600s |
| U-Net Run 2 (2+2+2+2, random d=3) | 1.1352 | 1.1481 | 0.013 | 16.4MB | 3524 | 1200s |
| U-Net Run 4 (2+2+2+2, fixed d=2) | 1.1316 | 1.1406 | 0.009 | 18.2MB | 4935 | 1200s |
| U-Net Run 5 (+ entry/exit skips) | - | - | - | - | killed | - |
| **U-Net Run 6 (+ per-iter ctrl params)** | **1.1321** | **1.1417** | **0.010** | **18.1MB** | 4629 | 1200s |

*Run 6 had 7% step overhead from find_unused_parameters. With del fix, expect ~4935 steps and better final BPB.*

## Run 7: ln_scale + XSA decoder, 1200s

**Config:** 2+2+2+2, dim=640, fixed depth=2. Added ln_scale (1/sqrt(effective_layer_idx+1) on
RMSNorm outputs) and XSA on decoder recurrent blocks (in addition to exit blocks).

**Training:**
| Step | Val BPB | Run 6 | Delta |
|------|---------|-------|-------|
| 1000 | 1.3061  | 1.2987 | +0.007 worse |
| 2000 | 1.2539  | 1.2477 | +0.006 worse |
| 3000 | 1.2329  | 1.2295 | +0.003 worse |
| 4000 | 1.2005  | 1.1902 | +0.010 worse |
| 4743 | 1.1546  | -      | Wallclock cap |

**Final results:**
- Pre-quant sliding window BPB: **1.1338** (stride 64)
- Int6 roundtrip BPB: **1.1644** (standard eval)
- Int6 sliding window BPB: **1.1420** (stride 64)
- **Quant penalty: 0.008** (1.1338 → 1.1420, best yet)
- Model size: **17.8MB** int6+zlib (over budget)
- Step avg: 253ms, 4743 steps in 1200s

**Observations:**
1. **Per-step convergence consistently worse** than Run 6 across all checkpoints.
2. **Best quant penalty yet** (0.008 vs Run 4's 0.009) — ln_scale may improve quantization friendliness.
3. **10ms/step overhead** from decoder XSA (253ms vs 243ms), costing ~200 steps.
4. **Late crossover vs Run 4** at step 4000 (1.2005 vs 1.2062) but not vs Run 6 (1.1902).

## Run 8: ln_scale only (no decoder XSA), 1200s

**Config:** Same as Run 7 but XSA_DECODER=0. Isolating ln_scale contribution.

**Training:**
| Step | Val BPB | Run 6 | Delta |
|------|---------|-------|-------|
| 1000 | 1.3078  | 1.2987 | +0.009 worse |
| 2000 | 1.2535  | 1.2477 | +0.006 worse |
| 3000 | 1.2350  | 1.2295 | +0.006 worse |
| 4000 | 1.2060  | 1.1902 | +0.016 worse |
| 4837 | 1.1533  | -      | Wallclock cap |

**Final results:**
- Pre-quant sliding window BPB: **1.1327** (stride 64)
- Int6 roundtrip BPB: **1.1636** (standard eval)
- Int6 sliding window BPB: **1.1412** (stride 64)
- **Quant penalty: 0.009** (1.1327 → 1.1412)
- Model size: **17.8MB** int6+zlib (over budget)
- Step avg: 248ms, 4837 steps in 1200s

**Observations:**
1. **ln_scale hurts per-step convergence** — gap vs Run 6 widens from +0.009 to +0.016 at step 4000.
2. **Step speed recovered** to 248ms (vs Run 7's 253ms), confirming decoder XSA was the overhead source.
3. **Final pre-quant 1.1327 slightly worse than Run 4's 1.1316** despite more steps (4837 vs 4935). ln_scale is a net negative for training quality.
4. **Quant penalty 0.009** — same as Run 4, so the 0.008 in Run 7 was from decoder XSA, not ln_scale.

**Conclusion:** ln_scale is harmful. The fixed 1/sqrt(idx) attenuation fights the learned per-iteration attn_scale/mlp_scale params. The baseline's ln_scale worked because layer indices were fixed; with recurrence + per-iter control params, the model can already learn its own signal regulation.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Quant Δ | Size | Steps | Wallclock |
|---|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~0.007 | ~16MB | ~6000+ | 600s |
| U-Net Run 4 (2+2+2+2, fixed d=2) | 1.1316 | 1.1406 | 0.009 | 18.2MB | 4935 | 1200s |
| U-Net Run 6 (+ per-iter ctrl params) | 1.1321 | 1.1417 | 0.010 | 18.1MB | 4629 | 1200s |
| U-Net Run 7 (+ ln_scale + XSA dec) | 1.1338 | 1.1420 | 0.008 | 17.8MB | 4743 | 1200s |
| U-Net Run 8 (+ ln_scale only) | 1.1327 | 1.1412 | 0.009 | 17.8MB | 4837 | 1200s |

## Run 9: XSA decoder only (no ln_scale), 1200s

**Config:** 2+2+2+2, dim=640, fixed depth=2, per-iter ctrl params.
XSA_DECODER=1, LN_SCALE=0. Isolating decoder XSA from Run 7.

**Training:**
| Step | Val BPB | Run 6 | Delta |
|------|---------|-------|-------|
| 1000 | 1.2995  | 1.2987 | +0.001 (same) |
| 2000 | 1.2463  | 1.2477 | -0.001 better |
| 3000 | 1.2281  | 1.2295 | -0.001 better |
| 4000 | 1.1958  | 1.1902 | +0.006 worse |
| 4739 | 1.1498  | -      | Wallclock cap |

**Final results:**
- Pre-quant sliding window BPB: **1.1289** (stride 64) — **new best**
- Int6 roundtrip BPB: **1.1591** (standard eval)
- Int6 sliding window BPB: **1.1368** (stride 64) — **new best**
- **Quant penalty: 0.008** (1.1289 → 1.1368, tied best with Run 7)
- Model size: **18.1MB** int6+zlib (over budget)
- Step avg: 253ms, 4739 steps in 1200s

**Observations:**
1. **New best on all final metrics.** Pre-quant 1.1289 (was 1.1316), int6 SW 1.1368 (was 1.1406). Decoder XSA is a clear win.
2. **Per-step convergence tracks Run 6 closely** through step 3000, then falls behind at step 4000 — but final results are better because decoder XSA improves the quality of the last ~700 training steps (post-QAT) more than expected.
3. **Quant penalty 0.008** — confirmed that this comes from decoder XSA, not ln_scale (Run 8 with ln_scale only had 0.009).
4. **10ms/step overhead** (253ms vs 243ms) costs ~200 steps but the per-step quality improvement more than compensates.
5. **ln_scale was the culprit in Run 7.** Run 9 removes it and immediately beats all prior runs. The damage in Run 7 was entirely from ln_scale fighting the per-iteration control params.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Quant Δ | Size | Steps | Wallclock |
|---|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~0.007 | ~16MB | ~6000+ | 600s |
| U-Net Run 4 (2+2+2+2, fixed d=2) | 1.1316 | 1.1406 | 0.009 | 18.2MB | 4935 | 1200s |
| U-Net Run 6 (+ per-iter ctrl params) | 1.1321 | 1.1417 | 0.010 | 18.1MB | 4629 | 1200s |
| U-Net Run 7 (+ ln_scale + XSA dec) | 1.1338 | 1.1420 | 0.008 | 17.8MB | 4743 | 1200s |
| U-Net Run 8 (+ ln_scale only) | 1.1327 | 1.1412 | 0.009 | 17.8MB | 4837 | 1200s |
| **U-Net Run 9 (+ XSA dec only)** | **1.1289** | **1.1368** | **0.008** | **18.1MB** | 4739 | 1200s |

## Run 10: 2+1+1+2 dim=704 depth=4, 1200s

**Config:** 2+1+1+2, dim=704, fixed depth=4 (so 2 + 4×1 + 4×1 + 2 = 12 effective layers).
XSA_DECODER=1, LN_SCALE=0. Fewer recurrent blocks, reinvested params into larger dim.

**Training:**
| Step | Val BPB | Run 9 | Delta |
|------|---------|-------|-------|
| 1000 | 1.3077  | 1.2995 | +0.008 worse |
| 2000 | 1.2542  | 1.2463 | +0.008 worse |
| 3000 | 1.2297  | 1.2281 | +0.002 worse |
| 4000 | 1.1651  | 1.1958 | -0.031 better (but Run 9 at step 4000 had more remaining) |
| 4068 | 1.1627  | -      | Wallclock cap |

**Final results:**
- Pre-quant sliding window BPB: **1.1422** (stride 64)
- Int6 roundtrip BPB: **1.1752** (standard eval)
- Int6 sliding window BPB: **1.1527** (stride 64)
- **Quant penalty: 0.011** (1.1422 → 1.1527, worst since depth=3)
- Model size: ~18MB int6+zlib (over budget)
- Step avg: 295ms, 4068 steps in 1200s

**Observations:**
1. **Worse than Run 9 on all final metrics.** Pre-quant 1.1422 vs 1.1289, int6 SW 1.1527 vs 1.1368.
2. **Depth=4 with 1 block is slower** (295ms vs 253ms) due to more sequential recurrence iterations — less parallelism within each step.
3. **Quant penalty regressed** to 0.011 — 4 recurrence iterations amplify quantization error more than 2.
4. **dim=704 didn't compensate** for losing block parallelism and gaining quant penalty.

**Conclusion:** 2+2+2+2 at depth=2 remains superior to 2+1+1+2 at depth=4. More blocks with fewer iterations beats fewer blocks with more iterations in wallclock-limited regime.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Quant Δ | Size | Steps | Wallclock |
|---|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~0.007 | ~16MB | ~6000+ | 600s |
| U-Net Run 4 (2+2+2+2, fixed d=2) | 1.1316 | 1.1406 | 0.009 | 18.2MB | 4935 | 1200s |
| **U-Net Run 9 (+ XSA dec only)** | **1.1289** | **1.1368** | **0.008** | **18.1MB** | 4739 | 1200s |
| U-Net Run 10 (2+1+1+2 d=4 dim=704) | 1.1422 | 1.1527 | 0.011 | ~18MB | 4068 | 1200s |

## Run 11: Shared recurrent 2+2s+2 dim=704 depth=2, 1200s

**Config:** 2+2shared+2, dim=704, fixed depth=2. Encoder and decoder reuse same 2 blocks,
differentiated by per-iteration control params. XSA on shared blocks (both phases).

**Training:**
| Step | Val BPB | Run 9 (253ms) | Run 10 (295ms) |
|------|---------|---------------|----------------|
| 1000 | 1.2977  | 1.2995        | 1.3077         |
| 2000 | 1.2451  | 1.2463        | 1.2542         |
| 3000 | 1.2207  | 1.2281        | 1.2297         |
| 4000 | 1.1588  | 1.1958        | 1.1651         |
| 4012 | 1.1586  | -             | -              |

**Final results:**
- Pre-quant sliding window BPB: **1.1376** (stride 64)
- Int6 roundtrip BPB: **1.1697** (standard eval)
- Int6 sliding window BPB: **1.1469** (stride 64)
- **Quant penalty: 0.009** (1.1376 → 1.1469)
- Model size: **16.2MB** int6+zlib (nearly within 16MB budget!)
- Step avg: 300ms, 4012 steps in 1200s

**Observations:**
1. **Best per-step convergence** — 0.037 better than Run 9 at step 4000. Shared blocks + dim=704 learn very efficiently.
2. **But 727 fewer steps** (4012 vs 4739) from dim=704 slowdown (300ms vs 253ms) erases the per-step advantage.
3. **Model size nearly fits budget** at 16.2MB — shared blocks saved ~2MB vs Run 9's 18.1MB.
4. **Quant penalty 0.009** — good, same as depth=2 non-shared.

## Run 12: Shared recurrent 1+4s+1 dim=704 depth=1, 1200s

**Config:** 1+4shared+1, dim=704, fixed depth=1. Complexity shifted to recurrent body,
fewer entry/exit blocks. 10 effective layers (vs 12 in Run 11).

**Training:**
| Step | Val BPB | Run 9 (253ms) | Run 11 (300ms) |
|------|---------|---------------|----------------|
| 1000 | 1.2980  | 1.2995        | 1.2977         |
| 2000 | 1.2475  | 1.2463        | 1.2451         |
| 3000 | 1.2327  | 1.2281        | 1.2207         |
| 4000 | 1.2038  | 1.1958        | 1.1588         |
| 4770 | 1.1505  | -             | -              |

**Final results:**
- Pre-quant sliding window BPB: **1.1361** (stride 64)
- Int6 roundtrip BPB: **1.1673** (standard eval)
- Int6 sliding window BPB: **1.1448** (stride 64)
- **Quant penalty: 0.009** (1.1361 → 1.1448)
- Step avg: 252ms, 4770 steps in 1200s

**Observations:**
1. **Depth=1 recovered step speed** (252ms, matching Run 9) but 10 effective layers isn't enough — falls behind Run 9 by 0.008 per-step at step 4000.
2. **Beats Run 11** on final BPB (1.1448 vs 1.1469) because 758 more steps compensate for 2 fewer effective layers.
3. **Doesn't beat Run 9** (1.1448 vs 1.1368). The 2 fewer effective layers cost ~0.008 BPB at the same step speed.

**Updated comparison:**
| Architecture | Pre-quant SW | Int6 SW | Quant Δ | Size | Steps | ms/step |
|---|---|---|---|---|---|---|
| Baseline (11 unique) | - | 1.1248 | ~0.007 | ~16MB | ~6000+ | ~100 |
| **U-Net Run 9 (2+2+2+2 d=2)** | **1.1289** | **1.1368** | **0.008** | **18.1MB** | 4739 | 253 |
| U-Net Run 10 (2+1+1+2 d=4 dim=704) | 1.1422 | 1.1527 | 0.011 | ~18MB | 4068 | 295 |
| U-Net Run 11 (2+2s+2 d=2 dim=704) | 1.1376 | 1.1469 | 0.009 | 16.2MB | 4012 | 300 |
| U-Net Run 12 (1+4s+1 d=1 dim=704) | 1.1361 | 1.1448 | 0.009 | ~16MB | 4770 | 252 |

## Key Insights

1. **U-Net skips work well in recurrent setting.** The encoder-decoder structure with skip connections gives depth-dependent information flow without the fixed-point problem of input injection.

2. **Quantization penalty scales with recurrence depth.** Weights reused N times accumulate N× the quantization error. Depth=2 has 0.009 penalty vs depth=3's 0.013. More QAT helps too.

3. **Fixed depth training is strictly better than random Poisson.** Every gradient update now optimizes the eval configuration directly. No wasted signal on unused depths, faster compilation, simpler code.

4. **Depth=2 beats depth=3 in wallclock-limited regime.** Despite weaker per-step convergence, 33% faster steps (243ms vs 322ms) yields 40% more steps (4935 vs ~3700), which more than compensates. Also has lower quant penalty.

5. **Step speed is the critical bottleneck.** Baseline gets ~6000+ steps at ~100ms. We get 4935 at 243ms. Any change that reduces step time is worth pursuing even if per-step learning slightly degrades.

6. **Per-iteration control params help in late training.** Giving each recurrence iteration its own attn_scale, mlp_scale, resid_mix, q_gain, skip_weight (~22K extra params) shows no benefit early but 0.016 BPB improvement by step 4000. The model learns to differentiate iteration behavior as it matures.

7. **Entry→exit skip connections don't help.** With only 2 entry and 2 exit blocks, the skip signal is negligible compared to the recurrent body.

8. **Decoder XSA is a clear win.** XSA on decoder blocks reduces quant penalty (0.008, best yet) and improves final BPB despite 10ms/step overhead. XSA prevents self-attention collapse in the decoder, which matters more than in the encoder (decoder is reading out, encoder is building representations).

9. **ln_scale is harmful with per-iteration control params.** Fixed 1/sqrt(idx) attenuation fights the learned per-iter attn_scale/mlp_scale. The model can already learn its own signal regulation — imposing a fixed schedule hurts.

10. **Shared recurrent blocks work well per-step** but dim=704 makes them too slow. Run 11 (shared, dim=704) had 0.037 better BPB than Run 9 at step 4000, but 727 fewer steps from the speed penalty. Per-iteration control params successfully differentiate encoder vs decoder behavior on shared weights.

11. **Effective layer count matters.** Run 12 (10 layers, 252ms) couldn't match Run 9 (12 layers, 253ms) despite identical speed. 12 effective layers at depth=2 is the sweet spot for this architecture.

## Next Steps to Consider

- **Shared recurrent at dim=640 depth=2**: same speed as Run 9 (253ms), 12 effective layers, smaller model (~14MB, room to grow dim). The definitive test of sharing.
- **Reduce model size**: Run 9 is 18.1MB, need 16MB. Shared blocks at dim=640 naturally fit. Or mixed int8/int6.
- **GPTQ**: smarter post-training quantization to reduce quant penalty
- **LoRA TTT to patch quant error**: small LoRA at eval time on the quantized model
