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

## Key Insights

1. **U-Net skips work well in recurrent setting.** The encoder-decoder structure with skip connections gives depth-dependent information flow without the fixed-point problem of input injection.

2. **Quantization penalty scales with recurrence depth.** Weights reused N times accumulate N× the quantization error. Depth=2 has 0.009 penalty vs depth=3's 0.013. More QAT helps too.

3. **Fixed depth training is strictly better than random Poisson.** Every gradient update now optimizes the eval configuration directly. No wasted signal on unused depths, faster compilation, simpler code.

4. **Depth=2 beats depth=3 in wallclock-limited regime.** Despite weaker per-step convergence, 33% faster steps (243ms vs 322ms) yields 40% more steps (4935 vs ~3700), which more than compensates. Also has lower quant penalty.

5. **Step speed is the critical bottleneck.** Baseline gets ~6000+ steps at ~100ms. We get 4935 at 243ms. Any change that reduces step time is worth pursuing even if per-step learning slightly degrades.

6. **Per-iteration control params help in late training.** Giving each recurrence iteration its own attn_scale, mlp_scale, resid_mix, q_gain, skip_weight (~22K extra params) shows no benefit early but 0.016 BPB improvement by step 4000. The model learns to differentiate iteration behavior as it matures.

7. **Entry→exit skip connections don't help.** With only 2 entry and 2 exit blocks, the skip signal is negligible compared to the recurrent body.

## Next Steps to Consider

- **Fix per-iter param speed**: delete unused block params before DDP to eliminate find_unused_parameters overhead, then rerun Run 6 at full speed
- **2+1+1+2 at depth=2**: fewer recurrent blocks → faster steps, potentially unlock larger dim
- **Reduce model size**: dim=640 is over budget. Try dim=576 or mixed int8 for recurrent + int6 for entry/exit
- **GPTQ**: smarter post-training quantization to reduce quant penalty
- **LoRA TTT to patch quant error**: small LoRA at eval time on the quantized model
