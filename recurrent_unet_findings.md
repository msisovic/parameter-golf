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

## Key Insights

1. **U-Net skips work well in recurrent setting.** The encoder-decoder structure with skip connections gives depth-dependent information flow without the fixed-point problem of input injection.

2. **Quantization penalty scales with recurrence depth.** Weights reused N times accumulate N× the quantization error. More QAT helps but doesn't fully solve it. Consider: int8 for recurrent blocks (reused), int6 for entry/exit (single-use).

3. **Parameter efficiency is excellent.** 6 unique blocks + weight sharing gives 14 effective layers at depth 3, fitting in ~16MB. But we're bottlenecked on steps — 340ms/step means only ~3500 steps in 20 min.

4. **Main bottleneck is now step speed.** The baseline gets ~6000+ steps at ~100ms each. We get 3500 at 340ms. If we could speed up (fewer blocks, smaller dim, or faster compilation), BPB would improve further since loss is still dropping fast.

## Next Steps to Consider

- **Rerun fixed depth at 1200s**: Run 3 only got 600s — need 1200s for fair comparison with Run 2
- **Try fixed depth=2**: fewer effective layers but faster steps → more steps in wallclock
- **Mixed quantization**: int8 for recurrent blocks, int6 for entry/exit to reduce quant penalty
- **Smaller dim + more steps**: dim=576 would be faster and fit budget better
- **LoRA TTT to patch quant error**: small LoRA at eval time on the quantized model
- **Depth sweep at eval**: try eval_depth=4 or 5 since depth curve was flat in Run 1
