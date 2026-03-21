# Depth Recurrence Findings

## Architecture
- 1 entry block + 1 recurrent block (iterated N times) + 1 exit block
- dim=640, 10 heads, 5 KV heads (head_dim=64), 3x MLP
- 12.1M params total, ~10MB int8+zlib (of 16MB budget)
- Keeps: partial RoPE (16 dims), ln_scale, resid_mix/x0, EMA, late QAT, Muon, sliding window eval, relu^2 MLP, QK RMSNorm, logit softcap, SmearGate, BigramHash
- Dropped: U-net skips, XSA, MTP, multiple unique layers

## Run 1: Uniform depth + curriculum (2024-03-21)

**Config:** min_depth=4, max_depth=10, eval_depth=20, uniform sampling with linear curriculum on max_depth.

**Training trajectory (4xH100, 1200s wallclock):**
| Step | Val BPB | Step Avg |
|------|---------|----------|
| 1000 | 3.0562  | 118ms    |
| 2000 | 1.8666  | 120ms    |
| 3000 | 1.5649  | 122ms    |
| 4000 | 1.3633  | 126ms    |
| 5000 | 1.3086  | 129ms    |
| 6000 | 1.3048  | 132ms    |
| 7000 | 1.2981  | 135ms    |
| 8000 | 1.2446  | 138ms    |
| 8605 | 1.2204  | 139ms    |

Stopped at step 8605 (wallclock cap). EMA applied.

**Depth sweep results (the critical test):**
| Depth | Val BPB | Sliding Window |
|-------|---------|----------------|
| 4     | 1.2317  | —              |
| **8** | **1.2150** | —           |
| 12    | 1.2288  | —              |
| 16    | 1.2546  | —              |
| 20    | 1.2968  | 1.2703         |

**Conclusion: Test-time compute scaling FAILED.** Performance peaks at depth 8 and degrades beyond. The model never saw depth > 10 during training, so it can't extrapolate. Depth 8 best = 1.2150 BPB vs baseline 1.1248.

**Submission size:** 10.05MB int8+zlib

## Root Cause Analysis

The model was trained with `random.randint(4, curr_max_depth)` where max_depth=10. It literally never experienced depth > 10. The recurrent block learned a transform that's useful for ~8 iterations but doesn't converge to a fixed point — extra iterations add noise/drift.

## Next Steps: Poisson Depth Sampling (Run 2)

**Hypothesis:** If we sample depth from a Poisson distribution (mean ~7, no hard cap, clipped to [2, ~24]), the model will occasionally train at high depths (14-20+), learning to produce useful representations at those depths.

**Distribution shape:** Most steps cheap (depth 5-9), ~5% hit depth 14+, ~1% hit 18+. Heavy tail gives exposure to deep iterations without blowing training budget.

## Run 2: Poisson depth sampling (2024-03-21)

**Config:** Poisson(mean=8), clipped to [2, 24], eval at depth 24 during training. Mid-training eval uses max_depth=24.

**Training trajectory (4xH100, 1200s wallclock):**
| Step | Val BPB (depth 24) | Step Avg |
|------|--------------------|----------|
| 1000 | 1.4251             | 219ms    |
| 2000 | 1.4077             | 205ms    |
| 3000 | 1.3469             | 204ms    |
| 4000 | 1.3543 (spike)     | 200ms    |
| 5000 | 1.3009             | 198ms    |
| 6000 | 1.2603             | 197ms    |
| 6093 | 1.2564             | 197ms    |

Stopped at step 6093 (wallclock cap). Fewer steps than Run 1 due to higher avg depth. EMA applied.
Peak memory: 55888 MiB (vs 26300 MiB in Run 1 — deep steps use more memory).

**Depth sweep results:**
| Depth | Run 2 (Poisson) | Run 1 (Uniform) | Δ       |
|-------|-----------------|-----------------|---------|
| 4     | 1.2424          | 1.2317          | -0.011  |
| **8** | **1.2196**      | **1.2150**      | -0.005  |
| 12    | 1.2228          | 1.2288          | +0.006  |
| 16    | 1.2314          | 1.2546          | +0.023  |
| 20    | 1.2437          | 1.2968          | +0.053  |

**Sliding window (depth 20, stride 64): 1.2186 BPB** (vs Run 1: 1.2703)

**Conclusion:** Poisson sampling massively improved depth extrapolation. Depth 20 improved by 0.053 BPB. The depth curve is now nearly flat (depth 8→20 only degrades 0.024 vs 0.082 in Run 1). However, depth 8 is still the best — more iterations don't yet *improve* performance, they just degrade much less. Best result: **1.2186 BPB** vs baseline 1.1248 (still 0.094 behind).

**Key observation:** Fewer training steps (6093 vs 8605) due to higher avg compute per step. The model saw ~6K steps vs ~8.6K — nearly 30% fewer weight updates. This may explain why shallow-depth performance (depth 4, 8) is slightly worse than Run 1.

**Submission size:** 9.91MB int8+zlib

## Ideas for Next Runs

**To close the gap to baseline (0.094 BPB):**
- Lightweight auxiliary loss at 1-2 random intermediate iterations (norm+logits only, no exit block — nearly free compute). Forces each iteration to produce useful predictions.
- Jacobian regularization on recurrent block (soft contraction constraint)
- Stronger x0 injection (clamp resid_mix[1] minimum to prevent forgetting input)
- Step-index conditioning (adaLN-style, let block know which iteration it's on)
- Try lower Poisson mean (e.g. 6) to get more training steps while keeping tail exposure
- Widen the model further (we use 9.9MB of 16MB budget)

## Reference: Baseline
- 9 specialized layers, U-net skips, int6 quantization
- **1.1248 BPB** (target to beat)
