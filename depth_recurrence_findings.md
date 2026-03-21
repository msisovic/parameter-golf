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

**Fallback if Poisson alone doesn't work:**
- Lightweight auxiliary loss at 1-2 random intermediate iterations (norm+logits only, no exit block — nearly free compute)
- Jacobian regularization on recurrent block (soft contraction constraint)
- Stronger x0 injection (clamp resid_mix[1] minimum)
- Step-index conditioning (adaLN-style, let block know which iteration it's on)

## Reference: Baseline
- 9 specialized layers, U-net skips, int6 quantization
- **1.1248 BPB** (target to beat)
