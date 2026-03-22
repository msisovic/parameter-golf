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

## Run 3: Equal-weighted aux loss at every iteration (FAILED)

**Config:** 2 entry blocks (no exit block) + recurrent. Poisson(mean=6). Aux cross-entropy loss at every recurrent iteration, equally weighted (averaged).

**Result: Training diverged.** Val BPB went 1.53 → 1.95 → 3.97 over steps 1000-3000. Killed at step ~3000.

**Root cause:** Equal-weighted aux loss creates conflicting gradients. Shallow iterations (depth 2-3) want representations optimized for immediate decoding. Deep iterations (depth 10+) want representations optimized for further refinement. These objectives conflict, destabilizing training.

**Key finding:** Equal-weighted aux loss at every iteration is **not used by any major paper** in this space.

## Literature Survey: Aux Loss Weighting Schemes

### Option A — Final-only loss + randomized depth (most proven at scale)
**Used by:** Geiping et al. 2025 ("Scaling up Test-Time Compute with Latent Reasoning", arXiv 2502.05171)
- Loss computed **only at the final iteration** after all N recurrent steps
- Depth N sampled from log-normal Poisson distribution (heavy-tailed)
- Truncated backprop through only the last k=8 iterations (saves memory)
- **Scales to 3.5B params / 800B tokens.** Most proven approach for LLMs.
- No aux losses means no conflicting gradients

### Option B — Stop-gradient consistency loss
**Used by:** LoopFormer (ICLR 2026, arXiv 2602.11451), Schwarzschild et al. 2022
- `loss = final_loss + 0.1 * ||stopgrad(h_deep) - h_shallow||²`
- Shallow iterations learn to *match* deep representations without corrupting deep path gradients
- LoopFormer also conditions on step index via adaLN (the block knows which iteration it's on)
- Schwarzschild: run prefix iterations with **detached gradients**, then continue with gradients. Prevents iteration-specific behavior.

### Option C — Linearly increasing weights
**Used by:** CALM (Schuster et al. 2022), RLTT (2025)
- `weight_t = t / sum(1..N)` — final iteration dominates, shallow iterations barely contribute
- CALM: for 8 layers, layer 1 gets weight 1/36, layer 8 gets 8/36
- RLTT progressive: `weight_t = t^alpha / sum(s^alpha)`

### Option D — Learned halting probability weighting
**Used by:** PonderNet (Banino et al. 2021), LoopLM
- Model learns per-step halting probability `lambda_n`
- Loss = `sum(p_n * L_n)` weighted by halting distribution
- KL regularizer against geometric prior prevents always using max steps
- More complex to implement, adds parameters

### Recommendation priority
1. **Option A first** — simplest, most proven, already close to our Run 2 setup
2. **Option B if A doesn't scale** — stop-gradient consistency is the cleanest way to add intermediate supervision
3. **Option C as middle ground** — simple to implement, mild regularization
4. **Option D if we need adaptive halting** — most complex, best for variable-difficulty inputs

## Run 4: Final-only loss, 2 entry blocks, no exit block (FAILED — diverged)

**Architecture change:** 2 entry blocks + 1 recurrent block (no exit block). Same 3 blocks = same 12.1M params.
- Entry blocks specialize on representation lifting
- Recurrent block specializes on iterative refinement
- Loss only at final iteration (Option A from literature)
- Poisson(mean=6) depth sampling

**Result: Training diverged.** Without exit block, the recurrent block must produce decodable representations at every depth — conflicting objectives at different depths.

**Root cause:** No exit block means the recurrent representation space is constrained to be "LM-head-ready" at every iteration. Combined with varied Poisson depths, different training steps want representations optimized for different depths, destabilizing training. This is the "School B" problem (LoopFormer-style) without LoopFormer's compensating tricks (adaLN conditioning, consistency loss).

## Run 5: Geiping-inspired — input injection, exit block, truncated backprop (next)

**Architecture change:** 1 entry block + 1 recurrent block + 1 exit block (3 blocks, same param budget as before).
- **Input injection (Geiping et al. 2025):** At every recurrent iteration, concat entry encoding `e` with recurrent state `s`, project `R^{2d} → R^d` via learned adapter. Anchors recurrence to input, prevents drift.
- **Exit block:** Decouples latent refinement space from decodable space. Recurrent block operates freely; exit block translates to LM-head-compatible representation.
- **Truncated backprop:** Only backprop through last k=8 recurrent iterations. Saves memory AND regularizes (prevents iteration-specific behavior).
- **Removed baseline artifacts:** attn_scale, mlp_scale, resid_mix, ln_scale_factor — these were designed for non-recurrent baselines with unique layers. In weight-shared recurrence they're either constant multipliers or redundant with input injection.
- **Fixed effective depth in init:** Output projections now scaled by `1/√(2*(2+mean_depth+1))` instead of hardcoded 7.
- Poisson(mean=6) depth sampling, final-only loss.

**Hypothesis:** Input injection (the critical missing piece from Runs 1-4) will anchor the recurrence and prevent drift, while the exit block allows the recurrent block to refine freely in latent space. This is "School A" (Geiping-style): proven at 3.5B scale.

**Key papers informing this run:**
- Geiping et al. 2025 ("Scaling up Test-Time Compute with Latent Reasoning", arXiv 2502.05171): 2 prelude + 4×r recurrent + 2 coda, concat+project input injection, truncated backprop k=8, sandwich RMSNorm, random s₀ init
- LoopFormer (ICLR 2026, arXiv 2602.11451): No exit block but uses adaLN + stop-gradient consistency loss to compensate
- Universal Transformer (ICLR 2019): Additive input injection + timestep encoding, simpler but unscaled

## Run 5 Results (2026-03-22)

**Training:** 7239 steps, 1200s wallclock, ~165ms/step avg. 12.88M params (dim=640). Poisson(mean=6).

**Depth sweep results:**
| Depth | Run 5 | Run 2 | Run 1 |
|-------|-------|-------|-------|
| 4     | 1.2333 | 1.2424 | 1.2317 |
| 8     | 1.2249 | 1.2196 | 1.2150 |
| 12    | 1.2246 | 1.2228 | 1.2288 |
| 16    | 1.2246 | 1.2314 | 1.2546 |
| 20    | 1.2247 | 1.2437 | 1.2968 |

**Sliding window (depth 20, stride 64): 1.2004 BPB** (best so far, Run 2: 1.2186)

**Conclusion:** Input injection completely solved drift — depth curve is flat (depth 8→20 spread = 0.0003 vs Run 2's 0.024). However, depth does NOT improve performance past ~8. The recurrence converges to a near-fixed-point by depth 8. Every iteration past 8 is approximately identity.

**Root cause:** Poisson(mean=6) puts ~85% of training at depth ≤8. The model never learns to use depth >8 productively because it almost never trains there. Geiping uses log-normal Poisson(mean=32) — fundamentally different depth distribution.

## Run 6: Log-normal Poisson + smaller model for more steps (next)

**Changes from Run 5:**
- **Log-normal Poisson sampling** (Geiping et al. 2025): `τ ~ N(log(μ) - σ²/2, σ), r ~ Poisson(exp(τ)) + 1` with mean=16, σ=0.5. Heavy tail reaches depth 30-48. Substantial training mass at depth 10-24.
- **Smaller model** (dim=512, 8 heads, 4 KV heads): ~36% fewer params → faster steps → more training steps in 1200s wallclock. Compensates for higher avg depth costing more per step.
- Max depth raised to 48, eval depth 32.
- Depth sweep extended to [4, 8, 12, 16, 24, 32].

**Hypothesis:** The model needs to frequently train at high depths (16-32+) to learn that later iterations should produce meaningfully different/better representations. Log-normal Poisson ensures ~50% of training at depth ≥16, unlike plain Poisson(mean=6) where only ~1% was at depth ≥14.

## Reference: Baseline
- 9 specialized layers, U-net skips, int6 quantization
- **1.1248 BPB** (target to beat)
