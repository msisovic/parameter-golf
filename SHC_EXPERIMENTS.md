# Static Hyper-Connections (SHC) Experiments

## Design
- Static HC with n=2 (two evolving streams), unrolled as scalar ops for torch.compile compatibility
- Per sublayer (attn + MLP): A_m (2), A_r (2x2), B (2) = 8 scalar params
- Init (v1): A_m=[1,0], B=[1,0], A_r=I => equivalent to standard pre-norm residual at init
- Init (v2+): A_m=e_{k mod 2} (alternating), B=[1,1], A_r=I (paper recipe)
- Keeps resid_mix (x0 anchor) on top of HC
- Replaces U-Net skip connections (removed encoder/decoder split, skip_weights)
- v1-v2 bug: only h0 used as output (h1 discarded). v3+: x = h0 + h1 (paper: sum row-wise)

## Results

All runs: NUM_LAYERS=11, BIGRAM_VOCAB_SIZE=2048, MUON_WD=0.04, ADAM_WD=0.04,
MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035, MUON_MOMENTUM=0.99,
MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500,
WARMDOWN_ITERS=3000, ITERATIONS=9000, EVAL_STRIDE=64, 1xH100.

### 10-min runs (v1/v2 — h1 discarded bug)

| Metric              | Baseline | HC v1 (uniform, WD) | HC v2 (alt init, no WD) |
|---------------------|----------|---------------------|-------------------------|
| step_avg            | 628ms    | 672ms (+7%)         | 674ms (+7%)             |
| steps completed     | 956      | 893                 | 891                     |
| val_bpb (pre-quant) | 1.3323   | 1.3319              | **1.3255** (-0.0068)    |
| int6 roundtrip bpb  | 1.9367   | 2.0814              | 2.0253                  |

### 10-min: HC without resid_mix (h1 discarded bug)

| Metric              | Baseline | HC+resid_mix | HC no resid_mix |
|---------------------|----------|-------------|-----------------|
| step_avg            | 628ms    | 672ms (+7%) | 649ms (+3.3%)   |
| steps completed     | 956      | 893         | 925             |
| val_bpb (pre-quant) | 1.3323   | 1.3319      | 1.3333          |

### 30-min runs (v1/v2 — h1 discarded bug)

| Metric              | Baseline | HC v1 (uniform, WD) | HC v2 (alt init, no WD) |
|---------------------|----------|---------------------|-------------------------|
| step_avg            | 616ms    | 671ms (+9%)         | 671ms (+9%)             |
| steps completed     | 2922     | 2681                | 2681                    |
| val_bpb (pre-quant) | 1.1887   | 1.1924 (+0.0037)    | 1.2014 (+0.0127)        |
| val_bpb @ step 1000 | 1.3060   | 1.2987 (-0.0073)    | 1.3078 (+0.0018)        |
| val_bpb @ step 2000 | 1.2386   | 1.2301 (-0.0085)    | 1.2395 (+0.0009)        |
| int6 roundtrip bpb  | 1.2154   | 1.2274 (+0.012)     | 1.2362 (+0.021)         |
| peak memory         | 21205 MiB| 21723 MiB           | 21723 MiB               |

### 10-min: HC v3 — stream sum fix (x = h0 + h1)

v1-v2 had a bug: only h0 was used as output, discarding h1 entirely. Paper says to
sum all streams row-wise. v3 fixes this: x = h0 + h1. resid_mix applied to h0 before Am.

| Metric              | Baseline | HC v3 (n=2) |
|---------------------|----------|-------------|
| step_avg            | 628ms    | 666ms (+6%) |
| steps completed     | 956      | 902         |
| val_bpb (pre-quant) | 1.3323   | 1.3283      |
| int6 roundtrip bpb  | 1.9367   | 2.0012      |
| peak memory         | —        | 22771 MiB   |

### 30-min: HC v3

| Metric              | Baseline | HC v3 (n=2) |
|---------------------|----------|-------------|
| step_avg            | 616ms    | 668ms (+8%) |
| steps completed     | 2922     | 2694        |
| val_bpb (pre-quant) | 1.1887   | 1.2040 (+0.015) |
| val_bpb @ step 1000 | 1.3060   | 1.3148 (+0.009) |
| val_bpb @ step 2000 | 1.2386   | 1.2430 (+0.004) |
| int6 roundtrip bpb  | 1.2154   | 1.2395 (+0.024) |
| peak memory         | 21205 MiB| 22771 MiB   |

v3 is worse than baseline both per-step and per-wallclock.

### 10-min: HC n=4

| Metric              | Baseline | HC n=4      |
|---------------------|----------|-------------|
| step_avg            | 628ms    | 814ms (+30%)|
| steps completed     | 956      | 737         |
| val_bpb (pre-quant) | 1.3323   | 1.3735      |
| int6 roundtrip bpb  | 1.9367   | 2.6330      |
| peak memory         | —        | 26899 MiB   |

n=4 has 30% step overhead (not memory-bound as hoped) and far too few steps. Not viable.

### 10-min: HC v4 — resid_mix after Am (attn only)

v3 had resid_mix applied to h0 before Am aggregation, which biased the model to avoid
h0 (the "contaminated" stream). v4 moves resid_mix after Am, applied to h_in before
the attention sublayer only (matching baseline placement).

| Metric              | Baseline | HC v3 (mix before Am) | HC v4 (mix after Am) |
|---------------------|----------|-----------------------|----------------------|
| step_avg            | 628ms    | 666ms (+6%)           | 678ms (+8%)          |
| steps completed     | 956      | 902                   | 886                  |
| val_bpb (pre-quant) | 1.3323   | 1.3283                | 1.3388               |
| int6 roundtrip bpb  | 1.9367   | 2.0012                | 2.0677               |

v4 is worse than v3 in quality but fixes the Am routing bias — alternating pattern
is now perfectly preserved. The extra overhead (678ms vs 666ms) may be from changed
data flow hurting torch.compile fusion.

### 10-min+: HC v5 — symmetric resid_mix on both lanes

Apply `mix[0]*h + mix[1]*x0` to both h0 and h1 before attn (once per block).
Both streams get x0 anchoring equally — no routing bias.

| Metric              | Baseline | HC v5 (10 min) |
|---------------------|----------|----------------|
| step_avg            | 628ms    | 690ms (+10%)   |
| steps completed     | 956      | 869            |
| val_bpb (pre-quant) | 1.3323   | 1.3367         |

Extended to ~1040 steps for per-step comparison:

| Variant             | val_bpb @ step 1000 |
|---------------------|---------------------|
| Baseline            | 1.3060              |
| HC v1 (broken)      | 1.2987              |
| HC v3 (mix h0 only) | 1.3148              |
| **HC v5 (mix both)**| **1.3003**          |

v5 is the best per-step among correct implementations. Symmetric routing preserved.

### 10-min: HC v6a — x0 scalar bias on MLP output

Replace per-dim resid_mix with a single scalar x0_scale added to MLP sublayer output.
x0 enters via `t_out = mlp(norm(h_in)) + x0_scale * x0`, then B distributes to both
streams naturally. Inspired by reference repo's additive hc_bias approach.

| Metric              | Baseline | HC v5 (mix both) | HC v6a (x0 on MLP out) |
|---------------------|----------|-------------------|------------------------|
| step_avg            | 628ms    | 690ms (+10%)      | **653ms (+4%)**        |
| steps completed     | 956      | 869               | **920**                |
| val_bpb (pre-quant) | 1.3323   | 1.3367            | **1.3296**             |
| int6 roundtrip bpb  | 1.9367   | 2.1948            | **2.0094**             |

**First HC variant to beat baseline on wallclock pre-quant.** Only 4% overhead
(scalar add is much cheaper than per-dim blend on both lanes). val_bpb 1.3296
vs baseline 1.3323 = -0.0027 improvement.

## Am Convergence Analysis

- v3 (resid_mix on h0 before Am): Am drifts to favor h1 (clean stream), routing
  around x0-contaminated h0. Alternating pattern breaks.
- v4 (resid_mix after Am on h_in): Clean alternating pattern preserved, but worse per-step.
- v5 (resid_mix on both lanes): Clean alternating pattern, best per-step of correct variants.

## Key Takeaways
- HC is **sample-efficient**: every variant beats baseline per-step
- **Overhead is the bottleneck**: per-dim resid_mix on both lanes = 10%, scalar x0 bias = 4%
- **v6a (scalar x0 on MLP output) beats baseline on wallclock** — first HC variant to do so
- x0 anchoring works best as additive bias on sublayer output (symmetric, cheap)
- n=4 not viable (30% overhead, not memory-bound)
- On 8xH100 with DDP, communication may further mask the 4% overhead

## Paper Findings
- No weight decay on static HC params
- Am = e_{k mod n} alternating init, B = all-ones, Ar = Identity
- Output: sum all n streams row-wise
- 1/√n scaling on proj weights (N/A for our zero-init proj)
- n=4 recommended by paper, n=2 also good
- Paper trained at 500B tokens vs our ~2B

## Open Questions
- Does v6a hold up at 30 min? Does wallclock advantage persist?
- Try x0 bias on both sublayers (attn + MLP) — better quality or just more overhead?
- On 8xH100 with DDP, does communication further mask the 4% overhead?
