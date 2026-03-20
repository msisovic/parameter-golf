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

### 10-min runs

| Metric              | Baseline | HC (n=2) |
|---------------------|----------|----------|
| step_avg            | 628ms    | 672ms (+7%) |
| steps completed     | 956      | 893      |
| val_bpb (pre-quant) | 1.3323   | 1.3319   |
| int6 roundtrip bpb  | 1.9367   | 2.0814   |

Post-quant penalty was large for both but worse for HC — caused by undertraining
(only 4 SWA checkpoints), not HC itself.

### 10-min: HC without resid_mix

Removed resid_mix (x0 anchor), init h1=zeros instead of clone.

| Metric              | Baseline | HC+resid_mix | HC no resid_mix |
|---------------------|----------|-------------|-----------------|
| step_avg            | 628ms    | 672ms (+7%) | 649ms (+3.3%)   |
| steps completed     | 956      | 893         | 925             |
| val_bpb (pre-quant) | 1.3323   | 1.3319      | 1.3333          |

Removing resid_mix cut overhead from 9% to 3.3%, but val_bpb regressed slightly —
the x0 anchor was providing value that HC didn't learn to replace in 925 steps.

### 10-min: HC v2 — no WD on HC params + alternating Am init (paper recipe)

Per paper: Am=e_{k mod 2} (alternating), B=[1,1], no weight decay on HC params.
Resid_mix kept.

| Metric              | Baseline | HC v1 (uniform, WD) | HC v2 (alt init, no WD) |
|---------------------|----------|---------------------|-------------------------|
| step_avg            | 628ms    | 672ms (+7%)         | 674ms (+7%)             |
| steps completed     | 956      | 893                 | 891                     |
| val_bpb (pre-quant) | 1.3323   | 1.3319              | **1.3255** (-0.0068)    |
| int6 roundtrip bpb  | 1.9367   | 2.0814              | 2.0253                  |

HC v2 is a clear per-step win. The alternating init + no WD lets HC learn
better routing even in ~900 steps.

HC param interpretability (converged values):
- Am alternation preserved: even blocks ≈ [1,0], odd blocks ≈ [0,1]
- MLP B values larger (~1.3) than attn B (~0.8) — MLP distributes more to both streams
- Ar develops negative off-diagonals — decorrelation between streams
- Late layers show strongest cross-stream effects (block 10: Ar[1,0]=-0.46)

### 30-min runs

| Metric              | Baseline | HC v1 (uniform, WD) | HC v2 (alt init, no WD) |
|---------------------|----------|---------------------|-------------------------|
| step_avg            | 616ms    | 671ms (+9%)         | 671ms (+9%)             |
| steps completed     | 2922     | 2681                | 2681                    |
| val_bpb (pre-quant) | 1.1887   | 1.1924 (+0.0037)    | 1.2014 (+0.0127)        |
| val_bpb @ step 1000 | 1.3060   | 1.2987 (-0.0073)    | 1.3078 (+0.0018)        |
| val_bpb @ step 2000 | 1.2386   | 1.2301 (-0.0085)    | 1.2395 (+0.0009)        |
| int6 roundtrip bpb  | 1.2154   | 1.2274 (+0.012)     | 1.2362 (+0.021)         |
| SWA checkpoints     | —        | 8                   | 8                       |
| peak memory         | 21205 MiB| 21723 MiB           | 21723 MiB               |

**HC v1 per-step wins** (~0.008 bpb better at matched steps), but per-wallclock baseline wins.
**HC v2 loses at 30 min** — the no-WD + alternating init that helped at 10 min hurts at 30 min.
Without WD, HC params drift too far: many Am values go negative, B values go negative in
later layers. The alternating init pattern is not preserved — params diverge significantly
from initialization. HC v1 (uniform init with WD) remains the better HC variant at scale.

### 10-min: HC v3 — stream sum fix (x = h0 + h1)

v1-v2 had a bug: only h0 was used as output, discarding h1 entirely. Paper says to
sum all streams row-wise. This meant h1 got zero gradient through the loss — half of
HC was dead. v3 fixes this: x = h0 + h1.

| Metric              | Baseline | HC v2 (h0 only) | HC v3 (h0 + h1) |
|---------------------|----------|-----------------|------------------|
| step_avg            | 628ms    | 674ms (+7%)     | 666ms (+6%)      |
| steps completed     | 956      | 891             | 902              |
| val_bpb (pre-quant) | 1.3323   | **1.3255**      | 1.3283           |
| int6 roundtrip bpb  | 1.9367   | 2.0253          | **2.0012**       |
| peak memory         | —        | —               | 22771 MiB        |

Stream sum slightly regressed pre-quant bpb vs v2 but improved int6 roundtrip.
HC params are much more stable — B values meaningful for both streams, Am stays
close to init. Step overhead dropped slightly (6% vs 7%). Needs 30-min validation.

**Note**: v1/v2 results are invalidated by the h1-discard bug — HC was not functioning
as designed. v3 is the first correct implementation.

## Key Takeaways
- v1/v2 had a critical bug: h1 was discarded, so HC was half-broken
- HC v3 (correct sum) adds ~6% step overhead with resid_mix
- 10-min int6 numbers are unreliable (only 4 SWA checkpoints) — 30-min needed
- Post-quant penalty normalizes with sufficient training
- On 8xH100 with DDP, overhead may be proportionally smaller (communication-dominated)

## Paper Findings (relevant to our setup)
- **No weight decay on static HC params** — paper explicitly says this
- **Alternating Am init**: Am = e_{k mod n} per layer
- **B init**: all-ones
- **Ar init**: Identity
- **Output scaling**: proj weights std scaled by 1/√n at init (N/A for our zero-init proj)
- **Final output**: sum all n streams row-wise (was missing in v1/v2!)
- **No special LR** for HC params in the paper

## Next Steps
- 30-min HC v3 run to validate stream sum fix with reliable int6 numbers
- Experiment with HC-specific learning rate (higher LR for routing convergence)
- Try n=4 (overhead may only be ~18% if memory-bound)
- Benchmark on 8xH100 where DDP communication may mask the overhead
