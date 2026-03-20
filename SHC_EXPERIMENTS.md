# Static Hyper-Connections (SHC) Experiments

## Design
- Static HC with n=2 (two evolving streams), unrolled as scalar ops for torch.compile compatibility
- Per sublayer (attn + MLP): A_m (2), A_r (2x2), B (2) = 8 scalar params
- Init: A_m=[1,0], B=[1,0], A_r=I => equivalent to standard pre-norm residual at init
- Keeps resid_mix (x0 anchor) on top of HC
- Replaces U-Net skip connections (removed encoder/decoder split, skip_weights)

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

| Metric              | Baseline | HC (n=2) | Delta |
|---------------------|----------|----------|-------|
| step_avg            | 616ms    | 671ms    | +9%   |
| steps completed     | 2922     | 2681     | -8%   |
| val_bpb (pre-quant) | 1.1887   | 1.1924   | +0.0037 |
| val_bpb @ step 1000 | 1.3060   | 1.2987   | -0.0073 |
| val_bpb @ step 2000 | 1.2386   | 1.2301   | -0.0085 |
| int6 roundtrip bpb  | 1.2154   | 1.2274   | +0.012 |
| sliding window bpb  | 1.1930   | 1.2047   | +0.012 |
| peak memory         | 21205 MiB| 21723 MiB| +518 MiB |

**Per-step HC wins** (~0.008 bpb better at matched steps).
**Per-wallclock baseline wins** (+9% step overhead = 241 fewer steps overcomes per-step gain).

## Key Takeaways
- HC adds ~9% step overhead with resid_mix, ~3.3% without
- HC is more sample-efficient (~0.008 bpb better per step)
- Post-quant penalty is similar when sufficiently trained
- Removing resid_mix saves overhead but loses some quality — x0 anchor has value
- On 8xH100 with DDP, overhead may be proportionally smaller (communication-dominated)

## Paper Findings (relevant to our setup)
- **No weight decay on static HC params** — paper explicitly says this
- **Alternating Am init**: Am = e_{k mod n} per layer (layer 0 reads stream 0, layer 1 reads stream 1, ...)
- **B init**: all-ones (not one-hot like our current [1,0])
- **Output scaling**: proj weights scaled by 1/√n at init
- **No special LR** for HC params in the paper

## Next Steps
- Remove weight decay from HC params (paper says to)
- Try alternating Am init (paper's recommended approach)
- Try resid_mix as fixed third lane in HC (cheap x0 anchor without per-dim params)
- Experiment with HC-specific learning rate
- Try n=4 (overhead may only be ~18% if memory-bound)
- Benchmark on 8xH100 where DDP communication may mask the overhead
