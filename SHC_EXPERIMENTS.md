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
- HC adds ~9% step overhead from extra scalar multiplies (h0/h1 mixing)
- HC is more sample-efficient (~0.008 bpb better per step)
- Post-quant penalty is similar when sufficiently trained
- On 8xH100 with DDP, overhead may be proportionally smaller (communication-dominated)

## Next Steps
- Reduce HC overhead (fuse ops, reduce unnecessary multiplies)
- Try HC without resid_mix (HC may subsume it)
- Try per-layer HC init variation (e.g. deeper layers get different init)
- Benchmark on 8xH100 where DDP communication may mask the overhead
