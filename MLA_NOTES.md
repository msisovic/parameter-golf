# MLA (Multi-head Latent Attention) Experiments

## Baseline Config
```
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1
MUON_WD=0.04 ADAM_WD=0.04
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=1200 EVAL_STRIDE=64
```

## Architecture
- GQA baseline: 8 query heads, 4 KV heads, head_dim=64, dim=512
- GQA attn params per layer: 786,432 (c_q:262K, c_k:131K, c_v:131K, proj:262K)
- MLP params per layer: 1,572,864 (fc:786K, proj:786K with mlp_mult=3.0)
- Attn ~33%, MLP ~67% of per-layer params

## MLA Design
Replace separate K/V projections with shared low-rank latent:
- `x -> W_dkv -> c_kv` (dim x d_c, shared across all heads)
- `c_kv -> W_uk -> K` (d_c x dim, all heads)
- `c_kv -> W_uv -> V` (d_c x dim, all heads)
- Q, output proj unchanged
- Partial RoPE applied same as GQA (first rope_dims of head_dim)
- FA3 compatible: Q/K/V all have same head_dim

## Parameter Savings (per layer / across 11 layers)
| d_c | MLA attn params | Saved vs GQA | Saved 11 layers |
|-----|----------------|--------------|-----------------|
| 64  | 622,592        | 163,840 (20.8%) | 1,802,240    |
| 96  | 671,744        | 114,688 (14.6%) | 1,261,568    |
| 128 | 720,896        | 65,536 (8.3%)   | 720,896      |
| 170 | ~786K          | ~0 (break-even) | ~0           |

## Experiment Results

### Exp 0: GQA Baseline (11 layers, no MLA)
- Status: DONE
- Steps: 7299 (wallclock capped at 1200s), step_avg: 164.41ms
- Params: 26,829,913
- val_bpb trajectory: 1.2614 (2k) → 1.2369 (3k) → 1.2296 (4k) → 1.2076 (5k) → 1.1828 (6k) → 1.1499 (7k) → 1.1415 (7.3k)
- final_int6_roundtrip val_bpb: 1.1483
- **final_int6_sliding_window val_bpb: 1.1248** (stride=64)
- Peak memory: 20713 MiB

### Exp 1: MLA d_c=128 (same layers/heads, just swap attn)
- Status: RUNNING
- Params: 26,109,017 (saved 720,896 vs baseline)
- step_avg: ~176ms (vs 164ms baseline, ~8% slower due to extra matmuls)
- val_bpb @ step 3000: 1.2519 (vs baseline 1.2369 @ 3k)
