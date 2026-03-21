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
- Status: DONE
- Params: 26,109,017 (saved 720,896 vs baseline)
- Steps: 7018 (vs 7299 baseline), step_avg: 170.99ms (~4% slower)
- val_bpb trajectory: 1.3464 (1k) → 1.2519 (3k) → 1.2409 (4k) → 1.1868 (6k) → 1.1533 (7k) → 1.1532 (7.0k)
- final_int6_roundtrip val_bpb: 1.1591
- **final_int6_sliding_window val_bpb: 1.1357** (stride=64)
- Peak memory: 23044 MiB
- **Result: 0.011 worse than baseline (1.1357 vs 1.1248)**
- Analysis: MLA adds overhead from 3 matmuls (W_dkv, W_uk, W_uv) vs 2 (c_k, c_v).
  Fewer steps + fewer params (not reinvested) = worse result.
  Need to reinvest saved params and/or reduce latency.
- NOTE: all runs used zlib (zstandard not installed). PR #315 used zstd-22 (15.6MB).
  Installed zstandard for future runs. Artifact sizes above are inflated.

### Exp 2: MLA d_c=64 + wider MLP (mlp_mult=3.31, fused W_ukv)
- Status: DONE
- Params: 26,807,385 (reinvested savings into MLP width)
- Steps: 6323 (vs 7299 baseline), step_avg: 189.79ms (~15% slower!)
- val_bpb trajectory: 1.2927 (2k) → 1.2657 (3k) → 1.2133 (5k) → 1.1791 (6k) → 1.1695 (6.3k)
- final_int6_roundtrip val_bpb: 1.1749
- **final_int6_sliding_window val_bpb: 1.1520** (stride=64)
- Peak memory: 23621 MiB
- **Result: 0.027 worse than baseline. d_c=64 too aggressive, plus 15% slower = ~1000 fewer steps.**

### Exp 3: Materialized MLA d_c=128 (W_k = W_down @ W_up_k in forward)
- Status: DONE
- Params: 25,388,121 (saved 1,441,792 vs baseline — 50% KV compression)
- Steps: 7253 (vs 7299 baseline), step_avg: 165.45ms (**same speed as baseline!**)
- val_bpb trajectory: 1.3430 (1k) → 1.2773 (2k) → 1.2186 (5k) → 1.1940 (6k) → 1.1625 (7k) → 1.1562 (7.3k)
- final_int6_roundtrip val_bpb: 1.1613
- **final_int6_sliding_window val_bpb: 1.1379** (stride=64)
- Artifact size: 16.68MB (zstd) — still over 16MB limit
- Peak memory: 20679 MiB
- **Result: 0.013 worse than baseline. Speed identical. Quality gap is purely from rank-128 KV constraint.**
- The materialized approach (computing W_down @ W_up_k inline) eliminates the speed overhead entirely.
  This is the right MLA implementation strategy for wallclock-limited settings.

### Exp 4: Materialized MLA d_c=128 + 12 layers (ABORTED)
- Killed early — 11-layer MLA already over 16MB, 12 layers would be worse
- Step speed was ~180ms (extra layer adds ~10%)

### Exp 5: Materialized MLA d_c=128 + 12 layers + force int6 quantization
- Status: DONE
- Params: 27,618,913
- Steps: 6701 (vs 7299 baseline), step_avg: 179.08ms
- val_bpb trajectory: 1.3449 (1k) → 1.2766 (2k) → 1.2484 (3k) → 1.2042 (5k) → 1.1749 (6k) → 1.1535 (6.7k)
- final_int6_roundtrip val_bpb: 1.1595
- **final_int6_sliding_window val_bpb: 1.1360** (stride=64)
- Artifact size: 16.54MB (zstd) — **still over 16MB limit**
- Peak memory: 22314 MiB
- **Result: 0.011 worse than baseline. Force int6 helped (16.5MB vs 16.68MB for 11L without it)**
  **but 12 layers has more total params (27.6M) so artifact still too large.**
- Note: val_bpb overtook baseline at step 5k (1.2042 vs 1.2076) — extra depth helps,
  but ~600 fewer steps (wallclock penalty) erases the advantage by end of training.

## Key Findings
1. Naive MLA (sequential matmuls) adds ~7-15% step time overhead — kills wallclock-limited runs
2. **Materialized MLA (W_k = W_down @ W_up_k computed inline) has ZERO speed overhead** — this is the way
3. d_c=64 compresses KV too aggressively, hurting quality
4. d_c=128 loses ~0.013 val_bpb from rank constraint, saves 1.4M params / ~50% KV compression
5. The competition is BOTH param-limited (16MB artifact) AND wallclock-limited (10min)
6. MLA artifact size is larger than baseline even with force int6 fix:
   - 11L MLA: 16.68MB without fix → would be ~15.5MB with fix (but quality too low)
   - 12L MLA: 16.54MB with fix (more params offsets compression savings)
   - Baseline 11L GQA: ~15.6MB
7. 12-layer MLA overtakes baseline mid-training (step 5k) but loses advantage due to fewer total steps

## Next Experiment: MLA with 8 KV heads (up from 4)

MLA makes increasing KV heads cheap. Per-layer KV param comparison:
- GQA 4 KV heads: 2 × 512×256 = 262,144
- GQA 8 KV heads: 2 × 512×512 = 524,288 (doubles!)
- MLA d_c=128, 8 KV heads: 512×128 + 128×512 + 128×512 = 196,608

So MLA with 8 KV heads uses **fewer params than GQA with 4 KV heads**, while doubling
KV expressiveness (less KV sharing → more specialized keys/values per query head group).
This is the natural way to reinvest MLA's param savings: more KV heads, not more layers.

### Exp 6 plan: 12L MLA d_c=128, 8 KV heads (up from 4)
- Expected params: ~27.6M + 11×(128×256 + 128×256) = ~27.6M + 720K ≈ 28.3M
  Wait — going from 4→8 KV heads with MLA increases w_up_k and w_up_v from (128,256)→(128,512),
  so delta per layer = 2 × 128 × 256 = 65,536. Over 12 layers = 786,432 extra params.
  But we also removed c_k/c_v savings... let me recalculate vs GQA 4KV baseline:
  - GQA 4KV: c_k(512×256) + c_v(512×256) = 262,144/layer
  - MLA 8KV: w_down(512×128) + w_up_k(128×512) + w_up_v(128×512) = 196,608/layer
  - Still saves 65,536/layer vs GQA 4KV — while having 2× the KV heads!
- Hypothesis: more KV heads improves quality enough to overcome the rank-128 constraint
- Risk: artifact size — more params in w_up tensors means bigger artifact

## Conclusion (so far)
MLA is not beneficial in its current configurations:
- Wallclock-limited: materialized MLA solves this (zero overhead), but extra layer costs ~10% steps
- Param-limited (16MB artifact): 11L MLA fits but quality gap too large; 12L MLA busts limit
- Quality: low-rank KV constraint hurts val_bpb by ~0.011-0.013
- Net effect: no configuration found yet that beats baseline on all three constraints
- **Next bet: use MLA's cheap KV heads to improve quality without adding net params**
