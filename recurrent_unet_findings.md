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

## Run 2: 2+2+2+2 dim=640, more QAT (NEXT)

**Changes from Run 1:**
- 2 entry + 2 encoder (×N) + 2 decoder (×N) + 2 exit blocks (8 unique blocks)
- Lower QAT threshold (0.3 instead of 0.1) for more quantization-aware training time
- Pre-quant sliding window eval added to measure exact quant penalty

**Hypothesis:** Extra entry/exit blocks use the 3.2MB budget headroom for better BPB. More QAT time should reduce the 0.015 quant penalty.
