"""Microbench for backward path: nologit recompute vs split-path stored-logits.
Profiles dx and dw independently + sweeps tile shapes for recompute kernels."""
import torch
import triton
from train_gpt import (
    fused_fp8_softcap_ce_dx_recompute_kernel,
    fused_fp8_softcap_ce_dw_recompute_kernel,
    fused_softcap_ce_dx_kernel,
    fused_softcap_ce_dw_kernel,
)


def _alloc(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(_alloc)

torch.manual_seed(0)
device = "cuda"
M, K, V = 98304, 512, 8192
softcap = 30.0
A = 2.0 * softcap
C = softcap / 2.0

x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.3
w = torch.randn(V, K, device=device, dtype=torch.bfloat16) * 0.1
targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)

x_scale = x.detach().abs().amax().float().clamp_min(1e-12) / 448.0
w_scale = w.detach().abs().amax().float().clamp_min(1e-12) / 448.0
x_fp8 = (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
w_fp8 = (w.float() / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
weight_fp8_t = w_fp8.t()
scale_ab = x_scale * w_scale

lse = torch.randn(M, dtype=torch.float32, device=device)
grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
logits_bf16 = torch._scaled_mm(
    x_fp8, w_fp8.t(),
    scale_a=x_scale.view(1, 1), scale_b=w_scale.view(1, 1),
    out_dtype=torch.bfloat16,
)

grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
grad_weight = torch.empty(V, K, dtype=torch.float32, device=device)


def bench(fn, name, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"  {name}: {ms:.3f} ms")
    return ms


# ============ Split-path baselines (separate dx / dw) ============
print("=== Split-path baselines (bf16 logits saved) ===")

def run_split_dx():
    grid = (triton.cdiv(M, 32), triton.cdiv(K, 64))
    fused_softcap_ce_dx_kernel[grid](
        grad_input, grad_out, lse, logits_bf16, targets, w,
        grad_input.stride(0), grad_input.stride(1),
        logits_bf16.stride(0), logits_bf16.stride(1),
        w.stride(0), w.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, num_warps=8,
    )

def run_split_dw():
    grid = (triton.cdiv(V, 128), triton.cdiv(K, 64))
    fused_softcap_ce_dw_kernel[grid](
        grad_weight, grad_out, lse, logits_bf16, targets, x,
        grad_weight.stride(0), grad_weight.stride(1),
        logits_bf16.stride(0), logits_bf16.stride(1),
        x.stride(0), x.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_warps=8,
    )

ms_split_dx = bench(run_split_dx, "split dx (BM=32,BN=256,BK=64)")
ms_split_dw = bench(run_split_dw, "split dw (BM=64,BN=128,BK=64)")
print(f"  TOTAL: {ms_split_dx + ms_split_dw:.3f} ms")


# ============ Nologit recompute: current configs ============
print("\n=== Nologit recompute (current configs) ===")

def run_nologit_dx(BM, BN, BK, warps):
    grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))
    fused_fp8_softcap_ce_dx_recompute_kernel[grid](
        grad_input, grad_out, lse, targets, x_fp8, w, weight_fp8_t, scale_ab,
        grad_input.stride(0), grad_input.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        w.stride(0), w.stride(1),
        weight_fp8_t.stride(0), weight_fp8_t.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, num_warps=warps,
    )

def run_nologit_dw(BM, BN, BK, warps):
    grid = (triton.cdiv(V, BN), triton.cdiv(K, BK))
    fused_fp8_softcap_ce_dw_recompute_kernel[grid](
        grad_weight, grad_out, lse, targets, x, x_fp8, weight_fp8_t, scale_ab,
        grad_weight.stride(0), grad_weight.stride(1),
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        weight_fp8_t.stride(0), weight_fp8_t.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, num_warps=warps,
    )

bench(lambda: run_nologit_dx(8, 256, 64, 4), "recompute dx CURRENT (BM=8,BN=256,BK=64,w=4)")
bench(lambda: run_nologit_dw(32, 128, 64, 8), "recompute dw CURRENT (BM=32,BN=128,BK=64,w=8)")


# ============ dx recompute sweep ============
print("\n=== dx recompute tile sweep ===")
# Register budget: logits_acc[BM,BN] + acc[BM,BK] + scalars
# BM=64, BN=128, BK=64 → 8K + 4K = 12K floats = 48KB — fits
# BM=32, BN=256, BK=64 → 8K + 2K = 10K = 40KB — fits
# BM=64, BN=256, BK=64 → 16K + 4K = 20K = 80KB — tight
dx_configs = [
    (16, 128, 64, 4),
    (16, 256, 64, 4),
    (32, 128, 64, 4),
    (32, 128, 64, 8),
    (32, 256, 64, 4),
    (32, 256, 64, 8),
    (32, 128, 128, 4),
    (32, 128, 128, 8),
    (64, 128, 64, 4),
    (64, 128, 64, 8),
    (64, 128, 128, 4),
    (64, 128, 128, 8),
    (64, 256, 64, 8),
]
for BM, BN, BK, warps in dx_configs:
    try:
        bench(lambda BM=BM, BN=BN, BK=BK, warps=warps: run_nologit_dx(BM, BN, BK, warps),
              f"dx BM={BM},BN={BN},BK={BK},w={warps}", warmup=5, iters=30)
    except Exception as e:
        print(f"  dx BM={BM},BN={BN},BK={BK},w={warps}: FAILED ({e})")


# ============ dw recompute sweep ============
print("\n=== dw recompute tile sweep ===")
# dw: grid (V/BN, K/BK), inner loop over M in blocks of BM
# logits_acc[BM,BN] + acc[BN,BK]
dw_configs = [
    (32, 128, 64, 4),
    (32, 128, 64, 8),
    (64, 64, 64, 4),
    (64, 64, 64, 8),
    (64, 128, 64, 4),
    (64, 128, 64, 8),
    (64, 128, 128, 4),
    (64, 128, 128, 8),
    (128, 64, 64, 4),
    (128, 64, 64, 8),
    (128, 128, 64, 4),
    (128, 128, 64, 8),
    (128, 128, 128, 8),
]
for BM, BN, BK, warps in dw_configs:
    try:
        bench(lambda BM=BM, BN=BN, BK=BK, warps=warps: run_nologit_dw(BM, BN, BK, warps),
              f"dw BM={BM},BN={BN},BK={BK},w={warps}", warmup=5, iters=30)
    except Exception as e:
        print(f"  dw BM={BM},BN={BN},BK={BK},w={warps}: FAILED ({e})")


# ============ Theoretical floor ============
print("\n=== Reference: _scaled_mm alone ===")
def run_scaled_mm():
    return torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.view(1, 1), scale_b=w_scale.view(1, 1),
        out_dtype=torch.bfloat16,
    )
bench(run_scaled_mm, "scaled_mm (M×K → M×V, the logit recompute cost)")
