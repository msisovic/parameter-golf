"""Extended tile sweep for backward recompute kernels — explore larger BK and wider tiles."""
import torch
import triton
from train_gpt import (
    fused_fp8_softcap_ce_dx_recompute_kernel,
    fused_fp8_softcap_ce_dw_recompute_kernel,
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


def run_dx(BM, BN, BK, warps, stages=1):
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


def run_dw(BM, BN, BK, warps, stages=1):
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


# Near-best from sweep 1 + new configs
print("=== dx recompute extended sweep ===")
dx_configs = [
    # Best from sweep 1
    (64, 128, 128, 4),
    # Try BK=256 (halve inner loop iterations)
    (64, 128, 256, 4),
    (64, 128, 256, 8),
    (32, 128, 256, 4),
    (32, 128, 256, 8),
    # Try bigger BM
    (128, 64, 128, 4),
    (128, 64, 128, 8),
    (128, 128, 128, 4),
    (128, 128, 128, 8),
    (128, 64, 256, 4),
    (128, 64, 256, 8),
    # BK=512 (single pass over K for logit recompute) — minimal grid K dim
    (64, 128, 512, 4),
    (64, 128, 512, 8),
    (32, 128, 512, 4),
]
for BM, BN, BK, warps in dx_configs:
    if BK > K:
        continue
    try:
        bench(lambda BM=BM, BN=BN, BK=BK, warps=warps: run_dx(BM, BN, BK, warps),
              f"dx BM={BM},BN={BN},BK={BK},w={warps}", warmup=5, iters=30)
    except Exception as e:
        print(f"  dx BM={BM},BN={BN},BK={BK},w={warps}: FAILED ({type(e).__name__}: {e})")


print("\n=== dw recompute extended sweep ===")
dw_configs = [
    # Best from sweep 1
    (128, 128, 128, 8),
    (64, 128, 128, 8),
    # Try BK=256
    (64, 128, 256, 4),
    (64, 128, 256, 8),
    (128, 128, 256, 4),
    (128, 128, 256, 8),
    (128, 64, 128, 4),
    (128, 64, 128, 8),
    (128, 64, 256, 4),
    (128, 64, 256, 8),
    # Larger BM for more M-reduction per program
    (256, 64, 128, 4),
    (256, 64, 128, 8),
    (256, 128, 128, 4),
    (256, 128, 128, 8),
    # BK=512
    (64, 128, 512, 4),
    (64, 128, 512, 8),
    (128, 128, 512, 8),
]
for BM, BN, BK, warps in dw_configs:
    if BK > K:
        continue
    try:
        bench(lambda BM=BM, BN=BN, BK=BK, warps=warps: run_dw(BM, BN, BK, warps),
              f"dw BM={BM},BN={BN},BK={BK},w={warps}", warmup=5, iters=30)
    except Exception as e:
        print(f"  dw BM={BM},BN={BN},BK={BK},w={warps}: FAILED ({type(e).__name__}: {e})")
