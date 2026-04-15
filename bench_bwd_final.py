"""Final backward bench: new configs vs split-path, both combined dx+dw."""
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
logits_bf16 = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale.view(1,1), scale_b=w_scale.view(1,1), out_dtype=torch.bfloat16)
grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
grad_weight = torch.empty(V, K, dtype=torch.float32, device=device)


def bench(fn, name, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"  {name}: {ms:.3f} ms")
    return ms


def run_nologit_bwd():
    grid_dx = (triton.cdiv(M, 64), triton.cdiv(K, 256))
    fused_fp8_softcap_ce_dx_recompute_kernel[grid_dx](
        grad_input, grad_out, lse, targets, x_fp8, w, weight_fp8_t, scale_ab,
        grad_input.stride(0), grad_input.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        w.stride(0), w.stride(1),
        weight_fp8_t.stride(0), weight_fp8_t.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, num_warps=8,
    )
    grid_dw = (triton.cdiv(V, 128), triton.cdiv(K, 256))
    fused_fp8_softcap_ce_dw_recompute_kernel[grid_dw](
        grad_weight, grad_out, lse, targets, x, x_fp8, weight_fp8_t, scale_ab,
        grad_weight.stride(0), grad_weight.stride(1),
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        weight_fp8_t.stride(0), weight_fp8_t.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, num_warps=8,
    )


def run_split_bwd():
    grid_dx = (triton.cdiv(M, 32), triton.cdiv(K, 64))
    fused_softcap_ce_dx_kernel[grid_dx](
        grad_input, grad_out, lse, logits_bf16, targets, w,
        grad_input.stride(0), grad_input.stride(1),
        logits_bf16.stride(0), logits_bf16.stride(1),
        w.stride(0), w.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, num_warps=8,
    )
    grid_dw = (triton.cdiv(V, 128), triton.cdiv(K, 64))
    fused_softcap_ce_dw_kernel[grid_dw](
        grad_weight, grad_out, lse, logits_bf16, targets, x,
        grad_weight.stride(0), grad_weight.stride(1),
        logits_bf16.stride(0), logits_bf16.stride(1),
        x.stride(0), x.stride(1),
        M, V, K, A, C,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_warps=8,
    )


print("=== Combined backward (dx + dw) ===")
ms_nologit = bench(run_nologit_bwd, "nologit recompute (new configs)")
ms_split = bench(run_split_bwd, "split-path (bf16 logits)")
print(f"\n  delta: {ms_nologit - ms_split:+.3f} ms")
print(f"  nologit/split: {ms_nologit/ms_split:.2f}x")
print(f"  speedup: {ms_split/ms_nologit:.2f}x nologit over split")
