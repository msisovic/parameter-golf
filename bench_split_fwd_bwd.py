"""Quick bench: split-path forward vs backward, each component separately."""
import torch
import triton
from train_gpt import (
    fused_softcap_ce_fwd_kernel,
    fused_softcap_ce_dx_kernel,
    fused_softcap_ce_dw_kernel,
    FusedSoftcapCrossEntropyFn,
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
scale_ab = x_scale * w_scale

# Forward: _scaled_mm + fused CE kernel
logits_bf16 = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale.view(1,1), scale_b=w_scale.view(1,1), out_dtype=torch.bfloat16)
losses = torch.empty(M, dtype=torch.float32, device=device)
lse = torch.empty(M, dtype=torch.float32, device=device)
grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
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


print("=== Split-path breakdown ===")

def run_scaled_mm():
    return torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale.view(1,1), scale_b=w_scale.view(1,1), out_dtype=torch.bfloat16)
ms_mm = bench(run_scaled_mm, "scaled_mm (fwd GEMM)")

def run_fwd_ce():
    fused_softcap_ce_fwd_kernel[(M,)](
        logits_bf16, losses, lse, targets,
        logits_bf16.stride(0), logits_bf16.stride(1),
        M, V, A, C, BLOCK_SIZE=2048, num_warps=2,
    )
ms_fwd_ce = bench(run_fwd_ce, "fused_softcap_ce_fwd_kernel")

def run_fwd_total():
    l = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale.view(1,1), scale_b=w_scale.view(1,1), out_dtype=torch.bfloat16)
    fused_softcap_ce_fwd_kernel[(M,)](
        l, losses, lse, targets,
        l.stride(0), l.stride(1),
        M, V, A, C, BLOCK_SIZE=2048, num_warps=2,
    )
ms_fwd = bench(run_fwd_total, "TOTAL fwd (mm + CE)")

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
ms_dx = bench(run_split_dx, "split dx")

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
ms_dw = bench(run_split_dw, "split dw")

print(f"  TOTAL bwd: {ms_dx + ms_dw:.3f} ms")
print(f"  TOTAL fwd+bwd: {ms_fwd + ms_dx + ms_dw:.3f} ms")
print(f"  bwd/fwd ratio: {(ms_dx + ms_dw)/ms_fwd:.1f}x")
