"""Isolated backward benches — run each path separately to avoid cross-contamination."""
import torch
import triton
import sys

def _alloc(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(_alloc)

torch.manual_seed(0)
device = "cuda"
M, K, V = 98304, 512, 8192
softcap = 30.0
A = 2.0 * softcap
C = softcap / 2.0

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


mode = sys.argv[1] if len(sys.argv) > 1 else "both"

if mode in ("split", "both"):
    from train_gpt import fused_softcap_ce_bwd_kernel
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(V, K, device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)
    lse = torch.randn(M, dtype=torch.float32, device=device)
    grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
    logits = torch.randn(M, V, device=device, dtype=torch.bfloat16)
    grad_logits = torch.empty_like(logits, dtype=torch.bfloat16)

    print(f"=== Split-path bwd (logits tensor: {logits.nelement()*2/1e9:.2f} GB) ===")
    def run_split():
        fused_softcap_ce_bwd_kernel[(M,)](
            grad_logits, grad_out, lse, logits, targets,
            logits.stride(0), logits.stride(1),
            grad_logits.stride(0), grad_logits.stride(1),
            M, V, A, C, BLOCK_SIZE=1024, num_warps=4,
        )
        gi = grad_logits @ w
        gw = grad_logits.t() @ x
    bench(run_split, "split bwd total (CE + 2x cuBLAS)")
    # Memory footprint
    print(f"  HBM for logits + grad_logits: {(logits.nelement() + grad_logits.nelement())*2/1e9:.2f} GB")
    del logits, grad_logits, x, w, targets, lse, grad_out
    torch.cuda.empty_cache()

if mode in ("nologit", "both"):
    from train_gpt import fused_fp8_softcap_ce_dx_recompute_kernel, fused_fp8_softcap_ce_dw_recompute_kernel
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(V, K, device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)
    lse = torch.randn(M, dtype=torch.float32, device=device)
    grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
    x_scale = x.detach().abs().amax().float().clamp_min(1e-12) / 448.0
    w_scale = w.detach().abs().amax().float().clamp_min(1e-12) / 448.0
    x_fp8 = (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    w_fp8 = (w.float() / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    weight_fp8_t = w_fp8.t()
    scale_ab = x_scale * w_scale
    grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    grad_weight = torch.empty(V, K, dtype=torch.float32, device=device)

    print(f"=== Nologit bwd (no logits tensor, x_fp8: {x_fp8.nelement()/1e6:.0f}MB) ===")
    def run_nologit():
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
    bench(run_nologit, "nologit bwd total (dx_recompute + dw_recompute)")
