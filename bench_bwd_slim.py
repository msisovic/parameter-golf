"""Bench slim persistent TMA backward kernels."""
import torch
import triton
from train_gpt import (
    fused_fp8_softcap_ce_dx_slim_kernel,
    fused_fp8_softcap_ce_dw_slim_kernel,
    fused_fp8_softcap_ce_dx_target_correction_kernel,
    fused_fp8_softcap_ce_dw_target_correction_kernel,
    fused_softcap_ce_bwd_kernel,
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
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.3
w = torch.randn(V, K, device=device, dtype=torch.bfloat16) * 0.1
targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)

x_scale = x.detach().abs().amax().float().clamp_min(1e-12) / 448.0
w_scale = w.detach().abs().amax().float().clamp_min(1e-12) / 448.0
x_fp8 = (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
w_fp8 = (w.float() / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
weight_fp8_rm = w_fp8.contiguous()  # [V, K] row-major
scale_ab = x_scale * w_scale

lse = torch.randn(M, dtype=torch.float32, device=device)
grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
grad_weight = torch.empty(V, K, dtype=torch.float32, device=device)


def bench(fn, name, warmup=10, iters=50):
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


# === dx slim sweep ===
print("=== dx slim (persistent TMA, no target in hot loop) ===")
dx_configs = [
    (64, 128, 128, 8, 4, True),
    (64, 128, 128, 8, 4, False),
    (128, 128, 128, 8, 4, True),
    (128, 128, 128, 8, 4, False),
    (128, 128, 128, 8, 8, True),
    (64, 128, 256, 8, 4, True),
    (64, 256, 128, 8, 4, True),
    (128, 64, 128, 8, 4, True),
    (256, 128, 128, 8, 4, True),
]
for BM, BV, BK, warps, GM, flatten in dx_configs:
    def run_dx(BM=BM, BV=BV, BK=BK, warps=warps, GM=GM, flatten=flatten):
        grad_input.zero_()
        fused_fp8_softcap_ce_dx_slim_kernel[(NUM_SMS,)](
            grad_input, grad_out, lse, x_fp8, weight_fp8_rm, w, scale_ab,
            M, V, K, softcap,
            NUM_SMS=NUM_SMS, FLATTEN=flatten,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=GM, num_warps=warps, num_stages=4,
        )
    try:
        bench(run_dx, f"dx BM={BM},BV={BV},BK={BK},w={warps},GM={GM},fl={flatten}")
    except Exception as e:
        print(f"  dx BM={BM},BV={BV},BK={BK},w={warps},GM={GM},fl={flatten}: FAILED ({type(e).__name__}: {e})")


# === dw slim sweep ===
print("\n=== dw slim (persistent TMA, no target in hot loop) ===")
dw_configs = [
    (64, 128, 128, 8, 4, True),
    (64, 128, 128, 8, 4, False),
    (128, 128, 128, 8, 4, True),
    (128, 64, 128, 8, 4, True),
    (64, 64, 128, 8, 4, True),
    (64, 128, 256, 8, 4, True),
    (128, 128, 128, 8, 8, True),
]
for BM, BV, BK, warps, GM, flatten in dw_configs:
    def run_dw(BM=BM, BV=BV, BK=BK, warps=warps, GM=GM, flatten=flatten):
        grad_weight.zero_()
        fused_fp8_softcap_ce_dw_slim_kernel[(NUM_SMS,)](
            grad_weight, grad_out, lse, x_fp8, x, weight_fp8_rm, scale_ab,
            grad_weight.stride(0), grad_weight.stride(1),
            M, V, K, softcap,
            NUM_SMS=NUM_SMS, FLATTEN=flatten,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=GM, num_warps=warps, num_stages=4,
        )
    try:
        bench(run_dw, f"dw BM={BM},BV={BV},BK={BK},w={warps},GM={GM},fl={flatten}")
    except Exception as e:
        print(f"  dw BM={BM},BV={BV},BK={BK},w={warps},GM={GM},fl={flatten}: FAILED ({type(e).__name__}: {e})")


# === target corrections ===
print("\n=== target correction kernels ===")
def run_dx_corr():
    fused_fp8_softcap_ce_dx_target_correction_kernel[(M,)](
        grad_input, grad_out, targets, x_fp8, weight_fp8_rm, w, scale_ab, lse,
        M, K, softcap, BLOCK_SIZE_K=512,
    )
bench(run_dx_corr, "dx target correction")

def run_dw_corr():
    fused_fp8_softcap_ce_dw_target_correction_kernel[(M,)](
        grad_weight, grad_out, targets, x_fp8, x, weight_fp8_rm, scale_ab,
        grad_weight.stride(0), grad_weight.stride(1),
        M, K, softcap, BLOCK_SIZE_K=512,
    )
bench(run_dw_corr, "dw target correction")


# === Reference: actual split-path bwd ===
print("\n=== Reference: split-path bwd (CE kernel + cuBLAS) ===")
logits = torch.randn(M, V, device=device, dtype=torch.bfloat16)
grad_logits = torch.empty_like(logits, dtype=torch.bfloat16)
def run_split():
    fused_softcap_ce_bwd_kernel[(M,)](
        grad_logits, grad_out, lse, logits, targets,
        logits.stride(0), logits.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        M, V, A, C, BLOCK_SIZE=1024, num_warps=4,
    )
    gi = grad_logits @ w
    gw = grad_logits.t() @ x
bench(run_split, "split bwd (CE + 2x cuBLAS)")
