"""Bench forward with logit writeback + cuBLAS backward vs original approaches."""
import torch
import triton
from train_gpt import (
    fused_fp8_softcap_ce_slim_stats_kernel,
    fused_fp8_softcap_ce_slim_stats_writeback_kernel,
    fused_fp8_softcap_ce_targets_kernel,
    fused_fp8_softcap_ce_finalize_nologits_kernel,
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
weight_fp8_rm = w_fp8.contiguous()
scale_ab = x_scale * w_scale

BM, BN, BK, GM = 256, 128, 128, 8
BN_fwd = BN
num_tiles = (V + BN_fwd - 1) // BN_fwd
partial_max = torch.empty(M, num_tiles, dtype=torch.float32, device=device)
partial_sum = torch.empty(M, num_tiles, dtype=torch.float32, device=device)
target_logit = torch.empty(M, dtype=torch.float32, device=device)
losses = torch.empty(M, dtype=torch.float32, device=device)
lse = torch.empty(M, dtype=torch.float32, device=device)
logits_buf = torch.empty(M, V, dtype=torch.bfloat16, device=device)
grad_logits = torch.empty(M, V, dtype=torch.bfloat16, device=device)
grad_out = torch.ones(M, dtype=torch.float32, device=device) / M


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


print("=== Forward comparison ===")

def run_fwd_slim():
    fused_fp8_softcap_ce_slim_stats_kernel[(NUM_SMS,)](
        x_fp8, weight_fp8_rm, scale_ab, partial_max, partial_sum,
        partial_max.stride(0), partial_max.stride(1),
        M, V, K, A, C,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=GM, num_warps=8, num_stages=4,
    )
    fused_fp8_softcap_ce_targets_kernel[(M,)](
        x_fp8, weight_fp8_rm, scale_ab, targets, target_logit, M, K, BLOCK_SIZE_K=512,
    )
    BT = 64 if num_tiles <= 64 else (128 if num_tiles <= 128 else 256)
    fused_fp8_softcap_ce_finalize_nologits_kernel[(M,)](
        losses, lse, partial_max, partial_sum, target_logit,
        partial_max.stride(0), partial_max.stride(1),
        M, num_tiles, A, C, BLOCK_SIZE_T=BT,
    )
ms_fwd = bench(run_fwd_slim, "fwd slim (no writeback)")


def run_fwd_writeback():
    fused_fp8_softcap_ce_slim_stats_writeback_kernel[(NUM_SMS,)](
        x_fp8, weight_fp8_rm, scale_ab, partial_max, partial_sum, logits_buf,
        partial_max.stride(0), partial_max.stride(1),
        logits_buf.stride(0), logits_buf.stride(1),
        M, V, K, A, C,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=GM, num_warps=8, num_stages=4,
    )
    fused_fp8_softcap_ce_targets_kernel[(M,)](
        x_fp8, weight_fp8_rm, scale_ab, targets, target_logit, M, K, BLOCK_SIZE_K=512,
    )
    BT = 64 if num_tiles <= 64 else (128 if num_tiles <= 128 else 256)
    fused_fp8_softcap_ce_finalize_nologits_kernel[(M,)](
        losses, lse, partial_max, partial_sum, target_logit,
        partial_max.stride(0), partial_max.stride(1),
        M, num_tiles, A, C, BLOCK_SIZE_T=BT,
    )
ms_fwd_wb = bench(run_fwd_writeback, "fwd slim + logit writeback")
print(f"  writeback overhead: {ms_fwd_wb - ms_fwd:+.3f} ms")
print(f"  logits buffer: {logits_buf.nelement()*2/1e9:.2f} GB")


print("\n=== Full fwd+bwd comparison ===")

# Approach A: writeback fwd + cuBLAS bwd
def run_writeback_full():
    # Forward with writeback
    fused_fp8_softcap_ce_slim_stats_writeback_kernel[(NUM_SMS,)](
        x_fp8, weight_fp8_rm, scale_ab, partial_max, partial_sum, logits_buf,
        partial_max.stride(0), partial_max.stride(1),
        logits_buf.stride(0), logits_buf.stride(1),
        M, V, K, A, C,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=GM, num_warps=8, num_stages=4,
    )
    fused_fp8_softcap_ce_targets_kernel[(M,)](
        x_fp8, weight_fp8_rm, scale_ab, targets, target_logit, M, K, BLOCK_SIZE_K=512,
    )
    BT = 64 if num_tiles <= 64 else (128 if num_tiles <= 128 else 256)
    fused_fp8_softcap_ce_finalize_nologits_kernel[(M,)](
        losses, lse, partial_max, partial_sum, target_logit,
        partial_max.stride(0), partial_max.stride(1),
        M, num_tiles, A, C, BLOCK_SIZE_T=BT,
    )
    # Backward: CE bwd kernel + cuBLAS
    fused_softcap_ce_bwd_kernel[(M,)](
        grad_logits, grad_out, lse, logits_buf, targets,
        logits_buf.stride(0), logits_buf.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        M, V, A, C, BLOCK_SIZE=1024, num_warps=4,
    )
    gi = grad_logits @ w
    gw = grad_logits.t() @ x
ms_wb_total = bench(run_writeback_full, "writeback fwd + cuBLAS bwd TOTAL")


# Approach B: split-path (scaled_mm + CE fwd + CE bwd + cuBLAS)
from train_gpt import fused_softcap_ce_fwd_kernel
def run_split_full():
    logits_mm = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale.view(1,1), scale_b=w_scale.view(1,1), out_dtype=torch.bfloat16)
    fused_softcap_ce_fwd_kernel[(M,)](
        logits_mm, losses, lse, targets,
        logits_mm.stride(0), logits_mm.stride(1),
        M, V, A, C, BLOCK_SIZE=2048, num_warps=2,
    )
    fused_softcap_ce_bwd_kernel[(M,)](
        grad_logits, grad_out, lse, logits_mm, targets,
        logits_mm.stride(0), logits_mm.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        M, V, A, C, BLOCK_SIZE=1024, num_warps=4,
    )
    gi = grad_logits @ w
    gw = grad_logits.t() @ x
ms_split_total = bench(run_split_full, "split-path (mm + CE fwd + CE bwd + cuBLAS) TOTAL")

print(f"\n  writeback approach: {ms_wb_total:.3f} ms")
print(f"  split-path: {ms_split_total:.3f} ms")
print(f"  delta: {ms_wb_total - ms_split_total:+.3f} ms")
