"""Benchmark fused dx+dw kernel vs split-path (CE bwd + cuBLAS)."""
import torch
import triton
from train_gpt import (
    fused_fp8_softcap_ce_dxdw_kernel,
    fused_softcap_ce_bwd_kernel,
    fused_fp8_softcap_ce_dx_slim_kernel,
    fused_fp8_softcap_ce_dw_slim_kernel,
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

lse = torch.randn(M, dtype=torch.float32, device=device)
grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
grad_weight = torch.zeros(V, K, dtype=torch.float32, device=device)


def bench(fn, name, warmup=10, iters=30):
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


# --- Reference: split-path backward (CE bwd kernel + cuBLAS) ---
print("=== Reference: split-path backward ===")
logits_buf = torch.randn(M, V, dtype=torch.bfloat16, device=device)
grad_logits = torch.empty(M, V, dtype=torch.bfloat16, device=device)

def run_split_bwd():
    fused_softcap_ce_bwd_kernel[(M,)](
        grad_logits, grad_out, lse, logits_buf, targets,
        logits_buf.stride(0), logits_buf.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        M, V, A, C, BLOCK_SIZE=1024, num_warps=4,
    )
    gi = grad_logits @ w
    gw = grad_logits.t() @ x
ms_split = bench(run_split_bwd, "split-path bwd (CE kernel + cuBLAS)")
del logits_buf, grad_logits
torch.cuda.empty_cache()


# --- Reference: separate slim dx + dw ---
print("\n=== Reference: separate slim dx + dw ===")
def run_slim_dx():
    fused_fp8_softcap_ce_dx_slim_kernel[(NUM_SMS,)](
        grad_input, grad_out, lse, x_fp8, weight_fp8_rm, w, scale_ab,
        M, V, K, softcap,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=64, BLOCK_SIZE_V=128, BLOCK_SIZE_K=256,
        GROUP_SIZE_M=4, num_warps=8, num_stages=4,
    )
ms_slim_dx = bench(run_slim_dx, "slim dx only")

def run_slim_dw():
    fused_fp8_softcap_ce_dw_slim_kernel[(NUM_SMS,)](
        grad_weight, grad_out, lse, x_fp8, x, weight_fp8_rm, scale_ab,
        grad_weight.stride(0), grad_weight.stride(1),
        M, V, K, softcap,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=64, BLOCK_SIZE_V=128, BLOCK_SIZE_K=256,
        GROUP_SIZE_M=4, num_warps=8, num_stages=4,
    )
ms_slim_dw = bench(run_slim_dw, "slim dw only")
ms_slim_sep = ms_slim_dx + ms_slim_dw
print(f"  slim dx + dw total: {ms_slim_sep:.3f} ms")


# --- Fused dxdw kernel sweep ---
print("\n=== Fused dx+dw kernel sweep ===")
configs = [
    # (BM, BV, BK, warps, stages, GM)
    (64, 128, 256, 8, 4, 4),
    (64, 128, 256, 8, 3, 4),
    (64, 128, 256, 4, 4, 4),
    (64, 128, 128, 8, 4, 4),
    (64, 128, 128, 8, 3, 4),
    (128, 128, 256, 8, 4, 4),
    (128, 128, 256, 8, 3, 4),
    (128, 128, 128, 8, 4, 4),
    (32, 128, 256, 8, 4, 4),
    (32, 128, 256, 8, 3, 4),
    (32, 128, 128, 8, 4, 4),
    (64, 256, 256, 8, 4, 4),
    (64, 64, 256, 8, 4, 4),
    (64, 128, 512, 8, 3, 4),
    (64, 128, 512, 8, 2, 4),
]

best_ms = 1e9
best_cfg = None
for BM, BV, BK, warps, stages, GM in configs:
    def run(BM=BM, BV=BV, BK=BK, warps=warps, stages=stages, GM=GM):
        grad_weight.zero_()
        fused_fp8_softcap_ce_dxdw_kernel[(NUM_SMS,)](
            grad_input, grad_weight, grad_out, lse, targets,
            x_fp8, x, weight_fp8_rm, w, scale_ab,
            grad_weight.stride(0), grad_weight.stride(1),
            M, V, K, softcap,
            NUM_SMS=NUM_SMS, FLATTEN=True,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=GM, num_warps=warps, num_stages=stages,
        )
    try:
        ms = bench(run, f"dxdw BM={BM},BV={BV},BK={BK},w={warps},s={stages}")
        if ms < best_ms:
            best_ms = ms
            best_cfg = (BM, BV, BK, warps, stages, GM)
    except Exception as e:
        print(f"  dxdw BM={BM},BV={BV},BK={BK},w={warps},s={stages}: FAIL ({type(e).__name__}: {str(e)[:100]})")

print(f"\n=== Summary ===")
print(f"  split-path bwd: {ms_split:.3f} ms")
print(f"  slim separate:  {ms_slim_sep:.3f} ms")
if best_cfg:
    print(f"  best fused:     {best_ms:.3f} ms  (BM={best_cfg[0]},BV={best_cfg[1]},BK={best_cfg[2]},w={best_cfg[3]},s={best_cfg[4]})")
    print(f"  vs split-path:  {best_ms - ms_split:+.3f} ms")
