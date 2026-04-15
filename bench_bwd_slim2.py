"""Extended sweep for slim backward: more configs, focus on reducing overhead."""
import torch
import triton
from train_gpt import (
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
grad_weight = torch.empty(V, K, dtype=torch.float32, device=device)


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


# BK=512 means single K_out tile — no logit recompute duplication
# But: acc[BM, 512] + logits_acc[BM, BV] register pressure
print("=== dx slim: extended sweep ===")
dx_configs = [
    # (BM, BV, BK, warps, stages, GM, flatten)
    # BK=256 winners from last sweep
    (64, 128, 256, 8, 4, 4, True),
    (64, 128, 256, 4, 4, 4, True),
    (64, 128, 256, 8, 3, 4, True),
    (64, 128, 256, 8, 2, 4, True),
    # BK=512 — single K tile, no duplication
    (32, 128, 512, 8, 4, 4, True),
    (32, 128, 512, 4, 4, 4, True),
    (32, 128, 512, 8, 3, 4, True),
    (32, 128, 512, 8, 2, 4, True),
    (32, 256, 512, 8, 4, 4, True),
    (64, 128, 512, 8, 4, 4, True),
    (64, 128, 512, 4, 4, 4, True),
    # Larger BV to amortize epilogue
    (64, 256, 256, 8, 4, 4, True),
    (128, 128, 256, 8, 4, 4, True),
    (128, 128, 256, 4, 4, 4, True),
    (128, 128, 256, 8, 3, 4, True),
    # Small BM, bigger BV
    (32, 256, 256, 8, 4, 4, True),
    (32, 128, 256, 8, 4, 4, True),
    (32, 128, 256, 4, 4, 4, True),
]
for BM, BV, BK, warps, stages, GM, flatten in dx_configs:
    def run(BM=BM, BV=BV, BK=BK, warps=warps, stages=stages, GM=GM, flatten=flatten):
        fused_fp8_softcap_ce_dx_slim_kernel[(NUM_SMS,)](
            grad_input, grad_out, lse, x_fp8, weight_fp8_rm, w, scale_ab,
            M, V, K, softcap,
            NUM_SMS=NUM_SMS, FLATTEN=flatten,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=GM, num_warps=warps, num_stages=stages,
        )
    try:
        bench(run, f"dx BM={BM},BV={BV},BK={BK},w={warps},s={stages}")
    except Exception as e:
        print(f"  dx BM={BM},BV={BV},BK={BK},w={warps},s={stages}: FAIL ({type(e).__name__}: {str(e)[:80]})")


print("\n=== dw slim: extended sweep ===")
dw_configs = [
    (64, 128, 256, 8, 4, 4, True),
    (64, 128, 256, 4, 4, 4, True),
    (64, 128, 256, 8, 3, 4, True),
    (64, 128, 256, 8, 2, 4, True),
    (32, 128, 512, 8, 4, 4, True),
    (32, 128, 512, 4, 4, 4, True),
    (64, 128, 512, 8, 4, 4, True),
    (128, 128, 256, 8, 4, 4, True),
    (128, 128, 256, 4, 4, 4, True),
    (32, 256, 256, 8, 4, 4, True),
    (32, 128, 256, 8, 4, 4, True),
    (32, 128, 256, 4, 4, 4, True),
]
for BM, BV, BK, warps, stages, GM, flatten in dw_configs:
    def run(BM=BM, BV=BV, BK=BK, warps=warps, stages=stages, GM=GM, flatten=flatten):
        fused_fp8_softcap_ce_dw_slim_kernel[(NUM_SMS,)](
            grad_weight, grad_out, lse, x_fp8, x, weight_fp8_rm, scale_ab,
            grad_weight.stride(0), grad_weight.stride(1),
            M, V, K, softcap,
            NUM_SMS=NUM_SMS, FLATTEN=flatten,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=GM, num_warps=warps, num_stages=stages,
        )
    try:
        bench(run, f"dw BM={BM},BV={BV},BK={BK},w={warps},s={stages}")
    except Exception as e:
        print(f"  dw BM={BM},BV={BV},BK={BK},w={warps},s={stages}: FAIL ({type(e).__name__}: {str(e)[:80]})")
