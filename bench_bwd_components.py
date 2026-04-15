"""Decompose backward kernel cost: FP8 GEMM alone vs FP8+epilogue vs full."""
import torch
import triton
import triton.language as tl
from train_gpt import _ptx_tanh, _grouped_pid_gm, fused_fp8_softcap_ce_slim_stats_kernel

def _alloc(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(_alloc)


# Minimal kernel: just FP8 GEMM recompute + bf16 grad matmul, no epilogue (dummy grad_logits=logits)
@triton.jit
def _dx_gemm_only_kernel(
    grad_input_ptr, x_fp8_ptr, weight_fp8_rm_ptr, weight_bf16_ptr, scale_ab_ptr,
    n_rows, n_cols, k_dim,
    NUM_SMS: tl.constexpr, FLATTEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_V: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(k_dim, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_k
    k_tiles_inner = tl.cdiv(k_dim, BLOCK_SIZE_K)
    scale_ab = tl.load(scale_ab_ptr)
    x_desc = tl.make_tensor_descriptor(x_fp8_ptr, [n_rows, k_dim], [k_dim, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    w_fp8_desc = tl.make_tensor_descriptor(weight_fp8_rm_ptr, [n_cols, k_dim], [k_dim, 1], [BLOCK_SIZE_V, BLOCK_SIZE_K])
    w_bf16_desc = tl.make_tensor_descriptor(weight_bf16_ptr, [n_cols, k_dim], [k_dim, 1], [BLOCK_SIZE_V, BLOCK_SIZE_K])
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN):
        pid_m, pid_k = _grouped_pid_gm(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M
        offs_k = pid_k * BLOCK_SIZE_K
        rows = offs_m + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for off_v in range(0, n_cols, BLOCK_SIZE_V):
            logits_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_V), dtype=tl.float32)
            for ki in range(k_tiles_inner):
                x_tile = x_desc.load([offs_m, ki * BLOCK_SIZE_K])
                w_tile = w_fp8_desc.load([off_v, ki * BLOCK_SIZE_K])
                logits_acc = tl.dot(x_tile, w_tile.T, logits_acc)
            # No epilogue — just cast logits as "grad_logits"
            grad_logits = (logits_acc * scale_ab).to(tl.bfloat16)
            w_bf16 = w_bf16_desc.load([off_v, offs_k])
            acc = tl.dot(grad_logits, w_bf16, acc)
        ks = offs_k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = ks < k_dim
        out_ptrs = grad_input_ptr + rows[:, None] * k_dim + ks[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & k_mask[None, :])


# FP8 GEMM only — no bf16 matmul, just recompute and throw away
@triton.jit
def _dx_fp8_recompute_only_kernel(
    dummy_ptr, x_fp8_ptr, weight_fp8_rm_ptr, scale_ab_ptr,
    n_rows, n_cols, k_dim,
    NUM_SMS: tl.constexpr, FLATTEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_V: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(k_dim, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_k
    k_tiles_inner = tl.cdiv(k_dim, BLOCK_SIZE_K)
    scale_ab = tl.load(scale_ab_ptr)
    x_desc = tl.make_tensor_descriptor(x_fp8_ptr, [n_rows, k_dim], [k_dim, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    w_fp8_desc = tl.make_tensor_descriptor(weight_fp8_rm_ptr, [n_cols, k_dim], [k_dim, 1], [BLOCK_SIZE_V, BLOCK_SIZE_K])
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN):
        pid_m, pid_k = _grouped_pid_gm(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M
        rows = offs_m + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for off_v in range(0, n_cols, BLOCK_SIZE_V):
            logits_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_V), dtype=tl.float32)
            for ki in range(k_tiles_inner):
                x_tile = x_desc.load([offs_m, ki * BLOCK_SIZE_K])
                w_tile = w_fp8_desc.load([off_v, ki * BLOCK_SIZE_K])
                logits_acc = tl.dot(x_tile, w_tile.T, logits_acc)
            acc += tl.sum(logits_acc, axis=1)[:, None]  # prevent dead code elimination
        ks = (pid_k * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
        k_mask = ks < k_dim
        out_ptrs = dummy_ptr + rows[:, None] * k_dim + ks[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & k_mask[None, :])


torch.manual_seed(0)
device = "cuda"
M, K, V = 98304, 512, 8192
softcap = 30.0
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.3
w = torch.randn(V, K, device=device, dtype=torch.bfloat16) * 0.1
x_scale = x.detach().abs().amax().float().clamp_min(1e-12) / 448.0
w_scale = w.detach().abs().amax().float().clamp_min(1e-12) / 448.0
x_fp8 = (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
w_fp8 = (w.float() / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
weight_fp8_rm = w_fp8.contiguous()
scale_ab = x_scale * w_scale
lse = torch.randn(M, dtype=torch.float32, device=device)
grad_out = torch.ones(M, dtype=torch.float32, device=device) / M
grad_input = torch.empty(M, K, dtype=torch.bfloat16, device=device)
targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)
dummy = torch.empty(M, K, dtype=torch.bfloat16, device=device)


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


BM, BV, BK, W, S, GM = 128, 128, 256, 8, 3, 4

print("=== Component decomposition (same tile shape BM=128,BV=128,BK=256,w=8,s=3) ===")

# Forward slim for reference (same tile config on FP8 GEMM)
A = 2.0 * softcap; C = softcap / 2.0
MAX_N_TILES = (V + BV - 1) // BV
partial_max = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
partial_sum = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
def run_fwd_slim():
    fused_fp8_softcap_ce_slim_stats_kernel[(NUM_SMS,)](
        x_fp8, weight_fp8_rm, scale_ab, partial_max, partial_sum,
        partial_max.stride(0), partial_max.stride(1),
        M, V, K, A, C,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BV, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=GM, num_warps=W, num_stages=S,
    )
bench(run_fwd_slim, "fwd slim (FP8 GEMM + tanh epilogue + partial write)")

# FP8 recompute only (no bf16 matmul, no epilogue)
def run_fp8_only():
    _dx_fp8_recompute_only_kernel[(NUM_SMS,)](
        dummy, x_fp8, weight_fp8_rm, scale_ab, M, V, K,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK, GROUP_SIZE_M=GM,
        num_warps=W, num_stages=S,
    )
bench(run_fp8_only, "FP8 recompute only (no epilogue, no bf16 matmul)")

# FP8 + bf16 matmul (no epilogue)
def run_gemm_only():
    _dx_gemm_only_kernel[(NUM_SMS,)](
        grad_input, x_fp8, weight_fp8_rm, w, scale_ab, M, V, K,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK, GROUP_SIZE_M=GM,
        num_warps=W, num_stages=S,
    )
bench(run_gemm_only, "FP8 recompute + bf16 grad matmul (no epilogue)")

# Full dx slim (FP8 + epilogue + bf16 matmul)
from train_gpt import fused_fp8_softcap_ce_dx_slim_kernel
def run_dx_full():
    fused_fp8_softcap_ce_dx_slim_kernel[(NUM_SMS,)](
        grad_input, grad_out, lse, x_fp8, weight_fp8_rm, w, scale_ab,
        M, V, K, softcap,
        NUM_SMS=NUM_SMS, FLATTEN=True,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_V=BV, BLOCK_SIZE_K=BK, GROUP_SIZE_M=GM,
        num_warps=W, num_stages=S,
    )
bench(run_dx_full, "dx slim full (FP8 + epilogue + bf16 matmul)")
