"""Clean matmul-only bench: can Triton beat scaled_mm when we don't write logits?

Strategy: use best-known Triton GEMM patterns (TMA descriptors, persistent
grouped scheduling, WGMMA via tl.dot) with a single scalar sink per tile so
the compute work is preserved but no output is written to HBM.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_libdevice

torch.manual_seed(0)
device = "cuda"
M, K, V = 98304, 512, 8192


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.jit
def _grouped_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _matmul_only_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm,
            },
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for gm in [4, 8]
        for ns in [3, 4]
        for nw in [4, 8]
    ]


@triton.autotune(
    configs=_matmul_only_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_only_persistent_kernel(
    x_fp8_ptr, w_fp8_cm_ptr, sink_ptr,
    M, N, K,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FLATTEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    x_desc = tl.make_tensor_descriptor(
        x_fp8_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    w_desc = tl.make_tensor_descriptor(
        w_fp8_cm_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    sink = 0.0
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m, pid_n = _grouped_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_m = pid_m * BLOCK_SIZE_M
        offs_n = pid_n * BLOCK_SIZE_N
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            x = x_desc.load([offs_m, offs_k])
            w = w_desc.load([offs_n, offs_k])
            acc = tl.dot(x, w.T, acc)
        # Single scalar sink — forces acc to be materialized but writes only 1 fp32 per CTA.
        sink += tl.sum(acc) * 1e-30
    tl.store(sink_ptr + start_pid, sink)


def bench(fn, iters=100, warmup=20):
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
    return s.elapsed_time(e) / iters


# Data
x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.3
w = torch.randn(V, K, device=device, dtype=torch.bfloat16) * 0.1
x_scale = x.abs().amax().float().clamp_min(1e-12) / 448.0
w_scale = w.abs().amax().float().clamp_min(1e-12) / 448.0
x_fp8 = (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
w_fp8 = (w.float() / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()  # [V, K] row-major

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
sink = torch.empty(NUM_SMS, dtype=torch.float32, device=device)


def run_matmul_only(ws, flatten):
    grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(V, meta["BLOCK_SIZE_N"])),)
    matmul_only_persistent_kernel[grid](
        x_fp8, w_fp8, sink,
        M, V, K,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=ws,
        FLATTEN=flatten,
    )


def run_scaled_mm():
    return torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.view(1, 1),
        scale_b=w_scale.view(1, 1),
        out_dtype=torch.bfloat16,
    )


# Scaled_mm reference
ms_mm = bench(run_scaled_mm)
flops = 2.0 * M * K * V
tflops_mm = flops / (ms_mm * 1e-3) / 1e12
print(f"scaled_mm:            {ms_mm:.3f} ms  ({tflops_mm:.1f} TFLOP/s, {100*tflops_mm/1979:.1f}% peak)")

# Try each WS / flatten combination
for ws in [False, True]:
    for fl in [False, True]:
        try:
            ms = bench(lambda: run_matmul_only(ws, fl))
            try:
                cfg = matmul_only_persistent_kernel.best_config
            except Exception:
                cfg = "?"
            tflops = flops / (ms * 1e-3) / 1e12
            print(f"matmul_only ws={ws} fl={fl}: {ms:.3f} ms  ({tflops:.1f} TFLOP/s, {100*tflops/1979:.1f}% peak)  best={cfg}")
        except Exception as e:
            print(f"matmul_only ws={ws} fl={fl}: FAILED ({type(e).__name__}: {str(e)[:120]})")
