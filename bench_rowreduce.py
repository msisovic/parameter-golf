"""Microbench for row-owner online-LSE kernel after tanh swap."""
import torch
import triton
from train_gpt import (
    fused_fp8_softcap_ce_rowreduce_nologits_kernel,
    fused_fp8_softcap_ce_rowreduce_desc_ws_kernel,
    fused_fp8_softcap_ce_rowreduce_desc_persistent_kernel,
    fused_fp8_softcap_ce_stats_nologits_desc_persistent_kernel,
    fused_fp8_softcap_ce_finalize_nologits_kernel,
    fused_fp8_softcap_ce_slim_stats_kernel,
    fused_fp8_softcap_ce_targets_kernel,
    fused_softcap_ce_fwd_kernel,
    FusedSoftcapCrossEntropyFn,
)

# Triton requires a device-side allocator for make_tensor_descriptor.
def _triton_alloc_fn(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_triton_alloc_fn)

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
w_fp8_t = w_fp8.t().contiguous().t()  # [K, V] with col-major layout of [V,K]? need [K,V] row-major? Check kernel.

# Match training path: fp8_lm_head_weight_t = w_fp8.t() (a view, not contiguous).
# This is [K, V] column-major (stride_k=1, stride_v=K).
w_fp8_kv = w_fp8.t()

scale_ab = (x_scale * w_scale).to(device)

losses = torch.empty(M, dtype=torch.float32, device=device)
lse = torch.empty_like(losses)


def run_rowreduce():
    fused_fp8_softcap_ce_rowreduce_nologits_kernel[
        lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    ](
        x_fp8, w_fp8_kv, scale_ab,
        losses, lse, targets,
        x_fp8.stride(0), x_fp8.stride(1),
        w_fp8_kv.stride(0), w_fp8_kv.stride(1),
        M, V, K, A, C,
    )


# Reference: scaled_mm + softcap + CE
def ref_losses():
    logits_fp8 = torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.view(1, 1),
        scale_b=w_scale.view(1, 1),
        out_dtype=torch.bfloat16,
    )
    logits_f = logits_fp8.float()
    # Use tanh form for reference to match (CE-invariant to constant).
    z = softcap * torch.tanh(logits_f / softcap)
    z_t = z.gather(1, targets[:, None]).squeeze(1)
    lse_ref = torch.logsumexp(z, dim=1)
    return lse_ref - z_t, lse_ref


# Correctness
run_rowreduce()
torch.cuda.synchronize()
ref_loss, ref_lse = ref_losses()
torch.cuda.synchronize()
max_abs_loss = (losses - ref_loss).abs().max().item()
max_abs_lse = (lse - ref_lse).abs().max().item()
print(f"max |loss - ref|: {max_abs_loss:.6f}")
print(f"max |lse  - ref|: {max_abs_lse:.6f}")
print(f"mean loss: {losses.mean().item():.4f}  ref mean: {ref_loss.mean().item():.4f}")

# Warmup / autotune (autotune explores configs on first call)
for _ in range(20):
    run_rowreduce()
torch.cuda.synchronize()

# Report chosen config
try:
    best = fused_fp8_softcap_ce_rowreduce_nologits_kernel.best_config
    print(f"best config: {best}")
except Exception as e:
    print(f"(no best_config): {e}")

# Bench
iters = 100
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    run_rowreduce()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / iters
print(f"rowreduce fwd: {ms:.3f} ms  ({ms/1000:.5f} s)")

# Apples-to-apples wrapper path: scaled_mm -> bf16 logits -> fused softcap+CE kernel.
def run_wrapper():
    logits = torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.view(1, 1),
        scale_b=w_scale.view(1, 1),
        out_dtype=torch.bfloat16,
    )
    return FusedSoftcapCrossEntropyFn.apply(logits, targets, softcap)

for _ in range(10):
    run_wrapper()
torch.cuda.synchronize()
start.record()
for _ in range(iters):
    run_wrapper()
end.record()
torch.cuda.synchronize()
ms_wrap = start.elapsed_time(end) / iters
print(f"wrapper (scaled_mm + fused CE): {ms_wrap:.3f} ms")

# scaled_mm alone (for reference)
def run_scaled_mm():
    return torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.view(1, 1),
        scale_b=w_scale.view(1, 1),
        out_dtype=torch.bfloat16,
    )

for _ in range(5):
    run_scaled_mm()
torch.cuda.synchronize()
start.record()
for _ in range(iters):
    run_scaled_mm()
end.record()
torch.cuda.synchronize()
ms_mm = start.elapsed_time(end) / iters
print(f"scaled_mm alone: {ms_mm:.3f} ms")

# --- Descriptor / TMA row-owner with warp specialization ---
# Requires weight_fp8_cm = row-major [V, K].
w_fp8_cm = w_fp8.contiguous()  # [V, K] row-major

def run_rowreduce_ws(ws):
    fused_fp8_softcap_ce_rowreduce_desc_ws_kernel[
        lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    ](
        x_fp8, w_fp8_cm, scale_ab,
        losses, lse, targets,
        M, V, K, A, C,
        WARP_SPECIALIZE=ws,
    )

for ws in [False]:  # ws=True on this kernel crashes Triton MLIR (axis=1 reduction conflict)
    # correctness
    run_rowreduce_ws(ws)
    torch.cuda.synchronize()
    d_loss = (losses - ref_loss).abs().max().item()
    d_lse = (lse - ref_lse).abs().max().item()
    # warmup
    for _ in range(20):
        run_rowreduce_ws(ws)
    torch.cuda.synchronize()
    try:
        best = fused_fp8_softcap_ce_rowreduce_desc_ws_kernel.best_config
        print(f"desc_ws ws={ws} best: {best}")
    except Exception:
        pass
    start.record()
    for _ in range(iters):
        run_rowreduce_ws(ws)
    end.record()
    torch.cuda.synchronize()
    ms_ws = start.elapsed_time(end) / iters
    print(f"rowreduce_desc_ws ws={ws}: {ms_ws:.3f} ms  (|loss|={d_loss:.4f}, |lse|={d_lse:.5f})")


# --- 2D persistent partial-stats + finalize (desc_persistent kernel, WS=False) ---
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def run_desc_persistent(ws, flatten):
    # Allocate partials sized for the chosen autotune BLOCK_N. We don't know the
    # exact BLOCK_N up front; size for the smallest we allow (64) = 128 tiles max.
    MAX_N_TILES = 128  # V/BLOCK_N with BLOCK_N >= 64
    partial_max = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
    partial_sum = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
    target_logit = torch.zeros(M, dtype=torch.float32, device=device)

    def launch():
        fused_fp8_softcap_ce_stats_nologits_desc_persistent_kernel[(NUM_SMS,)](
            x_fp8, w_fp8_cm, scale_ab,
            partial_max, partial_sum, target_logit, targets,
            partial_max.stride(0), partial_max.stride(1),
            M, V, K, A, C,
            NUM_SMS=NUM_SMS,
            WARP_SPECIALIZE=ws,
            FLATTEN=flatten,
        )
        # num_tiles depends on autotune-chosen BLOCK_N — read back from best_config.
        best = fused_fp8_softcap_ce_stats_nologits_desc_persistent_kernel.best_config
        bn = best.kwargs["BLOCK_SIZE_N"]
        num_tiles = (V + bn - 1) // bn
        BT = 64 if num_tiles <= 64 else 128
        fused_fp8_softcap_ce_finalize_nologits_kernel[(M,)](
            losses, lse, partial_max, partial_sum, target_logit,
            partial_max.stride(0), partial_max.stride(1),
            M, num_tiles, A, C,
            BLOCK_SIZE_T=BT,
        )

    # First launch to trigger autotune (stats), then warm finalize separately.
    launch()
    torch.cuda.synchronize()
    return launch


for ws in [False]:
    for flatten in [False, True]:
        launcher = run_desc_persistent(ws, flatten)
        # correctness
        launcher()
        torch.cuda.synchronize()
        d_loss = (losses - ref_loss).abs().max().item()
        d_lse = (lse - ref_lse).abs().max().item()
        for _ in range(20):
            launcher()
        torch.cuda.synchronize()
        try:
            best = fused_fp8_softcap_ce_stats_nologits_desc_persistent_kernel.best_config
            print(f"desc_persistent ws={ws} fl={flatten} best: {best}")
        except Exception:
            pass
        start.record()
        for _ in range(iters):
            launcher()
        end.record()
        torch.cuda.synchronize()
        ms_dp = start.elapsed_time(end) / iters
        print(f"desc_persistent (stats+finalize) ws={ws} flatten={flatten}: {ms_dp:.3f} ms  (|loss|={d_loss:.4f}, |lse|={d_lse:.5f})")


# --- Persistent row-owner: single-pass online-LSE + persistent outer loop + flatten ---
def run_rowreduce_persistent(flatten):
    fused_fp8_softcap_ce_rowreduce_desc_persistent_kernel[(NUM_SMS,)](
        x_fp8, w_fp8_cm, scale_ab,
        losses, lse, targets,
        M, V, K, A, C,
        NUM_SMS=NUM_SMS,
        FLATTEN=flatten,
    )


for flatten in [False, True]:
    # correctness
    run_rowreduce_persistent(flatten)
    torch.cuda.synchronize()
    d_loss = (losses - ref_loss).abs().max().item()
    d_lse = (lse - ref_lse).abs().max().item()
    for _ in range(20):
        run_rowreduce_persistent(flatten)
    torch.cuda.synchronize()
    try:
        best = fused_fp8_softcap_ce_rowreduce_desc_persistent_kernel.best_config
        print(f"rowreduce_persistent flatten={flatten} best: {best}")
    except Exception:
        pass
    start.record()
    for _ in range(iters):
        run_rowreduce_persistent(flatten)
    end.record()
    torch.cuda.synchronize()
    ms_rp = start.elapsed_time(end) / iters
    print(f"rowreduce_persistent flatten={flatten}: {ms_rp:.3f} ms  (|loss|={d_loss:.4f}, |lse|={d_lse:.5f})")


# --- Slim path: 2D persistent stats (no target) + separate targets kernel + finalize ---
def run_slim(flatten):
    MAX_N_TILES = 256  # V/BLOCK_N, conservative upper bound
    partial_max = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
    partial_sum = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
    target_logit = torch.empty(M, dtype=torch.float32, device=device)

    def launch():
        fused_fp8_softcap_ce_slim_stats_kernel[(NUM_SMS,)](
            x_fp8, w_fp8_cm, scale_ab,
            partial_max, partial_sum,
            partial_max.stride(0), partial_max.stride(1),
            M, V, K, A, C,
            NUM_SMS=NUM_SMS,
            FLATTEN=flatten,
        )
        # Compute per-row target logit (tiny kernel).
        fused_fp8_softcap_ce_targets_kernel[(M,)](
            x_fp8, w_fp8_cm, scale_ab,
            targets, target_logit,
            M, K,
            BLOCK_SIZE_K=512,
        )
        best = fused_fp8_softcap_ce_slim_stats_kernel.best_config
        bn = best.kwargs["BLOCK_SIZE_N"]
        num_tiles = (V + bn - 1) // bn
        BT = 64 if num_tiles <= 64 else (128 if num_tiles <= 128 else 256)
        fused_fp8_softcap_ce_finalize_nologits_kernel[(M,)](
            losses, lse, partial_max, partial_sum, target_logit,
            partial_max.stride(0), partial_max.stride(1),
            M, num_tiles, A, C,
            BLOCK_SIZE_T=BT,
        )
    launch()
    torch.cuda.synchronize()
    return launch


for flatten in [False, True]:
    launcher = run_slim(flatten)
    launcher()
    torch.cuda.synchronize()
    d_loss = (losses - ref_loss).abs().max().item()
    d_lse = (lse - ref_lse).abs().max().item()
    for _ in range(20):
        launcher()
    torch.cuda.synchronize()
    try:
        best = fused_fp8_softcap_ce_slim_stats_kernel.best_config
        print(f"slim flatten={flatten} best: {best}")
    except Exception:
        pass
    start.record()
    for _ in range(iters):
        launcher()
    end.record()
    torch.cuda.synchronize()
    ms_slim = start.elapsed_time(end) / iters
    print(f"slim (stats+targets+finalize) flatten={flatten}: {ms_slim:.3f} ms  (|loss|={d_loss:.4f}, |lse|={d_lse:.5f})")

# --- Isolate tanh.approx error from FP8 noise ---
# Compute logits once in bf16 (shared for both paths), then compare:
#   z_ref = softcap * torch.tanh(logits / softcap)            (IEEE tanh)
#   z_approx computed via our kernel path (but it runs over FP8 GEMM)
# Direct isolation: just compare PTX tanh vs torch.tanh on the real logit distribution.
with torch.no_grad():
    logits_bf16 = torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.view(1, 1),
        scale_b=w_scale.view(1, 1),
        out_dtype=torch.bfloat16,
    ).float()
    # torch uses IEEE tanh.
    z_ieee = softcap * torch.tanh(logits_bf16 / softcap)
    # For approx comparison, we need to run our kernel's tanh over the same logits.
    # Easiest approximation: torch.tanh is IEEE; PTX approx error has known bound ~5e-4 relative.
    # Instead, quantify via kernel output vs full IEEE reference on same inputs:
    lse_ieee = torch.logsumexp(z_ieee, dim=1)
    z_t_ieee = z_ieee.gather(1, targets[:, None]).squeeze(1)
    loss_ieee = lse_ieee - z_t_ieee

    # Our kernel output is already in `losses` / `lse` from the timing run.
    delta_loss = (losses - loss_ieee).abs()
    delta_lse = (lse - lse_ieee).abs()
    print(f"vs IEEE-tanh reference on same FP8 GEMM:")
    print(f"  loss: max {delta_loss.max():.5f}, mean {delta_loss.mean():.5f}")
    print(f"  lse : max {delta_lse.max():.5f}, mean {delta_lse.mean():.5f}")
