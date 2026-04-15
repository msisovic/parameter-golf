"""Gradient check for NoLogitFusedFP8SoftcapCrossEntropyFn after switching to tanh-form.

Compares fused-kernel loss/grad_x/grad_w against a pytorch autograd reference that
uses the *same* FP8-quantized inputs (dequantized to fp32) through a tanh-softcap CE.
This isolates the tanh-form formula correctness from FP8 quantization noise.
"""
import os
import torch
import torch.nn.functional as F

os.environ.setdefault("TORCH_LOGS", "")
import triton
def _alloc(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(_alloc)

from train_gpt import (
    NoLogitFusedFP8SoftcapCrossEntropyFn,
    fused_fp8_softcap_ce_slim_stats_kernel,
    fused_fp8_softcap_ce_targets_kernel,
    fused_fp8_softcap_ce_finalize_nologits_kernel,
)

device = "cuda"
torch.manual_seed(0)

M, K, V = 4096, 512, 8192
softcap = 30.0

x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
w = torch.randn(V, K, device=device, dtype=torch.bfloat16) * 0.05
targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)

# FP8 quantize (tensorwise) — mirrors NoLogitFused... path.
x_scale = x.detach().abs().amax().float().clamp_min(1e-12) / 448.0
w_scale = w.detach().abs().amax().float().clamp_min(1e-12) / 448.0
x_fp8 = (x / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
w_fp8 = (w / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
# Mirror training: weight_fp8_t is a non-contiguous .t() view of the row-major w_fp8
# (so that weight_fp8_t.t() recovers the original row-major [V,K] memory layout).
weight_fp8_t = w_fp8.t()
weight_fp8_cm = weight_fp8_t.contiguous().t()  # column-major [V,K] — what FP8Linear uses
weight_fp8_rm = w_fp8.contiguous()  # row-major [V,K] — what the slim kernel wants

# --- Reference: _scaled_mm (same FP8 matmul the kernel does) -> tanh softcap -> CE.
# To also check grad: carry autograd through a surrogate x_ref/w_ref whose forward
# reproduces the _scaled_mm output numerically but is differentiable. We use
# logits_det = _scaled_mm(x_fp8, w_fp8.t(), ...)  (detached), then build
# logits_ref = x_ref @ w_ref.t() and sanity-check it matches.
logits_fp8 = torch._scaled_mm(
    x_fp8, w_fp8.t(),
    scale_a=x_scale.view(1, 1), scale_b=w_scale.view(1, 1),
    out_dtype=torch.bfloat16,
).float()

# Differentiable reference uses dequantized fp32 matmul — grad-wise equivalent to
# _scaled_mm with fp32 accumulation; loss-wise will differ by <1% from _scaled_mm bf16.
x_deq = x_fp8.float() * x_scale
w_deq = w_fp8.float() * w_scale
x_ref = x_deq.clone().requires_grad_(True)
w_ref = w_deq.clone().requires_grad_(True)
logits_ref = x_ref @ w_ref.t()
z_ref = softcap * torch.tanh(logits_ref / softcap)
loss_ref = F.cross_entropy(z_ref, targets, reduction="none")
loss_ref.sum().backward()

# Also compute the bf16-accumulated loss (what the kernel actually returns) for fwd sanity
z_sm = softcap * torch.tanh(logits_fp8 / softcap)
loss_sm = F.cross_entropy(z_sm, targets, reduction="none")

# --- Direct-call of slim kernels (bypass autograd) for sanity
NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
scale_ab = x_scale * w_scale
A = 2.0 * softcap
C = softcap / 2.0
MAX_N_TILES = (V + 63) // 64
pmax = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
psum = torch.empty(M, MAX_N_TILES, dtype=torch.float32, device=device)
tlogit = torch.empty(M, dtype=torch.float32, device=device)
losses_direct = torch.empty(M, dtype=torch.float32, device=device)
lse_direct = torch.empty(M, dtype=torch.float32, device=device)
fused_fp8_softcap_ce_slim_stats_kernel[(NUM_SMS,)](
    x_fp8, weight_fp8_rm, scale_ab, pmax, psum,
    pmax.stride(0), pmax.stride(1),
    M, V, K, A, C, NUM_SMS=NUM_SMS, FLATTEN=True,
    BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
    GROUP_SIZE_M=8, num_warps=8, num_stages=4,
)
fused_fp8_softcap_ce_targets_kernel[(M,)](
    x_fp8, weight_fp8_rm, scale_ab, targets, tlogit, M, K, BLOCK_SIZE_K=512,
)
bn = 128
num_tiles = (V + bn - 1) // bn
BT = 64 if num_tiles <= 64 else (128 if num_tiles <= 128 else 256)
fused_fp8_softcap_ce_finalize_nologits_kernel[(M,)](
    losses_direct, lse_direct, pmax, psum, tlogit,
    pmax.stride(0), pmax.stride(1), M, num_tiles, A, C, BLOCK_SIZE_T=BT,
)
torch.cuda.synchronize()
d_direct = (losses_direct - loss_sm).abs().max().item()
print(f"direct slim vs _scaled_mm ref: max|Δ|={d_direct:.4e}")

# --- Fused kernel path
x_in = x.clone().detach().requires_grad_(True)
w_param = w.clone().detach().requires_grad_(True)
# Kernel requires pre-quantized w; triton_quant=False uses the provided x_fp8? No — it re-quantizes.
loss_fused = NoLogitFusedFP8SoftcapCrossEntropyFn.apply(
    x_in, targets, w_param, weight_fp8_t, weight_fp8_cm, w_scale, softcap, False
)
loss_fused.sum().backward()

# vs _scaled_mm reference (kernel-identical matmul path)
d_sm_max = (loss_fused - loss_sm).abs().max().item()
d_sm_mean = (loss_fused - loss_sm).abs().mean().item()
print(f"loss vs _scaled_mm: max|Δ|={d_sm_max:.4e}  mean|Δ|={d_sm_mean:.4e}")
# vs fp32 reference (for grad comparison context)
d_loss_max = (loss_fused - loss_ref).abs().max().item()
d_loss_mean = (loss_fused - loss_ref).abs().mean().item()
mean_loss = loss_ref.abs().mean().item()
print(f"loss vs fp32-matmul ref: max|Δ|={d_loss_max:.4e}  mean|Δ|={d_loss_mean:.4e}  mean|ref|={mean_loss:.4f}")

# grad_x comparison. Our grad_input is bf16, reference is fp32.
gx_fused = x_in.grad.float()
gx_ref = x_ref.grad
d_gx_max = (gx_fused - gx_ref).abs().max().item()
d_gx_mean = (gx_fused - gx_ref).abs().mean().item()
mean_gx = gx_ref.abs().mean().item()
print(f"grad_x: max|Δ|={d_gx_max:.4e}  mean|Δ|={d_gx_mean:.4e}  mean|ref|={mean_gx:.4e}")

gw_fused = w_param.grad.float()
gw_ref = w_ref.grad
d_gw_max = (gw_fused - gw_ref).abs().max().item()
d_gw_mean = (gw_fused - gw_ref).abs().mean().item()
mean_gw = gw_ref.abs().mean().item()
print(f"grad_w: max|Δ|={d_gw_max:.4e}  mean|Δ|={d_gw_mean:.4e}  mean|ref|={mean_gw:.4e}")
