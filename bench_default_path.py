"""Bench the default (non-FP8) LM head path: F.linear + tanh softcap + F.cross_entropy.
This is the baseline before any FP8 or fused kernel optimizations."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda"
M, K, V = 98304, 512, 8192
softcap = 30.0

x = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(V, K, device=device, dtype=torch.bfloat16, requires_grad=True)
targets = torch.randint(0, V, (M,), device=device, dtype=torch.int64)


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


# Forward only
def fwd_only():
    logits = F.linear(x, w)
    z = softcap * torch.tanh(logits / softcap)
    loss = F.cross_entropy(z.float().view(-1, V), targets, reduction="mean")
    return loss

# Forward + backward
def fwd_bwd():
    x.grad = None
    w.grad = None
    logits = F.linear(x, w)
    z = softcap * torch.tanh(logits / softcap)
    loss = F.cross_entropy(z.float().view(-1, V), targets, reduction="mean")
    loss.backward()

print("=== Default path (bf16 F.linear + tanh + F.cross_entropy) ===")
ms_fwd = bench(fwd_only, "forward only")
ms_fwd_bwd = bench(fwd_bwd, "forward + backward")
print(f"  implied backward: {ms_fwd_bwd - ms_fwd:.3f} ms")
print(f"  bwd/fwd ratio: {(ms_fwd_bwd - ms_fwd)/ms_fwd:.1f}x")

# Also bench just the matmuls for reference
print("\n=== Component breakdown ===")
def matmul_fwd():
    return x @ w.t()
bench(matmul_fwd, "bf16 matmul fwd (x @ w.T)")

logits_ref = (x @ w.t()).detach()
grad_logits = torch.randn_like(logits_ref)
def matmul_dx():
    return grad_logits @ w
def matmul_dw():
    return grad_logits.t() @ x.detach()
bench(matmul_dx, "bf16 matmul dx (grad_logits @ w)")
bench(matmul_dw, "bf16 matmul dw (grad_logits.T @ x)")
