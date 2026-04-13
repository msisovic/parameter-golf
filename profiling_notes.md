# Profiling The Experimental Fused FP8+Softcap+CE Path

This note is for profiling the experimental single-wrapper path that:

- uses the FP8 tied LM-head wrapper
- computes softcap+CE inside the same autograd function
- still materializes dense logits to HBM

It is not the default kept stack. Enable it with:

```bash
EXPERIMENTAL_FUSED_FP8_SOFTCAP_CE=1
```

## What We Are Trying To Learn

The main question is not "is this kernel fused?" It is:

- does the experimental fused path still spend enough time in dense logit traffic that it cannot beat the kept split path?

More concretely:

- Is the experimental fused kernel/wrapper still a major step-time consumer?
- Is it compute-bound or memory-bound?
- Does its profile look like a GEMM with some extra math, or like a bandwidth-heavy path dominated by dense logit stores/reads?

## Code Locations

- Experimental fused autograd wrapper:
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:656)
- Experimental fused entrypoint:
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:721)
- Training-path switch:
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:1550)
- CUDA profiler range markers:
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:166)
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:3175)

The key lines showing that this path still writes and saves dense logits are:

- logits materialization:
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:665)
- logits saved for backward:
  - [train_gpt.py](/root/parameter-golf/train_gpt.py:689)

## Recommended Real-Run Profile Setup

Use the real 8xH100 setup and real training shapes.

Suggested environment:

```bash
export TTT_ENABLED=0
export SEED=0
export PARALLEL_START_LAYER=8
export GPTQ_RESERVE_SECONDS=13
export FP8_LM_HEAD=1
export EXPERIMENTAL_FUSED_FP8_SOFTCAP_CE=1
export FUSED_SOFTCAP_CE=0
export WARMUP_STEPS=5
export PROFILE_CUDA_RANGE=1
export PROFILE_CUDA_STEPS=20
export TRAIN_LOG_EVERY=100000
export VAL_LOSS_EVERY=100000
export MAX_WALLCLOCK_SECONDS=120
```

Notes:

- `EXPERIMENTAL_FUSED_FP8_SOFTCAP_CE=1` selects the one-wrapper fused-with-logit-writes path.
- `FUSED_SOFTCAP_CE=0` avoids accidentally taking the kept split CE path.
- `PROFILE_CUDA_RANGE=1` uses `cudaProfilerStart/Stop` so `nsys` can capture only the steady-state step window.
- `PROFILE_CUDA_STEPS=20` is a practical first pass. Raise it if the trace is too noisy.

## `nsys` Command

Run this on the 8xH100 machine if `nsys` is installed there:

```bash
nsys profile \
  --output nsys_exp_fused_fp8_softcap_ce \
  --force-overwrite true \
  --trace cuda,nvtx,osrt,cublas,cudnn \
  --sample none \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Then get summary tables:

```bash
nsys stats --report cuda_gpu_kern_sum nsys_exp_fused_fp8_softcap_ce.nsys-rep
nsys stats --report cuda_api_sum nsys_exp_fused_fp8_softcap_ce.nsys-rep
nsys stats --report gpu_kern_exec_trace nsys_exp_fused_fp8_softcap_ce.nsys-rep
```

## What To Look For In `nsys`

Questions:

- Is the experimental fused wrapper path still a large fraction of steady-state step time?
- Which kernels dominate within that lane?
- Is there obvious extra kernel-launch fragmentation, or is nearly all time inside a few large kernels?

Useful interpretation patterns:

- If the hot time is still concentrated in the experimental fused forward/backward plus GEMM-adjacent kernels, this lane remains relevant.
- If the hot time is elsewhere, stop pushing this lane.
- If the wrapper path is already a small slice, a no-logit-write redesign is unlikely to pay enough.

`nsys` will not tell you "the dense logit write costs X%". It only tells you where time goes.

## `ncu` Command

After `nsys` identifies the hot kernel names, use `ncu` on those kernels.

Template:

```bash
ncu \
  --target-processes all \
  --set full \
  --kernel-name-base demangled \
  --kernel-name "<PUT_HOT_KERNEL_NAME_HERE>" \
  --export ncu_exp_fused_fp8_softcap_ce \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If `--set full` is too heavy, use a smaller metric set first.

## What To Look For In `ncu`

You are trying to infer whether the fused-with-logit-writes path is bandwidth-limited.

Key signals:

- high DRAM throughput
- low or moderate tensor-core utilization relative to expected GEMM work
- high memory-stall indicators
- kernel time that scales like a bandwidth-heavy path, not a clean compute-saturated GEMM

Interpretation:

- If DRAM traffic is very high and compute utilization is not saturated, dense logit stores/reads remain a likely limiter.
- If tensor-core utilization is high and DRAM pressure is not dominant, then removing logit writes may not buy much.

## What "Success" Looks Like

A successful profiling pass gives enough evidence to choose one of these:

1. The lane is still hot and bandwidth-heavy.
   - Next move: prototype a true no-dense-logits forward/backward design.

2. The lane is still hot, but mostly compute-saturated already.
   - Next move: stop chasing logit-write elimination here.

3. The lane is no longer a major slice of full-step time.
   - Next move: re-rank bottlenecks elsewhere in the model.
