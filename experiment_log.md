# Hopper logits/loss path experiments

## Baseline

- Source log: `logs/4d71d0b7-27ae-469b-8b40-73763fdd74af.txt`
- Baseline stop: `4838` steps in `587159 ms`
- Baseline throughput: `121.36 ms/step`, `6.48M tok/s`
- Baseline final capped validation: `4838/20000 val_loss: 2.7725 val_bpb: 1.0733`
- Baseline reference command:

```bash
TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Stage notes

- `TRAIN_CE_FLOAT=0` removes the training-only FP32 cast before CE. Eval stays FP32.
- `FP8_LM_HEAD=1` enables the training-only tied LM-head FP8 forward path. Eval stays on the reference path.
- `TTT_ENABLED=0` is used for these speed runs to avoid post-train TTT work contaminating wallclock.
- Compile latency note:
  - this model often has a long `torch.compile` / Inductor phase even on known-good baselines
  - capped wallclock starts after compilation in this setup
  - do not treat long compilation by itself as evidence that an experiment regressed
  - only judge experiments from post-compile training checkpoints and capped-run results
  - caveat: if a new variant repeatedly shows compile time that is clearly at least `2x` a comparable recent run, that is a real concern worth noting, but still not enough by itself to reject the variant without post-compile data

## Results

### Stage 1: FP8 tied LM head

```bash
TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Run log: `logs/ab7b8f7b-b973-4321-b0ac-2b4ec5661132.txt`
- Result: `4894` steps in `587174 ms`
- Throughput: `119.98 ms/step`, `6.55M tok/s`
- Final capped validation: `4894/20000 val_loss: 2.7728 val_bpb: 1.0734`
- Delta vs baseline: `+56` steps, about `+1.14%`
- Updated expectation after Stage 1: FP8 tied head looks like a real but modest win here; revise head-only expectation to roughly `+0.5%` to `+1.5%`, centered near `+1.1%`

### Stage 1a: cached FP8 tied-head weight refresh per optimizer step

```bash
TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Run log: `logs/da02ed03-abb5-4e05-95a0-cea890874231.txt`
- Result: `4894` steps in `587186 ms`
- Throughput: `119.98 ms/step`, `6.55M tok/s`
- Final capped validation: `4894/20000 val_loss: 2.7737 val_bpb: 1.0738`
- Delta vs prior Stage 1: no measurable speed improvement in capped wallclock terms
- Conclusion: quantizing the tied weight once per optimizer step is cleaner, but it does not improve end-to-end step count over the original FP8-head prototype

### Stage 1 + Stage 2: FP8 tied LM head + BF16 training CE

```bash
TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 TRAIN_CE_FLOAT=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Run log: `logs/74c84ec0-4e57-4fc4-8a71-5bc8602a9b64.txt`
- Result: `4893` steps in `587177 ms`
- Throughput: `120.00 ms/step`, `6.55M tok/s`
- Final capped validation: `4893/20000 val_loss: 2.7731 val_bpb: 1.0735`
- Delta vs baseline: `+55` steps, about `+1.14%`
- Delta vs Stage 1: `-1` step, so this does not improve the stack and should be left off for now
- Updated expectation after Stage 1+2: the stacked target should come down materially; with CE-cast removal not helping, the currently supported stack is still about `+1%`, not `+3%` to `+5%`

### Stage 1 + Stage 3 attempt: custom fused softcap + CE

```bash
TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Run log: `logs/9e525334-5085-411b-9eb3-56242c1725a8.txt`
- Run was stopped after the 4000-step validation because throughput was already clearly worse than the current best stack
- Partial throughput signal:
  - `4000/20000 train_loss: 2.8961 train_time: 8.0m tok/s: 6566690`
  - This is materially slower than the kept FP8-head stack at the same point (`6799827 tok/s` in the earlier Stage 1 run)
- Partial validation:
  - `4000/20000 val_loss: 2.8677 val_bpb: 1.1101`
- Conclusion: this Python-level custom autograd fusion is not a viable speed path; it loses to the optimized stock CE kernel

### Stage 1 + Stage 3b: Triton fused softcap + CE

```bash
TTT_ENABLED=0 SEED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Run log: `logs/fb5d3d09-4bb9-4fa0-9fb7-44785a8df84f.txt`
- Result: `4961` steps in `587107 ms`
- Throughput: `118.34 ms/step`, `6.65M tok/s`
- Final capped validation: `4961/20000 val_loss: 2.7724 val_bpb: 1.0732`
- Delta vs baseline: `+123` steps, about `+2.54%`
- Delta vs kept Stage 1 stack: `+67` steps, about `+1.37%`
- Conclusion: unlike the Python-level fusion, the Triton kernel is a real stacked win here and should stay in the active stack

## Notes

- Why the Python-level fusion regressed while Triton won:
  - The failed custom autograd version still computed the loss with generic PyTorch tensor ops (`tanh`, `logsumexp`, `gather`, `softmax`) and a dense Python-defined backward.
  - That replaced the highly optimized stock CE kernel with multiple full-vocab passes, so it added overhead instead of removing it.
  - The Triton version is materially different because it fuses softcap transform, rowwise reduction, target extraction, and loss-gradient construction into dedicated kernels.
  - In short: the positive result came from real kernel fusion, not from merely rewriting CE algebra in Python.

- Follow-up worth trying next:
  - Port the deeper modded-nanogpt `#207` idea: fuse LM-head quantization into the Triton loss path instead of keeping “FP8 head” and “fused softcap+CE” as separate stages.
  - That is the clearest remaining path if we want to push beyond the current `+2.54%` stacked win in the logits/loss lane.

## Current iteration

- `2026-04-12`: tried switching the training-only FP8 tied LM head from tensorwise activation scaling to rowwise activation scaling.
- Result: rejected due to throughput regression on the active stack `FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1`.
- Early-run throughput comparison:
  - previous kept stack: `500/1000/1500 tok/s = 8415493 / 8364411 / 8349760`
  - rowwise attempt: `500/1000/1500 tok/s = 8333582 / 8284839 / 8268338`
  - plain baseline without logits/loss kernels: `8241348 / 8211974 / 8210145`
- Takeaway:
  - rowwise `_scaled_mm` scaling is materially slower here and gives back too much of the head-path speedup
  - even though it may improve FP8 approximation quality, it is not a viable direction unless the projection is fused more deeply than the current standalone `_scaled_mm` path
- Decision:
  - revert to tensorwise FP8 LM-head scaling
  - keep the Triton fused softcap+CE path
  - next worthwhile direction remains full Triton fusion of FP8 projection plus loss rather than a slower rowwise `_scaled_mm` variant
- Why it was slower:
  - the rowwise attempt did not just change numerical granularity; it changed both the pre-matmul quantization work and the `_scaled_mm` scaling mode
  - compared with tensorwise scaling, rowwise scaling adds:
    - a per-row `amax(dim=1, keepdim=True)` reduction instead of one global `amax`
    - materialization of an `(M, 1)` activation-scale tensor
    - broadcast divide by per-row scales before the FP8 cast
    - rowwise `_scaled_mm` inputs with `(M, 1)` `scale_a` and `(1, vocab)` `scale_b`
  - targeted GPU microbenchmarks on this stack showed the slowdown comes from both pieces:
    - at approximately training-like shapes (`M=98304, K=512, N=8192`), `_scaled_mm` alone was about `0.000864 s` tensorwise vs `0.001434 s` rowwise
    - the scale+quantize step was about `0.000334 s` tensorwise vs `0.000462 s` rowwise
  - interpretation:
    - rowwise quantization is slower because it adds extra unfused memory traffic before the matmul
    - rowwise `_scaled_mm` also appears to take a materially slower kernel path than scalar-scale `_scaled_mm` on Hopper/PyTorch in this setup
  - implication for future work:
    - if finer-grained scaling is still desirable, it should likely be done inside a deeper Triton fusion of FP8 LM-head projection plus softcap+CE, rather than via standalone rowwise `_scaled_mm`

- `2026-04-12`: first full FP8-head + fused softcap+CE integration attempt was much slower at the first checkpoint:
  - `500/20000 train_loss: 3.2521 train_time: 1.4m tok/s: 4640985`
  - this is an implementation regression, not a verdict on fusion in general; the initial kernel shape is being reworked

- `2026-04-12`: after fixing the worst kernel-shape issue and reusing forward logits in backward, the 500-step fused check recovered most of the lost speed:
  - run log: `logs/fused_fp8_softcap_ce_storelogits_500_20260412.txt`
  - `500/500 train_loss: 3.2556 train_time: 0.8m tok/s: 8323715`
  - `500/500 val_loss: 3.2993 val_bpb: 1.2772`
  - comparison:
    - much better than the broken first fused attempt (`4640985 tok/s`)
    - still a bit behind the current kept stack at the same point (`8415493 tok/s`)
  - current read:
    - the catastrophic regression came from the original one-row-per-program forward kernel shape
    - fusion is now in the right performance neighborhood, but there is still a remaining gap before it beats the kept non-fused stack

- `2026-04-12`: simplified the experimental fusion into a wrapper around the current fast pieces (`_scaled_mm` projection + existing Triton softcap CE kernels) to isolate autograd-boundary overhead.
  - run log: `logs/fused_fp8_softcap_ce_wrapper_500_20260412.txt`
  - `500/500 train_loss: 3.2558 train_time: 0.8m tok/s: 8409948`
  - `500/500 val_loss: 3.2991 val_bpb: 1.2771`
  - read:
    - this is the fastest result so far in the experimental fused-LM-head sub-track
    - speed is still just under the kept stack (`8415493 tok/s`), so the wrapper itself does not create a real win
    - the remaining gap is likely from small composition overhead, not from the CE kernels themselves

- `2026-04-12`: replaced the eager tensorwise FP8 activation quantization (`div -> clamp -> cast -> contiguous`) with a Triton tensorwise quantizer behind `FP8_LM_HEAD_TRITON_QUANT=1`.
  - run log: `logs/fused_ce_fp8_triton_quant_500_20260412b.txt`
  - `500/500 train_loss: 3.2487 train_time: 0.8m tok/s: 8399429`
  - `500/500 val_loss: 3.2994 val_bpb: 1.2773`
  - comparison vs wrapper:
    - slightly slower throughput (`8399429` vs `8409948`)
    - materially better early train loss (`3.2487` vs `3.2558`)
  - interpretation:
    - the Triton quantizer is not numerically identical to the eager PyTorch FP8 cast path
    - on local checks it changed logits and gradients slightly, so this is acting as a different quantization/noise regime rather than a pure implementation cleanup
    - promising for optimization signal, but not yet a speed win and not enough by itself to claim a better capped result

- `2026-04-12`: full capped run of the Triton tensorwise quantizer on the active stack.
  - command:
    - `SEED=0 TTT_ENABLED=0 PARALLEL_RESIDUAL_START=8 GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 FP8_LM_HEAD_TRITON_QUANT=1 torchrun --standalone --nproc_per_node=8 train_gpt.py`
  - result:
    - `4933` steps in `587188 ms`
    - `diagnostic pre-quantization post-ema val_loss: 2.77000072`
    - `diagnostic quantized val_loss: 2.79848029`
  - read:
    - early 500-step train loss improvement did not carry through to the capped run
    - throughput and quality both ended up slightly behind the current kept stack
    - this is therefore not the next speed win, despite the promising short-run signal

- `2026-04-12`: local ceiling check for removing dynamic activation rescaling from the FP8 LM-head forward.
  - setup:
    - compared the current dynamic tensorwise `x_scale = amax(x)` path against a fixed precomputed `x_scale` on approximately training-like shapes (`M=98304, K=512, N=8192`)
    - both paths kept the existing `_scaled_mm` projection and Triton CE kernels
  - result:
    - current total: about `0.005310 s`
    - fixed-scale total: about `0.005272 s`
  - implication:
    - even a best-case removal of the per-forward `amax` only showed about a `0.7%` end-to-end ceiling locally
    - useful to know, but not large enough to explain the next leap in speed by itself

- `2026-04-12`: tried a more serious backward-side fusion for the FP8 LM head.
  - attempt:
    - wrapper path around `_scaled_mm` forward + Triton CE forward
    - in backward, avoided the previous “quantize then transpose with PyTorch ops” approach
    - instead, generated packed FP8 `grad_logits` layouts directly from the Triton CE backward math for fast `_scaled_mm` use
  - variants checked:
    - dual-packed FP8 backward for both `grad_input` and `grad_weight`
    - narrower version that only packed a transposed FP8 path for `grad_weight`
  - result:
    - both variants were clearly slower in local end-to-end microbench than the current split path
    - representative measurements:
      - baseline total: about `0.00184 s`
      - dual-packed FP8 backward: about `0.00261 s`
      - `grad_weight`-only packed variant: about `0.00299 s`
      - at a larger training-like shape, dual-packed FP8 backward was about `0.00770 s` vs baseline `0.00549 s`
  - conclusion:
    - the remaining layout/packing work in Triton still costs more than it saves from the faster FP8 GEMMs
    - backward-side fusion remains the right conceptual target, but these packing-based implementations are not the next win

- `2026-04-12`: revisited rowwise activation scaling with a cleaner implementation.
  - setup:
    - `FP8_LM_HEAD_ROWWISE=1` enabled per-row activation scales for the LM-head forward
    - cached the `(1, vocab)` `scale_b` tensor once in the model instead of rebuilding it each forward
    - optional Triton rowwise quantizer was used to avoid the eager broadcast-divide/cast path
  - 500-step run:
    - run log: `logs/fp8_rowwise_triton_quant_500_20260412.txt`
    - `500/500 train_loss: 3.2554 train_time: 0.8m tok/s: 8321268`
    - `500/500 val_loss: 3.3003 val_bpb: 1.2776`
  - full capped run:
    - `500/20000 train_loss: 3.2572 train_time: 0.8m tok/s: 8324041`
    - `4887/20000 val_loss: 2.7720 val_bpb: 1.0731`
    - `diagnostic pre-quantization post-ema val_loss: 2.77022295`
    - `diagnostic quantized val_loss: 2.79949592`
  - comparison:
    - slower than the current kept tensorwise stack throughout the run
    - no meaningful accuracy gain; final pre-quant and quantized losses were both slightly worse
  - read:
    - the cleaner implementation reduced avoidable overhead, but the rowwise `_scaled_mm` regime itself is still slower enough to matter
    - finer activation scaling alone is not improving the real objective in this codepath

## Current kept state

- Keep the active logits/loss path at:
  - `FP8_LM_HEAD=1`
  - `FUSED_SOFTCAP_CE=1`
  - tensorwise FP8 LM-head scaling
  - no rowwise scaling
  - no TorchAO float8 integration
- Current read:
  - the major win in this lane was the Triton fused softcap+CE path
  - the LM-head forward is now close enough to saturated that cheap local tweaks mostly trade tiny speed differences against small numeric changes without improving the capped result

## Follow-ups

- Profile a real training step and rank the next non-LM-head bottlenecks before spending more time in this lane.
- If staying in the logits/loss lane, only revisit backward-side fusion if the design avoids explicit layout packing and transpose materialization.
- If revisiting finer-grained scaling, treat `_scaled_mm` rowwise as closed for this repo and require a different kernel path from the start.

## Unexplored Big Gains

- A true GEMM-class custom kernel for LM-head projection with on-the-fly local scaling.
  - This is the only plausible route left for “finer-than-tensorwise scaling without paying the rowwise `_scaled_mm` penalty”.
- A real cuBLASLt outer-vector/block-scaling prototype.
  - This remains the most credible external-kernel path if we want per-row or per-block scaling and do not want to write a full GEMM kernel ourselves.
- Re-profiling the broader model.
  - The next meaningful speed leap may simply be outside the LM-head path at this point.

## External checks

- `2026-04-12`: TorchAO float8 was checked as a possible packaged rowwise/blockwise alternative.
  - On `torch==2.9.1`, the compatible TorchAO release is `0.15.0`.
  - Direct local training-shape benchmarks still came in much slower than the current `_scaled_mm` path:
    - `_scaled_mm` tensorwise: about `0.01020 s`
    - `_scaled_mm` rowwise: about `0.01084 s`
    - TorchAO tensorwise: about `0.03734 s`
    - TorchAO axiswise: about `0.03834 s`
    - TorchAO axiswise with higher-precision grad-weight style config: about `0.03827 s`
  - Decision:
    - TorchAO is not a viable speed path on this stack.
