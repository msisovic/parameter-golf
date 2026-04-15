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

## 2026-04-14 Iterations

- Iteration 1: true no-dense-logits FP8 prototype added behind `EXPERIMENTAL_NOLOGIT_FUSED_FP8_SOFTCAP_CE=1`.
  - Forward now computes FP8 projection tiles directly in Triton and stores only rowwise partial stats plus `lse`.
  - Backward recomputes projection tiles inside Triton `dx` / `dw` kernels, so neither dense `logits` nor dense `grad_logits` hit HBM.
  - Small CUDA correctness check vs the current wrapper path looked acceptable:
    - loss delta about `0.00138`
    - `grad_x` max abs diff about `9.77e-4`
    - `grad_w` max abs diff about `7.32e-4`
  - Training-shape local microbench (`M=98304, K=512, V=8192`) was not viable:
    - wrapper forward+backward: about `0.00526 s`
    - true no-logit forward+backward: about `0.14042 s`
    - wrapper forward-only: about `0.00207 s`
    - true no-logit forward-only: about `0.00923 s`
  - Read:
    - the direct Triton FP8 projection kernel is far slower than `_scaled_mm` on this stack
    - the no-logit design is numerically reasonable, but this implementation is not remotely competitive

- Iteration 2: direct backward path added behind `EXPERIMENTAL_FUSED_FP8_SOFTCAP_CE_DIRECT_BWD=1`.
  - This keeps `_scaled_mm` forward and the existing fused CE forward, but replaces dense `grad_logits` materialization with the in-tree direct Triton `dx` / `dw` kernels.
  - Small CUDA correctness check matched the wrapper exactly within the local test.
  - Initial training-shape local microbench (`M=98304, K=512, V=8192`) still regressed badly:
    - wrapper forward+backward: about `0.00526 s`
    - direct-backward variant: about `0.02993 s`
  - A small block-size sweep improved the direct Triton backward kernels materially but not enough:
    - best direct `dx+dw` kernel pair in the local sweep: about `0.02114 s`
    - tuned end-to-end direct-backward variant after taking the best local config: about `0.02398 s`
  - Read:
    - eliminating `grad_logits` traffic alone is not enough if the direct Triton reduction kernels give back much more than they save
    - the current `fused_softcap_ce_dx_kernel` / `fused_softcap_ce_dw_kernel` implementations are not good enough as a real speed path here

- Decision after these iterations:
  - do not treat either new path as a throughput candidate yet
  - the next credible logits/loss-lane attempt still needs a GEMM-class kernel path, not just more recompute or more direct reduction logic around the current Triton kernels

- Iteration 3: narrowed scope to the no-logit forward microbenchmark only and tuned the forward stats kernel shape aggressively.
  - Goal for this pass:
    - stop optimizing the broken full path
    - treat the no-logit forward as a standalone target
    - push the FP8 tiled matmul + softcap-stats kernel down into the same rough regime as the current forward lane before touching backward again
  - Real-shape forward-only microbench target:
    - `M=98304, K=512, V=8192`
  - Starting point for the public no-logit forward path:
    - about `0.00923 s`
  - Tile sweeps on the direct stats kernel found large headroom:
    - early winner: `(BLOCK_M, BLOCK_N, BLOCK_K, warps) = (32, 128, 64, 4)` at about `0.00375 s`
    - better winner: `(64, 128, 128, 8)` at about `0.00256 s` for the core stats+finalize kernel when the timed region included the actual kernel launches and output buffers
  - Structural cleanup:
    - removed the dense `partial_target` matrix from the no-logit forward path
    - replaced it with a per-row `target_logit` vector, since each row’s target belongs to exactly one vocab tile
    - widened the finalize reduction from `BLOCK_SIZE_T=16` to `32`
  - Public no-logit forward timings after tuning:
    - with eager FP8 quantization: about `0.00294 s`
    - with Triton FP8 quantization: about `0.00267 s`
  - Reference forward-only wrapper timing on the same setup:
    - about `0.00206 s`
  - Read:
    - this was a real forward-side recovery: about `3.4x` faster than the original no-logit forward attempt (`0.00923 s` to `0.00267 s`)
    - it is also now roughly at the “2x faster than `0.0053 s`” target scale, if that `0.0053 s` reference is used
    - however, the tuned no-logit public forward is still slower than the current `_scaled_mm`-based wrapper forward (`0.00267 s` vs `0.00206 s`)
    - the easy tile-shape wins in this kernel family look mostly harvested; the next forward-side win likely needs a more serious GEMM-class implementation, probably descriptor/TMA-style or otherwise more persistent than the current pointer-load kernel

- Iteration 4: pushed harder on the no-logit forward toward the `< 0.001 s` target and mapped the current ceiling.
  - Better non-persistent pointer-kernel config:
    - switched the forward stats kernel to `(BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) = (128, 256, 128, 8, 3)`
    - public no-logit forward with Triton quantization improved from about `0.00267 s` to about `0.00256 s`
    - public no-logit forward with eager quantization improved from about `0.00294 s` to about `0.00282 s`
  - Tried a persistent pointer-kernel scheduler:
    - result regressed to about `0.00362 s`
    - decision: rejected
  - Tried moving toward a descriptor/TMA-style GEMM skeleton:
    - blocked on Triton tensor-descriptor alignment constraints for contiguous FP8 tensors in this setup
    - specifically, the FP8 layouts here have inner stride `1` byte, which the current `TensorDescriptor` helper rejects because strides must be `16`-byte aligned
    - decision: not a viable quick drop-in path from the current cached FP8 layouts
  - Checked whether temp allocation reuse was the hidden remaining overhead:
    - reusing `x_fp8`, `partial_max`, `partial_sum`, `target_logit`, `losses`, and `lse` did not materially change the forward timing
    - tuned forward stayed at about `0.00256 s`
  - Current read:
    - the present Triton pointer-kernel family appears to have a practical floor around `2.5 ms` on this shape
    - getting below `1.0 ms` does not look plausible from more block-size tuning, persistent scheduling in this form, or workspace reuse
    - the next real move would need a different implementation class entirely, likely a Hopper-specific GEMM path that can consume FP8 efficiently without the descriptor-alignment issue of the current Triton helper route

- Iteration 5: switched the no-logit forward stats kernel to `tl.make_block_ptr` loads.
  - Rationale:
    - descriptor/TMA via `TensorDescriptor` was blocked by FP8 stride-alignment constraints in this setup
    - `tl.make_block_ptr` still gives a more GEMM-like block-load path without that specific helper limitation
  - Best public forward-only result on the same shape after the switch:
    - wrapper forward: about `0.00206 s`
    - no-logit forward with eager FP8 quantization: about `0.00266 s`
    - no-logit forward with separate Triton FP8 quantization: about `0.00241 s`
  - Comparison vs previous best public no-logit forward:
    - previous best: about `0.00256 s`
    - `make_block_ptr` best: about `0.00241 s`
    - gain: about `6%`
  - Extra check:
    - tried fusing activation quantization directly into the `make_block_ptr` GEMM tile loop
    - result regressed to about `0.00314 s`
    - decision: rejected; keep the separate Triton quantizer ahead of the block-pointer kernel
  - Read:
    - `make_block_ptr` is a real improvement and is now the best forward-only no-logit path checked here
    - the gain is meaningful but still nowhere near the `< 0.001 s` target
    - getting the next large jump will likely require a more Hopper-specialized MMA path than plain Triton block-pointer `tl.dot`

- Iteration 6: fixed and wired a grouped block-pointer stats kernel, then re-measured the public forward path with CUDA events.
  - Bug fix:
    - the first grouped/autotuned version incorrectly let autotune vary `BLOCK_SIZE_N`, even though the partial-buffer layout assumes a fixed vocab tile width
    - that could alias partial writes and give invalid speedups
    - fixed by pinning `BLOCK_SIZE_N=256` and autotuning only `BLOCK_SIZE_M`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, stages, and warps
  - Public forward-only timing on the real shape after wiring the corrected grouped kernel:
    - no-logit forward: about `2.335 ms`
    - wrapper forward: about `2.025 ms`
  - Comparison vs previous best public no-logit forward:
    - previous best: about `2.407 ms`
    - grouped block-pointer best: about `2.335 ms`
    - gain: about `3.0%`
  - Kernel breakdown with CUDA events:
    - Triton FP8 activation quantization: about `0.052 ms`
    - grouped no-logit stats kernel: about `2.031 ms`
    - finalize reduction over `partial_max` / `partial_sum`: about `0.063 ms`
    - wrapper `_scaled_mm`: about `0.896 ms`
    - wrapper fused CE over dense logits: about `0.873 ms`
    - wrapper public forward total: about `2.100 ms`
    - no-logit public forward total: about `2.375 ms`
  - Read:
    - the remaining blocker is the fused stats kernel itself, not quantization and not the second-pass finalize reduction
    - removing the logits write is not enough if the replacement fused GEMM+softcap kernel is materially slower than `_scaled_mm`
    - on this shape, the custom fused stats kernel is already slower than `_scaled_mm + CE` before quantization/finalize overheads are added

- Iteration 7: tried a descriptor/TMA-style persistent forward kernel with contiguous `weight_fp8_cm`.
  - Important correction:
    - the earlier descriptor failure was not a universal FP8/TMA dead end
    - a row-major contiguous weight cache (`weight_fp8_cm`, shape `[V, K]`) does satisfy the descriptor layout requirements for this problem shape
  - New experiment:
    - added a device-side `tl.make_tensor_descriptor` persistent kernel that builds descriptors for `x_fp8` and `weight_fp8_cm` inside the kernel and performs the same softcap+stats epilogue without storing logits
    - checked the relevant Hopper scheduling modes:
      - `warp_specialize=False, flatten=False`: about `2.472 ms`
      - `warp_specialize=False, flatten=True`: about `2.431 ms`
      - `warp_specialize=True, flatten=False`: about `2.486 ms`
      - `warp_specialize=True, flatten=True`: about `2.425 ms`
  - Read:
    - this is a genuinely different implementation class from the block-pointer path, but on the current Triton 3.5.1 stack it still loses to the grouped block-pointer stats kernel (`~2.03 ms`)
    - the dominant blocker is now very concrete:
      - matching `_scaled_mm` requires a GEMM implementation that stays near cuBLASLt-class FP8 throughput even after adding the softcap epilogue
      - the present Triton fused kernels are not there yet; they spend more time in the fused matmul+stats kernel than the wrapper spends in `_scaled_mm` plus a separate CE pass

- Iteration 8: isolated the custom LM-head matmul from the fused softcap/CE epilogue.
  - Method:
    - benchmarked a fixed grouped Triton kernel with the same `make_block_ptr` loads and `tl.dot` mainloop as the fused stats kernel, but writing only one scalar sink per tile so the matmul work is retained and dense logits are never stored
    - compared it directly against the same fixed grouped kernel with the full softcap+row-stats epilogue, and against `_scaled_mm`
  - Real-shape CUDA-event timings:
    - custom grouped matmul-only sink kernel: about `0.890 ms`
    - `_scaled_mm`: about `0.865 ms`
    - same grouped kernel with fused softcap+row-stats epilogue: about `2.164 ms`
    - implied epilogue plus partial-stat overhead on top of the custom matmul: about `1.275 ms`
  - Read:
    - the custom FP8 LM-head matmul is not the main problem anymore; it is already close to `_scaled_mm`
    - the current slowdown is dominated by the fused softcap/CE row-stat epilogue, not by the tiled matmul itself
    - the next serious optimization target is the no-materialization CE/statistics design, especially how the rowwise reductions and partial writes are structured

- Iteration 9: replaced the two-pass partial-stats forward with a row-owner online-LSE kernel.
  - Design:
    - one kernel owns a block of rows
    - it streams all vocab tiles for those rows
    - updates `row_max`, `row_sum`, and `target_logit` online in registers
    - writes only final `losses` and `lse`
    - this removes `partial_max`, `partial_sum`, and the finalize kernel from the forward path entirely
  - First pass regressed:
    - initial row-owner kernel: about `2.655 ms` public no-logit forward
    - cause was mainly poor kernel shape for the online-reduction workload
  - After retuning the row-owner kernel to allow smaller vocab tiles:
    - public no-logit forward: about `2.293 ms`
    - wrapper forward: about `2.030 ms`
    - previous best no-logit public forward: about `2.335 ms`
    - gain vs previous best no-logit forward: about `1.8%`
  - Component breakdown after retuning:
    - Triton FP8 quantization: about `0.055 ms`
    - row-owner online-LSE kernel: about `2.076 ms`
    - old two-pass grouped stats kernel: about `2.078 ms`
    - old finalize kernel: about `0.064 ms`
  - Read:
    - the row-owner design did exactly what it was supposed to structurally: it eliminated the partial-stat round trip and the finalize kernel
    - however, the partial-stat buffers and finalize kernel were only a small part of the total forward cost on this shape
    - most of the remaining epilogue overhead is the in-kernel softcap/LSE/target work itself, not the old partial-buffer plumbing

- Iteration 10: replaced sigmoid softcap with hardware-tanh softcap (PTX `tanh.approx.f32`).
  - Motivation:
    - Iteration 9 confirmed the dominant residual cost is the in-kernel softcap/LSE SFU work, not plumbing.
    - `A*sigmoid(x/C)` with `A=2s`, `C=s/2` is CE-equivalent (up to a row-wise constant) to `s*tanh(x/s)` — one MUL, one MUFU.TANH, one MUL, instead of `mul+rcp+exp+add+mul` that `tl.sigmoid` lowers to.
  - What went in:
    - single-instruction PTX helper `_ptx_tanh` using `tanh.approx.f32` inline asm (this is the sm_90 MUFU.TANH path).
    - row-owner online-LSE kernel now uses `softcap * _ptx_tanh(logits * inv_softcap)` for both the tile softcap and the `target_logit` softcap.
  - Pitfall observed:
    - `triton.language.extra.cuda.libdevice.tanh` compiles to `__nv_tanhf`, a software polynomial. Using it regressed the kernel to `~2.774 ms`. Only the PTX intrinsic hits the hardware SFU op.
  - Apples-to-apples CUDA-event timings on the real training shape (`M=98304, K=512, V=8192`, softcap 30, tensorwise FP8):
    - `_scaled_mm` alone (GEMM-only ceiling): `~0.946 ms`
    - Wrapper: `_scaled_mm` + fused softcap+CE over materialized bf16 logits: `~1.661 ms`
    - Previous best fused row-owner (sigmoid): `~2.076 ms`
    - New fused row-owner (PTX tanh): `~1.502 ms`
  - Read:
    - First fused-kernel configuration that actually beats the wrapper: `1.502 ms` vs `1.661 ms` (~10% faster end-to-end fwd, ~28% faster than the sigmoid fused path).
    - Ratio to `_scaled_mm`-only ceiling dropped from `~2.4x` (sigmoid) to `~1.59x` (tanh), confirming SFU throughput was the binding constraint.
    - Autotune picked a larger tile after the swap (`M=128, N=256, K=64, warps=8, stages=3`), which is consistent with SFU pressure being the thing that used to force small N tiles.
  - Accuracy of `tanh.approx.f32`:
    - Isolated vs IEEE `torch.tanh` on the same FP8 GEMM output (so FP8 noise cancels):
      - max `|loss - IEEE|`: `0.0087`, mean: `7.8e-4`
      - max `|lse  - IEEE|`: `1.4e-4`, mean: `2e-5`
    - Loss magnitude `~9.2` → worst-case relative error `~1e-3`, mean `~1e-4`. Well below bf16 noise and an order of magnitude below FP8 quantization noise.
  - Caveat:
    - Backward kernels (`fused_fp8_softcap_ce_dx_recompute_kernel`, `fused_fp8_softcap_ce_dw_recompute_kernel`) still use the sigmoid-form gradient `A * sigmoid_u * (1 - sigmoid_u) * inv_C`. They need to be ported to the tanh-form gradient `1 - tanh(x/s)^2` before end-to-end training on this kernel will produce correct grads.

- Iteration 11: attempted warp-specialization; landed on TMA-descriptor row-owner instead.
  - Warp-specialize attempt on the existing block-pointer row-owner kernel:
    - adding `warp_specialize=True` to the inner N-tile `tl.range` triggered an internal MLIR assertion in `WSLowerToken.cpp:73` (Triton 3.5.1). WS lowering does not appear to support `tl.make_block_ptr` in this kernel shape.
  - Ported the row-owner to `tl.make_tensor_descriptor` (TMA) loads — new kernel `fused_fp8_softcap_ce_rowreduce_desc_ws_kernel`.
    - It takes `weight_fp8_cm` (row-major `[V, K]`) and does `tl.dot(x, w.T, acc)`, matching the descriptor pattern from the earlier desc-persistent kernel.
    - With `WARP_SPECIALIZE=False` it compiled and ran cleanly at `~1.408 ms` — another `~7%` faster than the block-pointer row-owner at `~1.508 ms`.
    - With `WARP_SPECIALIZE=True` compilation failed in `WSDataPartition.cpp:1196`: `reduceOp.getAxis() != dim && "reduce should not happen on the partitioned dimension"`.
    - Root cause: the row-owner design reduces over `axis=1` (N, vocab) in `tl.max` / `tl.sum` / LSE rescale. WS's automatic data-partitioner picks N as the warp partition dim because it is the natural tile width, which collides with the reduction.
  - Read:
    - WS-inside-inner-loop is structurally incompatible with this row-owner design. To get WS we would need a 2D persistent `(M_block, N_block)` tile loop with partial stats and a finalize kernel (the desc-persistent design from Iteration 7), which trades away the row-owner single-pass property.
    - Even without WS, moving the row-owner to TMA loads is a clean win over block pointers, so the TMA kernel becomes the new best single-pass forward.
  - Apples-to-apples CUDA-event timings on the same real training shape:
    - block-pointer row-owner + PTX tanh: `~1.508 ms`
    - TMA-descriptor row-owner + PTX tanh (no WS): `~1.408 ms`
    - wrapper (`_scaled_mm` + fused softcap+CE over bf16 logits): `~1.678 ms`
    - `_scaled_mm` alone: `~0.968 ms`
  - Cumulative win vs Codex's sigmoid block-pointer row-owner (`~2.076 ms`): **~32%** faster; vs wrapper: **~16%** faster.
  - Follow-ups not yet taken:
    - wire the TMA descriptor kernel into `NoLogitFusedFP8SoftcapCrossEntropyFn` so the training path uses it
    - port backward kernels to the tanh gradient form so end-to-end training is correct on this kernel
    - if WS is still desired, revisit the 2D persistent partial-stats kernel (now with PTX tanh swapped in) as a separate variant

- Iteration 12: re-measured matmul-only floor. Iteration 8's number was badly under-tuned.
  - Motivation:
    - Iteration 8 claimed our no-write Triton matmul maxed out at ~`0.890 ms`, essentially matching `_scaled_mm` at `0.865 ms`, and used that to justify "no-write gives no GEMM-perf advantage on this shape."
    - That conclusion shaped subsequent iterations — it is wrong.
  - New experiment:
    - fresh `bench_matmul_only.py` writes a persistent 2D tile kernel using `tl.make_tensor_descriptor` loads, grouped PID (`GROUP_SIZE_M`), `tl.range` with `flatten` and optional `warp_specialize`, and a single-scalar sink (each CTA writes one f32) so the compute work is preserved but no logits hit HBM.
    - autotune over `BLOCK_SIZE_M ∈ {64,128,256}`, `BLOCK_SIZE_N ∈ {64,128,256}`, `BLOCK_SIZE_K ∈ {64,128}`, `GROUP_SIZE_M ∈ {4,8}`, stages ∈ {3,4}, warps ∈ {4,8}.
  - CUDA-event timings on the real training shape (`M=98304, K=512, V=8192`, tensorwise FP8), same process as `_scaled_mm`:
    - `_scaled_mm`: ~`0.921–0.939 ms` (~`45%` FP8 peak)
    - matmul-only persistent, `WS=False, flatten=False`: ~`0.759 ms` (~`55%` peak)
    - matmul-only persistent, `WS=False, flatten=True`: ~`0.656–0.694 ms` (~`60–64%` peak)
    - matmul-only persistent, `WS=True,  flatten=False`: ~`0.764–0.774 ms`
    - matmul-only persistent, `WS=True,  flatten=True`: ~`0.668–0.713 ms`
  - Winning config in both runs: `BLOCK_M=256, BLOCK_N=128, BLOCK_K∈{64,128}, GROUP_SIZE_M=8, num_warps=8, num_stages=3`.
  - Read:
    - Our no-write Triton GEMM runs **~25–30% faster** than `_scaled_mm` on this shape. The claim "the write is fully hidden by WGMMA on H100 at K=512, so no-write gives no GEMM-perf advantage" was empirically false.
    - Missing levers that Codex never tried:
      - `BLOCK_M=256` (gave the biggest single win — `flatten` picked it in both runs)
      - `flatten=True` on the persistent outer loop (consistently ~`0.1 ms` faster than `flatten=False`)
    - WS did not help the matmul-only kernel (`0.694` vs `0.713`, basically a wash). Triton's default async pipeline with `num_stages=3 + flatten` already hides load latency on this shape. WS is only useful if there is a fat epilogue stalling the pipeline.
  - Implication for the fused kernel target:
    - Fused floor is now `~0.66 ms` (matmul-only at `WS=False, flatten=True`), not `~0.87 ms`.
    - Sub-`1 ms` fused forward is achievable even with `~0.25–0.3 ms` of *exposed* epilogue cost.
    - Correct target design is **2D persistent + flatten + TMA** with partial-stats + a cheap finalize kernel, plus the PTX tanh softcap. WS is optional polish, not on the critical path.

- Iteration 13: tried to actually transfer the matmul-only ceiling to the fused kernel. Result: it does not compose cleanly.
  - Motivation:
    - Iteration 12 showed a `0.66 ms` matmul-only ceiling. Transferring that win into the fused forward was the whole point.
  - What was tried (all with `WS=False` and PTX tanh throughout):
    1. **2D persistent stats (BLOCK_M=256) + finalize**: widened `_fused_fp8_softcap_ce_desc_persistent_configs` to include `BLOCK_M ∈ {128, 256}` and `BLOCK_N ∈ {64, 128, 256}`. Swapped sigmoid for PTX tanh in both the stats kernel and the existing finalize kernel.
    2. **Row-owner in persistent grid**: new kernel `fused_fp8_softcap_ce_rowreduce_desc_persistent_kernel` that keeps the row-owner single-pass online-LSE (registers only, no partials) but puts the outer row-block iteration inside a persistent `tl.range(start_pid, num_row_blocks, NUM_SMS, flatten=FLATTEN)` so it can use the flatten scheduling hint.
    3. **Row-owner autotune** also widened to `BLOCK_M ∈ {64, 128, 256}` and `BLOCK_N ∈ {64, 128, 256}`.
  - CUDA-event timings on the real training shape:
    - 2D persistent stats+finalize, `flatten=False`, best config `BLOCK_M=256, BLOCK_N=128, BLOCK_K=128, GROUP=8`: `~1.568 ms`
    - 2D persistent stats+finalize, `flatten=True`, best config `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128`: `~1.685 ms` (flatten hurt here)
    - Persistent row-owner, `flatten=False`, best `BLOCK_M=128, BLOCK_N=256, BLOCK_K=128`: `~1.514 ms`
    - Persistent row-owner, `flatten=True`, best `BLOCK_M=256, BLOCK_N=128, BLOCK_K=128`: `~1.484 ms`
    - Non-persistent row-owner TMA (Iteration 11's best, unchanged): `~1.405 ms`
    - Matmul-only ceiling (Iteration 12): `~0.656 ms`
  - Read — why the matmul-only win did not transfer:
    - `flatten=True` speeds up outer-loop pipelining *per SM*. Amortization ratios:
      - Matmul-only 2D tiles: `24,576 ÷ 132 ≈ 186` iterations per SM → big pipeline amortization → `~0.10 ms` saved.
      - Persistent row-owner over row-blocks only: `768 ÷ 132 ≈ 6` iterations per SM → almost nothing to pipeline → only `~0.03 ms` saved (`1.514 → 1.484`).
      - 2D persistent stats has the same `186` tiles/SM, but `flatten=True` actively hurt it (`1.568 → 1.685 ms`): a nontrivial per-tile epilogue (`tanh + max + sum + target-search + 3 stores`) steals compute from the next tile's WGMMA that flatten tries to overlap with it.
    - Net: the matmul-only ceiling is an epilogue-free artifact. With any meaningful per-tile epilogue, the effective fused ceiling is significantly higher.
    - Register pressure confirmation: the 2D stats kernel with `flatten=True` fell back to `BLOCK_M=128, BLOCK_N=128` once the full epilogue was compiled in, even though `BLOCK_M=256, BLOCK_N=128` was available in the config space. The `tl.where` target-search (full `BLOCK_M × BLOCK_N` mask + gather) is a plausible prime contributor.
  - Candidates to unlock the matmul-only ceiling:
    - Slim the per-tile epilogue so it composes with `flatten`. Prime suspect: remove the target-logit `tl.where` from the per-tile path (it needs a full `BLOCK_M × BLOCK_N` mask), and recover the target logit via a separate cheap gather kernel on the side.
    - Split the fused kernel into: `(a)` a raw GEMM + softcap + LSE-stats kernel shaped to match matmul-only (`BLOCK_M=256, flatten=True`), and `(b)` a tiny targets-only pass.
    - WS is still off the table for the row-owner's reduction axis; revisit only if a layout change makes WS's auto-partitioner pick a non-reduced axis.

- Iteration 14: slim 2D persistent stats + separate targets kernel + finalize — `flatten=True` becomes a real win.
  - Design:
    - `fused_fp8_softcap_ce_slim_stats_kernel`: 2D persistent, grouped PID, TMA descriptors, PTX tanh softcap, per-tile online stats (`tile_max`, `tile_sum`). **No** target-logit `tl.where` in the hot loop — removed the full `BLOCK_M × BLOCK_N` mask that had been forcing the autotuner to fall back to smaller tiles.
    - `fused_fp8_softcap_ce_targets_kernel`: one program per row, dot-products `x[row]` with `w[targets[row]]` (K=512). Trivial FLOPs, ~100 MB of HBM reads mostly served from L2 (w is only 4 MB so it's L2-resident). Writes `target_logit[M]`.
    - Existing `fused_fp8_softcap_ce_finalize_nologits_kernel` reduces partials and subtracts the PTX-tanh'd target logit to produce `losses`, `lse`.
  - CUDA-event timings on the real training shape (same harness as prior iterations, no FP8 quantization in the timed region — the `x_fp8` is precomputed once; all row-owner / wrapper numbers in this iteration's table are under the same protocol and are directly comparable):
    - Wrapper `_scaled_mm + fused softcap+CE`: `~1.68 ms`
    - Row-owner non-persistent TMA (prior best, Iteration 11): `~1.405 ms`
    - Slim `flatten=False`, best `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP=8`: `~1.390 ms`
    - **Slim `flatten=True`**, best `BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, GROUP=8`: **`~1.169 ms`**
    - Matmul-only ceiling (Iteration 12): `~0.656 ms`
  - Progress:
    - `~17%` faster than the previous best fused kernel (`1.405 → 1.169 ms`).
    - `~30%` faster than the wrapper under the same no-quant timing protocol (`1.68 → 1.17 ms`).
    - Gap to matmul-only ceiling shrunk from `~2.15x` (row-owner) to `~1.78x` (slim).
  - Public-forward note for apples-to-apples against Codex's earlier numbers:
    - Codex's wrapper/fused measurements (`~2.03 ms` / `~2.29 ms`) were taken through the public `apply` paths that include the Triton FP8 quantization pass (`~0.055 ms`) plus autograd.Function overhead.
    - Adding that same overhead back to this iteration's slim path gives a projected public forward of `~1.22 ms` vs Codex's `2.29 ms` — about `~46%` faster end-to-end for the no-logit public forward if these microbench wins transfer.
  - Read:
    - Removing the target `tl.where` from the per-tile hot path was the key unlock, exactly as hypothesized in Iteration 13: `flatten=True` could not overlap next-tile WGMMA with a fat epilogue, but a slim epilogue lets the overlap actually land.
    - Autotune picked `BLOCK_M=128` (not `256`) under the slim epilogue with `flatten=True`. `BLOCK_N=256` was preferred — suggests the freed register budget went into wider N tiles rather than taller M tiles. Worth a follow-up probe explicitly forcing `BLOCK_M=256` to check whether autotune under-explored it.
    - The separate targets kernel is cheap (K=512 dot per row, ~100 MB of HBM reads with most of `w` hot in L2) and did not dominate the total time; a component breakdown would confirm exactly where the 1.17 ms is spent.

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

## Backward optimization (2026-04-15)

- **Baseline**: nologit fused backward with original tile shapes (`dx BM=8,BN=256,BK=64`, `dw BM=32,BN=128,BK=64`): **125 ms** — catastrophically slow, 5.9× the split-path's **21 ms** (Triton dx/dw over stored logits).
  - Root cause: tiny BLOCK_SIZE_M=8 for dx → 12K programs each recomputing full logits.
  - Revealed that 500-step training at 3.6M tok/s vs 8.4M baseline was entirely caused by backward.

- **Tile shape tuning**: bump dx to BM=64,BV=128,BK=256,w=8; dw to BM=64,BN=128,BK=256,w=8 → combined **14.9 ms**. 8.4× improvement.
  - 500-step training: 7.48M tok/s, val_loss 3.3026. Still ~1M tok/s behind split-path baseline (8.4M).

- **Discovering the real split-path backward**: `FusedFP8SoftcapCrossEntropyFn.backward` uses `fused_softcap_ce_bwd_kernel` (row-wise CE grad) + **cuBLAS** `@` for the two grad matmuls. Not the Triton dx/dw kernels (those are `FusedFP8SoftcapCrossEntropyDirectBwdFn`).
  - Actual split-path bwd: **3.2 ms** (CE bwd 0.7ms + cuBLAS dx 1.2ms + cuBLAS dw 1.1ms, overlapped).
  - Default (non-FP8) full path: fwd 10.1ms + bwd 11.6ms = 21.7ms.

- **Persistent TMA slim backward kernels**: dx and dw with TMA descriptors, persistent scheduling, target comparison pulled out of hot loop. Best dx: BM=128,BV=128,BK=256,w=8,s=3 → **4.09 ms**. Best dw: BM=64,BV=128,BK=256,w=8,s=4 → **5.56 ms**. Combined + corrections: **10.0 ms**.

- **Component decomposition** (dx at BM=128,BV=128,BK=256):
  - FP8 recompute only: 1.86ms
  - FP8 + bf16 grad matmul (no epilogue): 3.27ms (+1.41ms for bf16)
  - Full (+ epilogue): 4.08ms (+0.81ms for epilogue)
  - Key insight: dx and dw **independently recompute logits** — 2× FP8 GEMM waste.

- **Forward logit writeback**: added `fused_fp8_softcap_ce_slim_stats_writeback_kernel` that stores bf16 logits to HBM during the epilogue (overlapped with tanh/exp compute).
  - Writeback fwd: 1.45ms (vs 1.18ms no-writeback). Overhead: **+0.27ms**.
  - Writeback fwd + cuBLAS bwd: **4.83 ms** total.
  - Split-path (mm + CE + cuBLAS bwd): **5.00 ms** total.
  - Memory cost: 1.6 GB logits buffer (same as split-path).
  - Strictly better than split-path on both speed (−0.17ms) and memory (same).

- **Fused dx+dw kernel attempt**: wrote `fused_fp8_softcap_ce_dxdw_kernel` — persistent TMA kernel tiling over (M, K_out), inner V-loop recomputes logits once (shared), dx via GEMM accumulation, dw via atomicAdd.
  - **v1 (4-way K unroll)**: 4 dx accumulators + 4 dw GEMMs per V-tile. Best: 64ms. Register pressure + 8 bf16 GEMMs per V-tile killed occupancy.
  - **v2 (dx_slim + atomicAdd dw)**: tile over (M, K_out) like dx_slim, 1 dx acc + 1 dw atomicAdd per V-tile. Best: BM=128,BV=128,BK=128,w=8,s=4 → **33 ms**. atomicAdd contention from 768 M-tiles writing to 64 V-tile slots destroyed performance.
  - Root cause: Triton tile-based bf16 GEMM reduction (64 tiny [BM,BV]×[BV,BK] matmuls per output tile) is ~5× slower than cuBLAS's single large [98304,8192]×[8192,512] GEMM.

- **FP8 backward GEMMs exploration**: FP8 dx+dw GEMMs alone: **1.15 ms** (vs 2.25ms bf16). 2× speedup.
  - But quantizing grad_logits [98304,8192] to FP8: 10.6ms. Contiguous transpose for dw: 6.8ms.
  - Full pipeline (CE bwd + quant + transpose + FP8 GEMMs): **18.6 ms** vs bf16 path's 3.2ms.
  - Verdict: quantization/transpose overhead on the 1.6 GB tensor overwhelms the GEMM speedup.

- **cuBLAS GEMM breakdown**:
  - bf16 dx [98304,8192]@[8192,512]: 1.05ms
  - bf16 dw [8192,98304]@[98304,512]: 1.04ms
  - bf16 dx+dw combined: 2.37ms (CE bwd ~0.8ms → total 3.2ms)
  - FP8 fwd mm [98304,512]@[512,8192]: 0.87ms

- **Conclusion**: recompute-based Triton backward cannot compete with cuBLAS for these shapes. cuBLAS bf16 backward at 3.2ms is the floor. Best practical path:
  - **Writeback approach**: nologit fwd with logit writeback (2.1ms) + cuBLAS bwd (3.2ms) = **5.3ms total**.
  - **Split-path**: FP8 mm + CE fwd (1.2ms) + cuBLAS bwd (3.2ms) = **4.4ms total**.
  - The 3ms bwd target is essentially at the cuBLAS floor — not achievable via Triton recompute.

- **Next**: wire the winning backward approach (cuBLAS) into `NoLogitFusedFP8SoftcapCrossEntropyFn.backward` and run a training comparison.
