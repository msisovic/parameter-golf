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
