# Experiment Log

## 2026-04-19 - Seq-Len Curriculum + Recompile Investigation

### Goal

Evaluate `TRAIN_SEQ_LEN=2048 -> 4096` curriculum, understand the suspicious slowdown near the bump, and compare post-training eval / quantized eval / TTT against the non-curriculum baseline.

### Key Training Command

```bash
SEED=0 GPTQ_RESERVE_SECONDS=13 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Recompile Finding

- The large slowdown near the seq-len bump was real and was caused by a live recompile around the transition to `4096`.
- Recompile logs showed `_train_forward_bump_loop` recompiling when `cu_seqlens` changed from length `192 -> 256`.
- Simply adding `256` to the synthetic warmup bucket list was not sufficient by itself.
- The working fix was to keep the existing synthetic startup warmup, and additionally prime the live post-bump regime with real loader-produced batches right after the seq-len switch, excluding that one-time cost from the training timer.

### Result of Hybrid Warmup Fix

- After the fix, there were no recompiles after actual training started and no recompiles after the live seq-len bump.
- The fixed curriculum run reached `4770` steps before the wallclock cap, versus `4533` in the original suspicious run.

500-step training logs from the fixed curriculum run:

- `500`: `train_loss 3.2561`, `train_time 0.8m`, `tok/s 8220365`
- `1000`: `train_loss 3.0031`, `train_time 1.6m`, `tok/s 8181105`
- `1500`: `train_loss 3.0132`, `train_time 2.4m`, `tok/s 8177328`
- `2000`: `train_loss 2.9752`, `train_time 3.2m`, `tok/s 8178470`
- `2500`: `train_loss 3.0573`, `train_time 4.3m`, `tok/s 7620238`
- `3000`: `train_loss 2.8952`, `train_time 5.6m`, `tok/s 6998953`
- `3500`: `train_loss 2.9624`, `train_time 6.8m`, `tok/s 6758070`
- `4000`: `train_loss 2.8759`, `train_time 8.0m`, `tok/s 6586898`
- `4500`: `train_loss 2.8528`, `train_time 9.1m`, `tok/s 6449402`

Other key events:

- Looping enabled at step `2137`
- Seq-len bump to `4096` at step `3943`
- Stopped early at step `4770` due to wallclock cap

### Validation / Quantized / TTT Results

#### Curriculum Run, Validation At 2048

- End-of-training val: `2.7727` bpb `1.0734`
- Diagnostic pre-quantization post-EMA val: `2.77139888` bpb `1.07286029`
- Diagnostic quantized val: `2.80068618` bpb `1.08419796`
- Quantized TTT LoRA val: `2.77292349` bpb `1.07348509`

This remains the best TTT result observed so far.

#### Eval-Only From Saved Artifact, Validation At 4096

Command pattern:

```bash
EVAL_ONLY_PATH=final_model.pt \
EVAL_SEQ_LEN=4096 \
TTT_EVAL_SEQ_LEN=4096 \
...
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Pre/post quantized eval at `4096`:

- Diagnostic pre-quantization post-EMA val: `2.76580509` bpb `1.07069019`
- Diagnostic quantized val: `2.79578644` bpb `1.08229648`

This confirms that the quantized `4096` eval baseline is better than the quantized `2048` eval baseline.

### 4096 TTT Investigation

- `TTT_EVAL_SEQ_LEN=4096` with default `TTT_BATCH_SIZE=64` OOMed during TTT compile warmup.
- Reducing to `TTT_BATCH_SIZE=16` allowed `4096` TTT to complete.

Results at `TTT_BATCH_SIZE=16`:

- `TTT_LORA_LR=1e-4`: `2.77790876` bpb `1.07541132`
- `TTT_LORA_LR=5e-5`: `2.78081074` bpb `1.07653476`
- `TTT_LORA_LR=2.5e-5`: `2.78751265` bpb `1.07912927`
- `TTT_LORA_LR=1.25e-4`: `2.77826573` bpb `1.07554951`

Current conclusion:

- `TTT_LORA_LR=1e-4` is the best `4096` TTT setting tried so far.
- `4096` wins before TTT, but does not retain that lead after TTT.
- Best `2048` TTT result (`2.77292349`) is still better than best `4096` TTT result (`2.77790876`).
- The likely interpretation is that long-context TTT currently gives a smaller gain than 2048 TTT in this regime, even though the base and quantized `4096` evals are stronger.

### Working Conclusions

- The seq-len curriculum recompile issue appears fixed well enough to continue experimenting with this regime.
- The limiting issue is no longer the live seq-len bump recompile.
- The next priority should be improving the training baseline / quantized baseline under seq-len curriculum rather than continuing to chase TTT gains first.
- For now, it is reasonable to treat long-context TTT as a smaller win in this regime and focus on beating the non-curriculum training baseline.
