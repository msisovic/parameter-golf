# Experiment Log V3

## 2026-04-28 — Port v2 Mechanisms onto base_v3 → `train_gpt_v3.py`

### Context

- Imported `train_gpt_base_v3.py` as a new baseline (retuned hparams, fused softcapped-CE op, LQER asym rank-k, sparse_attn_gate, `_forward_hidden`/`_project_logits` split, `PREQUANT_ONLY`/`TTT_EVAL_ONLY` resume hooks, lrzip `pergroup` compressor).
- Ported the four v2 mechanisms onto it: fixed rotary cache regimes (slot 0 short / slot 1 long), seq-len curriculum (`TRAIN_SEQ_LEN`→`TRAIN_SEQ_LEN_END` at `SEQ_LEN_BUMP_FRAC`), `LOOP_UNTIE_EXTRA_SCALARS`, and the rotary/scalar-override threading through Attention/Block/`_forward_hidden`/`forward_ttt`/`_block_with_lora`.
- Did **not** port v2's `EVAL_ONLY` family — base_v3 already has `TTT_EVAL_ONLY`/`PREQUANT_ONLY` and the user prefers those.

### Hparam Reconciliation Decisions (vs v2)

- Switched output gate: dropped `GATED_ATTN_ENABLED` + `INIT_STD=0.005`, picked up base_v3's `SPARSE_ATTN_GATE_ENABLED=1 SCALE=0.5` (mutually exclusive per Block.__init__:953). Sparse gate uses `(num_heads, gate_window)=(8,12)` weights vs dense `(8, 512)` → ~2.3% the gate params, motivated by submission size.
- Kept `GATED_ATTN_QUANT_GATE=1` despite the misleading name — it's the int8-per-row quant routing for `attn_gate_w`, shared by both gate variants.
- Kept v2's long-seq TTT trio (`TTT_CHUNK_SIZE=64`, `TTT_LORA_LR=7.5e-5`) — the chunk-64 LR sweep in `experiment_log_v2.md` showed these were best at ~600s wallclock for 8192-eval. Adopted v3's `TTT_LORA_RANK=80` (was 96).
- Kept v2's `GPTQ_RESERVE_SECONDS=4` (vs v3's 0.5). Long-seq calibration is heavier; LQER pack adds more post-train work; 0.5s leaves no headroom.
- Adopted v3 retunes: `PHASED_TTT_PREFIX_DOCS=2500`, `EMBED_CLIP_SIGMAS=14.0`, `MLP_CLIP_SIGMAS=11.5`, `MIN_LR=0.1`, `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `TTT_BETA2=0.99`, `TTT_WEIGHT_DECAY=0.5` (v3 default became 1.0 — re-overrode to v2's old default).
- Seed: `42` (v3 baseline submission seed) instead of v2's 1337.

### Distributed Determinism Bug at Loop-On Transition (and the Refactor That Removed It)

The first run hit an NCCL ALLREDUCE timeout at step ~2100 (just past `frac=0.35`).

**Root cause**: `training_frac(step, elapsed_ms)` uses each rank's local `time.perf_counter()` (v3.py:3614–3617), so per-rank `elapsed_ms` differs by ~ms. At the threshold-crossing step, some ranks computed `frac >= enable_looping_at = 0.35` and flipped `base_model.looping_active = True`; others stayed False.

With the original v2-port LOOP_UNTIE design, `loop_extra_attn_scale` / `mlp_scale` / `resid_mix` were three GPT-root tensors *only* referenced via `_block_scalar_overrides` for `extra_pass >= 0` triples — i.e., only when `looping_active=True`. So at the threshold-crossing step, looping-active ranks had gradients on those tensors and looping-inactive ranks did not. `_all_reduce_packed_grads` filters `if p.grad is not None`, so its flat bucket size differed by exactly **12288 elements** (3072 + 3072 + 6144) — matching the NCCL timeout's `NumelIn` delta (`44231 − 31943 = 12288`).

**Why base_v3 doesn't deadlock alone**: without `LOOP_UNTIE`, no rank-conditional params. Looping just runs shared params more times — same bucket on every rank, just a one-step gradient-magnitude blip if ranks disagree. Latent silent race, not a deadlock.

**Why v2 didn't deadlock**: same code path (v2.py:3247–3257), same race. Got lucky on rank-skew at the threshold step in past runs.

**First fix (kept around for context)**: synchronize the trigger via `dist.all_reduce(MIN)` on a 1-elem int32 bool while the flag is still False. Worked, ~150ms total overhead across a 600s run.

**Real fix — refactor**: the per-step sync was a workaround for a structural issue. Replaced the three separate `loop_extra_*` GPT-root tensors with **per-block pass-indexed scalars**:

```
Block.attn_scale  : [pass_count, dim]      # row 0 = base, rows 1..n-1 = extras
Block.mlp_scale   : [pass_count, dim]
Block.resid_mix   : [pass_count, 2, dim]
```

Looping blocks (layers in `[loop_start, loop_end]`) under LOOP_UNTIE get `pass_count = 1 + num_loops`. All other blocks (and all blocks when LOOP_UNTIE off) keep `pass_count = 1`. Init: row 0 keeps original `ones` / `[1, 0]`; rows 1..n-1 are zeros for attn/mlp scale (spike suppression), `[1, 0]` for resid_mix (passthrough). Loop entries become `(layer_idx, pass_idx)` pairs; `Block.forward` reads `self.attn_scale[pass_idx]` etc.

The autograd-level invariant: even when forward only references `attn_scale[0]`, PyTorch allocates `.grad` for the **full Parameter** (shape `[pass_count, dim]`). So the all_reduce bucket size is structurally identical across ranks regardless of `looping_active`, regardless of whether each rank has crossed the threshold. No sync needed.

Net delta from the refactor:
- Dropped: `loop_extra_*` GPT-root params, `_block_scalar_overrides`, the optimizer scalar-group append, `CONTROL_TENSOR_NAME_PATTERNS` extension, and the `dist.all_reduce(MIN)` per-step sync.
- Added: a `pass_count` arg to `Block.__init__` and a `pass_idx` arg to `Block.forward` / `_block_with_lora`.
- Result is ~18 lines smaller, eliminates a synchronization point, and the off-state is functionally equivalent to `LOOP_UNTIE=0` shared looping.

### Run Command

```bash
SEED=42 \
RUN_ID=v3_seq8192_loop_untie_s42 \
ARTIFACT_DIR=artifacts/v3_seq8192_loop_untie_s42 \
NCCL_NET=Socket \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
GLOBAL_TTT_CHUNK_TOKENS=65536 GLOBAL_TTT_MOMENTUM=0.9 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 WARMDOWN_FRAC=0.85 \
BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 \
TTT_BATCH_SIZE=16 TTT_CHUNK_SIZE=64 TTT_LORA_LR=0.000075 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
GATED_ATTN_QUANT_GATE=1 GATE_WINDOW=12 SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=8192 SEQ_LEN_BUMP_FRAC=0.85 \
EVAL_SEQ_LEN=8192 TTT_EVAL_SEQ_LEN=8192 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048 \
LOOP_UNTIE_EXTRA_SCALARS=1 \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
torchrun --standalone --nproc_per_node=8 train_gpt_v3.py 2>&1 | tee logs/v3_seq8192_loop_untie_s42.log
```

### Takeaways

- The wallclock-based `training_frac` race is real and needs an explicit cross-rank sync at any flag flip whose decision affects which params are in the autograd graph. With `LOOP_UNTIE` the cost of "ranks disagreeing for one step" is a hard NCCL deadlock; without it, just one wonky training step. Both are bugs, but only one is loud.
- Result TBD — first clean run pending after the fix.
