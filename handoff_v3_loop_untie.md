# V3 LOOP_UNTIE Investigation — Handoff

**Branch**: `pr1530-exact-record`  
**Goal**: get LOOP_UNTIE_EXTRA_SCALARS working cleanly on top of base_v3 with no pre-loop train_loss drift, so the mechanism's gain (extras as separate learnable copies of the looped layer scalars) lands cleanly in the final eval. Long-term goal is to recover v2's ~0.002 final-bpb improvement; LOOP_UNTIE alone is one small contributor to that.

## Where things stand

### Versions tried and what each gave us

| variant | step 500 train_loss (Δ vs base_v3 = 2.5617) | post-loop train_loss | run completed? | final pre-quant val_bpb |
|---|---|---|---|---|
| `base_v3` | 2.5617 (0) | converges normally | yes | 1.06428436 |
| **kwargs+isNone** (Block.forward has `resid_mix=None, attn_scale=None, mlp_scale=None`, `mix_src = self.resid_mix if resid_mix is None else resid_mix`) | 2.5708 (+0.0091) | rejoins base_v3 by step 2500 | yes | **1.06409244** (−0.00019) |
| `LOOP_UNTIE=0` (same code, mechanism off) | 2.5647 (+0.0030) | rejoins | yes | (not measured to end) |
| **split methods** (`Block.forward` + `Block.forward_with_overrides`, `if extra_pass < 0:` branch in `_forward_hidden`) | 2.5638 (+0.0021) | hung in recompile after loop activation at step ~2160; 10+ min stuck, no progress | **no** | n/a |

### What we learned

1. **The kwargs+is-None pattern in `Block.forward` is the dominant pre-loop drift cause.** Splitting into two methods drops drift from +0.009 to +0.002. The math is identical but Dynamo/Inductor compiles a structurally different graph when `resid_mix is None`-style conditionals are present.
2. **The split-methods approach trades pre-loop drift for post-loop compile hang.** After looping_active flips at step ~2160, Dynamo has to recompile the looping graph. With two method bodies (`forward` + `forward_with_overrides`), it doesn't finish — observed 10+ min of 100% CPU on rank 0 with no progress, log buffered at step 2000, run had to be killed.
3. **`LOOP_UNTIE=0` drift pattern looks like noise** (~0.001–0.003 oscillation around base_v3, doesn't compound). So the dead-code overhead of having the triple-form `encoder_loop_entries` and the `_block_scalar_overrides` helper is benign by itself — it's specifically the `is None` conditional that perturbs the trace.

## Current state of `train_gpt_v3.py`

**Now contains the unified-forward refactor (UNTESTED)** — see commit message.

The split-methods version (which hangs post-loop) was committed as `d4b1670`. On top of that, `Block.forward` and `_block_with_lora` were unified back into single methods that take `resid_mix`, `attn_scale`, `mlp_scale` as **required positional args** (no defaults, no `is None`). Callers always pass tensors, sourced via the new `_resolve_block_scalars(block, extra_pass, loop_local_idx)` helper:
- Base entries (`extra_pass < 0`): returns `(block.resid_mix, block.attn_scale, block.mlp_scale)`.
- Extra entries: returns the `loop_extra_*[extra_pass, loop_local_idx]` slices.

Diff vs `train_gpt_base_v3.py` (current state):
- LOOP_UNTIE_EXTRA_SCALARS hparam
- triple-form `encoder_loop_entries` / `decoder_loop_entries`
- `loop_extra_attn_scale` / `loop_extra_mlp_scale` / `loop_extra_resid_mix` Parameters (init: ones / [1,0])
- `_resolve_block_scalars` helper (returns block scalars or loop_extra slot)
- `Block.forward` modified to take `resid_mix, attn_scale, mlp_scale` as required positional args
- `_block_with_lora` modified to take same 3 required positional args
- `_forward_hidden` and `forward_ttt` use `_resolve_block_scalars` + a single uniform call (no if/else branch)
- Optimizer scalar group append + `CONTROL_TENSOR_NAME_PATTERNS` extension
- `_clone_loop_extras_from_source` helper called at loop activation (copies block params + AdamW state into the loop_extra slots)

**Key open question for tomorrow**: does this unified version finish recompile after loop activation in reasonable time (like base_v3 does, ~30s)? If yes, also check pre-loop step 500 train_loss is ≤ 2.564 (matching base_v3 + noise). The split-methods version was at the right pre-loop value but never finished post-loop recompile.

## What to try next (in order)

### 1. Unified-forward refactor (the next experiment)

Replace the split-methods approach with a **single `Block.forward`** that takes `resid_mix`, `attn_scale`, `mlp_scale` as **required positional args** (no defaults, no `is None` check). All callers always pass tensors:
- Base entries (extra_pass < 0): pass `block.resid_mix, block.attn_scale, block.mlp_scale` directly.
- Extra entries: pass override tensors from `loop_extra_*`.

Drop `forward_with_overrides`, `_block_with_lora_with_overrides`, and the `if extra_pass < 0:` branches.

**Why this should work**:
- One method body → half the Inductor work compared to split → recompile after loop activation should finish in ~30s like base_v3 does.
- No `is None` conditional → same trace shape regardless of arg source → pre-loop should still match base_v3.
- Same compute as base_v3 (just plumbed via args instead of `self.*`).

**Risk**: if Dynamo treats `block.resid_mix` (read at call site) differently from `self.resid_mix` (read inside forward), drift could come back. Empirical question.

**How to test**: apply, run `LOOP_UNTIE_EXTRA_SCALARS=1`, watch step 500 (should be ~2.563–2.564) and step 2500 (post-loop, should rejoin base_v3 around 2.539). If both look right and the run completes, ship.

### 2. Fallback if (1) hangs the same way

The kwargs+is-None version (drift +0.009 but completes) gave **+0.00019 pre-quant val_bpb** vs base_v3. That's a real, if small, win. Reproduce it cleanly and use as the baseline LOOP_UNTIE result. Then go look at how to add curriculum back for the bigger gains.

### 3. Fallback if (1) drifts back to +0.005+

The drift is intrinsic to having more than one tensor source for the same logical "what scalar to use here" — Dynamo sees different inputs as different traces. Ideas to recover:
- Always pass `loop_extra_*` tensors (even pre-loop), with values pre-set to clones of `block.*`. So Block.forward always reads from loop_extra slots. No conditional, single source. Cost: extra memory (3× as many param Parameters), and `loop_extra_*` would need to be inited as clones from the start (currently we only clone at activation).
- Move the override into the Block constructor so `block.attn_scale` IS the loop_extra slot for looping layers. More invasive structural change.

## Reference numbers

```
base_v3 control (LOOP_UNTIE=0, ROPE_YARN=0):
  step 500/1000/1500/2000 train_loss: 2.5617 / 2.7945 / 2.6178 / 2.6484
  layer_loop:enabled step:2160
  step 2500/3000/3500 train_loss: 2.5394 / 2.5512 / 2.5528
  stop_step: 4892
  pre-quant post-EMA val_bpb: 1.06428436
  diagnostic quantized val_bpb:  1.07278898
  quantized_ttt_phased val_bpb:  1.06016974
```

## Run command (reuse base_v3 hparams + LOOP_UNTIE)

Use this exact command to test the unified-forward version. Replace `<variant>` with a label (e.g. `unified`).

```bash
SEED=42 RUN_ID=v3_loop_untie_unified_s42 ARTIFACT_DIR=artifacts/v3_loop_untie_unified_s42 \
NCCL_NET=Socket \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 GATE_WINDOW=12 \
GATED_ATTN_QUANT_GATE=1 SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
LOOP_UNTIE_EXTRA_SCALARS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt_v3.py > logs/v3_loop_untie_unified_s42.log 2>&1
```

To verify it works as expected, watch for:
- Step 500 train_loss ≈ 2.564 (within ~0.001–0.003 of base_v3's 2.5617)
- `layer_loop:enabled step:2160` log line appearing within ~30s of step 2000 (recompile must finish quickly)
- Continuous progress past step 2500 (train_loss should rejoin base_v3's ~2.539 trajectory)
- Run reaches `stopping_early: wallclock_cap` and produces `pre-quantization post-ema val_bpb`

Compare final `quantized_ttt_phased val_bpb` to base_v3's `1.06016974` and to the kwargs-only run's previous result (which finished the run with +0.00019 val_bpb improvement at the pre-quant stage).

To run with the mechanism off (sanity check that the new code path is functionally equivalent to base_v3):

```bash
LOOP_UNTIE_EXTRA_SCALARS=0 ...
```

This should produce step 500 train_loss within ~0.001–0.003 of base_v3 (oscillating noise). If it drifts more, there's a bug in `_resolve_block_scalars` falling back to `block.*` for base passes.

## Pointers

- Detailed step-by-step log: `investigation_log_v3.md`
- Working logs: `logs/v3_loop_untie_only_s42.log` (kwargs version, completed), `logs/v3_loop_untie_off_s42.log` (LOOP_UNTIE=0 baseline), `logs/v3_loop_untie_split_s42.log` (split version, hung)
- base_v3 reference: `logs/base_v3_control_s42.txt`

## Bigger picture

The pre-loop drift investigation has consumed a lot of effort. Even the "good" kwargs version only buys +0.00019 on pre-quant val_bpb — a fraction of v2's +0.002 total improvement. **The real lever for the v2-sized win is the seq-len curriculum + long-context TTT**, which were stripped to focus on diagnosing the LOOP_UNTIE port. After landing a clean LOOP_UNTIE result on this baseline, the next step is to add seq-len curriculum back **carefully**, ideally without the bump_prewarm + warm_eval_logits + wrapper-function compile that caused the original v3 to recompile-thrash.
