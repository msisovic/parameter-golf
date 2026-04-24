# TTT Wallclock Handoff - 2026-04-24

## Current Branch

- Branch: `pr1530-exact-record`
- Latest pushed commit before this handoff: `cf49e27 Prewarm TTT eval-mode compile`
- The worktree has untracked local artifacts (`datasets/`, `tokenizers/`, logs, `final_model.pt`, `final_model.int6.ptz`, etc.). They were intentionally not committed.

## Main Fix Landed

- `train_gpt_v2.py` now prewarms TTT in the same mode used by timed TTT:
  - calls `ttt_model.eval()` before TTT compile warmup
  - warms both `TTT_BATCH_SIZE` and the final partial tail batch shape
- This removed avoidable timed `_fwd_ttt_inner` recompiles from the TTT timer section.
- Validation: `python -m py_compile train_gpt_v2.py`

## Reference Commands / Inputs

Common data/model settings used in the recent runs:

```bash
DATA_DIR=. \
DATA_PATH=./datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
GLOBAL_TTT_CHUNK_TOKENS=65536 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=8192 SEQ_LEN_BUMP_FRAC=0.85 \
EVAL_SEQ_LEN=8192 TTT_EVAL_SEQ_LEN=8192 \
ROPE_YARN=1 ROPE_TRAIN_SEQ_LEN=2048
```

Eval-only runs use:

```bash
EVAL_ONLY=1 EVAL_ONLY_PATH=final_model.pt
```

## Important Results

### Original full run, seed 1, chunk 48, batch 16, before fix

- Log: `train_1024_seed1_seq8192_ttt_recompile.log`
- `TTT_CHUNK_SIZE=48`
- `TTT_BATCH_SIZE=16`
- Timed TTT:
  - `val_loss=2.32727725`
  - `val_bpb=1.06346814`
  - `eval_time=958.5s`
- Problem:
  - warmup compiled TTT while model was in train mode
  - timed TTT switched to eval mode and paid recompiles

### Eval-only, seed 1 checkpoint, chunk 48, batch 16, after fix

- Log: `evalonly_seed1_seq8192_ttt_chunk48_bsz16_20260424_161529.log`
- Internal log: `logs/evalonly_seed1_seq8192_ttt_chunk48_bsz16_20260424_161529.txt`
- `TTT_CHUNK_SIZE=48`
- `TTT_BATCH_SIZE=16`
- Timed TTT:
  - `val_loss=2.32727639`
  - `val_bpb=1.06346775`
  - `eval_time=765.7s`
- Same score as original, about 193s faster in the timed TTT section.
- Phase timings:
  - phase 1 pause: `403.3s`
  - phase 3 done: `559.0s`
  - final: `765.7s`

### Eval-only, seed 1 checkpoint, chunk 96, batch 16

- Log: `evalonly_seed1_seq8192_ttt_chunk96_bsz16_20260424_155906.log`
- `TTT_CHUNK_SIZE=96`
- `TTT_BATCH_SIZE=16`
- Timed TTT:
  - `val_loss=2.32812720`
  - `val_bpb=1.06385653`
  - `eval_time=433.5s`
- Fast enough, but score was worse by about `+0.00039 bpb`.
- This suggests chunk size materially affects local TTT quality.

### Full run, seed 1337, chunk 48, batch 20

- Log: `train_1024_seed1337_seq8192_ttt_no_recompile.log`
- `TTT_CHUNK_SIZE=48`
- `TTT_BATCH_SIZE=20`
- Timed TTT:
  - `val_loss=2.32717390`
  - `val_bpb=1.06342092`
  - `eval_time=858.0s`
- This was slower than batch 16 despite fewer doc batches (`2500` vs `3125`).
- Likely cause:
  - larger doc batches reduce scheduling granularity
  - sorted-by-length batches make long-doc stragglers worse
  - larger batch increases per-rank memory/compute pressure
  - phase overshoot also increased (`pd:816` vs `pd:780` at phase 1)
- Do not pursue `TTT_BATCH_SIZE=20` or `24` as the next speed lever.

## Current Understanding

- `TTT_BATCH_SIZE=16` is the better throughput point than `20` for this workload.
- `TTT_CHUNK_SIZE=48` preserves known score but is too slow (`765.7s` timed TTT).
- `TTT_CHUNK_SIZE=96` is fast enough but loses score.
- The next likely path under 600s is an intermediate chunk size plus LR compensation.

## Recommended Next Experiment

Run eval-only first:

```bash
TTT_BATCH_SIZE=16 \
TTT_CHUNK_SIZE=64 \
TTT_LORA_LR=0.000133
```

Rationale:

- `64` is between the known quality point (`48`) and fast point (`96`).
- Scaling LR by chunk size roughly preserves adaptation per token:
  - `1e-4 * 64 / 48 = 1.333e-4`
- Expected timed TTT: likely near the 600s boundary.
- If score is worse, sweep nearby:
  - `TTT_LORA_LR=0.00012`
  - `TTT_LORA_LR=0.00015`

## Operational Notes

- If measuring contest-style full wallclock, remember TTT compile warmup is outside the logged `total_eval_time`.
- The tail-batch warmup makes compile warmup longer but keeps timed TTT cleaner.
- If only final timed TTT matters, current patch is helpful.
- If full wallclock including warmup matters, consider making tail-batch prewarm optional or avoiding the tail shape if its timed recompile cost is smaller than compile warmup cost.
- GPU memory during TTT can be uneven because ranks dynamically claim doc batches. Use the tightest GPU, not the freest GPU, to judge OOM risk.
