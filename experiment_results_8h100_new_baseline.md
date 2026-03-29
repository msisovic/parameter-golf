# Experiment Results (8xH100, New GPTQ Baseline)

## Goal

Retune the delayed dual-recurrence recipe against the AR self-generated GPTQ + BigramHash `3072x112` stack.

Fixed settings for this sweep:

- Seed: `314`
- `BIGRAM_VOCAB_SIZE=3072`
- `BIGRAM_DIM=112`
- `TTT_ENABLED=0` during training runs
- Sweep `RECUR_START_STEP == WARMDOWN_ITERS` over `{2500, 3000, 3500}`

## Experiment 1: Equal Start/Warmdown Sweep

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB | Post-TTT BPB | Notes |
|-------|-----|--------------------|--------------------------------|--------------|-------|
| `RECUR_START_STEP=2500`, `WARMDOWN_ITERS=2500` | [sweep_logs/newbase_s314_eq2500.log](/root/parameter-golf/sweep_logs/newbase_s314_eq2500.log) | `1.1356` | `1.11615505` | pending | Clearly worse than 3000/3500. Earlier activation hurts. |
| `RECUR_START_STEP=3000`, `WARMDOWN_ITERS=3000` | [sweep_logs/newbase_s314_eq3000.log](/root/parameter-golf/sweep_logs/newbase_s314_eq3000.log) | `1.1344` | `1.11550547` | pending | Best of the seed-314 equal sweep, but still worse than the earlier seed-1337-style `1.11471611`. |
| `RECUR_START_STEP=3500`, `WARMDOWN_ITERS=3500` | [sweep_logs/newbase_s314_eq3500.log](/root/parameter-golf/sweep_logs/newbase_s314_eq3500.log) | `1.1343` | `1.11589748` | pending | Slightly worse than 3000 despite slightly better post-EMA. |

### Short Findings

- Equal `2500/2500`, `3000/3000`, `3500/3500` confirms the old center still holds: `3000/3000` is best among these three.
- The gain is not recovered by simple retuning alone on seed `314`; all three are worse than the earlier `1.11471611` no-TTT run.
- `3500/3500` improving post-EMA but regressing after GPTQ suggests the remaining gap is not purely a training-side issue.

## Experiment 2: Broad XSA, Skip Recurrent Layers

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB | Post-TTT BPB | Notes |
|-------|-----|--------------------|--------------------------------|--------------|-------|
| `RECUR_START_STEP=3000`, `WARMDOWN_ITERS=3000`, `XSA_LAST_N=13`, `XSA_SKIP_RECUR=1` | [logs/xsa_all_skip_recur.txt](/root/parameter-golf/logs/xsa_all_skip_recur.txt) | `1.1339` | `1.11477270` | `1.11452572` | Best structural tweak so far; broad XSA helps, but not on recurrent layers. |
| Same as above, rerun with scale dump | [logs/xsa_all_skip_recur_scales.txt](/root/parameter-golf/logs/xsa_all_skip_recur_scales.txt) | `1.1335` | `1.11425194` | pending | Faster rerun with slightly better final score; likely runtime variance, not an intentional training-path change. |

### Scale Dump Notes

- `attn_scale` and `mlp_scale` are smooth and positive across all virtual layers; there is no obvious dead branch.
- Repeated recurrent passes show a mild MLP increase relative to their first pass, but attention remains substantial:
  - phys 4 first pass: `attn=0.4100`, `mlp=0.2414`
  - phys 4 repeated: `attn=0.3858`, `mlp=0.2845`
  - phys 5 first pass: `attn=0.3548`, `mlp=0.2574`
  - phys 5 repeated: `attn=0.5062`, `mlp=0.3033`
- `final_int6` scale dumps exactly match `post_ema`, which is expected because these per-block scale vectors are not part of the int6-quantized weights.

### Updated Findings

- Skipping XSA on recurrent layers while using broad XSA elsewhere is the clearest recurrence-specific win on the new baseline.
- The scale dump does not support a simple story that repeated passes are "MLP only"; attention still carries a large residual weight on recurrent passes.
- The improved rerun score (`1.11425194`) is real, but the speedup appears to be run-to-run/runtime variance rather than a known code change in the training path.

## Experiment 3: Layer 0 MLP-Only (Provisional)

| Setup | Log | Status | Notes |
|-------|-----|--------|-------|
| `LAYER0_MLP_ONLY=1` on top of `XSA_LAST_N=13`, `XSA_SKIP_RECUR=1` | [logs/layer0_mlp_only_xsa_skip_recur.txt](/root/parameter-golf/logs/layer0_mlp_only_xsa_skip_recur.txt) | partial | Early training signal shows no obvious quality collapse and a real pre-recurrence speedup (`80.48ms` at step `3000` vs `85.06ms` baseline). The post-activation region is noisy because the recurrent compiled path changes. |
