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
