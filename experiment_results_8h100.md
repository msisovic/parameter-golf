# Experiment Results (8xH100)

## Experiment 1:

- **Date**: 2026-03-28
- **Hardware**: 8xH100 80GB
- **Log**: [logs/7aab75c8-4bb9-4969-b15b-6561736ae297.txt](/root/parameter-golf/logs/7aab75c8-4bb9-4969-b15b-6561736ae297.txt)

| Metric | Loss | BPB |
|--------|------|-----|
| Pre-TTT (int6 sliding window) | 1.89101066 | 1.11996599 |
| Post-TTT | 1.88689319 | 1.11752739 |


## Experiment 2: All-layer sandwich norm ablation on 8xH100

- **Date**: 2026-03-28
- **Hardware**: 8xH100 80GB
- **Key changes**: `SANDWICH_NORM=1` on all layers, with the same delayed dual-recurrence recipe as the recent 8xH100 baseline

### Notes
- Early signal is clearly negative: at step 4000, val_bpb worsened from **1.2079** to **1.2330** on the matched 8xH100 setup.
- Regression appears before recurrence activates, so broad sandwich norm is likely harmful rather than specifically incompatible with repeated layers.
- **Conclusion:** drop the all-layer sandwich norm line and move on.

## Experiment 3: Repeated-pass LoRA ablations on 8xH100

- **Date**: 2026-03-28
- **Hardware**: 8xH100 80GB
- **TTT**: not run on these ablations (`TTT_ENABLED=0`)

| Setup | Log | Final post-EMA BPB | Final int6 sliding-window BPB |
|--------|-----|--------------------|--------------------------------|
| Rank 4, shared scalar AdamW group | [logs/c2936af5-5831-466c-ba1e-d103055f7567.txt](/root/parameter-golf/logs/c2936af5-5831-466c-ba1e-d103055f7567.txt) | 1.1364 | 1.12070655 |
| Rank 4, separate AdamW, `lr=1e-3` | [logs/37531638-306c-47ea-a4a2-111a6ca31bf3.txt](/root/parameter-golf/logs/37531638-306c-47ea-a4a2-111a6ca31bf3.txt) | 1.1358 | not completed |
| Rank 4, separate AdamW, `lr=3e-4` | [logs/b8b7b63e-fecc-40fb-95fe-831e1bf447f5.txt](/root/parameter-golf/logs/b8b7b63e-fecc-40fb-95fe-831e1bf447f5.txt) | 1.1354 | 1.11925740 |

### Notes
- These LoRA runs used repeated-pass-only adapters: first pass uses base weights, second occurrence of recurrent layers 4 and 5 uses `W + ΔW`.
- The `lr=3e-4` run is the best LoRA result so far and slightly beats the no-LoRA 8xH100 sliding-window baseline (`1.11925740` vs `1.11996599`).
- The gain is small (~0.0007 BPB), so this line looks mildly positive rather than strongly promising.
