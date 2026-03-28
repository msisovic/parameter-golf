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
