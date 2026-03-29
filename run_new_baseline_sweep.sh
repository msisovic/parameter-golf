#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p sweep_logs

seed=314
bigram_vocab=3072
bigram_dim=112

for value in 2500 3000 3500; do
  run_id="newbase_s314_eq${value}"
  log_file="sweep_logs/${run_id}.log"
  echo "=== ${run_id} ===" | tee "${log_file}"
  env \
    RUN_ID="${run_id}" \
    SEED="${seed}" \
    BIGRAM_VOCAB_SIZE="${bigram_vocab}" \
    BIGRAM_DIM="${bigram_dim}" \
    WARMDOWN_ITERS="${value}" \
    RECUR_START_STEP="${value}" \
    TTT_ENABLED=0 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee -a "${log_file}"
done
