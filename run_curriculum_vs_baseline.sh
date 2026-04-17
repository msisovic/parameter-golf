#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

mkdir -p runs/curriculum runs/baseline logs

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo; echo "===== [$(timestamp)] $* ====="; echo; }

banner "STEP 1/4: Curriculum + YaRN training (sliding eval at 4096/64)"
TTT_ENABLED=0 \
TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 SEQ_LEN_BUMP_FRAC=0.8 \
ROPE_YARN=1 \
SLIDING_WINDOW_ENABLED=1 EVAL_SEQ_LEN=4096 EVAL_STRIDE=64 \
GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
ARTIFACT_DIR=./runs/curriculum RUN_ID=curriculum_v3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee logs/curriculum_v3.train.log

banner "STEP 2/4: Curriculum analysis (sliding 4096/64, doc-position buckets)"
EVAL_SEQ_LEN=4096 EVAL_STRIDE=64 \
ROPE_YARN=1 TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 \
ARTIFACT_DIR=./runs/curriculum RUN_ID=curriculum_stride64_s4096 \
python analysis/deeper_doc_eval.py \
  2>&1 | tee logs/curriculum_stride64_s4096.analysis.log

banner "STEP 3/4: Baseline training (sliding eval at 2048/64)"
TTT_ENABLED=0 \
ROPE_YARN=0 \
SLIDING_WINDOW_ENABLED=1 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
GPTQ_RESERVE_SECONDS=13 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
ARTIFACT_DIR=./runs/baseline RUN_ID=baseline_v3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee logs/baseline_v3.train.log

banner "STEP 4/4: Baseline analysis (sliding 2048/64, doc-position buckets)"
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ROPE_YARN=0 \
ARTIFACT_DIR=./runs/baseline RUN_ID=baseline_stride64_s2048 \
python analysis/deeper_doc_eval.py \
  2>&1 | tee logs/baseline_stride64_s2048.analysis.log

banner "ALL DONE"
echo "artifacts:"
echo "  ./runs/curriculum/    (curriculum+YaRN checkpoint)"
echo "  ./runs/baseline/      (baseline checkpoint)"
echo "analysis outputs:"
echo "  ./analysis/curriculum_stride64_s4096/"
echo "  ./analysis/baseline_stride64_s2048/"
echo "training + analysis logs:"
echo "  ./logs/{curriculum_v3,baseline_v3}.train.log"
echo "  ./logs/{curriculum_stride64_s4096,baseline_stride64_s2048}.analysis.log"
