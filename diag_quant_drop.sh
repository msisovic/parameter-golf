#!/usr/bin/env bash
# Diagnostic sweep for the "curriculum quant drop" mystery.
# Runs eval-only on an existing post-EMA checkpoint under four configs to
# isolate whether the drop is driven by YaRN, bit-precision, or calibration.
#
# Usage: ./diag_quant_drop.sh <path/to/final_model.pt> [<path/to/final_model.int6.ptz>]
#   arg1 (required): post-EMA checkpoint (what base_model.load_state_dict consumes)
#   arg2 (optional): existing int6 blob, reused in the "samequant" variants so
#                    the quant drop there is attributed only to the eval knob
set -euo pipefail

CKPT=${1:?usage: $0 <final_model.pt> [existing_blob.int6.ptz]}
BLOB=${2:-}

CKPT_ABS=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
BLOB_ABS=""
if [[ -n "$BLOB" ]]; then
    BLOB_ABS=$(cd "$(dirname "$BLOB")" && pwd)/$(basename "$BLOB")
fi
STAMP=$(date +%Y%m%d_%H%M%S)

run_variant () {
    local name=$1; shift
    local reuse_blob=$1; shift
    local dir=./runs/diag_${STAMP}/${name}
    mkdir -p "$dir" logs
    if [[ "$reuse_blob" == "yes" && -n "$BLOB_ABS" ]]; then
        ln -sf "$BLOB_ABS" "$dir/final_model.int6.ptz"
    fi
    echo
    echo "===== [$(date +%H:%M:%S)] diag/${name} ====="
    env TTT_ENABLED=0 ROPE_YARN=1 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
        GPTQ_RESERVE_SECONDS=30 SLIDING_WINDOW_ENABLED=1 EVAL_STRIDE=64 \
        TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 \
        ARTIFACT_DIR="$dir" RUN_ID="diag_${name}" \
        EVAL_ONLY_PATH="$CKPT_ABS" "$@" \
        torchrun --standalone --nproc_per_node=8 train_gpt.py \
        2>&1 | tee "logs/diag_${STAMP}_${name}.log"
}

# 1) eval at 2048 (YaRN inactive), reuse existing int6 blob.
#    -> isolates: does the quant drop shrink when YaRN is off?
run_variant eval2048_samequant yes EVAL_SEQ_LEN=2048

# 2) eval at 4096 (YaRN active), reuse existing int6 blob.
#    -> eval-only reference for the same-blob 4096 number.
run_variant eval4096_samequant yes EVAL_SEQ_LEN=4096

# 3) re-quant at int8, eval at 4096.
#    -> isolates: does bumping precision from int6 to int8 close the gap?
run_variant eval4096_int8 no EVAL_SEQ_LEN=4096 MATRIX_BITS=8

# 4) re-quant with 256 calibration batches, int6, eval at 4096.
#    -> isolates: does more calibration data close the gap?
run_variant eval4096_calib256 no EVAL_SEQ_LEN=4096 GPTQ_CALIBRATION_BATCHES=256

echo
echo "===== summary (diag_${STAMP}) ====="
for v in eval2048_samequant eval4096_samequant eval4096_int8 eval4096_calib256; do
    echo "--- $v ---"
    grep -E "diagnostic (pre-quantization post-ema|quantized)" "logs/diag_${STAMP}_${v}.log" || echo "  (no matching lines)"
done
