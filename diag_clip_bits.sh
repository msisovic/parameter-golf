#!/usr/bin/env bash
# Sweep MATRIX_BITS × MATRIX_CLIP_SIGMAS on a fixed post-EMA checkpoint.
# Goal: find a (bits, clip) point that fits in ~16 MB AND improves post-quant bpb.
#
# Based on PR #1394's SDClip insight: compressed size ≈ b − log₂(k) + const,
# so (7, 25.7) ≈ (6, 12.85) in size with finer middle-bin resolution. The clip
# sweep at int6 tests whether our curriculum-trained weight distribution has a
# different optimal k than the baseline's.
#
# Usage: ./diag_clip_bits.sh <path/to/final_model.pt>
set -euo pipefail

CKPT=${1:?usage: $0 <final_model.pt>}
CKPT_ABS=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
STAMP=$(date +%Y%m%d_%H%M%S)

run_variant () {
    local name=$1; shift
    local dir=./runs/diag_clipbits_${STAMP}/${name}
    mkdir -p "$dir" logs
    echo
    echo "===== [$(date +%H:%M:%S)] clipbits/${name} ====="
    env TTT_ENABLED=0 ROPE_YARN=1 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
        GPTQ_RESERVE_SECONDS=30 SLIDING_WINDOW_ENABLED=1 \
        EVAL_SEQ_LEN=4096 EVAL_STRIDE=64 \
        TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 \
        ARTIFACT_DIR="$dir" RUN_ID="clipbits_${name}" \
        EVAL_ONLY_PATH="$CKPT_ABS" "$@" \
        torchrun --standalone --nproc_per_node=8 train_gpt.py \
        2>&1 | tee "logs/diag_clipbits_${STAMP}_${name}.log"
}

# ----- Experiment A: PR #1394's equal-size trade -----
# (bits=7, clip=25.7) should compress to ~16 MB, same as (bits=6, clip=12.85).
run_variant A_int7_k25p7 MATRIX_BITS=7 MATRIX_CLIP_SIGMAS=25.7

# ----- Experiment B: clip_sigmas sweep at int6 -----
# Tests whether our weight distribution has a different SDClip optimum.
run_variant B_int6_k8    MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=8
run_variant B_int6_k10   MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=10
run_variant B_int6_k12p85 MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85
run_variant B_int6_k15   MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=15
run_variant B_int6_k20   MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=20

echo
echo "===== summary (diag_clipbits_${STAMP}) ====="
printf "%-22s %-12s %-22s %-22s %-22s\n" variant blob_bytes pre_quant_sliding quant_sliding drop
for v in A_int7_k25p7 B_int6_k8 B_int6_k10 B_int6_k12p85 B_int6_k15 B_int6_k20; do
    log="logs/diag_clipbits_${STAMP}_${v}.log"
    [[ -f "$log" ]] || continue
    blob=$(grep -oE "Serialized model quantized\+[a-z]+: [0-9]+ bytes" "$log" | awk '{print $4}')
    pre=$(grep "diagnostic pre-quantization post-ema sliding_window" "$log" | grep -oE "val_bpb:[0-9.]+" | head -1 | cut -d: -f2)
    post=$(grep "diagnostic quantized_sliding_window" "$log" | grep -oE "val_bpb:[0-9.]+" | head -1 | cut -d: -f2)
    drop="n/a"
    if [[ -n "$pre" && -n "$post" ]]; then
        drop=$(python3 -c "print(f'{float('$post') - float('$pre'):.5f}')")
    fi
    printf "%-22s %-12s %-22s %-22s %-22s\n" "$v" "${blob:-n/a}" "${pre:-n/a}" "${post:-n/a}" "$drop"
done
