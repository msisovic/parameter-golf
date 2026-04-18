#!/usr/bin/env bash
# Test hypotheses for why TTT recovery dropped from 94% (baseline) to 40% (ours).
#
# Primary question: is it the context-length change (2048 -> 4096), or something
# intrinsic to the curriculum-trained weights?
#
# H1 test (ctx2048): if recovery% jumps to ~90% when TTT runs at 2048 on OUR
# checkpoint, it's context-length mismatch. Fix: tune TTT hyperparameters for
# 4096. We then sweep TTT_LORA_LR / TTT_GRAD_STEPS / TTT_CHUNK_SIZE at 4096.
# If ctx2048 stays at ~40%, it's weight-intrinsic and harder.
#
# The blob is symlinked into each variant's artifact dir so serialize is skipped
# and every variant runs TTT on IDENTICAL quantized weights. That way any
# difference in the result is purely from the TTT knob being swept, not from
# quant noise.
#
# Usage: ./diag_ttt_recovery.sh <final_model.pt> <existing_blob.int6.ptz>
#
# Runtime: ~8 min for the 2048 variant, ~20 min each for the 4096 variants.
# Set VAL_DOC_FRACTION in-line (e.g. 0.3) to cut runtime at the cost of noise.
set -euo pipefail

CKPT=${1:?usage: $0 <final_model.pt> <existing_blob.int6.ptz>}
BLOB=${2:?usage: $0 <final_model.pt> <existing_blob.int6.ptz>}

CKPT_ABS=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
BLOB_ABS=$(cd "$(dirname "$BLOB")" && pwd)/$(basename "$BLOB")
STAMP=$(date +%Y%m%d_%H%M%S)

run_variant () {
    local name=$1; shift
    local dir=./runs/diag_ttt_${STAMP}/${name}
    mkdir -p "$dir" logs
    ln -sf "$BLOB_ABS" "$dir/final_model.int6.ptz"
    echo
    echo "===== [$(date +%H:%M:%S)] ttt/${name} ====="
    env TTT_ENABLED=1 ROPE_YARN=1 FP8_LM_HEAD=1 FUSED_SOFTCAP_CE=1 \
        GPTQ_RESERVE_SECONDS=30 SLIDING_WINDOW_ENABLED=0 \
        EVAL_STRIDE=64 \
        TRAIN_SEQ_LEN=2048 TRAIN_SEQ_LEN_END=4096 \
        TTT_BATCH_SIZE=32 \
        ARTIFACT_DIR="$dir" RUN_ID="ttt_${name}" \
        EVAL_ONLY_PATH="$CKPT_ABS" "$@" \
        torchrun --standalone --nproc_per_node=8 train_gpt.py \
        2>&1 | tee "logs/diag_ttt_${STAMP}_${name}.log"
}

# ----- H1 primary (already confirmed: 112% recovery at 2048) -----
# Commented out to save ~8 min on reruns. Uncomment to reproduce.
# run_variant ctx2048 EVAL_SEQ_LEN=2048 TTT_EVAL_SEQ_LEN=2048

# Reference: reproduces the 40% recovery at 4096 with defaults. Done: 59% recovery.
# run_variant ctx4096_default EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 TTT_LORA_LR=0.0001 TTT_GRAD_STEPS=1 TTT_CHUNK_SIZE=32

# ----- TTT hparam sweep at 4096 (only meaningful if H1 confirmed) -----
# Weaker-gradient hypothesis DISPROVEN: bigger LR destabilizes (gradient directions
# are noisy at 4096, bigger step averages back toward post-quant). Skipping LR sweep.
# run_variant ctx4096_lr3e-4 EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 TTT_LORA_LR=0.0003 TTT_GRAD_STEPS=1 TTT_CHUNK_SIZE=32
# run_variant ctx4096_lr1e-3 EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 TTT_LORA_LR=0.001  TTT_GRAD_STEPS=1 TTT_CHUNK_SIZE=32

# More grad steps per chunk -> more adaptation, same LR.
# Folded into the retraining run (post-reboot) so we don't duplicate work.
# run_variant ctx4096_gs2 EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 TTT_LORA_LR=0.0001 TTT_GRAD_STEPS=2 TTT_CHUNK_SIZE=32

# Bigger chunk -> fewer but stronger (less noisy) gradient signals per doc.
# Most diagnostic for the "noisy gradient" hypothesis.
run_variant ctx4096_chunk128 EVAL_SEQ_LEN=4096 TTT_EVAL_SEQ_LEN=4096 TTT_LORA_LR=0.0001 TTT_GRAD_STEPS=1 TTT_CHUNK_SIZE=128

echo
echo "===== summary (diag_ttt_${STAMP}) ====="
printf "%-22s %-12s %-12s %-12s %-12s %-10s\n" variant pre_quant post_quant ttt recovered rec_pct
for v in ctx4096_chunk128; do
    log="logs/diag_ttt_${STAMP}_${v}.log"
    [[ -f "$log" ]] || continue
    pre=$(grep "diagnostic pre-quantization post-ema val_loss" "$log" | head -1 | grep -oE "val_loss:[0-9.]+" | cut -d: -f2)
    quant=$(grep "diagnostic quantized val_loss" "$log" | head -1 | grep -oE "val_loss:[0-9.]+" | cut -d: -f2)
    ttt=$(grep "quantized_ttt_lora val_loss" "$log" | head -1 | grep -oE "val_loss:[0-9.]+" | cut -d: -f2)
    rec="n/a"; pct="n/a"
    if [[ -n "$pre" && -n "$quant" && -n "$ttt" ]]; then
        rec=$(python3 -c "print(f'{float('\"'\"'$quant'\"'\"') - float('\"'\"'$ttt'\"'\"'):.5f}')")
        pct=$(python3 -c "drop=float('\"'\"'$quant'\"'\"')-float('\"'\"'$pre'\"'\"'); rec=float('\"'\"'$quant'\"'\"')-float('\"'\"'$ttt'\"'\"'); print(f'{100*rec/drop:.1f}%' if drop > 1e-9 else 'n/a')")
    fi
    printf "%-22s %-12s %-12s %-12s %-12s %-10s\n" "$v" "${pre:-n/a}" "${quant:-n/a}" "${ttt:-n/a}" "$rec" "$pct"
done
