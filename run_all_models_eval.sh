#!/bin/bash
# =============================================================================
# run_all_models_eval.sh
#
# Run the UFO evaluation pipeline over a list of generated-image models, one
# after another. Each model's run uses the same eval config and VLM judge, and
# is written to its own log file.
#
# Usage:
#   chmod +x run_all_models_eval.sh
#   ./run_all_models_eval.sh
#
# Run in the background and capture everything:
#   nohup ./run_all_models_eval.sh > run_all_models.log 2>&1 &
#
# Override defaults without editing the file:
#   VLM=gemini CONFIG=config/eval_config.yaml ./run_all_models_eval.sh
#   MODELS="bagel uno" ./run_all_models_eval.sh
# =============================================================================

set -u  # treat unset variables as errors

# --- Resolve project root (directory of this script) so it runs from anywhere -
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Configuration (override via environment variables) ----------------------
PY_SCRIPT="scripts/run_eval_score.py"
CONFIG="${CONFIG:-config/eval_config.yaml}"   # falls back to the example below if missing
VLM="${VLM:-gemini}"                          # VLM judge: gpt | gemini | claude | doubao | qwen
LOG_DIR="${LOG_DIR:-logs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                   # e.g. EXTRA_ARGS="--debug"

# Generated-image models to evaluate (matches the directory names under
# <generated_root>/<model_name>/...). Override with: MODELS="a b c"
if [ -n "${MODELS:-}" ]; then
    # shellcheck disable=SC2206
    MODELS=(${MODELS})
else
    MODELS=(
        "doubao"
        "qwen"
        "nanobanana"
        "uno"
        "omnigen2"
        "bagel"
    )
fi

# --- Fall back to the example config if the user one is not present -----------
if [ ! -f "${CONFIG}" ]; then
    if [ -f "config/eval_config.example.yaml" ]; then
        echo "[WARN] ${CONFIG} not found, falling back to config/eval_config.example.yaml"
        CONFIG="config/eval_config.example.yaml"
    else
        echo "[ERROR] Eval config not found: ${CONFIG}"
        exit 1
    fi
fi

mkdir -p "${LOG_DIR}"

echo "=============================================================="
echo " UFO batch evaluation"
echo "   config : ${CONFIG}"
echo "   vlm    : ${VLM}"
echo "   models : ${MODELS[*]}"
echo "   logs   : ${LOG_DIR}/"
echo "=============================================================="

OK_MODELS=()
FAILED_MODELS=()

for MODEL in "${MODELS[@]}"; do
    TS="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo ">>> [${TS}] Running model_name=${MODEL}"

    LOG="${LOG_DIR}/eval_${VLM}_${MODEL}.log"

    # Run sequentially and wait for completion before moving to the next model.
    # shellcheck disable=SC2086
    python "${PY_SCRIPT}" \
        --config "${CONFIG}" \
        --vlm "${VLM}" \
        --model_name "${MODEL}" \
        ${EXTRA_ARGS} \
        > "${LOG}" 2>&1

    STATUS=$?
    if [ ${STATUS} -eq 0 ]; then
        echo "<<< Finished ${MODEL} (ok)  -> ${LOG}"
        OK_MODELS+=("${MODEL}")
    else
        echo "<<< FAILED  ${MODEL} (exit ${STATUS}) -> ${LOG}"
        FAILED_MODELS+=("${MODEL}")
        # Keep going with the remaining models instead of aborting the batch.
    fi
done

echo ""
echo "=============================================================="
echo " All models done."
echo "   succeeded: ${OK_MODELS[*]:-none}"
echo "   failed   : ${FAILED_MODELS[*]:-none}"
echo "=============================================================="

# Non-zero exit if any model failed, so callers/CI can detect it.
if [ ${#FAILED_MODELS[@]} -ne 0 ]; then
    exit 1
fi
