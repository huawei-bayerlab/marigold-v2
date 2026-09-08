#!/usr/bin/env bash
# PSNR/SSIM/LPIPS for an existing Hypersim albedo prediction directory.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash evaluation/albedo/run_qwen_albedo_eval.sh \
    PREDICTION_DIR METRICS_DIR [MAX_SAMPLES] [DATASET_BASE_DIR]

Evaluate Hypersim albedo predictions. MAX_SAMPLES is an optional positive
limit; DATASET_BASE_DIR defaults to
$DEPTH_ASSETS_DIR/datasets/marigold_train_albedo, or ./assets/datasets/
marigold_train_albedo when unset.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if (( $# < 2 || $# > 4 )); then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_DIR="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
PREDICTION_DIR="$1"
METRICS_DIR="$2"
MAX_SAMPLES="${3:-}"
DATASET_BASE_DIR="${4:-${ASSETS_DIR}/datasets/marigold_train_albedo}"

if [[ ! -d "${PREDICTION_DIR}" ]]; then
  echo "Prediction directory not found: ${PREDICTION_DIR}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_BASE_DIR}" ]]; then
  echo "Dataset directory not found: ${DATASET_BASE_DIR}" >&2
  exit 1
fi

if [[ -n "${MAX_SAMPLES}" && ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer: ${MAX_SAMPLES}" >&2
  exit 2
fi

EVAL_COMMAND=(
  python -u evaluation/albedo/eval_qwen_albedo_hypersim.py
  --prediction_dir "${PREDICTION_DIR}"
  --output_dir "${METRICS_DIR}"
  --base_data_dir "${DATASET_BASE_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  EVAL_COMMAND+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${REPO_ROOT}"
"${EVAL_COMMAND[@]}"

echo "Metrics: ${METRICS_DIR}/metrics.json"
