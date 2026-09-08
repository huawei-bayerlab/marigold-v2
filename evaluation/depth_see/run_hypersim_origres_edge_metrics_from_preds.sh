#!/usr/bin/env bash
# SEE metrics for an existing Hypersim prediction directory.
set -euo pipefail

usage() {
  echo "Usage: $0 PRED_DIR [MAX_SAMPLES] [DATASET_CONFIG] [METRICS_DIR]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 1 || $# > 4 )); then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_ROOT="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
if [[ "${ASSETS_ROOT}" != /* ]]; then
  ASSETS_ROOT="${REPO_ROOT}/${ASSETS_ROOT}"
fi

PRED_DIR="$1"
MAX_SAMPLES="${2:-100}"
DATASET_CONFIG="${3:-${PRED_DIR}/generated_eval_hypersim_test_origres.yaml}"
METRICS_DIR="${4:-$(cd "${PRED_DIR}/.." && pwd)/metrics}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-${ASSETS_ROOT}/datasets/marigold_train}"
DATASET_KEY="${DATASET_KEY:-hypersim_test_origres_rel_log_depth_qwen}"
PREDICTION_SPACE="${PREDICTION_SPACE:-log}"
SEE_KERNEL_SIZES="${SEE_KERNEL_SIZES:-1,3,5,7}"

if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "Dataset config not found: ${DATASET_CONFIG}" >&2
  exit 1
fi
mkdir -p "${METRICS_DIR}"
cd "${REPO_ROOT}"
METRICS_COMMAND=(
  python -u evaluation/depth_see/compute_hypersim_edge_metrics.py
  --dataset_config "${DATASET_CONFIG}"
  --dataset_key "${DATASET_KEY}"
  --prediction_root "${PRED_DIR}"
  --dataset_base_dir "${DATASET_BASE_DIR}"
  --output_dir "${METRICS_DIR}"
  --max_samples "${MAX_SAMPLES}"
  --prediction_space "${PREDICTION_SPACE}"
  --see_kernel_sizes "${SEE_KERNEL_SIZES}"
)
"${METRICS_COMMAND[@]}" 2>&1 | tee "${METRICS_DIR}/metrics.log"

echo "Metrics: ${METRICS_DIR}/hypersim_edge_metrics_summary.json"
