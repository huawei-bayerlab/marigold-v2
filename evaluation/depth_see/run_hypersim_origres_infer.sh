#!/usr/bin/env bash
# Native-resolution Hypersim inference only (no metrics).
set -euo pipefail

usage() {
  echo "Usage: $0 [CHECKPOINT] [MAX_SAMPLES] [NUM_GPUS]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# > 3 )); then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_ROOT="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
if [[ "${ASSETS_ROOT}" != /* ]]; then
  ASSETS_ROOT="${REPO_ROOT}/${ASSETS_ROOT}"
fi

CHECKPOINT="${1:-${ASSETS_ROOT}/checkpoints/Marigold-V2/depth/Log-stage2}"
MAX_SAMPLES="${2:-100}"
NUM_GPUS="${3:-1}"
SPLIT="${SPLIT:-test}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-${ASSETS_ROOT}/datasets/marigold_train}"
DATASET_KEY="${DATASET_KEY:-hypersim_test_origres_rel_log_depth_qwen}"

RUN_ROOT="${REPO_ROOT}/output/eval_runs/depth_see_infer_$(date +%y%m%dT%H%M%S)"
PRED_DIR="${PRED_DIR:-${RUN_ROOT}/predictions}"
mkdir -p "${PRED_DIR}"

cd "${REPO_ROOT}"
INFER_COMMAND=(
  python -u evaluation/depth_see/evaluate_hypersim_test_origres.py
  --checkpoint "${CHECKPOINT}"
  --output_dir "${PRED_DIR}"
  --max_samples "${MAX_SAMPLES}"
  --num_gpus "${NUM_GPUS}"
  --split "${SPLIT}"
  --dataset_name "${DATASET_KEY}"
  --dataset_base_dir "${DATASET_BASE_DIR}"
)
"${INFER_COMMAND[@]}" 2>&1 | tee "${RUN_ROOT}/inference.log"

echo "Predictions:    ${PRED_DIR}"
echo "Dataset config: ${PRED_DIR}/generated_eval_hypersim_test_origres.yaml"
