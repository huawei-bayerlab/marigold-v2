#!/usr/bin/env bash
# Native-resolution Hypersim test inference + Soft Edge Error (SEE_k).
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
MAX_SAMPLES="${2:-}"
NUM_GPUS="${3:-1}"
SPLIT="${SPLIT:-test}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-${ASSETS_ROOT}/datasets/marigold_train}"
DATASET_KEY="hypersim_test_origres_rel_log_depth_qwen"
PREDICTION_SPACE="${PREDICTION_SPACE:-log}"
SEE_KERNEL_SIZES="${SEE_KERNEL_SIZES:-1,3,5,7}"

STAMP="$(date +%y%m%dT%H%M%S)"
RUN_ROOT="${REPO_ROOT}/output/eval_runs/depth_see_${STAMP}${RUN_SUFFIX:+_${RUN_SUFFIX}}"
PRED_DIR="${RUN_ROOT}/predictions"
METRICS_DIR="${RUN_ROOT}/metrics"
mkdir -p "${PRED_DIR}" "${METRICS_DIR}"

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARGS=(--max_samples "${MAX_SAMPLES}")
fi

cd "${REPO_ROOT}"
echo "[1/2] Inference on Hypersim ${SPLIT} at native resolution"
INFER_COMMAND=(
  python -u evaluation/depth_see/evaluate_hypersim_test_origres.py
  --checkpoint "${CHECKPOINT}"
  --output_dir "${PRED_DIR}"
  --num_gpus "${NUM_GPUS}"
  --split "${SPLIT}"
  --dataset_name "${DATASET_KEY}"
  --dataset_base_dir "${DATASET_BASE_DIR}"
)
"${INFER_COMMAND[@]}" "${MAX_SAMPLES_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/inference.log"

echo "[2/2] SEE metrics"
METRICS_COMMAND=(
  python -u evaluation/depth_see/compute_hypersim_edge_metrics.py
  --dataset_config "${PRED_DIR}/generated_eval_hypersim_test_origres.yaml"
  --dataset_key "${DATASET_KEY}"
  --prediction_root "${PRED_DIR}"
  --dataset_base_dir "${DATASET_BASE_DIR}"
  --output_dir "${METRICS_DIR}"
  --prediction_space "${PREDICTION_SPACE}"
  --see_kernel_sizes "${SEE_KERNEL_SIZES}"
)
"${METRICS_COMMAND[@]}" "${MAX_SAMPLES_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/metrics.log"

echo "Predictions: ${PRED_DIR}"
echo "Metrics:     ${METRICS_DIR}/hypersim_edge_metrics_summary.json"
