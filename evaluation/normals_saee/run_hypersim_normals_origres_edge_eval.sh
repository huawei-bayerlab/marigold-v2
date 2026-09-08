#!/usr/bin/env bash
# Native-resolution Hypersim normals inference + angular and soft-edge (SAEE) metrics.
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
ASSETS_DIR="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
CHECKPOINT="${1:-${ASSETS_DIR}/checkpoints/Marigold-V2/normals}"
MAX_SAMPLES="${2:-}"
NUM_GPUS="${3:-1}"
SPLIT="${SPLIT:-test}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-${ASSETS_DIR}/datasets/marigold_train_normals}"
DATASET_NAME="${DATASET_NAME:-hypersim_test_normals_origres_qwen}"
EDGE_ANGLE_THRESHOLD_DEG="${EDGE_ANGLE_THRESHOLD_DEG:-5.0}"
EDGE_DILATE_RADIUS="${EDGE_DILATE_RADIUS:-0}"
SOFT_EDGE_KERNEL_SIZES="${SOFT_EDGE_KERNEL_SIZES:-1,3,5,7}"
NUM_WORKERS="${NUM_WORKERS:-0}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-${ASSETS_DIR}/checkpoints/Qwen-Image-Edit-2509}"

RUN_NAME="normals_saee_$(date +%y%m%dT%H%M%S)${RUN_SUFFIX:+_${RUN_SUFFIX}}"
RUN_ROOT="${OUTPUT_BASE:-${REPO_ROOT}/output/eval_runs}/${RUN_NAME}"
PRED_DIR="${RUN_ROOT}/predictions"
METRICS_DIR="${RUN_ROOT}/metrics"
mkdir -p "${PRED_DIR}" "${METRICS_DIR}"
MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  MAX_SAMPLES_ARGS=(--max_samples "${MAX_SAMPLES}")
fi

cd "${REPO_ROOT}"
echo "[1/2] Inference on Hypersim ${SPLIT} normals at native resolution"
INFER_COMMAND=(
  python -u evaluation/normals_saee/evaluate_hypersim_normals_test_origres.py
  --checkpoint "${CHECKPOINT}"
  --output_dir "${PRED_DIR}"
  --dataset_base_dir "${DATASET_BASE_DIR}"
  --dataset_name "${DATASET_NAME}"
  --qwen_checkpoint "${QWEN_CHECKPOINT}"
  --split "${SPLIT}"
  --num_workers "${NUM_WORKERS}"
  --num_gpus "${NUM_GPUS}"
)
"${INFER_COMMAND[@]}" "${MAX_SAMPLES_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/inference.log"

echo "[2/2] Angular and soft-edge metrics"
METRICS_COMMAND=(
  python -u evaluation/normals_saee/compute_hypersim_normals_edge_metrics.py
  --prediction_dir "${PRED_DIR}"
  --dataset_name "${DATASET_NAME}"
  --base_data_dir "${DATASET_BASE_DIR}"
  --split "${SPLIT}"
  --output_dir "${METRICS_DIR}"
  --edge_angle_threshold_deg "${EDGE_ANGLE_THRESHOLD_DEG}"
  --edge_dilate_radius "${EDGE_DILATE_RADIUS}"
  --soft_edge_kernel_sizes "${SOFT_EDGE_KERNEL_SIZES}"
)
"${METRICS_COMMAND[@]}" "${MAX_SAMPLES_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/metrics.log"
echo "Predictions: ${PRED_DIR}"
echo "Metrics:     ${METRICS_DIR}/metrics.json"
