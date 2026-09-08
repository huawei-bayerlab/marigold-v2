#!/usr/bin/env bash
# Depth inference + AbsRel/delta1 on the five zero-shot benchmarks.
set -euo pipefail

usage() {
  echo "Usage: $0 [CHECKPOINT] [MAX_SAMPLES] [NUM_GPUS] [EVAL_CONFIG] [DATASETS] [USE_PPD_PROTOCOL]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# > 6 )); then
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
EVAL_CONFIG="${4:-evaluation/config/inference_depth.yaml}"
DATASETS="${5:-all}"
USE_PPD_PROTOCOL="${6:-true}"

case "$(echo "${USE_PPD_PROTOCOL}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) USE_PPD_PROTOCOL=true ;;
  0|false|no|n|off|"") USE_PPD_PROTOCOL=false ;;
  *)
    echo "Invalid USE_PPD_PROTOCOL: '${USE_PPD_PROTOCOL}'" >&2
    exit 1
    ;;
esac

MARIGOLD_DEPTH_EVAL_ROOT="${MARIGOLD_DEPTH_EVAL_DATASET_ROOT:-${ASSETS_ROOT}/datasets/marigold_depth_eval}"
STAMP="$(date +%y%m%dT%H%M%S)"
RUN_ROOT="${REPO_ROOT}/output/eval_runs/depth_${STAMP}${RUN_SUFFIX:+_${RUN_SUFFIX}}"
PRED_DIR="${RUN_ROOT}/predictions"
METRICS_DIR="${RUN_ROOT}/metrics"
TMP_DIR="${RUN_ROOT}/tmp_eval"
mkdir -p "${PRED_DIR}" "${METRICS_DIR}" "${TMP_DIR}"

if [[ "${USE_PPD_PROTOCOL}" == "true" ]]; then
  ALIGNMENT_MODE="ppd_ransac_log"
  RESIZE_PREDICTIONS="false"
  EVAL_EXTRA_ARGS=(--ppd_eval_protocol)
else
  ALIGNMENT_MODE="least_square_log"
  RESIZE_PREDICTIONS="true"
  EVAL_EXTRA_ARGS=()
fi

cd "${REPO_ROOT}"
echo "[1/4] Inference"
INF_CMD=(python -u evaluation/depth/evaluate_relative_testsets.py
  --checkpoint "${CHECKPOINT}" --output_dir "${PRED_DIR}"
  --marigold_depth_eval_root "${MARIGOLD_DEPTH_EVAL_ROOT}"
  --num_gpus "${NUM_GPUS}" --eval_config "${EVAL_CONFIG}" --datasets "${DATASETS}")
if [[ -n "${MAX_SAMPLES}" ]]; then INF_CMD+=(--max_samples "${MAX_SAMPLES}"); fi
"${INF_CMD[@]}" 2>&1 | tee "${RUN_ROOT}/inference.log"

echo "[2/4] Subset configs matching the available predictions"
SUBSET_COMMAND=(
  python evaluation/depth/build_eval_subsets.py
  --repo_root "${REPO_ROOT}"
  --pred_root "${PRED_DIR}"
  --out_dir "${TMP_DIR}"
  --base_data_dir "${MARIGOLD_DEPTH_EVAL_ROOT}"
  --datasets "${DATASETS}"
  --resize_predictions "${RESIZE_PREDICTIONS}"
)
"${SUBSET_COMMAND[@]}"

echo "[3/4] Metrics with ${ALIGNMENT_MODE} alignment"
if [[ "${DATASETS}" == "all" ]]; then
  EVAL_DATASETS=(diode eth3d kitti nyuv2 scannet)
else
  IFS=',' read -r -a EVAL_DATASETS <<< "${DATASETS}"
fi
for ds in "${EVAL_DATASETS[@]}"; do
  ds="$(echo "${ds}" | xargs | tr '[:upper:]' '[:lower:]')"
  EVAL_COMMAND=(
    python -u evaluation/depth/eval.py
    --prediction_dir "${PRED_DIR}/${ds}_rel_depth_qwen_768"
    --dataset_config "${TMP_DIR}/data_${ds}_subset.yaml"
    --base_data_dir "${MARIGOLD_DEPTH_EVAL_ROOT}"
    --output_dir "${METRICS_DIR}/${ds}"
    --alignment "${ALIGNMENT_MODE}"
  )
  "${EVAL_COMMAND[@]}" "${EVAL_EXTRA_ARGS[@]}" 2>&1 | tee -a "${RUN_ROOT}/eval.log"
done

echo "[4/4] Summary"
python -u evaluation/depth/summarize_eval_metrics.py "${METRICS_DIR}" --out "${METRICS_DIR}/summary.json"
echo "Predictions: ${PRED_DIR}"
echo "Metrics:     ${METRICS_DIR}"
