#!/usr/bin/env bash
# Score an existing prediction directory (steps 2-4 of run_infer_and_eval.sh).
set -euo pipefail

usage() {
  echo "Usage: $0 PREDICTION_ROOT [DATASETS] [USE_PPD_PROTOCOL] [DEPTH_EVAL_ROOT] [OUT_ROOT] [ALIGNMENT]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 1 || $# > 6 )); then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_ROOT="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
if [[ "${ASSETS_ROOT}" != /* ]]; then
  ASSETS_ROOT="${REPO_ROOT}/${ASSETS_ROOT}"
fi

PRED_ROOT="$(cd "$1" && pwd)"
DATASETS="${2:-all}"
USE_PPD_PROTOCOL="${3:-true}"
MARIGOLD_DEPTH_EVAL_ROOT="${4:-${ASSETS_ROOT}/datasets/marigold_depth_eval}"
OUT_ROOT="${5:-$(dirname "${PRED_ROOT}")/eval_from_preds_$(date +%y%m%dT%H%M%S)}"
ALIGNMENT_OVERRIDE="${6:-}"

case "$(echo "${USE_PPD_PROTOCOL}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) USE_PPD_PROTOCOL=true ;;
  0|false|no|n|off|"") USE_PPD_PROTOCOL=false ;;
  *)
    echo "Invalid USE_PPD_PROTOCOL: '${USE_PPD_PROTOCOL}'" >&2
    exit 1
    ;;
esac

if [[ "${USE_PPD_PROTOCOL}" == "true" ]]; then
  ALIGNMENT_MODE="ppd_ransac_log"
  RESIZE_PREDICTIONS="false"
  EVAL_EXTRA_ARGS=(--ppd_eval_protocol)
else
  ALIGNMENT_MODE="least_square_log"
  RESIZE_PREDICTIONS="true"
  EVAL_EXTRA_ARGS=()
fi
if [[ -n "${ALIGNMENT_OVERRIDE}" ]]; then
  ALIGNMENT_MODE="${ALIGNMENT_OVERRIDE}"
fi

TMP_DIR="${OUT_ROOT}/tmp_eval"
METRICS_DIR="${OUT_ROOT}/metrics"
mkdir -p "${TMP_DIR}" "${METRICS_DIR}"
cd "${REPO_ROOT}"

echo "[2/4] Subset configs matching the available predictions"
SUBSET_COMMAND=(
  python evaluation/depth/build_eval_subsets.py
  --repo_root "${REPO_ROOT}"
  --pred_root "${PRED_ROOT}"
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
  pred_dir="${PRED_ROOT}/${ds}_rel_depth_qwen_768"
  if [[ ! -d "${pred_dir}" ]]; then
    echo "Skipping ${ds}: ${pred_dir} missing"
    continue
  fi

  EVAL_COMMAND=(
    python -u evaluation/depth/eval.py
    --prediction_dir "${pred_dir}"
    --dataset_config "${TMP_DIR}/data_${ds}_subset.yaml"
    --base_data_dir "${MARIGOLD_DEPTH_EVAL_ROOT}"
    --output_dir "${METRICS_DIR}/${ds}"
    --alignment "${ALIGNMENT_MODE}"
  )
  "${EVAL_COMMAND[@]}" "${EVAL_EXTRA_ARGS[@]}" 2>&1 | tee -a "${OUT_ROOT}/eval.log"
done

echo "[4/4] Summary"
python -u evaluation/depth/summarize_eval_metrics.py "${METRICS_DIR}" --out "${METRICS_DIR}/summary.json"
echo "Metrics: ${METRICS_DIR}"
