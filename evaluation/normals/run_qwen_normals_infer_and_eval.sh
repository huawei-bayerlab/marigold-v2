#!/usr/bin/env bash
# Surface-normal inference + angular metrics on the Marigold V1 normals benchmarks.
set -euo pipefail

usage() {
  echo "Usage: $0 [CHECKPOINT] [MAX_SAMPLES] [NUM_GPUS] [BASE_DATA_DIR] [EMBED_DIR] [DATASETS]"
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
ASSETS_DIR="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
CHECKPOINT="${1:-${ASSETS_DIR}/checkpoints/Marigold-V2/normals}"
MAX_SAMPLES="${2:-}"
NUM_GPUS="${3:-1}"
BASE_DATA_DIR="${4:-${MARIGOLD_NORMALS_EVAL_DATASET_ROOT:-${ASSETS_DIR}/datasets/marigold_normals_eval}}"
EMBED_DIR="${5:-${ASSETS_DIR}/checkpoints/Marigold-V2/qwen_text_embeddings}"
read -r -a DATASETS <<< "${6:-ibims nyuv2 scannet sintel}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-${ASSETS_DIR}/checkpoints/Qwen-Image-Edit-2509}"

if [[ ! -f "${CHECKPOINT}/trainables.safetensors" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}/trainables.safetensors" >&2
  exit 1
fi
if [[ ! -d "${BASE_DATA_DIR}" ]]; then
  echo "Dataset directory not found: ${BASE_DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${QWEN_CHECKPOINT}" ]]; then
  echo "Qwen checkpoint not found: ${QWEN_CHECKPOINT}" >&2
  exit 1
fi

RUN_ROOT="${OUTPUT_BASE:-${REPO_ROOT}/output/eval_runs}/normals_$(date +%y%m%dT%H%M%S)_$$"
PRED_ROOT="${RUN_ROOT}/predictions"
METRICS_ROOT="${RUN_ROOT}/metrics"
CONFIG_ROOT="${RUN_ROOT}/configs"
mkdir -p "${PRED_ROOT}" "${METRICS_ROOT}" "${CONFIG_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${RUN_ROOT}"

cd "${REPO_ROOT}"
for dataset in "${DATASETS[@]}"; do
  case "${dataset}" in
    ibims) disp_name="ibims_normals_test" ;;
    nyuv2) disp_name="nyu_normals_test" ;;
    scannet) disp_name="scannet_normals_test" ;;
    sintel) disp_name="sintel_normals_test" ;;
    *) echo "Unknown dataset '${dataset}'" >&2; exit 1 ;;
  esac
  config_path="${CONFIG_ROOT}/${dataset}.yaml"

  echo "[inference] ${dataset}"
  build_cmd=(
    python -u evaluation/normals/build_qwen_normals_eval_config.py
    --dataset "${dataset}"
    --base_data_dir "${BASE_DATA_DIR}"
    --output_dir "${PRED_ROOT}"
    --embed_dir "${EMBED_DIR}"
    --qwen_checkpoint "${QWEN_CHECKPOINT}"
    --config_out "${config_path}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    build_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  "${build_cmd[@]}"

  if (( NUM_GPUS > 1 )); then
    infer_cmd=(
      accelerate launch --num_processes "${NUM_GPUS}"
      -m evaluation.depth.evaluate_pipeline
    )
  else
    infer_cmd=(python -u -m evaluation.depth.evaluate_pipeline)
  fi
  "${infer_cmd[@]}" \
    --config "${config_path}" \
    --output_dir "${PRED_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    2>&1 | tee -a "${RUN_ROOT}/inference.log"

  echo "[evaluation] ${dataset}"
  eval_cmd=(
    python -u evaluation/normals/eval_normals.py
    --prediction_dir "${PRED_ROOT}/${disp_name}"
    --dataset_config "evaluation/normals/config/${dataset}_test.yaml"
    --base_data_dir "${BASE_DATA_DIR}"
    --output_dir "${METRICS_ROOT}/${dataset}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    eval_cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  "${eval_cmd[@]}" 2>&1 | tee -a "${RUN_ROOT}/eval.log"
done

python -u evaluation/normals/summarize_normals_eval.py "${METRICS_ROOT}" --out "${METRICS_ROOT}/summary.json"
echo "Predictions: ${PRED_ROOT}"
echo "Metrics:     ${METRICS_ROOT}/summary.json"
