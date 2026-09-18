#!/usr/bin/env bash
# LayeredDepth-Syn layer-8 inference and aligned log-depth evaluation.
set -euo pipefail

usage() {
  echo "Usage: $0 [MAX_SAMPLES] [NUM_GPUS]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# > 2 )); then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_ROOT="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
if [[ "${ASSETS_ROOT}" != /* ]]; then
  ASSETS_ROOT="${REPO_ROOT}/${ASSETS_ROOT}"
fi

MAX_SAMPLES="${1:-}"
NUM_GPUS="${2:-1}"
DATASET_BASE_DIR="${LAYEREDDEPTH_DATASET_ROOT:-${ASSETS_ROOT}/datasets/LayeredDepth-Syn}"
EMBED_DIR="${EMBED_DIR:-${ASSETS_ROOT}/checkpoints/Marigold-V2/qwen_text_embeddings}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-${ASSETS_ROOT}/checkpoints/Qwen-Image-Edit-2509}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${ASSETS_ROOT}/checkpoints/Marigold-V2/depth/Log-stage2}"
LAYERED_CHECKPOINT="${LAYERED_CHECKPOINT:-${ASSETS_ROOT}/checkpoints/Marigold-V2/depth/Log-layered}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/output/eval_runs}"

if [[ -n "${MAX_SAMPLES}" && ! "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer: ${MAX_SAMPLES}" >&2
  exit 1
fi
if [[ ! "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer: ${NUM_GPUS}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_BASE_DIR}/extracted/val/manifest_layer8.csv" ]]; then
  echo "LayeredDepth-Syn manifest not found under ${DATASET_BASE_DIR}" >&2
  exit 1
fi
for checkpoint in "${STAGE2_CHECKPOINT}" "${LAYERED_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}/trainables.safetensors" ]]; then
    echo "Checkpoint not found: ${checkpoint}/trainables.safetensors" >&2
    exit 1
  fi
done
if [[ ! -d "${QWEN_CHECKPOINT}" ]]; then
  echo "Qwen checkpoint not found: ${QWEN_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${EMBED_DIR}" ]]; then
  echo "Prompt embedding directory not found: ${EMBED_DIR}" >&2
  exit 1
fi

RUN_ROOT="${OUTPUT_BASE}/depth_see_through_$(date +%y%m%dT%H%M%S)_$$"
mkdir -p "${RUN_ROOT}"

cd "${REPO_ROOT}"

run_one() {
  local name="$1"
  local checkpoint="$2"
  local output_dir="${RUN_ROOT}/${name}"
  local config_path="${output_dir}/generated_eval_layereddepth_l8.yaml"
  mkdir -p "${output_dir}"

  echo "[${name}] Building evaluation config"
  local build_command=(
    python -u evaluation/depth_see_through/build_layereddepth_l8_eval_config.py
    --output_dir "${output_dir}"
    --dataset_base_dir "${DATASET_BASE_DIR}"
    --embed_dir "${EMBED_DIR}"
    --qwen_checkpoint "${QWEN_CHECKPOINT}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    build_command+=(--max_samples "${MAX_SAMPLES}")
  fi
  "${build_command[@]}"

  echo "[${name}] Evaluating ${checkpoint}"
  local eval_args=(
    --config "${config_path}"
    --checkpoint "${checkpoint}"
    --output_dir "${output_dir}"
  )
  if (( NUM_GPUS > 1 )); then
    accelerate launch --num_processes "${NUM_GPUS}" \
      -m evaluation.depth.evaluate_pipeline "${eval_args[@]}"
  else
    python -u -m evaluation.depth.evaluate_pipeline "${eval_args[@]}"
  fi 2>&1 | tee "${output_dir}/evaluation.log"
}

run_one log_stage2 "${STAGE2_CHECKPOINT}"
run_one log_layered "${LAYERED_CHECKPOINT}"

echo "Evaluation complete: ${RUN_ROOT}"
