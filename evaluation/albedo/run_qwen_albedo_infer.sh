#!/usr/bin/env bash
# Native-resolution albedo inference on the Hypersim test split.
set -euo pipefail

usage() {
  echo "Usage: $0 CHECKPOINT OUTPUT_DIR [MAX_SAMPLES] [NUM_GPUS] [DATASET_BASE_DIR] [EMBED_DIR] [QWEN_CHECKPOINT]"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 2 || $# > 7 )); then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSETS_DIR="${DEPTH_ASSETS_DIR:-${REPO_ROOT}/assets}"
CHECKPOINT="$1"
OUTPUT_DIR="$2"
MAX_SAMPLES="${3:-}"
NUM_GPUS="${4:-1}"
DATASET_BASE_DIR="${5:-${ASSETS_DIR}/datasets/marigold_train_albedo}"
EMBED_DIR="${6:-${ASSETS_DIR}/checkpoints/Marigold-V2/qwen_text_embeddings}"
QWEN_CHECKPOINT="${7:-${ASSETS_DIR}/checkpoints/Qwen-Image-Edit-2509}"

INFER_COMMAND=(
  python -u evaluation/albedo/infer_qwen_albedo_hypersim.py
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_base_dir "${DATASET_BASE_DIR}"
  --embed_dir "${EMBED_DIR}"
  --qwen_checkpoint "${QWEN_CHECKPOINT}"
  --num_gpus "${NUM_GPUS}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  INFER_COMMAND+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${REPO_ROOT}"
"${INFER_COMMAND[@]}"

echo "Predictions: ${OUTPUT_DIR}/hypersim_test_albedo_qwen_native"
