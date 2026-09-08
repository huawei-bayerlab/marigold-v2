#!/usr/bin/env bash
# Download Hypersim geometry and build the surface-normal training set with the
# Marigold V1.1 preprocessor. Environment overrides: DATASETS_DIR, WORK_DIR,
# OUTPUT_DIR, TOOLS_DIR, WORKERS, PYTHON.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASETS_DIR="${DATASETS_DIR:-${DEPTH_ASSETS_DIR:-${REPO_DIR}/assets}/datasets}"
WORK_DIR="${WORK_DIR:-${DATASETS_DIR}/hypersim_normals_work}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASETS_DIR}/marigold_train_normals}"
TOOLS_DIR="${TOOLS_DIR:-${WORK_DIR}/upstream}"
WORKERS="${WORKERS:-16}"
MARIGOLD_COMMIT="${MARIGOLD_COMMIT:-ac915287aaa4e5fd5fe578322735c7c411a1f263}"
HYPERSIM_COMMIT="${HYPERSIM_COMMIT:-c85b2879c8da8ded2f4c24d2630c1ab4451999b2}"

mkdir -p "${TOOLS_DIR}" "${OUTPUT_DIR}"

fetch() {
    local url=$1 destination=$2
    if [[ -s "${destination}" ]]; then return; fi
    curl --fail --location --retry 8 --retry-all-errors --connect-timeout 30 \
        --output "${destination}.part" "${url}"
    mv -- "${destination}.part" "${destination}"
}

HYPERSIM_RAW="https://raw.githubusercontent.com/apple/ml-hypersim/${HYPERSIM_COMMIT}"
MARIGOLD_RAW="https://raw.githubusercontent.com/prs-eth/Marigold/${MARIGOLD_COMMIT}/script/normals/dataset_preprocess/hypersim"
fetch "${HYPERSIM_RAW}/contrib/99991/download.py" "${TOOLS_DIR}/download.py"
fetch "${HYPERSIM_RAW}/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv" \
    "${TOOLS_DIR}/metadata_images_split_scene_v1.csv"
fetch "${MARIGOLD_RAW}/preprocess_hypersim_normals.py" "${TOOLS_DIR}/preprocess_hypersim_normals.py"
fetch "${MARIGOLD_RAW}/hypersim_util.py" "${TOOLS_DIR}/hypersim_util.py"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec "${PYTHON}" "${REPO_DIR}/scripts/hypersim_normals/run_hypersim_normals_pipeline.py" \
    --metadata-csv "${TOOLS_DIR}/metadata_images_split_scene_v1.csv" \
    --selective-downloader "${TOOLS_DIR}/download.py" \
    --preprocess-script "${TOOLS_DIR}/preprocess_hypersim_normals.py" \
    --filtered-list-dir "${REPO_DIR}/evaluation/data_split/hypersim_normals" \
    --work-dir "${WORK_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}"
