#!/usr/bin/env bash
# Download Hypersim diffuse reflectance and build the albedo training set.
# Requires the preprocessed Hypersim RGB dataset (marigold_train). Environment
# overrides: DATASETS_DIR, RGB_DATASET_DIR, WORK_DIR, OUTPUT_DIR, TOOLS_DIR,
# WORKERS, PYTHON.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASETS_DIR="${DATASETS_DIR:-${DEPTH_ASSETS_DIR:-${REPO_DIR}/assets}/datasets}"
WORK_DIR="${WORK_DIR:-${DATASETS_DIR}/hypersim_albedo_work}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASETS_DIR}/marigold_train_albedo}"
RGB_DATASET_DIR="${RGB_DATASET_DIR:-${DATASETS_DIR}/marigold_train}"
TOOLS_DIR="${TOOLS_DIR:-${WORK_DIR}/upstream}"
WORKERS="${WORKERS:-16}"
HYPERSIM_COMMIT="${HYPERSIM_COMMIT:-c85b2879c8da8ded2f4c24d2630c1ab4451999b2}"

if [[ ! -d "${RGB_DATASET_DIR}" ]]; then
    echo "Missing preprocessed RGB dataset: ${RGB_DATASET_DIR} (run scripts/download_assets.py first)" >&2
    exit 1
fi
mkdir -p "${TOOLS_DIR}" "${OUTPUT_DIR}"

DOWNLOADER="${TOOLS_DIR}/download.py"
if [[ ! -s "${DOWNLOADER}" ]]; then
    curl --fail --location --retry 8 --retry-all-errors --connect-timeout 30 \
        --output "${DOWNLOADER}.part" \
        "https://raw.githubusercontent.com/apple/ml-hypersim/${HYPERSIM_COMMIT}/contrib/99991/download.py"
    mv -- "${DOWNLOADER}.part" "${DOWNLOADER}"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec "${PYTHON}" "${REPO_DIR}/scripts/hypersim_albedo/run_hypersim_albedo_pipeline.py" \
    --selective-downloader "${DOWNLOADER}" \
    --filtered-list-dir "${REPO_DIR}/evaluation/data_split/hypersim_iid" \
    --rgb-dataset-dir "${RGB_DATASET_DIR}" \
    --work-dir "${WORK_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    --output-dtype float32
