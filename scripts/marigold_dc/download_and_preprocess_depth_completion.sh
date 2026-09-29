#!/usr/bin/env bash
# Download and preprocess the four Marigold-DC depth-completion datasets.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASETS_DIR="${DATASETS_DIR:-${DEPTH_ASSETS_DIR:-${REPO_DIR}/assets}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASETS_DIR}/marigold_depth_completion}"
RSYNC_PASSWORD="${RSYNC_PASSWORD:-m1455541}"
DATASETS="ibims1,nyudepthv2,kittidc,ddad"
CLEAN_RAW=false

usage() {
    cat <<'EOF'
Usage: download_and_preprocess_depth_completion.sh [OPTIONS]

Download and preprocess iBims-1, NYUv2, KITTI DC, and DDAD using the
Marigold-DC depth-completion protocol.

Options:
  --datasets LIST       Comma-separated subset of ibims1,nyudepthv2,kittidc,ddad
  --clean-raw           Remove downloaded archives and extracted source data
  -h, --help            Show this help message

Environment overrides:
  OUTPUT_DIR            Dataset root (default: $DEPTH_ASSETS_DIR/datasets/
                        marigold_depth_completion, or assets/...)
  DATASETS_DIR          Parent directory used by the OUTPUT_DIR default
  PYTHON                Python executable used for preprocessing
  RSYNC_PASSWORD        iBims-1 rsync password (default: m1455541)
EOF
}

while (($#)); do
    case "$1" in
        --datasets)
            if (($# < 2)); then
                echo "--datasets requires a comma-separated value" >&2
                exit 2
            fi
            DATASETS="$2"
            shift 2
            ;;
        --clean-raw)
            CLEAN_RAW=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

IFS=',' read -r -a SELECTED_DATASETS <<< "${DATASETS}"
declare -A VALID_DATASETS=(
    [ibims1]=1
    [nyudepthv2]=1
    [kittidc]=1
    [ddad]=1
)
for dataset in "${SELECTED_DATASETS[@]}"; do
    dataset="${dataset,,}"
    if [[ -z "${VALID_DATASETS[${dataset}]:-}" ]]; then
        echo "Unsupported dataset: ${dataset}" >&2
        echo "Choose from: ibims1, nyudepthv2, kittidc, ddad" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"

download_http() {
    local url=$1
    local destination=$2
    if [[ -s "${destination}" ]]; then
        echo "[skip] ${destination}"
        return
    fi
    mkdir -p "$(dirname -- "${destination}")"
    echo "[download] ${url} -> ${destination}"
    curl --fail --location --retry 8 --retry-all-errors --connect-timeout 30 \
        --continue-at - --output "${destination}.part" "${url}"
    mv -- "${destination}.part" "${destination}"
}

download_ibims1() {
    local archive="${OUTPUT_DIR}/ibims1_core_mat.zip"
    local attempt
    if [[ ! -s "${archive}" ]]; then
        echo "[download] iBims-1 -> ${archive}"
        for attempt in {1..8}; do
            if RSYNC_PASSWORD="${RSYNC_PASSWORD}" rsync --partial --progress \
                "rsync://m1455541@dataserv.ub.tum.de/m1455541/ibims1_core_mat.zip" \
                "${archive}.part"; then
                mv -- "${archive}.part" "${archive}"
                break
            fi
            if ((attempt == 8)); then
                echo "iBims-1 rsync failed after ${attempt} attempts" >&2
                return 1
            fi
            echo "[retry] iBims-1 rsync (attempt $((attempt + 1))/8)" >&2
        done
    else
        echo "[skip] ${archive}"
    fi
    if ! compgen -G "${OUTPUT_DIR}/ibims1_core_mat/ibims1_core_mat/*.mat" \
        > /dev/null; then
        echo "[extract] ${archive}"
        unzip -n -q "${archive}" -d "${OUTPUT_DIR}/ibims1_core_mat"
    fi
}

download_nyudepthv2() {
    download_http \
        "https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_img_gt.h5" \
        "${OUTPUT_DIR}/nyu_img_gt.h5"
    download_http \
        "https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/download/v0.1.0/nyu_pred_with_500.h5" \
        "${OUTPUT_DIR}/nyu_pred_with_500.h5"
}

download_kittidc() {
    local archive="${OUTPUT_DIR}/data_depth_selection.zip"
    download_http \
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip" \
        "${archive}"
    if [[ ! -d "${OUTPUT_DIR}/depth_selection/val_selection_cropped" ]]; then
        echo "[extract] ${archive}"
        unzip -n -q "${archive}" -d "${OUTPUT_DIR}"
    fi
}

download_ddad() {
    local archive="${OUTPUT_DIR}/ddad_pregenerated.zip"
    download_http \
        "https://drive.usercontent.google.com/download?id=1y8Rt3Hld8zVTSKxx9d9yYXSzr5niKN7i&confirm=t" \
        "${archive}"
    if [[ ! -d "${OUTPUT_DIR}/pregenerated/val" ]]; then
        echo "[extract] ${archive}"
        unzip -n -q "${archive}" -d "${OUTPUT_DIR}"
    fi
}

for dataset in "${SELECTED_DATASETS[@]}"; do
    case "${dataset,,}" in
        ibims1) download_ibims1 ;;
        nyudepthv2) download_nyudepthv2 ;;
        kittidc) download_kittidc ;;
        ddad) download_ddad ;;
    esac
done

PREPROCESS_ARGS=(--root "${OUTPUT_DIR}" --datasets "${DATASETS}")
if [[ "${CLEAN_RAW}" == true ]]; then
    PREPROCESS_ARGS+=(--clean-raw)
fi
exec "${PYTHON}" "${SCRIPT_DIR}/preprocess_depth_completion.py" "${PREPROCESS_ARGS[@]}"
