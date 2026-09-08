#!/usr/bin/env bash
# Create a conda environment and install Marigold V2 in editable mode.
# Usage: bash setup/setup_env.sh [ENV_NAME] [CUDA_VERSION]   (defaults: marigold-v2 cu128)
set -euo pipefail

ENV_NAME="${1:-marigold-v2}"
CUDA_VERSION="${2:-cu128}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda create -y -n "${ENV_NAME}" --override-channels -c conda-forge python=3.10
# shellcheck disable=SC1091
. "$(conda info --base)/etc/profile.d/conda.sh"
# Hooks of an already-active env may read unset variables; -u would abort here.
set +u
conda activate "${ENV_NAME}"
set -u

python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
python -m pip install -e "${REPO_ROOT}"
python -m pip check

echo "Done. Activate with: conda activate ${ENV_NAME}"
