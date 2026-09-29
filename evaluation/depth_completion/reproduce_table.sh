#!/usr/bin/env bash
# Reproduce the zero-shot metric depth-completion table. See evaluation/depth_completion/REPRODUCING_DEPTH_COMPLETION.md.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: reproduce_table.sh subset [32gb|80gb|all]   a few images per cell, then check the result
       reproduce_table.sh full   [32gb|80gb|all]   every image of every cell
       reproduce_table.sh <baseline|hires|tiled|ours>     one row of the table at a time

  subset   Scores a few images of each cell and compares them with results_reference/. Takes a
           few hours on one GPU and catches a wrong checkpoint, missing data, or changed preprocessing.
  full     The whole table, days of GPU time. Run `subset` first.

The table spans two GPU classes, so it is split into two disjoint halves. Memory follows the
number of pixels the prior renders, which is a property of the cell: `hires` on KITTI-DC fits a
32 GB card, while `hires` on iBims-1 needs more.

  32gb   the 8 cells that fit a 32 GB card                             (default)
  80gb   the 3 cells that need an 80 GB card
  all    both halves, the complete table

Run 32gb on the card you have and 80gb wherever a larger one is available, and together they cover
the table. Peak memory is set by a single image, so a half selects which cells run.

Samples per cell for `subset` (a tiled sample costs 15-20x more, hence the lower default):
  SUBSET_N=5        baseline and hires
  SUBSET_N_TILED=1  tiled
  SUBSET_DIR=output/eval_runs/depth_completion_subset    kept apart from the full runs

Other environment: PY selects the interpreter. GPUS=0,1,2,3 spreads the work across local
devices, one shard per device for `full` and one cell per device for `subset`.
EOF
}

if [[ $# -lt 1 || $# -gt 2 || "${1:-}" =~ ^(--help|-h)$ ]]; then
  usage
  [[ "${1:-}" =~ ^(--help|-h)$ ]] && exit 0
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-python}"
GPUS="${GPUS:-}"                   # e.g. 0,1,2,3: shard a full row across these local devices
SETS=(ibims1 nyudepthv2 kittidc ddad)
SUBSET_N="${SUBSET_N:-5}"
SUBSET_N_TILED="${SUBSET_N_TILED:-1}"
SUBSET_DIR="${SUBSET_DIR:-output/eval_runs/depth_completion_subset}"

# Fragmentation accounts for ~1.6 GiB, which decides whether the larger cells fit a 32 GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The table itself, shared with the cell helpers.
# shellcheck source=evaluation/depth_completion/cells.sh
source "$(dirname "${BASH_SOURCE[0]}")/cells.sh"

# --- runners -------------------------------------------------------------------------------
# One configuration of our method. With GPUS set the dataset is split into one contiguous shard per
# device and the shards run concurrently; each writes its own results file and the summary merges
# them by sample id. Samples are completed independently, so this changes nothing but the wall clock.
ours() {
  local run=("${PY}" -u -m evaluation.depth_completion.run_marigold_v2)
  if [[ -z "${GPUS}" ]]; then
    "${run[@]}" "$@"
    return
  fi
  local devices pids=() rc=0 i
  IFS=',' read -r -a devices <<< "${GPUS}"
  for i in "${!devices[@]}"; do
    CUDA_VISIBLE_DEVICES="${devices[i]}" "${run[@]}" "$@" --shard "${i}/${#devices[@]}" &
    pids+=("$!")
  done
  for i in "${pids[@]}"; do wait "${i}" || rc=1; done
  return "${rc}"
}

run_subset() {  # run_subset <32|80|all>
  local want="$1"
  local todo=() row ds tier n
  while read -r row ds; do
    tier="$(cell_tier "${row}" "${ds}")"
    cell_selected "${row}" "${ds}" "${want}" || {
      printf '  skip  %-9s %-11s (%s)\n' "${row}" "${ds}" "$(skip_reason "${tier}" "${want}")"
      continue; }
    n="${SUBSET_N}"; [[ "${row}" == tiled ]] && n="${SUBSET_N_TILED}"
    todo+=("${row} ${ds} ${n}")
  done < <(cells)
  echo "Subset reproduction: $(half_label "${want}") -> ${SUBSET_DIR}/"

  # Cells are independent, so with GPUS set they run one per device rather than one after the
  # other. That matters here: a single tiled sample is half an hour, so the serial wall clock is
  # dominated by two cells that could have run side by side.
  local devices=() rc=0 i
  [[ -n "${GPUS}" ]] && IFS=',' read -r -a devices <<< "${GPUS}"
  local ndev="${#devices[@]}"
  if ((ndev == 0)); then devices=("") ndev=1; fi   # no GPUS: one worker, inheriting the environment
  local pids=()
  for ((i = 0; i < ndev; i++)); do
    (
      # An assignment prefix has to be literal, so pin the device with env(1) instead.
      local pre=() failed=0
      [[ -n "${devices[i]}" ]] && pre=(env "CUDA_VISIBLE_DEVICES=${devices[i]}")
      for ((j = i; j < ${#todo[@]}; j += ndev)); do
        read -r row ds n <<< "${todo[j]}"
        printf '  run   %-9s %-11s %s image(s)%s\n' "${row}" "${ds}" "${n}" \
          "${devices[i]:+ on GPU ${devices[i]}}"
        # shellcheck disable=SC2046  # cell_args deliberately word-splits into arguments
        "${pre[@]}" "${PY}" -u -m evaluation.depth_completion.run_marigold_v2 \
          $(cell_args "${row}" "${ds}") --limit "${n}" --resume --out-dir "${SUBSET_DIR}" \
          || { printf '  FAILED %-9s %-11s\n' "${row}" "${ds}" >&2; failed=1; }
      done
      exit "${failed}"
    ) &
    pids+=("$!")
  done
  for i in "${pids[@]}"; do wait "${i}" || rc=1; done
  echo
  "${PY}" -u -m evaluation.depth_completion.check_reference --results-dir "${SUBSET_DIR}" || rc=$?
  return "${rc}"
}

run_full() {  # run_full <32|80|all>
  local want="$1"
  echo "Full reproduction: $(half_label "${want}")."
  local row ds tier
  while read -r row ds; do
    tier="$(cell_tier "${row}" "${ds}")"
    cell_selected "${row}" "${ds}" "${want}" || {
      printf '  skip  %-9s %-11s (%s)\n' "${row}" "${ds}" "$(skip_reason "${tier}" "${want}")"
      continue; }
    # shellcheck disable=SC2046
    ours $(cell_args "${row}" "${ds}") --resume
  done < <(cells)
}

run_row() {  # one row of the table across all four datasets, whatever the GPU
  local row="$1" r ds
  while read -r r ds; do
    [[ "${r}" == "${row}" ]] || continue
    # shellcheck disable=SC2046
    ours $(cell_args "${row}" "${ds}") --resume
  done < <(cells)
}

# The GPU class is parsed here, outside any `||` list: inside `run_subset ... || rc=$?` the -e
# option is suspended and a bad value would go unnoticed.
case "$1" in
  subset|full) HALF="$(parse_half "${2:-32gb}")" || exit 2 ;;
esac

case "$1" in
  subset)    rc=0; run_subset "${HALF}" || rc=$?; exit "${rc}" ;;  # its own check; no table below
  full)      run_full "${HALF}" ;;
  baseline)  run_row baseline ;;
  hires)     run_row hires ;;
  tiled)     run_row tiled ;;
  ours)      run_row baseline; run_row hires; run_row tiled ;;
  *) usage >&2; exit 2 ;;
esac

"${PY}" -u -m evaluation.depth_completion.summarize
