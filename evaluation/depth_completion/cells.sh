# Shared definition of the depth-completion table, sourced by reproduce_table.sh so that one
# description of each cell drives the launcher. Keeping this in one place keeps one protocol per
# results file.
#
# A cell is one (row, dataset) pair. The three rows are baseline, hires and tiled.

cells() {
  echo "baseline ibims1"; echo "baseline nyudepthv2"; echo "baseline kittidc"; echo "baseline ddad"
  echo "hires ibims1";    echo "hires nyudepthv2";    echo "hires kittidc";    echo "hires ddad"
  echo "tiled ibims1";                                echo "tiled kittidc";    echo "tiled ddad"
}

# Smallest GPU each cell fits, measured one image at a time on a 32 GB RTX 5000 Ada. See the
# guide's "GPU memory per cell". hires:ibims1 peaks at 30.8 GiB and wants the card to itself, so it
# is grouped with the large half.
cell_tier() {
  case "$1:$2" in
    baseline:*)                     echo 32 ;;
    hires:nyudepthv2|hires:kittidc) echo 32 ;;
    tiled:ibims1|tiled:kittidc)     echo 32 ;;
    hires:ibims1)                   echo 48 ;;
    hires:ddad|tiled:ddad)          echo 80 ;;
    *) echo "no tier for cell $1:$2" >&2; exit 3 ;;
  esac
}

# Does this cell belong to the half being run? The two halves partition the table.
cell_selected() {  # cell_selected <row> <dataset> <32|80|all>
  local tier; tier="$(cell_tier "$1" "$2")"
  case "$3" in
    32)  ((tier <= 32)) ;;
    80)  ((tier > 32)) ;;
    all) true ;;
  esac
}

# The runner arguments for one cell. The tag is what results_reference/ and summarize.py key on.
cell_args() {
  case "$1" in
    baseline) echo "--dataset $2 --tag dc_baseline" ;;
    hires)
      case "$2" in
        ibims1)     echo "--dataset ibims1     --proc-long-side 1280 --loss-at-native --tag dc_hires" ;;
        nyudepthv2) echo "--dataset nyudepthv2 --proc-long-side 1152 --loss-at-native --tag dc_hires" ;;
        kittidc)    echo "--dataset kittidc    --proc-long-side 1824 --loss-at-native --tag dc_hires" ;;
        ddad)       echo "--dataset ddad       --proc-long-side 0    --loss-at-native --tag dc_hires" ;;
      esac ;;
    tiled)
      case "$2" in
        ibims1)  echo "--dataset ibims1  --proc-preset hires --tiles 3x3 --tile-scale 4.0 --tile-overlap 0.15 --tag dc_tiled" ;;
        kittidc) echo "--dataset kittidc --proc-preset hires --tiles 2x2 --tile-scale 1.5 --tile-overlap 0.25 --tag dc_tiled" ;;
        ddad)    echo "--dataset ddad    --proc-preset hires --tiles 2x2 --tile-scale 2.0 --tile-overlap 0.25 --tag dc_tiled" ;;
      esac ;;
  esac
}

parse_half() {
  case "${1:-32gb}" in
    32gb) echo 32 ;; 80gb) echo 80 ;; all) echo all ;;
    *) echo "expected 32gb, 80gb or all, got '${1}'" >&2; exit 2 ;;
  esac
}

# Why a cell was left out: too large for this half, or covered by the other one.
skip_reason() {  # skip_reason <cell tier> <half>
  local article="a"; [[ "$1" == 80 ]] && article="an"
  if [[ "$2" == 32 ]]; then echo "needs ${article} $1 GB card"; else echo "in the 32gb half"; fi
}

half_label() {
  case "$1" in
    32) echo "the 8 cells that fit a 32 GB card" ;;
    80) echo "the 3 cells needing an 80 GB card" ;;
    all) echo "all 11 cells" ;;
  esac
}
