# Reproducing the depth-completion results

Sparse depth in, dense metric depth out, on four benchmarks. Every row is scored under one
protocol: the benchmark's own sparse points as input, scoring at native resolution, and metric
scale recovered by anchoring to those sparse points.

Marigold V2 predicts relative depth, so the sparse points supply the metric scale. The released
prior and its trained depth LoRA stay frozen, a second zero-initialised LoRA is injected, and that
adapter is optimised against the sparse points at test time. Each image gets its own adapter.

- [Quick start](#quick-start)
- [Results](#results)
- [Getting the data](#getting-the-data)
- [Data and splits](#data-and-splits)
- [Protocol](#protocol)
- [Reproducing the numbers](#reproducing-the-numbers): [subset](#subset-reproduction),
  [full](#full-reproduction), [GPU memory](#gpu-memory-per-cell), [cost](#cost-per-sample)
- [The method](#the-method)
- [Reproducibility](#reproducibility)
- [Files](#files)

## Quick start

From the repository root, with the datasets in place:

```bash
# 1. verify the setup against the published per-sample results (~45 min on 4 GPUs)
GPUS=0,1,2,3 bash evaluation/depth_completion/reproduce_table.sh subset 32gb

# 2. reproduce the full table (days; shard it across GPUs)
GPUS=0,1,2,3 bash evaluation/depth_completion/reproduce_table.sh full 32gb

# 3. print the table at any point
python -m evaluation.depth_completion.summarize
```

Set `DEPTH_ASSETS_DIR` if your assets tree is somewhere other than `./assets`, as for the rest of
`evaluation/`. Run step 1 first: it takes a few hours on one GPU and catches a wrong checkpoint,
missing data, or changed preprocessing.

## Results

MAE in metres, lower is better. iBims-1 and NYUv2 are their complete test sets; KITTI-DC and DDAD
are fixed 150-sample subsets (see [Data and splits](#data-and-splits)).

| Method | iBims-1 (n=100) | NYUv2 (n=654) | KITTI-DC (n=150) | DDAD (n=150) |
|---|---|---|---|---|
| Marigold-V2-DC (LoRA) | 0.042 | 0.045 | 0.349 | 1.549 |
| + high-res inference | 0.034 | 0.044 | 0.340 | 1.465 |
| + tiled local adaptation | 0.030 | 0.044 | 0.318 | 1.226 |

Both increments are significant under a paired bootstrap (4000 resamples): high-res
-19.5 % / -3.4 % / -2.7 % / -5.4 %, tiling a further -10.9 % / -6.4 % / -16.3 % on the three sets
it applies to. `summarize.py` prints the table to full precision with those intervals, and
`--metric rmse` (or `absrel`, `delta1`) gives the same table for the other metrics.

The paper compares these rows against external methods. This directory covers Marigold V2 alone.

## Getting the data

One command, from the repository root:

```bash
bash scripts/marigold_dc/download_and_preprocess_depth_completion.sh
```

It downloads and preprocesses all four datasets into
`${DEPTH_ASSETS_DIR:-assets}/datasets/marigold_depth_completion/`, where the runners look for them.
Use `--datasets ibims1,nyudepthv2` for a subset. Downloads resume from `.part` files and completed
archives are skipped. The host needs `curl`, `rsync` and `unzip`; the Python dependencies come from
the project environment. See `scripts/marigold_dc/README.md` for the overrides and the upstream
source of each dataset.

```text
marigold_depth_completion/
├── ibims1/        100 samples
├── nyudepthv2/    654
├── kittidc/      1000
└── ddad/         3950
```

Each dataset holds matching `rgb/`, `gt/`, `sparse/` and `intrinsics/` directories, with ground
truth and sparse maps as float32 metre `.npy` arrays where 0 marks "no measurement". This root is
separate from `marigold_depth_eval/`, which holds the zero-shot monocular-depth benchmarks.

The preprocessing reproduces the upstream tree exactly. KITTI-DC `gt` maps are bit-exact against
the official archive's `groundtruth_depth` ÷ 256, and the `sparse` maps reproduce through
`filter_heuristic_depth`, the VPP4DC 7×7 local-window filter (threshold 1.5 m) that drops a return
sitting more than 1.5 m behind the nearest one in its neighbourhood, about 7 % of the raw Velodyne
points. Five KITTI samples were spot-checked against an independently built copy of the tree and
came out identical.

## Data and splits

| dataset | evaluated | test set | source of the list |
|---|---|---|---|
| iBims-1 | 100 | 100 | the complete test set, enumerated from the data |
| NYUv2 | 654 | 654 | the complete test set, enumerated from the data |
| KITTI-DC | 150 | 1000 | fixed subset, `data_split/kittidc_dc.txt` |
| DDAD | 150 | 3950 | fixed subset, `data_split/ddad_dc.txt` |

For iBims-1 and NYUv2, `load_split` lists `<dataset_root>/rgb` and sorts by sample id, which stays
in step with the data. KITTI-DC (1000 images) and DDAD (3950 per camera) are subsampled to 150, and
those ids are pinned in `data_split/`, one line per sample as `rgb gt sparse`.

Both subsets are an evenly spaced draw over the sorted ids, `np.round(np.linspace(0, N-1, n))`,
i.e. index `round(i × (N-1) / (n-1))` for `i` in `0 … n-1`. The draw is deterministic, so each
subset follows from the dataset alone and always contains the first and last sample. The committed
lists are the authority, and they can be checked against the data:

```bash
python -m evaluation.depth_completion.make_data_split          # verify, non-zero exit on a mismatch
python -m evaluation.depth_completion.make_data_split --write  # rebuild after a dataset re-export
```

`datasets.json` maps each dataset to its root, its split file where it has one, and the expected
sample count. `load_split` enforces that count on every run, so a truncated download fails
immediately and a table row is always scored over the full set of samples.

The sparse input differs between the indoor and outdoor sets:

| dataset | points per image | origin |
|---|---|---|
| iBims-1 | exactly 1000 | uniform random subsample of the dense GT, so every point equals GT |
| NYUv2 | exactly 500 | the same |
| KITTI-DC | 14.7k-18.8k | official raw LiDAR; 38 % of points coincide with GT, 47 % land where GT is invalid and serve as input only |
| DDAD | 4.6k-5.6k | LiDAR sweeps; 98 % coincide with GT |

iBims-1 and NYUv2 ship no official sparse map, so their point budget is a benchmark-design choice.

## Protocol

Identical for every method in the table:

- **Input.** The benchmark's provided sparse depth, read at native resolution.
- **Scoring.** Native GT resolution, full frame. MAE, RMSE, AbsRel and δ₁ over the pixels where GT
  is finite and greater than 1 mm, through one implementation, `common.compute_metrics`.
- **Metric scale.** Recovered from the sparse points through a log-affine fit.

The numbers above are specific to this protocol: 100/654/150/150 samples, scored at native
resolution. Published depth-completion figures generally come from different sample counts, crops
and sparse patterns, so compare rows within this table.

## Reproducing the numbers

Two entry points:

| | what it covers | one GPU | four GPUs |
|---|---|---|---|
| [`subset`](#subset-reproduction) | a few images of each cell, checked against the published per-sample results | ~2.3 h | ~45 min |
| [`full`](#full-reproduction) | every image of every cell | 8-12 days | 2-3 days |

The table spans two GPU classes, so each entry point takes the card you have as its argument and
runs the cells that fit it. Memory follows the number of pixels the prior renders, which is a
property of the individual cell: `hires` on KITTI-DC fits a 32 GB card, while `hires` on iBims-1
needs more. See [GPU memory per cell](#gpu-memory-per-cell).

| half | cells | card |
|---|---|---|
| `32gb` | 8 of 11: all of `baseline`, `hires` on NYUv2 and KITTI-DC, `tiled` on iBims-1 and KITTI-DC | 32 GB |
| `80gb` | 3 of 11: `hires` on iBims-1 and DDAD, `tiled` on DDAD | 80 GB |
| `all` | both halves | 80 GB |

The two halves are disjoint, so the commands compose: run `32gb` on the card you have and `80gb`
wherever a larger one is available, and together they cover the table.

```bash
# on a 32 GB card
bash evaluation/depth_completion/reproduce_table.sh subset 32gb    # the default
bash evaluation/depth_completion/reproduce_table.sh full   32gb

# on an 80 GB card, for the three remaining cells
bash evaluation/depth_completion/reproduce_table.sh subset 80gb
bash evaluation/depth_completion/reproduce_table.sh full   80gb
```

Each run lists the cells it covers and the cells it leaves to the other half:

```text
  skip  hires     ibims1      (needs a 48 GB card)
  skip  hires     ddad        (needs an 80 GB card)
  skip  tiled     ddad        (needs an 80 GB card)
Subset reproduction: the 8 cells that fit a 32 GB card -> output/eval_runs/depth_completion_subset/
```

Peak memory is set by a single image, since every sample is completed independently and its graph
freed afterwards. A half therefore selects which cells run, and the sample count is free to choose
on cost alone.

Results are one JSON line per sample in
`output/eval_runs/depth_completion/<tag>_<dataset>_o<offset>.jsonl`, carrying the metrics and the
configuration that produced them, next to the predicted depth maps under `predictions/` (see
[Predicted depth maps](#predicted-depth-maps)). `summarize` reads whatever is present, so a
partially-run table prints the rows it finds and `--` for the rest:

```bash
python -m evaluation.depth_completion.summarize                  # table + paired bootstrap
python -m evaluation.depth_completion.summarize --no-bootstrap   # the table alone
python -m evaluation.depth_completion.summarize --metric rmse    # or absrel, delta1
```

### Subset reproduction

Scores the first few images of every cell your card runs, then compares them with
`results_reference/`, which holds the per-sample results behind every number in the table above.

```bash
bash evaluation/depth_completion/reproduce_table.sh subset 32gb
GPUS=0,1,2,3 bash evaluation/depth_completion/reproduce_table.sh subset 32gb   # one cell per GPU
```

| half | cells | one GPU | four GPUs |
|---|---|---|---|
| `32gb` | 8 | ~2.3 h | ~45 min |
| `80gb` | 3 | ~1.2 h | ~40 min |
| `all` | 11 | ~3.5 h | ~1 h |

Five images per cell, and one for tiled, where a single sample takes over half an hour on iBims-1
and DDAD. Output goes to `output/eval_runs/depth_completion_subset/`, apart from the full runs.
Set `SUBSET_N` and `SUBSET_N_TILED` to change the counts:

```bash
SUBSET_N=5 SUBSET_N_TILED=2 bash evaluation/depth_completion/reproduce_table.sh subset 32gb
```

Two shared samples are enough to bootstrap an interval and return a verdict. At one sample a row
reports its ratio and `1 sample: gross errors only`, which covers a wrong checkpoint or a broken
tile merge. Raising `SUBSET_N_TILED=2` adjudicates the tiled rows too, for about an extra hour.

The run ends with the comparison. Below an example output:

```text
  row                        dataset         n     yours  published   ratio    95% interval  verdict
  Marigold-V2-DC (LoRA)      iBims-1         2    0.0757     0.0744    1.08    [0.84, 1.31]  consistent
  Marigold-V2-DC (LoRA)      NYUv2           2    0.0157     0.0157    1.01    [0.99, 1.03]  consistent
  Marigold-V2-DC (LoRA)      KITTI-DC        2    0.2470     0.2590    0.95    [0.95, 0.95]  consistent
  Marigold-V2-DC (LoRA)      DDAD            2    1.5945     1.5396    1.02    [0.95, 1.10]  consistent
  + high-res inference       NYUv2           2    0.0147     0.0152    0.95    [0.89, 1.01]  consistent
  + high-res inference       KITTI-DC        2    0.2332     0.2290    1.01    [0.94, 1.09]  consistent
  + tiled local adaptation   iBims-1         1    0.0605     0.0557    1.09              --  1 sample: gross errors only
  + tiled local adaptation   NYUv2           2    0.0147     0.0152    0.95    [0.89, 1.01]  consistent
  + tiled local adaptation   KITTI-DC        1    0.1868     0.1943    0.96              --  1 sample: gross errors only

7 row(s) checked, all consistent with the published results. 2 row(s) had too few shared samples to judge.
```

Nine rows come out of eight cells, because the `tiled` NYUv2 row reports the high-res result, which
is its row-3 value. Exit code 0 when every compared row is consistent, 1 when one differs, and 2
when the run shares no samples with the reference, which points at a different `--out-dir` or
`--tag`.

**Reading the verdict.** The check compares the median per-sample ratio to the published value,
bootstraps it, and flags a row when that interval lies entirely outside ±5 %. `consistent` therefore
means "within 5 % of the published result", and a wide interval means only gross failures have been
ruled out at that sample count; raise `SUBSET_N` to tighten it. [Reproducibility](#reproducibility)
explains the choice of statistic.

To check results you already have:

```bash
python -m evaluation.depth_completion.check_reference                       # against output/eval_runs/depth_completion/
python -m evaluation.depth_completion.check_reference --results-dir DIR
```

It compares each of the three rows on every dataset and lists the rows where both sides have
results.

### Full reproduction

The complete table, once the subset checks out.

```bash
bash evaluation/depth_completion/reproduce_table.sh full 32gb
GPUS=0,1,2,3 bash evaluation/depth_completion/reproduce_table.sh full 80gb
python -m evaluation.depth_completion.summarize
```

| half | cells | one GPU |
|---|---|---|
| `32gb` | 8 | ~7.6 days |
| `80gb` | 3 | ~4.5 days |
| `all` | 11 | ~12.1 days |

Divide by the number of GPUs, since every sample is completed independently and the work shards
perfectly. See [Multiple GPUs](#multiple-gpus).

Individual rows run on any card:

```bash
bash evaluation/depth_completion/reproduce_table.sh baseline    # one row, all four datasets
bash evaluation/depth_completion/reproduce_table.sh hires
bash evaluation/depth_completion/reproduce_table.sh tiled
bash evaluation/depth_completion/reproduce_table.sh ours        # the three above
```

### GPU memory per cell

Peak memory follows the number of pixels the prior renders. For a tiled run that is `--tile-scale`
times the tile's own long side, which makes tiling cheaper than high-res on KITTI-DC and dearer on
DDAD.

| cell | prior renders | peak GiB | smallest card | half |
|---|---|---|---|---|
| `baseline` iBims-1 | 480×640, 0.31 Mpx | 17.4 | 32 GB | `32gb` |
| `tiled` KITTI-DC | 330×1140 per tile, 0.38 | 18.6 | 32 GB | `32gb` |
| `baseline` KITTI-DC | 352×1216, 0.43 | 19.2 | 32 GB | `32gb` |
| `baseline` NYUv2 | 576×768, 0.44 | 19.4 | 32 GB | `32gb` |
| `baseline` DDAD | 643×1024, 0.66 | 22.8 | 32 GB | `32gb` |
| `tiled` iBims-1 | 736×980 per tile, 0.72 | 26.4 | 32 GB | `32gb` |
| `hires` KITTI-DC | 528×1824, 0.96 | 27.0 | 32 GB | `32gb` |
| `hires` NYUv2 | 864×1152, 1.00 | 27.5 | 32 GB | `32gb` |
| `hires` iBims-1 | 960×1280, 1.23 | 30.8 | 48 GB | `80gb` |
| `hires` DDAD | 1216×1936, 2.35 | > 31.5 | 80 GB | `80gb` |
| `tiled` DDAD | 1520×2420 per tile, 3.68 | > 31.5 | 80 GB | `80gb` |

Sorted by cost, which tracks the megapixel count. `tiled` KITTI-DC is the second cheapest cell in
the table and `tiled` DDAD the most expensive, because tiling subdivides the first and magnifies
the second.

### Cost per sample

Median seconds per sample.

| seconds | iBims-1 | NYUv2 | KITTI-DC | DDAD |
|---|---|---|---|---|
| `baseline` | 96 | 130 | 129 | 197 |
| `hires` | 143 | 300 | 290 | 300 |
| `tiled` | 842-2058 | see `hires` | 466 | 2187 |

Treat these as indicative. They track GPU contention more than card class, and the tiled figures
most of all: the same iBims-1 tiled cell has been measured at both ends of the range above on the
same hardware. Two images of each dataset on an idle RTX 5000 Ada ran the `baseline` row in
77 / 103 / 100 / 152 s against the 96 / 130 / 129 / 197 s here.

### Multiple GPUs

Every sample is completed independently, so the work shards perfectly and sharding affects the wall
clock alone. `--shard I/N` takes the Ith of N contiguous slices:

```bash
GPUS=0,1,2,3 bash evaluation/depth_completion/reproduce_table.sh full 32gb
```

That launches one process per listed device, each with its own `--shard`. For `subset`, `GPUS`
assigns one cell per device instead, which suits a handful of images spread over cells that differ
widely in cost. Each process writes its own `<tag>_<dataset>_o<offset>.jsonl`, and `summarize`
merges them by sample id.

Across several machines, give each process a slice of one global split. With eleven GPUs in total,
the first machine runs shards `0/11 … 1/11`, the next `2/11 … 6/11`, and so on:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.depth_completion.run_marigold_v2 --dataset ddad --shard 0/11 --resume --tag dc_baseline
```

`--offset` and `--limit` cover an uneven split, which helps when the GPUs differ in speed. Every
runner takes `--resume`, which skips samples already in its own shard, so an interrupted job
continues where it stopped.

### Predicted depth maps

Every run writes one `pred_<sample>.npy` per sample, float32 metric depth at native GT
resolution, to `<out-dir>/predictions/<tag>_<dataset>/`. A row takes about 1.8 GiB across the
four datasets, 1.3 GiB of it on DDAD. `--pred-dir DIR` changes the location and `--no-save-pred`
skips the maps.

## The method

`run_marigold_v2.py` runs one denoising step of the frozen `depth/Log-stage2` prior at the trained
timestep t = 499/1000. The prior and its rank-128 depth LoRA stay frozen. A fresh rank-16 LoRA goes
into the **last 12 of the 60 transformer blocks**, zero-initialised so the first forward pass
reproduces the prior's own prediction. That adapter and a two-parameter log-affine `(a, b)` are
optimised together for 100 Adam steps, `lr 1e-3` for the adapter and `3e-2` for `(a, b)`, against
`L1 + L2` on the sparse pixels. A least-squares fit initialises `(a, b)`, so the optimisation starts
from the best global alignment available.

The three rows differ in where the prior runs and what it is optimised over.

**Row 1, baseline.** One pass at the per-dataset default: NYUv2 upscaled to 768 to reach the
resolution the prior operates at, DDAD downscaled to 1024 for memory, iBims-1 and KITTI-DC at
native. Where the processing grid differs from the native one, the sparse map is splatted onto it
and the loss is computed there.

**Row 2, high-res inference.** `--proc-long-side` raises the resolution the prior runs at, and
`--loss-at-native` holds the supervision on the original pixels: the sparse map stays on its own
grid, and the prediction returns to native resolution inside the optimisation loop, so the loss and
the metric see the same pixels. This keeps the anchor density of the original sparse map.

```bash
python -m evaluation.depth_completion.run_marigold_v2 --dataset ibims1     --proc-long-side 1280 --loss-at-native --tag dc_hires
python -m evaluation.depth_completion.run_marigold_v2 --dataset nyudepthv2 --proc-long-side 1152 --loss-at-native --tag dc_hires
python -m evaluation.depth_completion.run_marigold_v2 --dataset kittidc    --proc-long-side 1824 --loss-at-native --tag dc_hires
python -m evaluation.depth_completion.run_marigold_v2 --dataset ddad       --proc-long-side 0    --loss-at-native --tag dc_hires
```

**Row 3, tiled local adaptation.** The whole procedure runs independently on each tile of an
overlapping grid, with its own adapter and its own log-affine fitted to the sparse points inside
that tile and optimised at `--tile-scale` times the tile's own long side. The metric predictions
are then cross-faded across the overlap. Every tile is metric on its own, so the merge only has to
hide small disagreements. A tile with fewer than 8 sparse points is not completed and its pixels
come from the untiled prediction instead. `--proc-preset hires` sets the per-dataset resolution and
implies `--loss-at-native`.

```bash
python -m evaluation.depth_completion.run_marigold_v2 --dataset ibims1  --proc-preset hires --tiles 3x3 --tile-scale 4.0 --tile-overlap 0.15 --tag dc_tiled
python -m evaluation.depth_completion.run_marigold_v2 --dataset kittidc --proc-preset hires --tiles 2x2 --tile-scale 1.5 --tile-overlap 0.25 --tag dc_tiled
python -m evaluation.depth_completion.run_marigold_v2 --dataset ddad    --proc-preset hires --tiles 2x2 --tile-scale 2.0 --tile-overlap 0.25 --tag dc_tiled
```

On NYUv2 the 500 points per image spread thin across tiles, so its row-3 cell reports the row-2
result and `summarize.py` reads it from there.

**Where the error concentrates.** Accuracy tracks the depth range of the scene: AbsRel grows with
the p99 depth across the four sets (8.6 / 4.5 / 58.7 / 143.5 m), which points at the
piecewise-affine metric parameterisation reaching its range on the outdoor sets. Tiling subdivides
that parameterisation, which is why it gains the most on DDAD.

## Reproducibility

The prior runs 4-bit quantised and its attention has no deterministic backward, so a re-run of the
same image on the same GPU moves by up to about 0.05 MAE on iBims-1, and by more on a different
GPU. Dataset means are stable to about 1e-3.

Per-image MAE spans a factor of twenty on iBims-1, so the mean over a handful of images is decided
by which images they are. `check_reference` therefore compares the median per-sample ratio to the
published value and bootstraps it. On a known-good setup that interval is about [0.84, 1.15] at
five iBims-1 samples and [0.96, 1.05] at five DDAD samples. A row is flagged only when the
interval lies entirely outside ±5 %; without that band a full run would flag on a 1 % hardware
difference. The differences between rows in [Results](#results) are tested the same way, with a
paired bootstrap that resamples the shared samples and scores both rows on each draw.

## Files

```text
run_marigold_v2.py    the method: test-time LoRA on the frozen prior, one process per dataset shard
common.py             split and sample loading, metrics, result files (numpy and PIL only)
tiling.py             tile geometry and the cross-fade merge for the tiled row
reproduce_table.sh    launcher for the subset check, the full table, or one row; cells.sh defines the cells
summarize.py          the table and the paired bootstrap from the per-sample results
check_reference.py    a run compared with results_reference/
make_data_split.py    verify or rebuild the KITTI-DC and DDAD subsets in data_split/
data_split/           the fixed KITTI-DC and DDAD subsets, one `rgb gt sparse` line per sample
datasets.json         dataset roots, split files and expected sample counts
results_reference/    the per-sample results behind every number above
```
