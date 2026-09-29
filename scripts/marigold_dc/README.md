# Marigold-DC depth-completion datasets

This workflow downloads and preprocesses the four depth-completion datasets
used by Marigold-DC: iBims-1, NYUv2, KITTI DC, and DDAD. The transformations
follow the upstream [Marigold-DC dataset instructions](https://github.com/prs-eth/Marigold-DC/blob/main/DATASETS.md).

## Download and preprocess

From the repository root:

```bash
bash scripts/marigold_dc/download_and_preprocess_depth_completion.sh
```

The default output is:
`${DEPTH_ASSETS_DIR:-assets}/datasets/marigold_depth_completion`.
Use `OUTPUT_DIR=/path/to/root` to select another location. Prepare only a
subset with:

```bash
bash scripts/marigold_dc/download_and_preprocess_depth_completion.sh \
  --datasets ibims1,nyudepthv2
```

Downloads resume when their `.part` files are present, completed archives are
skipped, and raw archives remain available for reproducibility. After a
successful run, `--clean-raw` removes only the selected archives and extracted
source directories; processed outputs are retained.

### Overrides

Set `OUTPUT_DIR` to change the processed dataset root and `DATASETS_DIR` to
change where raw archives and extracted sources are staged. Set `PYTHON` when
the active `python` does not provide the required dependencies, for example:

```bash
PYTHON=/path/to/python \
  bash scripts/marigold_dc/download_and_preprocess_depth_completion.sh
```

The iBims-1 download uses its upstream default rsync password. Set
`RSYNC_PASSWORD` only when a different credential is required.

The host needs `curl`, `rsync`, and `unzip`. Python dependencies are provided
by the project environment (`h5py`, NumPy, Pillow, SciPy, and tqdm).

## Sources

- iBims-1: [TUM MediaTUM record](https://mediatum.ub.tum.de/1455541), downloaded
  from `rsync://m1455541@dataserv.ub.tum.de/m1455541/ibims1_core_mat.zip`.
- NYUv2: Andrea Conti's [preprocessed HDF5 release](https://github.com/andreaconti/sparsity-agnostic-depth-completion/releases/tag/v0.1.0).
- KITTI DC: the [KITTI Depth Completion validation split](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion),
  downloaded from the `avg-kitti` S3 archive.
- DDAD: the [VPP4DC-preprocessed archive](https://drive.google.com/open?id=1y8Rt3Hld8zVTSKxx9d9yYXSzr5niKN7i)
  provided by the Marigold-DC instructions.

## Output layout

```text
marigold_depth_completion/
├── ibims1/
├── nyudepthv2/
├── kittidc/
└── ddad/
```

Each processed dataset contains matching `rgb/`, `sparse/`, `gt/`, and
`intrinsics/` directories. The expected sample counts are 100, 654, 1,000,
and 3,950 respectively. This root is separate from
`marigold_depth_eval/`, which contains the existing zero-shot monocular-depth
benchmarks.
