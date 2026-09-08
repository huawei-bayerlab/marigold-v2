# Hypersim surface-normal dataset

Builds `assets/datasets/marigold_train_normals` with the Marigold V1.1 Hypersim
normals preprocessor (pinned commit), scene by scene, from the curated lists in
`evaluation/data_split/hypersim_normals`. Raw HDF5 files are deleted after each
scene, and finished scenes are skipped on rerun.

```bash
bash scripts/hypersim_normals/download_and_preprocess_hypersim_normals.sh
```

Needs about 1.2 TB for the output and 200 GB of staging space. Defaults can be
overridden through the environment, e.g.
`WORKERS=8 OUTPUT_DIR=/data/marigold_train_normals bash scripts/hypersim_normals/download_and_preprocess_hypersim_normals.sh`.

Output: RGB/normal pairs under `<split>/<scene>/` plus the manifests
`hypersim_filtered_{train,val,test,all}.txt`.
