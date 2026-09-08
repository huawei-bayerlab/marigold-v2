# Hypersim albedo dataset

Builds `assets/datasets/marigold_train_albedo` following the Marigold V1.1
IID-lighting preprocessing: albedo is Hypersim's linear-RGB
`diffuse_reflectance`, clipped to `[0, 1]` and stored as HWC float32 `.npy`.
Only the reflectance files are downloaded; RGB images are hard-linked from the
preprocessed depth dataset (`marigold_train`), which must exist first. Finished
scenes are skipped on rerun.

```bash
bash scripts/hypersim_albedo/download_and_preprocess_hypersim_albedo.sh
```

Needs about 500 GB of free space. Defaults can be overridden through the
environment, e.g. `WORKERS=8 RGB_DATASET_DIR=/data/marigold_train bash scripts/hypersim_albedo/download_and_preprocess_hypersim_albedo.sh`.

Output manifests (`rgb_path albedo_path`): `hypersim_filtered_train.txt`
(23,842 pairs), `hypersim_filtered_test.txt` (5,238), `hypersim_filtered_val.txt`
(70-image subset of test), `hypersim_filtered_vis.txt` (5 images), and
`hypersim_filtered_all.txt`.
