#!/usr/bin/env python3
"""Download models, checkpoints, prompt embeddings, and datasets into assets/.

Resumable: existing files are skipped, partial downloads continue, and tar
shards are extracted in place (and deleted unless --keep-archives).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import fnmatch
import io
import multiprocessing
import os
import tarfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image

# Finite socket timeouts so a stalled connection fails and gets retried.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import (
    get_token,
    hf_hub_download,
    list_repo_files,
    snapshot_download,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets"
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_WORKERS = 4
DOWNLOAD_FILE_TIMEOUT = int(os.environ.get("DEPTH_DOWNLOAD_FILE_TIMEOUT", "900"))
LAYERED_DEPTH_PREPARE_WORKERS = max(
    1,
    int(os.environ.get("DEPTH_PREPARE_WORKERS", str(min(8, os.cpu_count() or 1)))),
)
LAYERED_DEPTH_NUM_LAYERS = 8
LAYERED_DEPTH_INVALID_MIN = 30000


def _nonempty_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "--assets-dir is empty (unset shell variable?)"
        )
    return Path(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=_nonempty_path,
        default=None,
        help="Asset root, relative to the repository (default: assets/ or $DEPTH_ASSETS_DIR).",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset snapshots and extraction.",
    )
    parser.add_argument(
        "--skip-training-data",
        action="store_true",
        help="Download evaluation datasets but skip the Hypersim training dataset.",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Skip model and fine-tuned checkpoint downloads.",
    )
    parser.add_argument(
        "--include-dinov3",
        action="store_true",
        help="Also download the gated DINOv3 checkpoint used by iREPA (needs `hf auth login`).",
    )
    parser.add_argument(
        "--include-layereddepth-syn",
        action="store_true",
        help=(
            "Also download and preprocess the optional gated LayeredDepth-Syn "
            "dataset into dense per-layer PNGs and manifests."
        ),
    )
    parser.add_argument(
        "--layereddepth-layers",
        type=int,
        nargs="+",
        choices=range(1, LAYERED_DEPTH_NUM_LAYERS + 1),
        default=[LAYERED_DEPTH_NUM_LAYERS],
        metavar="LAYER",
        help=(
            "LayeredDepth-Syn layers to materialize (default: layer 8). "
            "Requested layer k still reads sparse layers 1..k."
        ),
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep downloaded tar archives without extracting them.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep tar archives after successful extraction (archives are deleted by default).",
    )
    return parser.parse_args()


def _download_snapshot(
    repo_id: str,
    destination: Path,
    *,
    repo_type: str,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    watchdog: bool = False,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    effective_ignore_patterns = list(ignore_patterns or [])
    # Archives deleted after extraction leave a .<name>.extracted marker; skip them.
    marker_suffix = ".extracted"
    for marker in destination.rglob(f".*.tar{marker_suffix}"):
        archive_name = marker.name[1 : -len(marker_suffix)]
        relative_archive = (marker.parent / archive_name).relative_to(destination)
        effective_ignore_patterns.append(relative_archive.as_posix())
    if watchdog:
        _download_snapshot_with_watchdog(
            repo_id,
            destination,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            ignore_patterns=effective_ignore_patterns,
        )
        return

    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        print(
            f"[download] {repo_id} -> {destination} "
            f"(attempt {attempt}/{DOWNLOAD_ATTEMPTS})",
            flush=True,
        )
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(destination),
                allow_patterns=allow_patterns,
                ignore_patterns=effective_ignore_patterns,
                max_workers=DOWNLOAD_WORKERS,
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                delay = 2**attempt
                print(
                    f"[retry] {repo_id} failed ({exc!r}); resuming in {delay}s",
                    flush=True,
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {repo_id} after {DOWNLOAD_ATTEMPTS} attempts. "
        "Gated repositories need `hf auth login` and accepted terms."
    ) from last_exc


def _download_file_worker(
    repo_id: str,
    filename: str,
    destination: str,
    repo_type: str,
) -> None:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=destination,
    )


def _download_snapshot_with_watchdog(
    repo_id: str,
    destination: Path,
    *,
    repo_type: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> None:
    """Download files one by one in child processes with a hard per-file timeout.

    ``snapshot_download`` can hang on a stalled connection; a killed child
    process is retried and resumes from the partial file.
    """
    files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
    selected = [
        filename
        for filename in files
        if (
            not allow_patterns
            or any(fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns)
        )
        and (
            not ignore_patterns
            or not any(
                fnmatch.fnmatch(filename, pattern) for pattern in ignore_patterns
            )
        )
    ]
    if not selected:
        raise RuntimeError(f"No files matched the requested patterns in {repo_id}")

    context = multiprocessing.get_context("spawn")
    for index, filename in enumerate(selected, start=1):
        target = destination / filename
        marker = target.with_name(f".{target.name}.extracted")
        if target.is_file() or marker.is_file():
            print(f"[skip] {repo_id} {index}/{len(selected)} {filename}", flush=True)
            continue

        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            print(
                f"[download] {repo_id} {index}/{len(selected)} {filename} "
                f"(attempt {attempt}/{DOWNLOAD_ATTEMPTS}, timeout {DOWNLOAD_FILE_TIMEOUT}s)",
                flush=True,
            )
            process = context.Process(
                target=_download_file_worker,
                args=(repo_id, filename, str(destination), repo_type),
            )
            process.start()
            process.join(DOWNLOAD_FILE_TIMEOUT)
            if process.is_alive():
                print(
                    f"[timeout] {repo_id} {filename}; terminating stalled transfer",
                    flush=True,
                )
                process.terminate()
                process.join(30)
                if process.is_alive():
                    process.kill()
                    process.join()
            elif process.exitcode == 0 and target.is_file():
                break

            if attempt < DOWNLOAD_ATTEMPTS:
                delay = 2**attempt
                print(f"[retry] {repo_id} {filename}; resuming in {delay}s", flush=True)
                time.sleep(delay)
        else:
            raise RuntimeError(
                f"Failed to download {repo_id}/{filename} after {DOWNLOAD_ATTEMPTS} attempts. "
                "Rerun to resume."
            )


def _safe_members(archive: Path) -> Iterable[tarfile.TarInfo]:
    root = archive.parent.resolve()
    with tarfile.open(archive, mode="r") as tar:
        members = tar.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise RuntimeError(f"Unsafe path in archive {archive}: {member.name}")
            # Skip the Hugging Face upload cache packed into some shards.
            if member.name == ".cache" or member.name.startswith(".cache/"):
                continue
            yield member


def _extract_archives(root: Path, *, keep_archives: bool = False) -> None:
    archives = sorted(root.rglob("*.tar"))
    for archive in archives:
        marker = archive.with_name(f".{archive.name}.extracted")
        if marker.exists():
            if not keep_archives and archive.exists():
                print(f"[cleanup] {archive}", flush=True)
                archive.unlink()
            continue
        print(f"[extract] {archive}", flush=True)
        members = list(_safe_members(archive))
        with tarfile.open(archive, mode="r") as tar:
            tar.extractall(path=archive.parent, members=members)
        marker.touch()
        if not keep_archives:
            print(f"[cleanup] {archive}", flush=True)
            archive.unlink()


def _write_layered_png(path: Path, image_array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(path, format="PNG")


def _decode_layered_png(value: dict, *, depth: bool) -> np.ndarray:
    """Decode a Hugging Face Image feature returned from a parquet row."""
    if not isinstance(value, dict) or not value.get("bytes"):
        raise RuntimeError("LayeredDepth-Syn parquet row contains an empty image")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        if depth:
            return np.array(image)
        return np.array(image.convert("RGB"))


def _write_layereddepth_manifests(
    split_root: Path, records: list[tuple[str, str]], layers: tuple[int, ...]
) -> None:
    fieldnames = ["scene", "camera", "frame", "rgb_path", "depth_path", "mask_path"]
    for layer in layers:
        manifest_path = split_root / f"manifest_layer{layer}.csv"
        with manifest_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            writer.writerows(
                (
                    scene,
                    "cam",
                    frame,
                    f"rgb/{scene}_{frame}.png",
                    f"depth_l{layer}/{scene}_{frame}.png",
                    f"rgb/{scene}_{frame}.png",
                )
                for scene, frame in records
            )


def _prepare_layereddepth_shard(
    split_root: Path, parquet_path: Path, layers: tuple[int, ...]
) -> list[tuple[str, str]]:
    """Extract one parquet shard and create its dense depth layers.

    Raw LayeredDepth-Syn depth files are sparse: layer ``k`` only contains
    pixels belonging to that layer.  For each requested layer, fill zero
    pixels from the next lower layer so that layer 8 represents the complete
    scene depth behind the transparent object.
    """
    records: list[tuple[str, str]] = []
    max_layer = max(layers)
    columns = (
        ["image.png"]
        + [f"depth_{layer}.png" for layer in range(1, max_layer + 1)]
        + ["__key__"]
    )
    parquet_file = parquet.ParquetFile(parquet_path)
    missing_columns = set(columns) - set(parquet_file.schema_arrow.names)
    if missing_columns:
        raise RuntimeError(
            f"{parquet_path} is missing LayeredDepth-Syn columns: "
            f"{sorted(missing_columns)}"
        )
    scene = parquet_path.stem
    print(f"[prepare] {parquet_path}", flush=True)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=32):
        for row in batch.to_pylist():
            frame = str(row["__key__"])
            if Path(frame).name != frame or frame in {"", ".", ".."}:
                raise RuntimeError(f"Unsafe LayeredDepth-Syn frame key: {frame!r}")
            file_stem = f"{scene}_{frame}"

            # Rewrite outputs until the completion marker exists, repairing
            # directories produced by the old sparse depth_8 extraction.
            rgb = _decode_layered_png(row["image.png"], depth=False)
            rgb_path = split_root / "rgb" / f"{file_stem}.png"
            _write_layered_png(rgb_path, rgb)

            raw_depths: list[np.ndarray] = []
            for layer in range(1, max_layer + 1):
                depth = _decode_layered_png(row[f"depth_{layer}.png"], depth=True)
                if depth.ndim != 2:
                    raise RuntimeError(
                        f"Expected a single-channel depth image in {parquet_path}, "
                        f"row {frame}, layer {layer}; got shape {depth.shape}"
                    )
                depth = depth.astype(np.uint16, copy=True)
                # LayeredDepth-Syn reserves the upper uint16 range for invalid
                # depth sentinels; they must not block fill-forward.
                depth[depth >= LAYERED_DEPTH_INVALID_MIN] = 0
                raw_depths.append(depth)

            previous_dense: np.ndarray | None = None
            for target_layer, raw_depth in enumerate(raw_depths, start=1):
                dense_depth = raw_depth.copy()
                if previous_dense is not None:
                    fill_mask = (dense_depth == 0) & (previous_dense > 0)
                    dense_depth[fill_mask] = previous_dense[fill_mask]
                if target_layer in layers:
                    depth_path = (
                        split_root / f"depth_l{target_layer}" / f"{file_stem}.png"
                    )
                    _write_layered_png(depth_path, dense_depth)
                previous_dense = dense_depth

            records.append((scene, frame))

    return records


def _prepare_layereddepth_split(
    split_root: Path,
    parquet_files: list[Path],
    layers: tuple[int, ...],
) -> int:
    """Prepare parquet shards concurrently and write manifests once."""
    split_root.mkdir(parents=True, exist_ok=True)
    worker_count = min(LAYERED_DEPTH_PREPARE_WORKERS, len(parquet_files))
    worker_args = [(split_root, parquet_path, layers) for parquet_path in parquet_files]
    records: list[tuple[str, str]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        for shard_records in executor.map(
            _prepare_layereddepth_shard, *zip(*worker_args)
        ):
            records.extend(shard_records)

    if not records:
        raise RuntimeError(f"No parquet rows found under {split_root.parent / 'data'}")
    _write_layereddepth_manifests(split_root, records, layers)
    return len(records)


def _prepare_layereddepth_syn(dataset_root: Path, layers: tuple[int, ...]) -> None:
    """Convert LayeredDepth-Syn parquet shards to the training layout."""
    layer_tag = "-".join(str(layer) for layer in layers)
    marker = dataset_root / f".layereddepth_syn_dense_v5_{layer_tag}"
    if marker.is_file():
        print(
            f"[skip] LayeredDepth-Syn is already prepared: {dataset_root}",
            flush=True,
        )
        return

    data_root = dataset_root / "data"
    split_files = {
        "train": sorted(data_root.glob("train-*.parquet")),
        "val": sorted(data_root.glob("validation-*.parquet")),
    }
    missing_splits = [split for split, files in split_files.items() if not files]
    if missing_splits:
        raise RuntimeError(
            "LayeredDepth-Syn parquet shards are incomplete; missing "
            f"{', '.join(missing_splits)} under {data_root}."
        )

    counts = {
        split: _prepare_layereddepth_split(
            dataset_root / "extracted" / split, files, layers
        )
        for split, files in split_files.items()
    }

    marker.write_text(
        "dense per-layer LayeredDepth-Syn preparation complete\n"
        f"train_rows={counts['train']}\n"
        f"val_rows={counts['val']}\n"
    )
    print(
        "[prepare] LayeredDepth-Syn dense layout complete: "
        f"{counts['train']} train rows, {counts['val']} validation rows",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    assets_arg = args.assets_dir
    if assets_arg is None:
        assets_arg = Path(os.environ.get("DEPTH_ASSETS_DIR", DEFAULT_ASSETS_DIR))
    assets_arg = assets_arg.expanduser()
    if not assets_arg.is_absolute():
        assets_arg = REPO_ROOT / assets_arg
    assets = assets_arg.resolve()
    checkpoints = assets / "checkpoints"
    datasets = assets / "datasets"
    if not args.skip_checkpoints:
        _download_snapshot(
            "Qwen/Qwen-Image-Edit-2509",
            checkpoints / "Qwen-Image-Edit-2509",
            repo_type="model",
        )
        if args.include_dinov3:
            if get_token():
                _download_snapshot(
                    "facebook/dinov3-vitb16-pretrain-lvd1689m",
                    checkpoints / "dinov3-vitb16-pretrain-lvd1689m",
                    repo_type="model",
                )
            else:
                print(
                    "[skip] DINOv3 was requested for iREPA, but no Hugging Face "
                    "token is available. Run `hf auth login`, accept the DINOv3 "
                    "terms, and rerun with `--include-dinov3`.",
                    flush=True,
                )
        _download_snapshot(
            "huawei-bayerlab/marigold-v2-0",
            checkpoints / "Marigold-V2",
            repo_type="model",
        )

    if not args.skip_datasets:
        if not args.skip_training_data:
            hypersim_root = datasets / "marigold_train"
            _download_snapshot(
                "obukhovai/hypersim_relative_depth",
                hypersim_root,
                repo_type="dataset",
                allow_patterns=["hypersim-*.tar"],
                watchdog=True,
            )
            if not any(hypersim_root.glob("hypersim-*.tar")):
                raise RuntimeError(f"No hypersim-*.tar shards found in {hypersim_root}")
        _download_snapshot(
            "obukhovai/vkitti2",
            datasets / "vkitti2",
            repo_type="dataset",
            watchdog=True,
        )
        _download_snapshot(
            "obukhovai/marigold_depth_eval",
            datasets / "marigold_depth_eval",
            repo_type="dataset",
        )
        _download_snapshot(
            "obukhovai/marigold_normals_eval",
            datasets / "marigold_normals_eval",
            repo_type="dataset",
        )
        if args.include_layereddepth_syn:
            layereddepth_root = datasets / "LayeredDepth-Syn"
            _download_snapshot(
                "princeton-vl/LayeredDepth-Syn",
                layereddepth_root,
                repo_type="dataset",
            )
            layers = tuple(sorted(set(args.layereddepth_layers)))
            _prepare_layereddepth_syn(layereddepth_root, layers)
        if not args.no_extract:
            if not args.skip_training_data:
                _extract_archives(
                    datasets / "marigold_train", keep_archives=args.keep_archives
                )
            _extract_archives(datasets / "vkitti2", keep_archives=args.keep_archives)
            _extract_archives(
                datasets / "marigold_depth_eval", keep_archives=args.keep_archives
            )

    print(f"[done] Assets are available under {assets}", flush=True)


if __name__ == "__main__":
    main()
