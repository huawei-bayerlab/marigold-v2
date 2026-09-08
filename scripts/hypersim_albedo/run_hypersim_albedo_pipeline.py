#!/usr/bin/env python3
"""Build the Hypersim RGB/albedo dataset from the Marigold V1.1 IID split.

Downloads only ``diffuse_reflectance`` per scene, clips it to [0, 1] like the
V1.1 preprocessor, and hard-links RGB from the preprocessed depth dataset.
Scenes are skipped on rerun once complete.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

ALBEDO_MODALITY = ".diffuse_reflectance.hdf5"
MIN_FREE_BYTES = 500_000_000_000  # ~135 GB staging + ~280 GB float32 targets
SAMPLE_RE = re.compile(
    r"^(?P<split>train|val|test)/(?P<scene>ai_\d{3}_\d{3})/"
    r"rgb_(?P<camera>cam_\d{2})_fr(?P<frame>\d{4})\.png$"
)
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Sample:
    split: str
    scene: str
    camera: str
    frame: int
    rgb_path: str
    albedo_path: str


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selective-downloader", type=Path, required=True)
    parser.add_argument("--filtered-list-dir", type=Path, required=True)
    parser.add_argument("--rgb-dataset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="float32 is lossless for Hypersim's float16 source",
    )
    return parser.parse_args()


def parse_sample_line(path: Path, line_number: int, line: str) -> Sample:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(
            f"{path}:{line_number}: expected five IID paths, got {len(fields)}"
        )
    rgb_path, albedo_path, shading_path, residual_path, stats_path = fields
    match = SAMPLE_RE.fullmatch(rgb_path)
    if match is None:
        raise ValueError(f"{path}:{line_number}: malformed RGB path: {rgb_path}")

    metadata = match.groupdict()
    frame = int(metadata["frame"])
    expected_stem = f"{metadata['camera']}_fr{frame:04d}"
    expected = (
        f"{metadata['split']}/{metadata['scene']}/albedo_{expected_stem}.npy",
        f"{metadata['split']}/{metadata['scene']}/shading_{expected_stem}.npy",
        f"{metadata['split']}/{metadata['scene']}/residual_{expected_stem}.npy",
        f"{metadata['split']}/{metadata['scene']}/shading_stats_{expected_stem}.json",
    )
    actual = (albedo_path, shading_path, residual_path, stats_path)
    if actual != expected:
        raise ValueError(
            f"{path}:{line_number}: modalities do not identify the same sample; "
            f"expected {expected}, got {actual}"
        )
    return Sample(
        split=metadata["split"],
        scene=metadata["scene"],
        camera=metadata["camera"],
        frame=frame,
        rgb_path=rgb_path,
        albedo_path=albedo_path,
    )


def read_list(path: Path) -> list[Sample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    samples = [
        parse_sample_line(path, line_number, line)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.strip()
    ]
    if not samples:
        raise ValueError(f"{path}: empty sample list")
    if len(samples) != len({sample.rgb_path for sample in samples}):
        raise ValueError(f"{path}: duplicate RGB entries")
    return samples


def load_curated_lists(
    filtered_list_dir: Path,
) -> tuple[dict[str, list[Sample]], dict[str, list[Sample]]]:
    named = {
        "train": read_list(filtered_list_dir / "hypersim_train_filtered.txt"),
        "test": read_list(filtered_list_dir / "hypersim_test.txt"),
        "val": read_list(filtered_list_dir / "hypersim_val.txt"),
        "vis": read_list(filtered_list_dir / "hypersim_vis.txt"),
    }
    canonical = named["train"] + named["test"]
    by_rgb = {sample.rgb_path: sample for sample in canonical}
    if len(by_rgb) != len(canonical):
        raise ValueError("train and test lists overlap")
    for list_name in ("val", "vis"):
        unknown = [
            sample.rgb_path
            for sample in named[list_name]
            if sample.rgb_path not in by_rgb
        ]
        if unknown:
            raise ValueError(
                f"{list_name}: {len(unknown)} samples are absent from train/test"
            )

    by_scene: dict[str, list[Sample]] = defaultdict(list)
    for sample in canonical:
        by_scene[sample.scene].append(sample)
    return named, dict(by_scene)


def check_free_space(output_dir: Path) -> None:
    free = shutil.disk_usage(output_dir.parent).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Need {MIN_FREE_BYTES / 1e9:.0f} GB free under {output_dir.parent}; "
            f"{free / 1e9:.0f} GB available"
        )


def raw_albedo_path(raw_dir: Path, sample: Sample) -> Path:
    return (
        raw_dir
        / sample.scene
        / "images"
        / f"scene_{sample.camera}_final_hdf5"
        / f"frame.{sample.frame:04d}.diffuse_reflectance.hdf5"
    )


def validate_nonempty(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(
            f"{label}: {len(missing)} missing/empty files; first paths:\n{preview}"
        )


def run_with_retries(command: list[str], log_path: Path, retries: int) -> None:
    for attempt in range(1, retries + 1):
        with log_path.open("a") as handle:
            handle.write(f"\nAttempt {attempt}: {' '.join(command)}\n")
            handle.flush()
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode == 0:
            return
        if attempt < retries:
            delay = min(60, 5 * (2 ** (attempt - 1)))
            log(f"[retry] command failed ({result.returncode}); retrying in {delay}s")
            time.sleep(delay)
    raise subprocess.CalledProcessError(result.returncode, command)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def save_albedo(source: Path, destination: Path, dtype: np.dtype, split: str) -> None:
    with h5py.File(source, "r") as handle:
        if "dataset" not in handle:
            raise ValueError(f"{source}: missing HDF5 dataset")
        albedo = np.asarray(handle["dataset"])
    if albedo.ndim != 3 or albedo.shape[-1] != 3:
        raise ValueError(f"{source}: expected HWC RGB albedo, got {albedo.shape}")

    # Same target as the V1.1 preprocessor; test samples may keep NaN pixels.
    albedo = np.clip(albedo.astype(dtype, copy=False), 0.0, 1.0)
    if split in {"train", "val"} and not np.isfinite(albedo).all():
        raise ValueError(f"{source}: non-finite value in curated training albedo")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, albedo, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def validate_scene_outputs(output_dir: Path, samples: list[Sample], scene: str) -> None:
    paths = [
        path
        for sample in samples
        for path in (output_dir / sample.rgb_path, output_dir / sample.albedo_path)
    ]
    validate_nonempty(paths, f"{scene} outputs")


def process_scene(
    scene: str,
    samples: list[Sample],
    args: argparse.Namespace,
) -> tuple[str, str]:
    raw_dir = args.work_dir / "raw"
    state_dir = args.work_dir / "state"
    scene_stage = args.output_dir / ".staging" / scene
    done_path = state_dir / "done" / f"{scene}.done"
    raw_ready_path = state_dir / "raw_ready" / f"{scene}.ready"
    scene_log = state_dir / "logs" / f"{scene}.log"

    if done_path.is_file():
        validate_scene_outputs(args.output_dir, samples, scene)
        return scene, "already complete"

    done_path.parent.mkdir(parents=True, exist_ok=True)
    raw_ready_path.parent.mkdir(parents=True, exist_ok=True)
    scene_log.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_raw = [raw_albedo_path(raw_dir, sample) for sample in samples]

    if raw_ready_path.is_file():
        validate_nonempty(expected_raw, f"{scene} reusable raw albedo")
    else:
        command = [
            sys.executable,
            str(args.selective_downloader),
            "--directory",
            str(raw_dir),
            "--scene",
            scene,
            "--contains",
            ALBEDO_MODALITY,
            "--overwrite",
            "--silent",
        ]
        run_with_retries(command, scene_log, args.download_retries)
        validate_nonempty(expected_raw, f"{scene} raw albedo")
        raw_ready_path.write_text(
            f"scene={scene}\nrequired_files={len(expected_raw)}\n"
        )

    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    scene_stage.mkdir(parents=True)
    dtype = np.dtype(args.output_dtype)
    for sample in samples:
        source_rgb = args.rgb_dataset_dir / sample.rgb_path
        if not source_rgb.is_file() or source_rgb.stat().st_size == 0:
            raise FileNotFoundError(f"missing preprocessed RGB: {source_rgb}")
        staged_rgb = scene_stage / Path(sample.rgb_path).name
        staged_albedo = scene_stage / Path(sample.albedo_path).name
        link_or_copy(source_rgb, staged_rgb)
        save_albedo(
            raw_albedo_path(raw_dir, sample), staged_albedo, dtype, sample.split
        )

    for split in {sample.split for sample in samples}:
        split_samples = [sample for sample in samples if sample.split == split]
        target_scene = args.output_dir / split / scene
        target_scene.parent.mkdir(parents=True, exist_ok=True)
        split_stage = scene_stage / split
        split_stage.mkdir()
        for sample in split_samples:
            for name in (Path(sample.rgb_path).name, Path(sample.albedo_path).name):
                os.replace(scene_stage / name, split_stage / name)
        if target_scene.exists():
            shutil.rmtree(target_scene)
        os.replace(split_stage, target_scene)

    validate_scene_outputs(args.output_dir, samples, scene)
    done_path.write_text(f"scene={scene}\nsamples={len(samples)}\n")

    raw_scene = raw_dir / scene
    if raw_scene.exists():
        shutil.rmtree(raw_scene)
    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    return scene, f"processed {len(samples)} samples"


def write_manifests(output_dir: Path, named: dict[str, list[Sample]]) -> None:
    output_names = {
        "train": "hypersim_filtered_train.txt",
        "test": "hypersim_filtered_test.txt",
        "val": "hypersim_filtered_val.txt",
        "vis": "hypersim_filtered_vis.txt",
    }
    for list_name, filename in output_names.items():
        lines = [
            f"{sample.rgb_path} {sample.albedo_path}" for sample in named[list_name]
        ]
        (output_dir / filename).write_text("\n".join(lines) + "\n")
    canonical = named["train"] + named["test"]
    (output_dir / "hypersim_filtered_all.txt").write_text(
        "\n".join(f"{sample.rgb_path} {sample.albedo_path}" for sample in canonical)
        + "\n"
    )


def validate_dataset(output_dir: Path, named: dict[str, list[Sample]]) -> None:
    canonical = named["train"] + named["test"]
    paths = [
        path
        for sample in canonical
        for path in (output_dir / sample.rgb_path, output_dir / sample.albedo_path)
    ]
    validate_nonempty(paths, "final RGB/albedo dataset")
    if len(named["train"]) != 23_842 or len(named["test"]) != 5_238:
        raise RuntimeError(
            "upstream IID split count changed: "
            f"train={len(named['train'])}, test={len(named['test'])}"
        )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in (
        args.selective_downloader,
        args.filtered_list_dir,
        args.rgb_dataset_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".staging").mkdir(exist_ok=True)
    check_free_space(args.output_dir)
    named, by_scene = load_curated_lists(args.filtered_list_dir)
    log(
        f"Starting {len(by_scene)} scenes with {args.workers} workers; "
        f"{len(named['train'])} train, {len(named['test'])} test, "
        f"{len(named['val'])} validation-subset samples"
    )

    failures: list[tuple[str, BaseException]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_scene, scene, samples, args): scene
            for scene, samples in sorted(by_scene.items())
        }
        for future in as_completed(futures):
            scene = futures[future]
            try:
                _, status = future.result()
                completed += 1
                log(f"[{completed}/{len(futures)}] {scene}: {status}")
            except BaseException as error:
                failures.append((scene, error))
                log(f"[FAILED] {scene}: {error!r}")

    if failures:
        for scene, error in failures:
            log(f"Failure summary: {scene}: {error!r}")
        raise RuntimeError(
            f"{len(failures)} scenes failed; rerun to retry incomplete scenes"
        )

    write_manifests(args.output_dir, named)
    validate_dataset(args.output_dir, named)
    shutil.rmtree(args.output_dir / ".staging")
    (args.output_dir / "PREPROCESS_COMPLETE").write_text(
        f"scenes={len(by_scene)}\ntrain_samples={len(named['train'])}\n"
        f"test_samples={len(named['test'])}\nval_subset_samples={len(named['val'])}\n"
        f"dtype={args.output_dtype}\n"
    )
    log("All RGB/albedo pairs validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
