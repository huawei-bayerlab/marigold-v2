#!/usr/bin/env python3
"""Download Hypersim geometry per scene and run the Marigold V1.1 normals preprocessor.

Scenes are processed independently and skipped on rerun once complete.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

MIN_OUTPUT_BYTES = 1_200_000_000_000
MIN_WORK_BYTES = 200_000_000_000

# Fixes for the pinned upstream preprocessor: per-scene calls produce empty
# splits (pandas 2.3 fails on them) and some frames have non-finite positions.
UPSTREAM_PATCHES = (
    (
        """            lines = split_meta_df.apply(
                lambda r: f"{r['rgb_path']} {r['normal_path']}", axis=1
            ).tolist()
            f.writelines("\\n".join(lines))
""",
        """            if split_meta_df.empty:
                lines = []
            else:
                lines = split_meta_df.apply(
                    lambda r: f"{r['rgb_path']} {r['normal_path']}", axis=1
                ).tolist()
            f.writelines("\\n".join(lines))
""",
    ),
    (
        """        split_meta_df["rgb_path"] = None
        split_meta_df["rgb_mean"] = np.nan
""",
        """        split_meta_df["rgb_path"] = None
        split_meta_df["normal_path"] = None
        split_meta_df["rgb_mean"] = np.nan
""",
    ),
    (
        """                surface_to_cam_world_normalized_1d_ = sklearn.preprocessing.normalize(
                    camera_position - position_1d_
                )
""",
        """                surface_to_cam_world_1d_ = camera_position - position_1d_
                surface_to_cam_norm_1d_ = np.linalg.norm(
                    surface_to_cam_world_1d_, axis=1
                )
                surface_to_cam_valid_1d_ = (
                    np.all(np.isfinite(surface_to_cam_world_1d_), axis=1)
                    & np.isfinite(surface_to_cam_norm_1d_)
                    & (surface_to_cam_norm_1d_ > 0)
                )
                surface_to_cam_world_normalized_1d_ = np.zeros_like(
                    surface_to_cam_world_1d_
                )
                surface_to_cam_world_normalized_1d_[surface_to_cam_valid_1d_] = (
                    surface_to_cam_world_1d_[surface_to_cam_valid_1d_]
                    / surface_to_cam_norm_1d_[surface_to_cam_valid_1d_, None]
                )
""",
    ),
)

REQUIRED_MODALITIES = (
    ".color.hdf5",
    ".normal_cam.hdf5",
    ".normal_world.hdf5",
    ".position.hdf5",
    ".render_entity_id.hdf5",
    "camera_keyframe_positions.hdf5",
)

PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--selective-downloader", type=Path, required=True)
    parser.add_argument("--preprocess-script", type=Path, required=True)
    parser.add_argument("--filtered-list-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--download-retries", type=int, default=5)
    return parser.parse_args()


def patch_upstream_preprocessor(path: Path) -> None:
    source = path.read_text()
    for old, new in UPSTREAM_PATCHES:
        if old in source:
            source = source.replace(old, new, 1)
        elif new not in source:
            raise RuntimeError(f"Could not patch {path}; upstream script changed")
    path.write_text(source)


def check_free_space(output_dir: Path, work_dir: Path) -> None:
    output_parent, work_parent = output_dir.parent, work_dir.parent
    output_free = shutil.disk_usage(output_parent).free
    work_free = shutil.disk_usage(work_parent).free
    if output_parent.stat().st_dev == work_parent.stat().st_dev:
        if output_free < MIN_OUTPUT_BYTES + MIN_WORK_BYTES:
            raise RuntimeError(
                f"Need {(MIN_OUTPUT_BYTES + MIN_WORK_BYTES) / 1e12:.1f} TB free under "
                f"{output_parent}; {output_free / 1e12:.2f} TB available"
            )
    elif output_free < MIN_OUTPUT_BYTES or work_free < MIN_WORK_BYTES:
        raise RuntimeError(
            f"Need {MIN_OUTPUT_BYTES / 1e12:.1f} TB under {output_parent} and "
            f"{MIN_WORK_BYTES / 1e9:.0f} GB under {work_parent}"
        )


def load_public_rows(
    metadata_csv: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    rows_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_by_scene: dict[str, str] = {}
    with metadata_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["included_in_public_release"].lower() != "true":
                continue
            scene = row["scene_name"]
            split = row["split_partition_name"]
            previous_split = split_by_scene.setdefault(scene, split)
            if previous_split != split:
                raise ValueError(f"{scene} occurs in both {previous_split} and {split}")
            rows_by_scene[scene].append(row)
    return dict(rows_by_scene), split_by_scene


def load_curated_samples(
    filtered_list_dir: Path,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    by_scene: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_split: dict[str, list[tuple[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    # Two-column (rgb normal) list from the Marigold V1.1 README.
    path = filtered_list_dir / "hypersim_filtered_all.txt"
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected two paths")
        rgb_path, normal_path = fields
        split = Path(rgb_path).parts[0]
        if split not in by_split or Path(normal_path).parts[0] != split:
            raise ValueError(f"{path}:{line_number}: invalid or mismatched split")
        if Path(rgb_path).parts[1] != Path(normal_path).parts[1]:
            raise ValueError(f"{path}:{line_number}: RGB/normal scene mismatch")
        sample = (rgb_path, normal_path)
        by_split[split].append(sample)
        by_scene[Path(rgb_path).parts[1]].append(sample)
    return dict(by_scene), by_split


def expected_raw_paths(
    scene: str, rows: list[dict[str, str]], raw_dir: Path
) -> list[Path]:
    paths: list[Path] = []
    camera_keyframes: set[str] = set()
    for row in rows:
        camera = row["camera_name"]
        frame = int(row["frame_id"])
        final_dir = raw_dir / scene / "images" / f"scene_{camera}_final_hdf5"
        geometry_dir = raw_dir / scene / "images" / f"scene_{camera}_geometry_hdf5"
        paths.extend(
            (
                final_dir / f"frame.{frame:04d}.color.hdf5",
                geometry_dir / f"frame.{frame:04d}.normal_cam.hdf5",
                geometry_dir / f"frame.{frame:04d}.normal_world.hdf5",
                geometry_dir / f"frame.{frame:04d}.position.hdf5",
                geometry_dir / f"frame.{frame:04d}.render_entity_id.hdf5",
            )
        )
        camera_keyframes.add(camera)
    paths.extend(
        raw_dir / scene / "_detail" / camera / "camera_keyframe_positions.hdf5"
        for camera in camera_keyframes
    )
    return paths


def validate_nonempty_files(paths: list[Path], label: str) -> None:
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


def write_scene_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_curated_scene_outputs(
    output_dir: Path, samples: list[tuple[str, str]], scene: str
) -> None:
    paths = [output_dir / relative for pair in samples for relative in pair]
    validate_nonempty_files(paths, f"{scene} curated outputs")


def process_scene(
    scene: str,
    rows: list[dict[str, str]],
    split: str,
    curated_samples: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[str, str]:
    raw_dir = args.work_dir / "raw"
    scene_stage = args.output_dir / ".staging" / scene
    state_dir = args.work_dir / "state"
    done_path = state_dir / "done" / f"{scene}.done"
    raw_ready_path = state_dir / "raw_ready" / f"{scene}.ready"
    scene_log = state_dir / "logs" / f"{scene}.log"
    meta_path = state_dir / "metadata" / f"{scene}.csv"
    target_scene_dir = args.output_dir / split / scene

    if done_path.is_file():
        validate_curated_scene_outputs(args.output_dir, curated_samples, scene)
        return scene, "already complete"

    scene_log.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    raw_ready_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    expected_raw = expected_raw_paths(scene, rows, raw_dir)
    if raw_ready_path.is_file():
        validate_nonempty_files(expected_raw, f"{scene} reusable raw inputs")
        log(f"[reuse] {scene}: keeping previously validated HDF5 inputs")
    else:
        # --overwrite replaces partial files left by an interrupted download.
        for modality in REQUIRED_MODALITIES:
            command = [
                sys.executable,
                str(args.selective_downloader),
                "--directory",
                str(raw_dir),
                "--scene",
                scene,
                "--contains",
                modality,
                "--overwrite",
                "--silent",
            ]
            run_with_retries(command, scene_log, args.download_retries)
        validate_nonempty_files(expected_raw, f"{scene} raw inputs")
        raw_ready_path.write_text(f"scene={scene}\nfiles={len(expected_raw)}\n")

    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    scene_stage.mkdir(parents=True)
    scene_csv = args.work_dir / "state" / "scene_csv" / f"{scene}.csv"
    write_scene_csv(scene_csv, rows)

    command = [
        sys.executable,
        str(args.preprocess_script),
        "--split_csv",
        str(scene_csv),
        "--dataset_dir",
        str(raw_dir),
        "--output_dir",
        str(scene_stage),
    ]
    with scene_log.open("a") as handle:
        handle.write(f"\nPreprocess: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    source_scene_dir = scene_stage / split / scene
    if not source_scene_dir.is_dir() and curated_samples:
        raise RuntimeError(f"{scene}: preprocessor did not create {source_scene_dir}")

    per_scene_meta = scene_stage / split / f"filename_meta_{split}.csv"
    if not per_scene_meta.is_file():
        raise RuntimeError(f"{scene}: preprocessor did not create {per_scene_meta}")
    shutil.copy2(per_scene_meta, meta_path)

    if source_scene_dir.is_dir():
        target_scene_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_scene_dir.exists():  # moved before the done marker was written
            validate_curated_scene_outputs(args.output_dir, curated_samples, scene)
            shutil.rmtree(source_scene_dir)
        else:
            os.replace(source_scene_dir, target_scene_dir)
            validate_curated_scene_outputs(args.output_dir, curated_samples, scene)
    else:
        log(f"[empty] {scene}: upstream skipped every NaN-normal frame")

    done_path.write_text(
        f"scene={scene}\nsplit={split}\npublic_rows={len(rows)}\n"
        f"curated_samples={len(curated_samples)}\n"
    )

    raw_scene_dir = raw_dir / scene
    if raw_scene_dir.exists():
        shutil.rmtree(raw_scene_dir)
    if scene_stage.exists():
        shutil.rmtree(scene_stage)
    return scene, f"processed {len(rows)} public rows"


def finalize_metadata(
    output_dir: Path,
    work_dir: Path,
    metadata_csv: Path,
    split_by_scene: dict[str, str],
    curated_by_split: dict[str, list[tuple[str, str]]],
) -> None:
    metadata_dir = work_dir / "state" / "metadata"
    source_metadata = pd.read_csv(metadata_csv)
    source_metadata["_source_index"] = source_metadata.index
    source_index = source_metadata[
        ["scene_name", "camera_name", "frame_id", "_source_index"]
    ]
    for split in ("train", "val", "test"):
        split_scenes = sorted(
            scene for scene, value in split_by_scene.items() if value == split
        )
        frames = [pd.read_csv(metadata_dir / f"{scene}.csv") for scene in split_scenes]
        metadata = pd.concat(frames, ignore_index=True)
        metadata = metadata.drop(columns=["Unnamed: 0"], errors="ignore")
        metadata = metadata.merge(
            source_index,
            on=["scene_name", "camera_name", "frame_id"],
            how="left",
            validate="one_to_one",
        )
        if metadata["_source_index"].isna().any():
            raise RuntimeError(
                f"{split}: could not recover all source metadata indices"
            )
        metadata = metadata.set_index("_source_index").sort_index()
        metadata.index.name = None
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(split_dir / f"filename_meta_{split}.csv")

        # The curated lists exclude the NaN-normal frames the preprocessor skips.
        local_lines = []
        for rgb_path, normal_path in curated_by_split[split]:
            local_rgb = Path(*Path(rgb_path).parts[1:])
            local_normal = Path(*Path(normal_path).parts[1:])
            local_lines.append(f"{local_rgb} {local_normal}")
        (split_dir / f"hypersim_filtered_{split}.txt").write_text(
            "\n".join(local_lines) + "\n"
        )


def write_and_validate_root_manifests(
    output_dir: Path,
    curated_by_split: dict[str, list[tuple[str, str]]],
) -> None:
    all_lines: list[str] = []
    for split in ("train", "val", "test"):
        lines = [f"{rgb} {normal}" for rgb, normal in curated_by_split[split]]
        (output_dir / f"hypersim_filtered_{split}.txt").write_text(
            "\n".join(lines) + "\n"
        )
        all_lines.extend(lines)
    (output_dir / "hypersim_filtered_all.txt").write_text("\n".join(all_lines) + "\n")

    all_paths = [
        output_dir / relative
        for samples in curated_by_split.values()
        for pair in samples
        for relative in pair
    ]
    validate_nonempty_files(all_paths, "final curated dataset")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in (
        args.metadata_csv,
        args.selective_downloader,
        args.preprocess_script,
        args.filtered_list_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.output_dir / split).mkdir(exist_ok=True)
    patch_upstream_preprocessor(args.preprocess_script)
    check_free_space(args.output_dir, args.work_dir)

    rows_by_scene, split_by_scene = load_public_rows(args.metadata_csv)
    curated_by_scene, curated_by_split = load_curated_samples(args.filtered_list_dir)
    unknown_scenes = set(curated_by_scene) - set(rows_by_scene)
    if unknown_scenes:
        raise RuntimeError(
            f"curated list contains unknown scenes: {sorted(unknown_scenes)}"
        )

    log(
        f"Starting {len(rows_by_scene)} public scenes with {args.workers} workers; "
        f"{sum(map(len, rows_by_scene.values()))} public frames and "
        f"{sum(map(len, curated_by_split.values()))} curated samples"
    )

    failures: list[tuple[str, BaseException]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_scene,
                scene,
                rows,
                split_by_scene[scene],
                curated_by_scene.get(scene, []),
                args,
            ): scene
            for scene, rows in sorted(rows_by_scene.items())
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

    finalize_metadata(
        args.output_dir,
        args.work_dir,
        args.metadata_csv,
        split_by_scene,
        curated_by_split,
    )
    write_and_validate_root_manifests(args.output_dir, curated_by_split)
    (args.output_dir / "PREPROCESS_COMPLETE").write_text(
        f"public_scenes={len(rows_by_scene)}\n"
        f"public_frames={sum(map(len, rows_by_scene.values()))}\n"
        f"curated_samples={sum(map(len, curated_by_split.values()))}\n"
    )
    log("All scenes and all curated RGB/normal pairs validated successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
