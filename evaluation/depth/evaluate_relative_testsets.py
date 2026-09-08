#!/usr/bin/env python3
"""Run depth inference on the five Marigold V1 zero-shot benchmarks.

Datasets: DIODE, ETH3D, KITTI, NYUv2, ScanNet. Writes one .npy per image,
which ``eval.py`` then scores.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from marigoldv2.util.config_resolvers import recursive_load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HF_REPO_RE = re.compile(r"^(?:hf://)?([a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+)(?:/(.*))?$")


def _resolve_checkpoint_path(checkpoint_path: str | None) -> str | None:
    """Resolve a local path or HuggingFace repo ID to a local directory."""
    if checkpoint_path is None:
        return None
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    m = _HF_REPO_RE.match(checkpoint_path)
    if m is None:
        return checkpoint_path
    repo_id = m.group(1)
    subfolder = m.group(2)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to load checkpoints from the HF Hub. "
            "Install it with: pip install huggingface_hub"
        )
    kwargs = {"repo_id": repo_id}
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**/*"]
    print(f"[checkpoint] Downloading HuggingFace checkpoint '{repo_id}' ...")
    local_dir = snapshot_download(**kwargs)
    if subfolder:
        local_dir = os.path.join(local_dir, subfolder)
    print(f"[checkpoint] Resolved to local path: {local_dir}")
    return local_dir


def _assets_dir() -> Path:
    """Repository-local assets/ unless DEPTH_ASSETS_DIR points elsewhere."""
    assets = os.environ.get("DEPTH_ASSETS_DIR")
    if assets:
        return Path(assets).expanduser()
    return Path(__file__).resolve().parents[2] / "assets"


DEFAULT_MARIGOLD_DEPTH_EVAL_ROOT = os.environ.get(
    "MARIGOLD_DEPTH_EVAL_DATASET_ROOT",
    str(_assets_dir() / "datasets" / "marigold_depth_eval"),
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "DEPTH_OUTPUT_ROOT",
    str(Path(__file__).resolve().parents[2] / "output" / "eval_relative_testsets"),
)
SUPPORTED_DATASETS = ("diode", "eth3d", "kitti", "nyuv2", "scannet")


def _base_runtime_paths_from_eval_config(eval_config_path: str | None) -> dict:
    if eval_config_path is None:
        return {}

    cfg = recursive_load_config(str(Path(eval_config_path).resolve()))
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg_dict, dict):
        return {}

    paths_cfg = cfg_dict.get("paths", {})
    if not isinstance(paths_cfg, dict):
        return {}

    use_paths_from = paths_cfg.get("use_paths_from")
    if use_paths_from and isinstance(paths_cfg.get(use_paths_from), dict):
        return dict(paths_cfg.get(use_paths_from))
    if isinstance(paths_cfg.get("runtime"), dict):
        return dict(paths_cfg.get("runtime"))
    return {}


def _common_prediction_steps(dataset_base_dir: str, name_mode: str):
    return [
        {"RunInference": {}},
        {
            "SaveDepthNpy": {
                "batch_key": "out/depth_rel_pred_m11",
                "file_path_key": "rgb_path",
                "base_dir": dataset_base_dir,
                "name_mode": name_mode,
            }
        },
    ]


def _relative_transforms_for_npy_depth_npy_mask():
    return [
        {
            "ReadRGBImage": {
                "key": "rgb_path",
                "name": "rgb",
                "use_rel_dir": False,
                "set_as_orig_res": True,
                "save_orig": True,
            }
        },
        {
            "ReadDepthNpyWithNpyMask": {
                "depth_key": "depth_path",
                "mask_key": "mask_path",
                "output_key": "depth_m",
                "inverse_output_key": "inv_depth_m",
                "min_depth": 0.001,
                "max_depth": 80.0,
                "create_inverse": True,
            }
        },
        {
            "NormalizeDepthPercentileAffine": {
                "key": "depth_m",
                "output_key": "depth_rel_m11",
                "mask_key": "valid_mask",
                "low_percentile": 2.0,
                "high_percentile": 98.0,
                "dst_min": -1.0,
                "dst_max": 1.0,
            }
        },
    ]


def _relative_transforms_for_nyuv2_png_depth():
    return [
        {
            "ReadRGBImage": {
                "key": "rgb_path",
                "name": "rgb",
                "use_rel_dir": False,
                "set_as_orig_res": True,
                "save_orig": True,
            }
        },
        {
            "ReadDepthPng16": {
                "depth_key": "depth_path",
                "output_key": "depth_m",
                "inverse_output_key": "inv_depth_m",
                "depth_scale": 1000.0,
                "min_depth": 0.001,
                "max_depth": 10.0,
                "create_inverse": True,
            }
        },
        {
            "NormalizeDepthPercentileAffine": {
                "key": "depth_m",
                "output_key": "depth_rel_m11",
                "mask_key": "valid_mask",
                "low_percentile": 2.0,
                "high_percentile": 98.0,
                "dst_min": -1.0,
                "dst_max": 1.0,
            }
        },
    ]


def _relative_transforms_for_kitti_png_depth():
    return [
        {
            "ReadRGBImage": {
                "key": "rgb_path",
                "name": "rgb",
                "use_rel_dir": False,
                "set_as_orig_res": True,
                "save_orig": True,
            }
        },
        {
            "ReadDepthPng16": {
                "depth_key": "depth_path",
                "output_key": "depth_m",
                "inverse_output_key": "inv_depth_m",
                "depth_scale": 256.0,
                "min_depth": 0.001,
                "max_depth": 80.0,
                "create_inverse": True,
            }
        },
        {
            "KittiBMCrop": {
                "height": 352,
                "width": 1216,
                "exclude_keys": [],
            }
        },
        {
            "NormalizeDepthPercentileAffine": {
                "key": "depth_m",
                "output_key": "depth_rel_m11",
                "mask_key": "valid_mask",
                "low_percentile": 2.0,
                "high_percentile": 98.0,
                "dst_min": -1.0,
                "dst_max": 1.0,
            }
        },
    ]


def _relative_transforms_for_scannet_png_depth():
    return [
        {
            "ReadRGBImage": {
                "key": "rgb_path",
                "name": "rgb",
                "use_rel_dir": False,
                "set_as_orig_res": True,
                "save_orig": True,
            }
        },
        {
            "ReadDepthPng16": {
                "depth_key": "depth_path",
                "output_key": "depth_m",
                "inverse_output_key": "inv_depth_m",
                "depth_scale": 1000.0,
                "min_depth": 0.1,
                "max_depth": 10.0,
                "create_inverse": True,
            }
        },
        {
            "NormalizeDepthPercentileAffine": {
                "key": "depth_m",
                "output_key": "depth_rel_m11",
                "mask_key": "valid_mask",
                "low_percentile": 2.0,
                "high_percentile": 98.0,
                "dst_min": -1.0,
                "dst_max": 1.0,
            }
        },
    ]


def _relative_transforms_for_eth3d_depth():
    return [
        {
            "ReadRGBImage": {
                "key": "rgb_path",
                "name": "rgb",
                "use_rel_dir": False,
                "set_as_orig_res": True,
                "save_orig": True,
            }
        },
        {
            "ReadDepthFileAuto": {
                "depth_key": "depth_path",
                "mask_key": "mask_path",
                "output_key": "depth_m",
                "inverse_output_key": "inv_depth_m",
                "png_depth_scale": 1000.0,
                "min_depth": 0.00001,
                "max_depth": 150.0,
                "create_inverse": True,
            }
        },
        {
            "Reshape": {
                "height": 672,
                "width": 1008,
                "exclude_keys": [],
                "use_lanczos": True,
            }
        },
        {
            "NormalizeDepthPercentileAffine": {
                "key": "depth_m",
                "output_key": "depth_rel_m11",
                "mask_key": "valid_mask",
                "low_percentile": 2.0,
                "high_percentile": 98.0,
                "dst_min": -1.0,
                "dst_max": 1.0,
            }
        },
    ]


def _split_line_tokens(line: str) -> list[str]:
    return [tok for tok in line.strip().split() if tok]


def _write_manifest_csv_from_split_file(
    split_file: Path,
    dataset_root: Path,
    out_csv: Path,
    rgb_idx: int,
    depth_idx: int,
    mask_idx: int | None,
) -> int:
    rows = []
    with split_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            cols = _split_line_tokens(stripped)
            needed = [rgb_idx, depth_idx] + ([] if mask_idx is None else [mask_idx])
            if len(cols) <= max(needed):
                continue

            rgb_rel = cols[rgb_idx]
            depth_rel = cols[depth_idx]
            if depth_rel == "None":
                continue

            # Without a mask column, point the mask at the RGB file so that the
            # mask-reading transform gets a readable image (it is then all-valid).
            mask_rel = cols[mask_idx] if mask_idx is not None else rgb_rel

            rgb_path = (dataset_root / rgb_rel).resolve()
            depth_path = (dataset_root / depth_rel).resolve()
            mask_path = (dataset_root / mask_rel).resolve()

            if not (rgb_path.exists() and depth_path.exists() and mask_path.exists()):
                continue

            rel_parts = Path(rgb_rel).parts
            scene = rel_parts[0] if len(rel_parts) > 0 else "scene"
            camera = rel_parts[1] if len(rel_parts) > 1 else "camera"
            frame = Path(rgb_rel).stem
            rows.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "frame": f"{frame}_{line_no:06d}",
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "mask_path": str(mask_path),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "camera",
                "frame",
                "rgb_path",
                "depth_path",
                "mask_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _prepare_manifests(
    repo_root: Path,
    marigold_depth_eval_root: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
) -> dict[str, Path]:
    split_root = repo_root / "evaluation" / "data_split"
    out_root = output_dir / "generated_manifests"

    manifest_specs = {
        "diode": {
            "split_file": split_root
            / "diode_depth"
            / "diode_val_all_filename_list.txt",
            "dataset_root": marigold_depth_eval_root / "diode",
            "out_csv": out_root / "diode_manifest.csv",
            "rgb_idx": 0,
            "depth_idx": 1,
            "mask_idx": 2,
        },
        "eth3d": {
            "split_file": split_root / "eth3d_depth" / "eth3d_filename_list.txt",
            "dataset_root": marigold_depth_eval_root / "eth3d",
            "out_csv": out_root / "eth3d_manifest.csv",
            "rgb_idx": 0,
            "depth_idx": 1,
            "mask_idx": None,
        },
        "kitti": {
            "split_file": split_root / "kitti_depth" / "eigen_test_files_with_gt.txt",
            "dataset_root": marigold_depth_eval_root / "kitti",
            "out_csv": out_root / "kitti_manifest.csv",
            "rgb_idx": 0,
            "depth_idx": 1,
            "mask_idx": None,
        },
        "nyuv2": {
            "split_file": split_root
            / "nyu_depth"
            / "labeled"
            / "filename_list_test.txt",
            "dataset_root": marigold_depth_eval_root / "nyuv2",
            "out_csv": out_root / "nyuv2_manifest.csv",
            "rgb_idx": 0,
            "depth_idx": 1,
            "mask_idx": 2,
        },
        "scannet": {
            "split_file": split_root
            / "scannet_depth"
            / "scannet_val_sampled_list_800_1.txt",
            "dataset_root": marigold_depth_eval_root / "scannet",
            "out_csv": out_root / "scannet_manifest.csv",
            "rgb_idx": 0,
            "depth_idx": 1,
            "mask_idx": None,
        },
    }

    selected_set = set(datasets) if datasets is not None else set(manifest_specs.keys())
    manifests: dict[str, Path] = {}
    for ds_name, spec in manifest_specs.items():
        if ds_name not in selected_set:
            continue
        if not spec["split_file"].exists():
            raise FileNotFoundError(
                f"Split file not found for {ds_name}: {spec['split_file']}"
            )
        if not spec["dataset_root"].exists():
            raise FileNotFoundError(
                f"Dataset root not found for {ds_name}: {spec['dataset_root']}"
            )

        count = _write_manifest_csv_from_split_file(
            split_file=spec["split_file"],
            dataset_root=spec["dataset_root"],
            out_csv=spec["out_csv"],
            rgb_idx=spec["rgb_idx"],
            depth_idx=spec["depth_idx"],
            mask_idx=spec["mask_idx"],
        )
        if count <= 0:
            raise RuntimeError(
                f"No valid samples found while generating {ds_name} manifest from {spec['split_file']}"
            )
        manifests[ds_name] = spec["out_csv"]

    if not manifests:
        raise RuntimeError(
            f"No manifests prepared. Selected datasets: {', '.join(sorted(selected_set))}"
        )

    return manifests


DEFAULT_EMBED_DIR = os.environ.get(
    "QWEN_TEXT_EMBEDDINGS_DIR",
    str(_assets_dir() / "checkpoints" / "Marigold-V2" / "qwen_text_embeddings"),
)
DEFAULT_QWEN_IMAGE_EDIT_CKPT = os.environ.get(
    "QWEN_IMAGE_EDIT_CHECKPOINT",
    str(_assets_dir() / "checkpoints" / "Qwen-Image-Edit-2509"),
)


def build_eval_config(
    output_dir: str,
    marigold_depth_eval_root: str,
    manifests: dict[str, str],
    num_workers: int,
    max_samples: int | None,
    datasets: list[str] | None = None,
    eval_config_path: str | None = None,
    embed_dir: str | None = None,
):
    base_runtime_paths = _base_runtime_paths_from_eval_config(eval_config_path)
    runtime_paths = dict(base_runtime_paths)
    runtime_paths.update(
        {
            "base_data_dir": os.environ.get(
                "DEPTH_DATASETS_ROOT", str(_assets_dir() / "datasets")
            ),
            "out_dir": output_dir,
            "override_vis_dir": output_dir,
            "checkpoint_dit": "",
        }
    )
    if embed_dir is not None:
        runtime_paths["embed_dir"] = embed_dir
    elif "embed_dir" not in runtime_paths:
        runtime_paths["embed_dir"] = DEFAULT_EMBED_DIR

    if "ckpt_qwen_image_edit" not in runtime_paths:
        runtime_paths["ckpt_qwen_image_edit"] = DEFAULT_QWEN_IMAGE_EDIT_CKPT

    vis = {}

    if "diode" in manifests:
        vis["diode_rel_depth_qwen_768"] = {
            "name": "GenericDataset",
            "disp_name": "diode_rel_depth_qwen_768",
            "max_io_retries": 200,
            "manifest_cfg": {
                "name": "GenericManifest",
                "manifest_graph": [
                    {
                        "LoadHypersimManifest": {
                            "manifest_csv": manifests["diode"],
                            "base_dir": str(Path(marigold_depth_eval_root) / "diode"),
                            "skip_missing_files": True,
                        }
                    }
                ],
            },
            "transform": _relative_transforms_for_npy_depth_npy_mask(),
            "validation_steps": _common_prediction_steps(
                str(Path(marigold_depth_eval_root) / "diode"), name_mode="id"
            ),
        }

    if "eth3d" in manifests:
        vis["eth3d_rel_depth_qwen_768"] = {
            "name": "GenericDataset",
            "disp_name": "eth3d_rel_depth_qwen_768",
            "max_io_retries": 200,
            "manifest_cfg": {
                "name": "GenericManifest",
                "manifest_graph": [
                    {
                        "LoadHypersimManifest": {
                            "manifest_csv": manifests["eth3d"],
                            "base_dir": str(Path(marigold_depth_eval_root) / "eth3d"),
                            "skip_missing_files": True,
                        }
                    }
                ],
            },
            "transform": _relative_transforms_for_eth3d_depth(),
            "validation_steps": _common_prediction_steps(
                str(Path(marigold_depth_eval_root) / "eth3d"), name_mode="id"
            ),
        }

    if "kitti" in manifests:
        vis["kitti_rel_depth_qwen_768"] = {
            "name": "GenericDataset",
            "disp_name": "kitti_rel_depth_qwen_768",
            "max_io_retries": 200,
            "manifest_cfg": {
                "name": "GenericManifest",
                "manifest_graph": [
                    {
                        "LoadHypersimManifest": {
                            "manifest_csv": manifests["kitti"],
                            "base_dir": str(Path(marigold_depth_eval_root) / "kitti"),
                            "skip_missing_files": True,
                        }
                    }
                ],
            },
            "transform": _relative_transforms_for_kitti_png_depth(),
            "validation_steps": _common_prediction_steps(
                str(Path(marigold_depth_eval_root) / "kitti"), name_mode="id"
            ),
        }

    if "nyuv2" in manifests:
        vis["nyuv2_rel_depth_qwen_768"] = {
            "name": "GenericDataset",
            "disp_name": "nyuv2_rel_depth_qwen_768",
            "max_io_retries": 200,
            "manifest_cfg": {
                "name": "GenericManifest",
                "manifest_graph": [
                    {
                        "LoadHypersimManifest": {
                            "manifest_csv": manifests["nyuv2"],
                            "base_dir": str(Path(marigold_depth_eval_root) / "nyuv2"),
                            "skip_missing_files": True,
                        }
                    }
                ],
            },
            "transform": _relative_transforms_for_nyuv2_png_depth(),
            "validation_steps": _common_prediction_steps(
                str(Path(marigold_depth_eval_root) / "nyuv2"), name_mode="rgb_id"
            ),
        }

    if "scannet" in manifests:
        vis["scannet_rel_depth_qwen_768"] = {
            "name": "GenericDataset",
            "disp_name": "scannet_rel_depth_qwen_768",
            "max_io_retries": 200,
            "manifest_cfg": {
                "name": "GenericManifest",
                "manifest_graph": [
                    {
                        "LoadHypersimManifest": {
                            "manifest_csv": manifests["scannet"],
                            "base_dir": str(Path(marigold_depth_eval_root) / "scannet"),
                            "skip_missing_files": True,
                        }
                    }
                ],
            },
            "transform": _relative_transforms_for_scannet_png_depth(),
            "validation_steps": _common_prediction_steps(
                str(Path(marigold_depth_eval_root) / "scannet"), name_mode="id"
            ),
        }

    if not vis:
        raise ValueError(
            f"No datasets selected. Available: {', '.join(SUPPORTED_DATASETS)}"
        )

    if max_samples is not None:
        for dataset_cfg in vis.values():
            for manifest_step in dataset_cfg["manifest_cfg"]["manifest_graph"]:
                step_cfg = next(iter(manifest_step.values()))
                step_cfg["max_samples"] = int(max_samples)
                step_cfg["seed"] = 42

    config = {
        "base_config": [
            (
                str(eval_config_path)
                if eval_config_path is not None
                else "evaluation/config/inference_depth.yaml"
            )
        ],
        "paths": {
            "use_paths_from": "runtime",
            "runtime": runtime_paths,
        },
        "dataset": {
            "train": {},
            "val": {},
            "vis": vis,
        },
        "dataloader": {
            "num_workers": int(num_workers),
            "effective_batch_size": 1,
            "max_train_batch_size": 1,
            "seed": 2025,
        },
        "eval": {
            "eval_metrics": [],
            "freeze_network_components": True,
        },
        "validation": {
            "file_path_key": "rgb_path",
            "main_val_metric": "",
            "main_val_metric_goal": "minimize",
        },
    }
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate depth predictions for diode, eth3d, kitti, nyuv2, scannet."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory or HuggingFace repo ID (e.g. 'org/repo').",
    )
    parser.add_argument(
        "--embed_dir",
        default=None,
        help="Path to qwen_text_embeddings directory. Auto-detected from the checkpoint if omitted.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where prediction outputs are written.",
    )
    parser.add_argument(
        "--marigold_depth_eval_root", default=DEFAULT_MARIGOLD_DEPTH_EVAL_ROOT
    )
    parser.add_argument(
        "--eval_config",
        default=str(
            Path(__file__).resolve().parents[1] / "config" / "inference_depth.yaml"
        ),
        help="Inference config with the model definition.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs/processes for distributed prediction generation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap per dataset split for quick smoke tests.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset list to run (choices: diode,eth3d,kitti,nyuv2,scannet) or 'all'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.datasets.strip().lower() == "all":
        selected_datasets = list(SUPPORTED_DATASETS)
    else:
        selected_datasets = [
            d.strip().lower() for d in args.datasets.split(",") if d.strip()
        ]
        invalid = [d for d in selected_datasets if d not in SUPPORTED_DATASETS]
        if invalid:
            raise ValueError(
                f"Unsupported dataset(s): {', '.join(invalid)}. Supported: {', '.join(SUPPORTED_DATASETS)}"
            )
        if not selected_datasets:
            raise ValueError("No datasets provided via --datasets")

    repo_root = _REPO_ROOT
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint = _resolve_checkpoint_path(args.checkpoint)

    embed_dir = args.embed_dir
    if embed_dir is None:
        candidate = os.path.join(resolved_checkpoint, "qwen_text_embeddings")
        if os.path.isdir(candidate):
            print(f"[embed_dir] Auto-detected from checkpoint snapshot: {candidate}")
            embed_dir = candidate
        else:
            embed_dir = DEFAULT_EMBED_DIR

    marigold_depth_eval_root = Path(args.marigold_depth_eval_root).resolve()
    if not marigold_depth_eval_root.exists():
        raise FileNotFoundError(
            f"Depth benchmark root does not exist: {marigold_depth_eval_root}"
        )

    manifests = _prepare_manifests(
        repo_root=repo_root,
        marigold_depth_eval_root=marigold_depth_eval_root,
        output_dir=output_dir,
        datasets=selected_datasets,
    )

    config = build_eval_config(
        output_dir=str(output_dir),
        marigold_depth_eval_root=str(marigold_depth_eval_root),
        manifests={k: str(v) for k, v in manifests.items()},
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        datasets=selected_datasets,
        eval_config_path=str(Path(args.eval_config).resolve()),
        embed_dir=embed_dir,
    )

    generated_cfg_path = output_dir / "generated_eval_relative_testsets.yaml"
    OmegaConf.save(config=OmegaConf.create(config), f=str(generated_cfg_path))

    if args.num_gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            str(args.num_gpus),
            "-m",
            "evaluation.depth.evaluate_pipeline",
            "--config",
            str(generated_cfg_path),
            "--checkpoint",
            str(resolved_checkpoint),
            "--output_dir",
            str(output_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-c",
            "from evaluation.depth.evaluate_pipeline import main; main()",
            "--config",
            str(generated_cfg_path),
            "--checkpoint",
            str(resolved_checkpoint),
            "--output_dir",
            str(output_dir),
        ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
