#!/usr/bin/env python3
"""Run Marigold V2 depth, normals, or albedo inference on a folder of images."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).expanduser()
BASE_CONFIG = REPO_ROOT / "evaluation" / "config" / "inference_depth.yaml"
QWEN_CHECKPOINT = ASSETS_DIR / "checkpoints" / "Qwen-Image-Edit-2509"
MARIGOLD_CHECKPOINTS = ASSETS_DIR / "checkpoints" / "Marigold-V2"
EMBED_DIR = MARIGOLD_CHECKPOINTS / "qwen_text_embeddings"

MODALITIES = {
    "depth": {
        "checkpoint": MARIGOLD_CHECKPOINTS / "depth" / "Log-stage2",
        "embed_prefix": "qwen_edit_2509_qwen_depth_realimg512",
        "prediction_key": "out/depth_pred",
        "adapter": "FolderDepthPrediction",
        "visualizer": "VisualizeFolderDepth",
        "visualization_folder": "visualizations/depth_spectral",
    },
    "normals": {
        "checkpoint": MARIGOLD_CHECKPOINTS / "normals",
        "embed_prefix": "qwen_edit_2509_qwen_normals_dummy512",
        "prediction_key": "out/normal_pred",
        "adapter": "FolderNormalizeSurfaceNormals",
        "visualizer": "VisualizeFolderNormals",
        "visualization_folder": "visualizations/normals",
    },
    "albedo": {
        "checkpoint": MARIGOLD_CHECKPOINTS / "albedo",
        "embed_prefix": "qwen_edit_2509_qwen_albedo_rgb_dummy512",
        "prediction_key": "out/albedo_pred",
        "adapter": "FolderRGBAlbedoPrediction",
        "output_color_space": "srgb",
        "visualizer": "VisualizeFolderAlbedo",
        "visualization_folder": "visualizations/albedo",
    },
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _count_images(image_dir: Path) -> int:
    return sum(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in image_dir.rglob("*")
    )


def _validate_inputs(image_dir: Path, checkpoint: str, modality: str) -> None:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    if _count_images(image_dir) == 0:
        raise ValueError(f"No supported images found under: {image_dir}")
    if not QWEN_CHECKPOINT.is_dir():
        raise FileNotFoundError(
            f"Qwen checkpoint not found: {QWEN_CHECKPOINT}; run scripts/download_assets.py"
        )
    prefix = MODALITIES[modality]["embed_prefix"]
    for suffix in ("_prompt_embeds.pt", "_prompt_mask.pt"):
        path = EMBED_DIR / f"{prefix}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {modality} prompt embedding: {path}")
    if Path(checkpoint).exists() and not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"Checkpoint is not a directory: {checkpoint}")


def _build_config(
    image_dir: Path,
    output_dir: Path,
    modality: str,
    checkpoint: str,
    width: int | None,
    height: int | None,
    seed: int,
) -> dict:
    config = MODALITIES[modality]
    native_resolution = width is None
    read_rgb = {
        "ReadRGBImage": {
            "key": "rgb_path",
            "name": "rgb",
            "use_rel_dir": False,
            "set_as_orig_res": True,
            "save_orig": True,
        }
    }
    if native_resolution:
        resize = {"ReshapeToMultiple": {"multiple": 16, "use_lanczos": True}}
    else:
        resize = {"Reshape": {"height": height, "width": width, "use_lanczos": True}}

    prediction_key = config["prediction_key"]
    output_step = {
        "batch_key": prediction_key,
        "base_dir": str(image_dir),
        "resize_to_orig_res": native_resolution,
    }
    validation_steps = [
        {"RunInference": {}},
        {"SaveFolderPredictionNpy": {**output_step, "folder_name": "predictions_npy"}},
        {
            config["visualizer"]: {
                **output_step,
                "folder_name": config["visualization_folder"],
                "inverse_spectral": modality == "depth"
                and "disparity" in checkpoint.lower(),
            }
        },
    ]

    base_config = OmegaConf.load(str(BASE_CONFIG))
    register_modules = list(base_config.get("register_modules") or [])
    register_modules.append("marigoldv2.validation.folder_steps")

    dataset_name = "images"
    dataset = {
        "name": "GenericDataset",
        "disp_name": dataset_name,
        "manifest_cfg": {
            "name": "GenericManifest",
            "manifest_graph": [
                {
                    "LoadVKittiManifest": {
                        "base_dir": str(image_dir),
                        "split": ".",
                        "skip_missing_files": False,
                    }
                }
            ],
        },
        "transform": [read_rgb, resize],
        "validation_steps": validation_steps,
    }
    return {
        "base_config": [str(BASE_CONFIG)],
        "register_modules": register_modules,
        "random_seed": seed,
        "dataset": {"vis": {dataset_name: dataset}},
        "dataloader": {
            "num_workers": 0,
            "effective_batch_size": 1,
            "max_train_batch_size": 1,
            "seed": seed,
        },
        "eval": {"eval_metrics": []},
        "validation": {
            "file_path_key": "rgb_path",
            "main_val_metric": "",
            "main_val_metric_goal": "minimize",
        },
        "network_graph": {
            "QwenImageEdit2509Step": {"kwargs": {"prefix": config["embed_prefix"]}},
            config["adapter"]: {
                "kwargs": {
                    "input_key": "out/pixel_pred",
                    "output_key": prediction_key,
                    "output_color_space": config.get("output_color_space", "linear"),
                }
            },
        },
        "paths": {
            "use_paths_from": "runtime",
            "runtime": {
                "out_dir": str(output_dir),
                "override_vis_dir": str(output_dir),
                "ckpt_qwen_image_edit": str(QWEN_CHECKPOINT),
                "embed_dir": str(EMBED_DIR),
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=tuple(MODALITIES), default="depth")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory or Hugging Face repo[/subfolder]. Default per modality.",
    )
    parser.add_argument("--image_dir", required=True, help="Folder of RGB images.")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "output" / "infer"),
        help="Where predictions and visualizations are written.",
    )
    parser.add_argument(
        "--width", type=int, help="Fixed inference width (with --height)."
    )
    parser.add_argument(
        "--height", type=int, help="Fixed inference height (with --width)."
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Seed for the VAE encoder noise."
    )
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if args.width is not None and (args.width <= 0 or args.height <= 0):
        parser.error("--width and --height must be positive")
    return args


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint or str(MODALITIES[args.modality]["checkpoint"])
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _validate_inputs(image_dir, checkpoint, args.modality)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        image_dir,
        output_dir,
        args.modality,
        checkpoint,
        args.width,
        args.height,
        args.seed,
    )
    config_path = output_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(config), f=str(config_path))

    resolution = "native" if args.width is None else f"{args.width}x{args.height}"
    print(f"{_count_images(image_dir)} images under {image_dir}")
    print(
        f"Modality: {args.modality}; checkpoint: {checkpoint}; resolution: {resolution}"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.depth.evaluate_pipeline",
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
            "--checkpoint",
            checkpoint,
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
