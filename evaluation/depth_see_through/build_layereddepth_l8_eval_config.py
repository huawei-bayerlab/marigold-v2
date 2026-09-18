#!/usr/bin/env python3
"""Build the runtime config for LayeredDepth-Syn layer-8 evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_DIR = Path(
    os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")
).expanduser()
if not DEFAULT_ASSETS_DIR.is_absolute():
    DEFAULT_ASSETS_DIR = REPO_ROOT / DEFAULT_ASSETS_DIR

DEFAULT_DATASET_BASE_DIR = DEFAULT_ASSETS_DIR / "datasets" / "LayeredDepth-Syn"
DEFAULT_EMBED_DIR = (
    DEFAULT_ASSETS_DIR / "checkpoints" / "Marigold-V2" / "qwen_text_embeddings"
)
DEFAULT_QWEN_CHECKPOINT = DEFAULT_ASSETS_DIR / "checkpoints" / "Qwen-Image-Edit-2509"
INFERENCE_CONFIG = REPO_ROOT / "evaluation" / "config" / "inference_depth.yaml"
DATASET_CONFIG = (
    REPO_ROOT
    / "marigoldv2"
    / "config"
    / "datasets"
    / "20260619_layereddepth_syn_qwen_log_depth_768.yaml"
)
DATASET_NAME = "layereddepth_syn_l8_val_rgb_supervision_768"
EVAL_METRICS = [
    "aligned_log_abs_relative_difference",
    "aligned_log_delta1_acc",
]


def _absolute_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def build_config(
    *,
    output_dir: Path,
    dataset_base_dir: Path,
    embed_dir: Path,
    qwen_checkpoint: Path,
    max_samples: int | None,
) -> dict:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    manifest_path = dataset_base_dir / "extracted" / "val" / "manifest_layer8.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"LayeredDepth-Syn manifest does not exist: {manifest_path}"
        )

    inference_cfg = OmegaConf.to_container(
        OmegaConf.load(str(INFERENCE_CONFIG)), resolve=False
    )
    dataset_cfg = OmegaConf.to_container(
        OmegaConf.load(str(DATASET_CONFIG)), resolve=False
    )
    if not isinstance(inference_cfg, dict) or not isinstance(dataset_cfg, dict):
        raise ValueError("Evaluation configs must contain mapping objects")

    vis_dataset = dataset_cfg[DATASET_NAME]
    if not isinstance(vis_dataset, dict):
        raise ValueError(f"Dataset definition is not a mapping: {DATASET_NAME}")

    manifest_step = vis_dataset["manifest_cfg"]["manifest_graph"][0][
        "LoadHypersimManifest"
    ]
    manifest_step["manifest_csv"] = str(manifest_path)
    if max_samples is None:
        manifest_step.pop("max_samples", None)
    else:
        manifest_step["max_samples"] = int(max_samples)
        manifest_step["seed"] = 42

    config = dict(inference_cfg)
    config["dataset"] = {
        "train": {},
        "val": {},
        "vis": {DATASET_NAME: vis_dataset},
    }
    config["dataloader"] = {
        "num_workers": 0,
        "effective_batch_size": 1,
        "max_train_batch_size": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "seed": 2025,
    }
    config["eval"] = {"eval_metrics": EVAL_METRICS}
    config["validation"] = {
        "file_path_key": "rgb_path",
        "main_val_metric": "",
        "main_val_metric_goal": "minimize",
    }
    config["paths"] = {
        "use_paths_from": "runtime",
        "runtime": {
            "base_data_dir": str(dataset_base_dir.parent),
            "out_dir": str(output_dir),
            "override_vis_dir": str(output_dir),
            "checkpoint_dit": "",
            "embed_dir": str(embed_dir),
            "ckpt_qwen_image_edit": str(qwen_checkpoint),
        },
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", required=True, help="Evaluation output directory."
    )
    parser.add_argument(
        "--dataset_base_dir",
        default=str(DEFAULT_DATASET_BASE_DIR),
        help="LayeredDepth-Syn dataset root.",
    )
    parser.add_argument("--embed_dir", default=str(DEFAULT_EMBED_DIR))
    parser.add_argument("--qwen_checkpoint", default=str(DEFAULT_QWEN_CHECKPOINT))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config_out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(
        output_dir=output_dir,
        dataset_base_dir=_absolute_path(args.dataset_base_dir),
        embed_dir=_absolute_path(args.embed_dir),
        qwen_checkpoint=_absolute_path(args.qwen_checkpoint),
        max_samples=args.max_samples,
    )
    config_out = (
        _absolute_path(args.config_out)
        if args.config_out is not None
        else output_dir / "generated_eval_layereddepth_l8.yaml"
    )
    config_out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=str(config_out))
    print(config_out)


if __name__ == "__main__":
    main()
