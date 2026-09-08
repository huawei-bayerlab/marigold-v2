#!/usr/bin/env python3
"""Run depth inference on the Hypersim test split at native resolution.

Writes predictions only; ``compute_hypersim_edge_metrics.py`` scores them.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

_HF_REPO_RE = re.compile(r"^(?:hf://)?([a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+)(?:/(.*))?$")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _assets_dir() -> Path:
    """Repository-local assets/ unless DEPTH_ASSETS_DIR points elsewhere."""
    assets = os.environ.get("DEPTH_ASSETS_DIR")
    if assets:
        return Path(assets).expanduser()
    return _REPO_ROOT / "assets"


DEFAULT_BASE_CONFIG = str(_REPO_ROOT / "evaluation" / "config" / "inference_depth.yaml")
DEFAULT_DATASET_BASE_DIR = os.environ.get(
    "MARIGOLD_DATASET_ROOT", str(_assets_dir() / "datasets" / "marigold_train")
)
DEFAULT_EMBED_DIR = os.environ.get(
    "QWEN_TEXT_EMBEDDINGS_DIR",
    str(_assets_dir() / "checkpoints" / "Marigold-V2" / "qwen_text_embeddings"),
)


def _embed_dir_from_base_config(base_config: str) -> str | None:
    cfg = OmegaConf.load(str(Path(base_config).resolve()))
    paths = cfg.get("paths")
    if paths is None:
        return None

    use_paths_from = paths.get("use_paths_from")
    if use_paths_from and paths.get(use_paths_from):
        scoped = paths.get(use_paths_from)
        if scoped and scoped.get("embed_dir"):
            return str(scoped.get("embed_dir"))

    if paths.get("embed_dir"):
        return str(paths.get("embed_dir"))
    return None


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    m = _HF_REPO_RE.match(checkpoint_path)
    if m is None:
        return checkpoint_path

    repo_id = m.group(1)
    subfolder = m.group(2)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF checkpoints. Install via: pip install huggingface_hub"
        ) from exc

    kwargs = {"repo_id": repo_id}
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**/*"]

    print(f"[checkpoint] Downloading HuggingFace checkpoint '{repo_id}' ...")
    local_dir = snapshot_download(**kwargs)
    if subfolder:
        local_dir = os.path.join(local_dir, subfolder)
    print(f"[checkpoint] Resolved to local path: {local_dir}")
    return local_dir


def _build_eval_config(
    output_dir: str,
    base_config: str,
    dataset_base_dir: str,
    split: str,
    max_samples: int | None,
    num_workers: int,
    embed_dir: str,
    dataset_name: str,
) -> dict:
    base_cfg_obj = OmegaConf.load(str(Path(base_config).resolve()))
    cfg = OmegaConf.to_container(base_cfg_obj, resolve=False)
    if not isinstance(cfg, dict):
        cfg = {}

    manifest_step = {
        "LoadMarigoldHypersimManifest": {
            "base_dir": dataset_base_dir,
            "split": split,
            "skip_missing_files": True,
        }
    }
    if max_samples is not None and max_samples > 0:
        manifest_step["LoadMarigoldHypersimManifest"]["max_samples"] = int(max_samples)
        manifest_step["LoadMarigoldHypersimManifest"]["seed"] = 42

    vis_dataset = {
        "name": "GenericDataset",
        "disp_name": dataset_name,
        "manifest_cfg": {
            "name": "GenericManifest",
            "manifest_graph": [manifest_step],
        },
        "transform": [
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
                    "output_key": "depth_m",
                    "inverse_output_key": "inv_depth_m",
                    "png_depth_scale": 1000.0,
                    "min_depth": 0.00001,
                    "max_depth": 65.0,
                    "create_inverse": True,
                }
            },
            {
                "MetricDepthToLogDepth": {
                    "key": "depth_m",
                    "output_key": "log_depth_m",
                    "eps": 1e-6,
                }
            },
            {
                "NormalizeDepthPercentileAffine": {
                    "key": "log_depth_m",
                    "output_key": "depth_rel_m11",
                    "mask_key": "valid_mask",
                    "low_percentile": 2.0,
                    "high_percentile": 98.0,
                    "dst_min": -1.0,
                    "dst_max": 1.0,
                }
            },
        ],
        "validation_steps": [
            {"RunInference": {}},
            {
                "SaveDepthNpy": {
                    "batch_key": "out/depth_rel_pred_m11",
                    "file_path_key": "rgb_path",
                    "base_dir": dataset_base_dir,
                    "name_mode": "id",
                }
            },
        ],
    }

    cfg.pop("base_config", None)

    cfg["paths"] = {
        "use_paths_from": "runtime",
        "runtime": {
            "base_data_dir": os.environ.get(
                "DEPTH_DATASETS_ROOT", str(_assets_dir() / "datasets")
            ),
            "out_dir": output_dir,
            "override_vis_dir": output_dir,
            "checkpoint_dit": "",
            "embed_dir": embed_dir,
            "ckpt_qwen_image_edit": os.environ.get(
                "QWEN_IMAGE_EDIT_CHECKPOINT",
                str(_assets_dir() / "checkpoints" / "Qwen-Image-Edit-2509"),
            ),
        },
    }

    cfg["dataset"] = {
        "train": {},
        "val": {},
        "vis": {
            dataset_name: vis_dataset,
        },
    }

    cfg["dataloader"] = {
        "num_workers": int(num_workers),
        "effective_batch_size": 1,
        "max_train_batch_size": 1,
        "seed": 2025,
    }

    cfg["validation"] = {
        "file_path_key": "rgb_path",
        "main_val_metric": "",
        "main_val_metric_goal": "minimize",
    }

    cfg["eval"] = {
        "eval_metrics": [],
    }

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Checkpoint path or HuggingFace repo id."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory root for predictions."
    )
    parser.add_argument(
        "--split", default="test", help="Hypersim split to evaluate (default: test)."
    )
    parser.add_argument("--dataset_base_dir", default=DEFAULT_DATASET_BASE_DIR)
    parser.add_argument(
        "--dataset_name", default="hypersim_test_origres_rel_log_depth_qwen"
    )
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--embed_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = _REPO_ROOT
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    embed_dir = args.embed_dir
    if embed_dir is None:
        cfg_embed_dir = _embed_dir_from_base_config(args.base_config)
        if cfg_embed_dir:
            embed_dir = cfg_embed_dir

    if embed_dir is None:
        candidate = os.path.join(checkpoint_path, "qwen_text_embeddings")
        if os.path.isdir(candidate):
            print(f"[embed_dir] Auto-detected from checkpoint snapshot: {candidate}")
            embed_dir = candidate
        else:
            embed_dir = DEFAULT_EMBED_DIR

    config = _build_eval_config(
        output_dir=str(out_dir),
        base_config=str(Path(args.base_config).resolve()),
        dataset_base_dir=str(Path(args.dataset_base_dir).resolve()),
        split=args.split,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        embed_dir=embed_dir,
        dataset_name=args.dataset_name,
    )

    cfg_path = out_dir / "generated_eval_hypersim_test_origres.yaml"
    OmegaConf.save(config=OmegaConf.create(config), f=str(cfg_path))

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
            str(cfg_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output_dir",
            str(out_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-c",
            "from evaluation.depth.evaluate_pipeline import main; main()",
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output_dir",
            str(out_dir),
        ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
