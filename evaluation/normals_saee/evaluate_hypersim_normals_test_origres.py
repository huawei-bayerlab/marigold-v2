#!/usr/bin/env python3
"""Run normals inference on the Hypersim test split at native resolution.

Normals counterpart of ``evaluation/depth_see/evaluate_hypersim_test_origres.py``.
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
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).expanduser()
DEFAULT_BASE_CONFIG = str(
    REPO_ROOT / "marigoldv2/experiments/20260728_qwen_normals/training_normals.yaml"
)
DEFAULT_DATASET_BASE_DIR = str(ASSETS_DIR / "datasets/marigold_train_normals")
DEFAULT_EMBED_DIR = str(ASSETS_DIR / "checkpoints/Marigold-V2/qwen_text_embeddings")
DEFAULT_QWEN_CHECKPOINT = str(ASSETS_DIR / "checkpoints/Qwen-Image-Edit-2509")
DEFAULT_DATASET_NAME = "hypersim_test_normals_origres_qwen"


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _embed_dir_from_base_config(base_config: str) -> str | None:
    cfg = OmegaConf.load(str(_resolve_repo_path(base_config).resolve()))
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
    match = _HF_REPO_RE.match(checkpoint_path)
    if match is None:
        return checkpoint_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for HF checkpoints") from exc
    kwargs = {"repo_id": match.group(1)}
    if match.group(2):
        kwargs["allow_patterns"] = [f"{match.group(2)}/*", f"{match.group(2)}/**/*"]
    local_dir = snapshot_download(**kwargs)
    if match.group(2):
        local_dir = os.path.join(local_dir, match.group(2))
    return local_dir


def _build_eval_config(
    *,
    output_dir: str,
    base_config: str,
    dataset_base_dir: str,
    split: str,
    max_samples: int | None,
    num_workers: int,
    embed_dir: str,
    qwen_checkpoint: str,
    dataset_name: str,
) -> dict:
    base_cfg = OmegaConf.to_container(
        OmegaConf.load(str(_resolve_repo_path(base_config).resolve())), resolve=False
    )
    cfg = base_cfg if isinstance(base_cfg, dict) else {}
    cfg.pop("base_config", None)

    manifest_kwargs = {
        "base_dir": dataset_base_dir,
        "split": split,
        "verify_files": True,
    }
    if max_samples is not None:
        manifest_kwargs.update({"max_samples": int(max_samples), "seed": 42})

    vis_dataset = {
        "name": "GenericDataset",
        "disp_name": dataset_name,
        "manifest_cfg": {
            "name": "GenericManifest",
            "manifest_graph": [{"LoadHypersimNormalsManifest": manifest_kwargs}],
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
                "ReadSurfaceNormalsNpy": {
                    "key": "normal_path",
                    "output_key": "normal_gt",
                    "mask_key": "valid_mask",
                }
            },
        ],
        "validation_steps": [
            {"RunInference": {}},
            {
                "SaveDepthNpy": {
                    "batch_key": "out/normal_pred",
                    "folder_name": "",
                    "file_path_key": "rgb_path",
                    "base_dir": dataset_base_dir,
                    "name_mode": "basename",
                    "resize_to_orig_res": False,
                }
            },
        ],
    }

    cfg["evaluation_dataset_names"] = [dataset_name]
    cfg["paths"] = {
        "use_paths_from": "runtime",
        "runtime": {
            "base_data_dir": dataset_base_dir,
            "out_dir": output_dir,
            "override_vis_dir": output_dir,
            "embed_dir": embed_dir,
            "ckpt_qwen_image_edit": qwen_checkpoint,
        },
    }
    cfg["dataset"] = {"train": {}, "val": {}, "vis": {dataset_name: vis_dataset}}
    cfg["dataloader"] = {
        "num_workers": int(num_workers),
        "effective_batch_size": 1,
        "max_train_batch_size": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "seed": 2025,
    }
    cfg["validation"] = {
        "file_path_key": "rgb_path",
        "main_val_metric": "",
        "main_val_metric_goal": "minimize",
    }
    cfg["eval"] = {"eval_metrics": []}
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset_base_dir", default=DEFAULT_DATASET_BASE_DIR)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--embed_dir", default=None)
    parser.add_argument("--qwen_checkpoint", default=DEFAULT_QWEN_CHECKPOINT)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    embed_dir = (
        args.embed_dir
        or _embed_dir_from_base_config(args.base_config)
        or DEFAULT_EMBED_DIR
    )
    config = _build_eval_config(
        output_dir=str(out_dir),
        base_config=args.base_config,
        dataset_base_dir=str(Path(args.dataset_base_dir).resolve()),
        split=args.split,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        embed_dir=embed_dir,
        qwen_checkpoint=str(Path(args.qwen_checkpoint).expanduser().resolve()),
        dataset_name=args.dataset_name,
    )
    cfg_path = out_dir / "generated_eval_hypersim_normals_test_origres.yaml"
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
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
