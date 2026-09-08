#!/usr/bin/env python3
"""Run albedo inference on the Hypersim test split at native resolution.

Predictions land in ``<output_dir>/<dataset_name>/`` mirroring the RGB layout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).expanduser()
DEFAULT_BASE_CONFIG = (
    REPO_ROOT / "marigoldv2/experiments/20260803_qwen_albedo/training_albedo.yaml"
)
DEFAULT_DATASET_BASE_DIR = Path(ASSETS_DIR / "datasets/marigold_train_albedo")
DEFAULT_FILELIST = REPO_ROOT / "evaluation/data_split/hypersim_iid/hypersim_test.txt"
DEFAULT_EMBED_DIR = ASSETS_DIR / "checkpoints/Marigold-V2/qwen_text_embeddings"
DEFAULT_QWEN_CHECKPOINT = ASSETS_DIR / "checkpoints/Qwen-Image-Edit-2509"
ALBEDO_EMBED_PREFIX = "qwen_edit_2509_qwen_albedo_rgb_dummy512"


def build_eval_config(
    *,
    output_dir: Path,
    base_config: Path,
    dataset_base_dir: Path,
    filelist: Path,
    dataset_name: str,
    max_samples: int | None,
    seed: int,
    num_workers: int,
    embed_dir: Path,
    qwen_checkpoint: str,
) -> dict:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if not dataset_base_dir.is_dir():
        raise FileNotFoundError(
            f"Hypersim albedo root does not exist: {dataset_base_dir}"
        )
    if not filelist.is_file():
        raise FileNotFoundError(f"Hypersim IID file list does not exist: {filelist}")
    if not embed_dir.is_dir():
        raise FileNotFoundError(f"Qwen embedding directory does not exist: {embed_dir}")
    for suffix in ("_prompt_embeds.pt", "_prompt_mask.pt"):
        embedding_path = embed_dir / f"{ALBEDO_EMBED_PREFIX}{suffix}"
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"Missing albedo prompt embedding: {embedding_path}"
            )

    manifest_kwargs = {
        "base_dir": str(dataset_base_dir),
        "split": "test",
        "source_split": "test",
        "filelist_path": str(filelist),
        "selection": "all",
        "verify_files": True,
        # The Marigold V1 IID file list has five columns; only RGB/albedo are used.
        "allow_extra_columns": True,
        "seed": int(seed),
    }
    if max_samples is not None:
        manifest_kwargs["max_samples"] = int(max_samples)

    return {
        "base_config": [str(base_config)],
        "evaluation_dataset_names": [dataset_name],
        "dataset": {
            "train": {},
            "val": {},
            "vis": {
                dataset_name: {
                    "name": "GenericDataset",
                    "disp_name": dataset_name,
                    "max_io_retries": 20,
                    "manifest_cfg": {
                        "name": "GenericManifest",
                        "manifest_graph": [
                            {"LoadHypersimAlbedoManifest": manifest_kwargs}
                        ],
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
                            "ReadRGBAlbedoNpy": {
                                "key": "albedo_path",
                                "output_key": "albedo_gt",
                                "mask_key": "valid_mask",
                            }
                        },
                    ],
                    "validation_steps": [
                        {"RunInference": {}},
                        {
                            "SaveDepthNpy": {
                                "batch_key": "out/albedo_pred",
                                "folder_name": "",
                                "file_path_key": "rgb_path",
                                "base_dir": str(dataset_base_dir),
                                "name_mode": "basename",
                                "resize_to_orig_res": True,
                            }
                        },
                    ],
                }
            },
        },
        "dataloader": {
            "num_workers": int(num_workers),
            "effective_batch_size": 1,
            "max_train_batch_size": 1,
            "pin_memory": False,
            "persistent_workers": False,
            "seed": int(seed),
        },
        "eval": {"eval_metrics": []},
        "validation": {
            "file_path_key": "rgb_path",
            "main_val_metric": "",
            "main_val_metric_goal": "minimize",
        },
        "paths": {
            "use_paths_from": "runtime",
            "runtime": {
                "base_data_dir": str(dataset_base_dir),
                "out_dir": str(output_dir),
                "override_vis_dir": str(output_dir),
                "embed_dir": str(embed_dir),
                "ckpt_qwen_image_edit": qwen_checkpoint,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Qwen albedo checkpoint directory."
    )
    parser.add_argument("--output_dir", required=True, help="Inference output root.")
    parser.add_argument("--dataset_base_dir", default=str(DEFAULT_DATASET_BASE_DIR))
    parser.add_argument("--filelist", default=str(DEFAULT_FILELIST))
    parser.add_argument("--dataset_name", default="hypersim_test_albedo_qwen_native")
    parser.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--embed_dir", default=str(DEFAULT_EMBED_DIR))
    parser.add_argument("--qwen_checkpoint", default=str(DEFAULT_QWEN_CHECKPOINT))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_eval_config(
        output_dir=output_dir,
        base_config=Path(args.base_config).resolve(),
        dataset_base_dir=Path(args.dataset_base_dir).resolve(),
        filelist=Path(args.filelist).resolve(),
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        seed=args.seed,
        num_workers=args.num_workers,
        embed_dir=Path(args.embed_dir).resolve(),
        qwen_checkpoint=args.qwen_checkpoint,
    )
    config_path = output_dir / "generated_eval_hypersim_albedo.yaml"
    OmegaConf.save(config=OmegaConf.create(config), f=str(config_path))

    common_args = [
        "--config",
        str(config_path),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--output_dir",
        str(output_dir),
    ]
    if args.num_gpus > 1:
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(args.num_gpus),
            str(REPO_ROOT / "evaluation/depth/evaluate_pipeline.py"),
            *common_args,
        ]
    else:
        command = [
            sys.executable,
            str(REPO_ROOT / "evaluation/depth/evaluate_pipeline.py"),
            *common_args,
        ]

    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
