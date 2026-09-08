#!/usr/bin/env python3
"""Write the inference config for one Marigold V1 normals benchmark.

The model definition comes from the normals training config; only the dataset
is replaced.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_DIR = Path(
    os.environ.get("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")
).expanduser()
DEFAULT_NORMALS_EVAL_DATASET_ROOT = Path(
    os.environ.get(
        "MARIGOLD_NORMALS_EVAL_DATASET_ROOT",
        str(DEFAULT_ASSETS_DIR / "datasets" / "marigold_normals_eval"),
    )
).expanduser()
DEFAULT_QWEN_CHECKPOINT = DEFAULT_ASSETS_DIR / "checkpoints" / "Qwen-Image-Edit-2509"
NORMALS_EXPERIMENT_CONFIG = (
    REPO_ROOT
    / "marigoldv2"
    / "experiments"
    / "20260728_qwen_normals"
    / "training_normals.yaml"
)


DATASETS = {
    "ibims": {
        "disp_name": "ibims_normals_test",
        "dataset_dir": "ibims/ibims",
        "filelist": "evaluation/data_split/ibims_normals/ibims_test.txt",
        "resize_to_multiple": 16,
    },
    "nyuv2": {
        "disp_name": "nyu_normals_test",
        "dataset_dir": "nyuv2/test",
        "filelist": "evaluation/data_split/nyu_normals/nyuv2_test.txt",
        "resize_to_multiple": 16,
    },
    "scannet": {
        "disp_name": "scannet_normals_test",
        "dataset_dir": "scannet",
        "filelist": "evaluation/data_split/scannet_normals/scannet_test.txt",
        "resize_to_multiple": 16,
    },
    "sintel": {
        "disp_name": "sintel_normals_test",
        "dataset_dir": "sintel",
        "filelist": "evaluation/data_split/sintel_normals/sintel_filtered.txt",
        "resize_to_multiple": 16,
    },
}


def build_config(
    *,
    dataset: str,
    base_data_dir: Path,
    output_dir: Path,
    embed_dir: Path,
    qwen_checkpoint: Path,
    max_samples: int | None,
    seed: int,
) -> dict:
    spec = DATASETS[dataset]
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    dataset_root = base_data_dir / spec["dataset_dir"]
    filelist = REPO_ROOT / spec["filelist"]
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if not filelist.is_file():
        raise FileNotFoundError(f"Dataset file list does not exist: {filelist}")

    manifest_kwargs = {
        "base_dir": str(dataset_root),
        "filelist_path": str(filelist),
        "skip_missing_files": False,
        "seed": int(seed),
    }
    if max_samples is not None:
        manifest_kwargs["max_samples"] = int(max_samples)

    transforms = [
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
                "key": "depth_path",
                "output_key": "normal_gt",
                "mask_key": "valid_mask",
            }
        },
    ]
    if spec.get("resize_to_multiple"):
        transforms.append(
            {
                "ResizeSurfaceNormalsToMultiple": {
                    "multiple": int(spec["resize_to_multiple"]),
                }
            }
        )

    return {
        "evaluation_dataset_names": [spec["disp_name"]],
        "base_config": [str(NORMALS_EXPERIMENT_CONFIG)],
        "dataset": {
            "vis": {
                spec["disp_name"]: {
                    "name": "GenericDataset",
                    "disp_name": spec["disp_name"],
                    "max_io_retries": 20,
                    "manifest_cfg": {
                        "name": "GenericManifest",
                        "manifest_graph": [
                            {"LoadRGBDepthPairsFromFileList": manifest_kwargs}
                        ],
                    },
                    "transform": transforms,
                    "validation_steps": [
                        {"RunInference": {}},
                        {
                            "SaveDepthNpy": {
                                "batch_key": "out/normal_pred",
                                "folder_name": "",
                                "file_path_key": "rgb_path",
                                "base_dir": str(dataset_root),
                                "name_mode": "basename",
                                "resize_to_orig_res": True,
                            }
                        },
                    ],
                }
            }
        },
        "dataloader": {
            "num_workers": 0,
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
                "base_data_dir": str(base_data_dir),
                "out_dir": str(output_dir),
                "override_vis_dir": str(output_dir),
                "embed_dir": str(embed_dir),
                "ckpt_qwen_image_edit": str(qwen_checkpoint),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--base_data_dir",
        default=str(DEFAULT_NORMALS_EVAL_DATASET_ROOT),
        help="Normals benchmark root (or set MARIGOLD_NORMALS_EVAL_DATASET_ROOT).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embed_dir", required=True)
    parser.add_argument("--qwen_checkpoint", default=str(DEFAULT_QWEN_CHECKPOINT))
    parser.add_argument("--config_out", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(
        dataset=args.dataset,
        base_data_dir=Path(args.base_data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        embed_dir=Path(args.embed_dir).resolve(),
        qwen_checkpoint=Path(args.qwen_checkpoint).resolve(),
        max_samples=args.max_samples,
        seed=args.seed,
    )
    config_out = Path(args.config_out).resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=str(config_out))
    print(config_out)


if __name__ == "__main__":
    main()
