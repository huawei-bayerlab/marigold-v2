import os

# Work around NCCL peer-to-peer issues on some systems during multi-GPU eval.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import argparse
import logging
import torch
import torch.distributed as dist
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List
from safetensors.torch import load_file
import re

from marigoldv2.util.config_resolvers import recursive_load_config
from marigoldv2.core.builder import (
    build_transforms,
    build_dataset,
    register_experiment_modules,
)
from marigoldv2.core.registry import REGISTRY
from marigoldv2.dataset.dataloading.custom_collate import custom_collate

from marigoldv2.validation.metric import MetricTracker
from marigoldv2.validation import metric
from marigoldv2.validation.validate import validate_dataloader
from marigoldv2.network.change_network_mode import set_to_eval
from marigoldv2.script.train.util import make_load_trainables_hook

_HF_REPO_RE = re.compile(r"^(?:hf://)?([a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+)(?:/(.*))?$")


def _resolve_checkpoint_path(checkpoint_path: str | None) -> str | None:
    """If *checkpoint_path* looks like a HuggingFace repo ID, download it and
    return the local snapshot directory; otherwise return it unchanged."""
    if checkpoint_path is None:
        return None

    # Already a local path that exists — use as-is.
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    m = _HF_REPO_RE.match(checkpoint_path)
    if m is None:
        return checkpoint_path  # Neither HF nor existing path; let the loader error naturally.

    repo_id = m.group(1)
    subfolder = m.group(2)  # may be None

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


def _reduce_metric_tracker(metric_tracker: MetricTracker, device: torch.device) -> None:
    """Reduce metric totals/counts across distributed ranks and recompute averages."""
    keys = list(metric_tracker._data.index)
    if len(keys) == 0:
        return

    totals = torch.tensor(
        [float(metric_tracker._data.loc[k, "total"]) for k in keys],
        dtype=torch.float64,
        device=device,
    )
    counts = torch.tensor(
        [float(metric_tracker._data.loc[k, "counts"]) for k in keys],
        dtype=torch.float64,
        device=device,
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    for i, key in enumerate(keys):
        total = float(totals[i].item())
        count = float(counts[i].item())
        avg = total / count if count > 0 else 0.0
        metric_tracker._data.loc[key, "total"] = total
        metric_tracker._data.loc[key, "counts"] = count
        metric_tracker._data.loc[key, "average"] = avg


def _checkpoint_has_vae_trainables(checkpoint_path: str) -> bool:
    state = load_file(checkpoint_path, device="cpu")
    return any(key.startswith("VAE.") for key in state.keys())


def _freeze_network_components() -> None:
    for module in REGISTRY.get("network_components", {}).values():
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(False)


def main():
    t_start = datetime.now()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Started at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run a Marigold V2 inference config over its datasets"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to evaluation config YAML."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save outputs."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory, or a HuggingFace repo ID (e.g. 'org/repo') or 'hf://org/repo' (overrides config).",
    )
    args = parser.parse_args()

    # -------------------- Config --------------------
    cfg = recursive_load_config(args.config)
    if "paths" not in cfg:
        raise ValueError("Invalid config: missing top-level 'paths'.")
    cfg.paths = cfg.paths[cfg.paths.use_paths_from]

    checkpoint_path = args.checkpoint or cfg.paths.get("checkpoint_dit", None)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    output_dir = args.output_dir
    if output_dir is None and "override_vis_dir" in cfg.paths:
        output_dir = cfg.paths.override_vis_dir
    if output_dir is None:
        output_dir = os.path.join(cfg.paths.get("out_dir", "."), "evaluation_output")

    os.makedirs(output_dir, exist_ok=True)

    # Build Accelerator early so evaluation can shard dataloaders when launched with accelerate.
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="bf16")
    REGISTRY["accelerator"] = accelerator

    REGISTRY["cfg"] = cfg
    register_experiment_modules(cfg)

    # Build model components/graph before loading weights.
    build_transforms(cfg.network_components, registry_category="network_components")({})
    network_graph = {
        "train": build_transforms(cfg.network_graph, registry_category="network_graph")
    }
    REGISTRY["built_network_graphs"] = network_graph

    cfg["in_args"] = args.__dict__
    cfg["base_data_dir"] = cfg.paths.get("base_data_dir", None)

    # Save config snapshot
    config_output_path = os.path.join(output_dir, "config.yaml")
    with open(config_output_path, "w+") as f:
        OmegaConf.save(config=cfg, f=f)
    logging.info(f"Config saved to {config_output_path}")

    # -------------------- Device --------------------
    device = str(accelerator.device)
    cfg["device"] = device
    logging.info(f"device = {device}")
    logging.info(
        "distributed eval context: process_index=%s num_processes=%s is_main_process=%s",
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.is_main_process,
    )

    # -------------------- Load checkpoint --------------------
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        trainables_filename = "trainables.safetensors"
        trainables_path = os.path.join(checkpoint_path, trainables_filename)
        if not os.path.exists(trainables_path):
            raise FileNotFoundError(
                f"No {trainables_filename} in {checkpoint_path}. "
                "Omit --checkpoint to run with pretrained weights only."
            )
        has_vae_trainables = _checkpoint_has_vae_trainables(trainables_path)
        print(
            "[eval] Loading Diffuser"
            + (" and VAE" if has_vae_trainables else "")
            + " weights from checkpoint."
        )
        load_hook = make_load_trainables_hook(
            REGISTRY,
            accelerator,
            filename=trainables_filename,
            exclude_components=[] if has_vae_trainables else ["VAE"],
        )
        load_hook([], checkpoint_path)
    else:
        logging.warning(
            "No checkpoint specified. Running with pretrained weights only."
        )

    if cfg.get("eval", {}).get("freeze_network_components", True):
        _freeze_network_components()
        logging.info(
            "Froze all network components for inference after checkpoint loading."
        )

    set_to_eval()

    # -------------------- Data --------------------
    random.seed(cfg.get("random_seed", 2025))
    np.random.seed(cfg.get("random_seed", 2025))
    torch.manual_seed(cfg.get("random_seed", 2025))
    torch.cuda.manual_seed_all(cfg.get("random_seed", 2025))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_data = cfg.dataset

    # Build visualization/evaluation loaders
    vis_loaders: List[DataLoader] = []
    for _vis_dict in cfg_data.get("vis", {}).values():
        _vis_dataset = build_dataset(_vis_dict)
        _sampler = None
        if accelerator.num_processes > 1:
            _sampler = DistributedSampler(
                _vis_dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
                drop_last=False,
            )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            sampler=_sampler,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=custom_collate,
        )
        vis_loaders.append(_vis_loader)

    metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
    val_metrics = MetricTracker(*[m.__name__ for m in metric_funcs])

    # -------------------- Evaluate --------------------
    if accelerator.is_main_process:
        print(f"\nEvaluating on {len(vis_loaders)} dataset(s)...\n")

    for i, vis_loader in enumerate(vis_loaders):
        vis_dataset_name = vis_loader.dataset.disp_name
        vis_out_dir = os.path.join(output_dir, vis_dataset_name)
        if accelerator.is_main_process:
            os.makedirs(vis_out_dir, exist_ok=True)
        REGISTRY["cfg"]["vis_out_dir"] = vis_out_dir

        if hasattr(vis_loader, "sampler") and hasattr(vis_loader.sampler, "set_epoch"):
            vis_loader.sampler.set_epoch(0)

        validate_dataloader(
            vis_loader,
            val_metrics,
            enable_visualization=accelerator.is_main_process,
        )

        _reduce_metric_tracker(val_metrics, accelerator.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if accelerator.is_main_process:
            val_metric_dict = val_metrics.result()
            print(f"\n{'=' * 60}")
            print(f"  Results for: {vis_dataset_name}")
            print(f"{'=' * 60}")
            for metric_name, metric_value in val_metric_dict.items():
                print(f"  {metric_name}: {metric_value:.6f}")
            print(f"{'=' * 60}\n")

            # Save metrics to file
            metrics_path = os.path.join(vis_out_dir, "metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Dataset: {vis_dataset_name}\n")
                f.write(f"Checkpoint: {checkpoint_path}\n\n")
                for metric_name, metric_value in val_metric_dict.items():
                    f.write(f"{metric_name}: {metric_value:.6f}\n")

        val_metrics.reset()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if accelerator.is_main_process:
        print(f"\nEvaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
