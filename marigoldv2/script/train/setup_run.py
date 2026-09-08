import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from marigoldv2.core.registry import REGISTRY
from marigoldv2.util.config_resolvers import recursive_load_config
from marigoldv2.util.logging_util import config_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Marigold V2 training")
    parser.add_argument("--config", type=str, help="Path to the training config.")
    parser.add_argument(
        "--resume_run",
        type=str,
        default=None,
        help="Checkpoint directory to resume from; --config is then ignored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Parent directory for the run (default: paths.out_dir from the config).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Run folder name (default: config file stem).",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Prefix the run folder with the start time.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA.")
    args = parser.parse_args()
    if args.resume_run is None and args.config is None:
        parser.error("--config is required unless --resume_run is given")
    return args


def setup_run():
    t_start = datetime.now()
    logging.info(f"Started at {t_start}")
    args = parse_args()
    is_rank0 = int(os.environ.get("RANK", "0")) == 0

    if args.resume_run is not None:
        logging.info(f"Resuming run: {args.resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(os.path.dirname(args.resume_run)))
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
        if "use_paths_from" in cfg.paths:
            cfg.paths = cfg.paths[cfg.paths.use_paths_from]
    else:
        cfg = recursive_load_config(args.config)
        if "use_paths_from" in cfg.paths:
            cfg.paths = cfg.paths[cfg.paths.use_paths_from]
        experiment_name = clean_folder_name(
            args.experiment_name or Path(args.config).stem
        )
        if args.add_datetime_prefix:
            experiment_name = f"{t_start.strftime('%y%m%dT%H%M%S')}_{experiment_name}"
        parent = args.output_dir if args.output_dir is not None else cfg.paths.out_dir
        out_dir_run = os.path.join(parent, experiment_name)
        os.makedirs(out_dir_run, exist_ok=False)

    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    out_dir_acc = os.path.join(out_dir_run, "accelerate")
    for d in (out_dir_ckpt, out_dir_eval, out_dir_vis, out_dir_acc):
        os.makedirs(d, exist_ok=True)  # distributed ranks reach this concurrently

    config_logging(cfg.get("logging", {}), out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    cfg["device"] = "cuda" if cuda_avail else "cpu"
    logging.info(f"device = {cfg['device']}")

    if args.resume_run is None and is_rank0:
        config_path = os.path.join(out_dir_run, "config.yaml")
        with open(config_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {config_path}")
        snapshot_code(out_dir_run)

    cfg["in_args"] = vars(args)
    cfg["out_dir_eval"] = out_dir_eval
    cfg["out_dir_vis"] = out_dir_vis
    cfg["out_dir_run"] = out_dir_run
    cfg["out_dir_acc"] = out_dir_acc
    cfg["base_data_dir"] = cfg.paths.base_data_dir

    REGISTRY["cfg"] = cfg
    return cfg


def snapshot_code(out_dir_run):
    """Archive the working tree (minus .gitignore'd files) next to the run."""
    temp_dir = os.path.join(out_dir_run, "code_tar")
    snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
    os.system(
        "rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' "
        f". '{temp_dir}'"
    )
    os.system(f"tar -cf '{snapshot_path}' '{temp_dir}'")
    os.system(f"rm -rf '{temp_dir}'")
    logging.info(f"Code snapshot saved to: {snapshot_path}")


def clean_folder_name(name: str, max_length: int = 80) -> str:
    name = name.strip().replace(os.sep, "_")
    if os.path.altsep:
        name = name.replace(os.path.altsep, "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._")
    return (name or "experiment")[:max_length]
