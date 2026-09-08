import os
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Work around NCCL peer-to-peer issues on some systems.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import random
import numpy as np
import bitsandbytes as bnb

from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, LambdaLR

from typing import List
import transformers
import diffusers

from marigoldv2.trainer import get_trainer_cls

from marigoldv2.core.builder import (
    build_transforms,
    build_dataset,
    register_experiment_modules,
)
from marigoldv2.script.train.setup_run import setup_run
from marigoldv2.core.registry import REGISTRY
from marigoldv2.dataset.dataloading.custom_collate import custom_collate
from marigoldv2.dataset.mixed_sampler import MixedBatchSampler
from marigoldv2.script.train.util import (
    build_trainable_param_groups_from_network_components,
    make_save_trainables_hook,
    make_load_trainables_hook,
)


def main():
    setup_run()
    cfg = REGISTRY["cfg"]
    register_experiment_modules(cfg)

    # Enable wandb only when explicitly allowed by CLI args.
    in_args = cfg.get("in_args", {})
    enable_wandb = not bool(in_args.get("no_wandb", False))

    # Resolve world size early so effective-batch logging is accurate even before Accelerator init.
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)
    cfg.optimization["accumulation_steps"] = accumulation_steps

    global_batch_size = (
        cfg.dataloader.max_train_batch_size * accumulation_steps * world_size_env
    )
    logging.info(
        f"Per-process target batch: {eff_bs}, accumulation steps: {accumulation_steps}, "
        f"estimated global batch size: {global_batch_size}"
    )

    # -------------------- Accelerator --------------------
    log_with = ["tensorboard", "wandb"] if enable_wandb else "tensorboard"

    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision="bf16",
        log_with=log_with,
        project_config=ProjectConfiguration(
            project_dir=cfg.out_dir_run, logging_dir=Path(cfg.out_dir_acc)
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if (
        accelerator.is_main_process
        and torch.cuda.device_count() > 1
        and accelerator.num_processes == 1
    ):
        logging.warning(
            "Multiple GPUs detected but running single process. "
            "Use `accelerate launch --multi_gpu marigoldv2/script/train/train.py ...` to enable multi-GPU."
        )

    # -------------------- Dynamic scaling for multi-GPU --------------------
    def _scale_step_count(value, factor):
        value_i = int(value)
        return max(1, int(math.ceil(value_i / factor)))

    def _scale_points(points, factor, max_step=None):
        if points is None:
            return points
        scaled = sorted({_scale_step_count(p, factor) for p in points if int(p) > 0})
        if max_step is not None:
            scaled = [p for p in scaled if p <= int(max_step)]
        return scaled

    gpu_scaling_cfg = cfg.optimization.get("gpu_scaling", {})
    gpu_scaling_enabled = bool(
        gpu_scaling_cfg.get("enabled", accelerator.num_processes > 1)
    )
    base_world_size = max(1, int(gpu_scaling_cfg.get("base_world_size", 1)))
    scale_factor = float(accelerator.num_processes) / float(base_world_size)

    if gpu_scaling_enabled and scale_factor > 1.0:
        old_max_iter = int(cfg.optimization.max_iter)
        cfg.optimization.max_iter = _scale_step_count(
            cfg.optimization.max_iter, scale_factor
        )

        if "vis_period" in cfg.optimization:
            cfg.optimization.vis_period = _scale_step_count(
                cfg.optimization.vis_period, scale_factor
            )
        if "val_period" in cfg.optimization:
            cfg.optimization.val_period = _scale_step_count(
                cfg.optimization.val_period, scale_factor
            )

        if "ckpt_points" in cfg.optimization:
            cfg.optimization.ckpt_points = _scale_points(
                cfg.optimization.ckpt_points,
                scale_factor,
                max_step=cfg.optimization.max_iter,
            )

        sched_cfg = cfg.optimization.get("scheduler", None)
        scale_scheduler_steps = bool(
            gpu_scaling_cfg.get("scale_scheduler_steps", False)
        )
        if sched_cfg is not None and scale_scheduler_steps:
            if "warmup_steps" in sched_cfg:
                sched_cfg.warmup_steps = _scale_step_count(
                    sched_cfg.warmup_steps, scale_factor
                )
            if "decay_steps" in sched_cfg:
                sched_cfg.decay_steps = _scale_step_count(
                    sched_cfg.decay_steps, scale_factor
                )

            if bool(gpu_scaling_cfg.get("scale_lr", False)) and "base_lr" in sched_cfg:
                sched_cfg.base_lr = float(sched_cfg.base_lr) * scale_factor

        if accelerator.is_main_process:
            logging.info(
                "Applied multi-GPU scaling: "
                f"world_size={accelerator.num_processes}, base_world_size={base_world_size}, factor={scale_factor:.2f}, "
                f"max_iter {old_max_iter}->{int(cfg.optimization.max_iter)}, "
                f"scale_scheduler_steps={scale_scheduler_steps}"
            )

    # -------------------- Ensure integer optimization fields --------------------
    def _to_int_maybe(v):
        try:
            if isinstance(v, float):
                return int(round(v))
            return int(v)
        except Exception:
            return v

    opt = cfg.optimization
    # Scalar integer fields
    for key in ["max_iter", "max_epoch", "vis_period", "val_period"]:
        if key in opt:
            opt[key] = _to_int_maybe(opt[key])

    # Scheduler nested integer fields
    sched = opt.get("scheduler", None)
    if sched is not None:
        for sk in ["warmup_steps", "decay_steps"]:
            if sk in sched:
                sched[sk] = _to_int_maybe(sched[sk])

    # List-valued points: convert each element to int when possible
    for list_key in ["vis_points", "val_points", "ckpt_points"]:
        if list_key in opt and opt[list_key] is not None:
            try:
                opt[list_key] = [_to_int_maybe(x) for x in list(opt[list_key])]
            except Exception:
                pass

    seed = cfg.get("random_seed", 2025)
    set_seed(seed, device_specific=True)

    if accelerator.is_main_process:
        print(f"Training with base seed: {seed}")

    loader_generator = torch.Generator().manual_seed(seed + accelerator.process_index)

    loader_num_workers = int(cfg.dataloader.get("num_workers", 0))
    loader_pin_memory = bool(cfg.dataloader.get("pin_memory", True))
    loader_prefetch_factor = int(cfg.dataloader.get("prefetch_factor", 2))
    loader_persistent_workers = bool(
        cfg.dataloader.get("persistent_workers", loader_num_workers > 0)
    )

    # In multi-GPU runs, each rank creating many workers can easily OOM host RAM.
    # Cap per-rank workers using the CPUs visible to the local process.
    world_size = max(1, int(accelerator.num_processes))
    if world_size > 1:
        available_cpus = os.cpu_count() or 1
        per_rank_cpus = max(1, available_cpus // world_size)
        max_workers_for_rank = max(0, per_rank_cpus - 1)
        loader_num_workers = min(loader_num_workers, max_workers_for_rank)

        if accelerator.is_main_process:
            logging.info(
                "DataLoader worker cap (multi-GPU): "
                f"world_size={world_size}, available_cpus={available_cpus}, "
                f"num_workers={loader_num_workers}"
            )

    # Keep DataLoader options valid across num_workers=0 and >0.
    loader_persistent_workers = (
        loader_persistent_workers if loader_num_workers > 0 else False
    )
    loader_prefetch_factor = loader_prefetch_factor if loader_num_workers > 0 else None

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    cfg_data = cfg.dataset

    # -------------------- Model --------------------
    print("Loading network components..")
    build_transforms(cfg.network_components, registry_category="network_components")({})
    print("Loaded network components")

    network_graph = {}
    network_graph["train"] = build_transforms(
        cfg.network_graph, registry_category="network_graph"
    )
    REGISTRY["built_network_graphs"] = network_graph
    print("Loaded network graph")
    loss_graph = build_transforms(cfg.loss_graph, registry_category="loss")
    print("Loaded loss graph")

    name_spec_ls = list(cfg.dataset.train.items())
    datasets = [build_dataset(spec) for name, spec in name_spec_ls]
    train_dataset = ConcatDataset(datasets)

    # build per-sample weights and scale by per-dataset sampling_weight
    parts = []
    for (name, spec), ds in zip(name_spec_ls, datasets):
        Li = len(ds)

        w = getattr(ds, "sample_weights", None)
        w = (
            torch.as_tensor(w, dtype=torch.double)
            if w is not None
            else torch.ones(Li, dtype=torch.double)
        )

        w = w / (w.sum() + 1e-12)
        w = w * float(getattr(spec, "sampling_weight", 1.0))

        parts.append(w)

    weights = torch.cat(parts, dim=0)

    if cfg.dataloader.max_train_batch_size > 1 and len(datasets) > 1:
        # Keep each batch within one dataset so that mixed resolutions do not
        # turn tensors such as valid_mask into ragged lists.
        sampler_prob = []
        for _, spec in name_spec_ls:
            sampler_prob.append(float(getattr(spec, "sampling_weight", 1.0)))

        batch_sampler = MixedBatchSampler(
            src_dataset_ls=datasets,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            shuffle=True,
            prob=sampler_prob,
            generator=loader_generator,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            collate_fn=custom_collate,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
            persistent_workers=loader_persistent_workers,
            prefetch_factor=loader_prefetch_factor,
        )
    else:
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=loader_generator,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            shuffle=False,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            collate_fn=custom_collate,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
            persistent_workers=loader_persistent_workers,
            prefetch_factor=loader_prefetch_factor,
        )

    # Validation loaders
    val_loaders: List[DataLoader] = []
    for disp_name, _val_dict in cfg_data.val.items():
        _val_dataset = build_dataset(_val_dict)
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
            persistent_workers=loader_persistent_workers,
            prefetch_factor=loader_prefetch_factor,
        )
        val_loaders.append(_val_loader)

    # Visualization loaders
    vis_loaders: List[DataLoader] = []
    for disp_name, _vis_dict in cfg_data.get("vis", {}).items():
        _vis_dataset = build_dataset(_vis_dict)
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=loader_num_workers,
            pin_memory=loader_pin_memory,
            persistent_workers=loader_persistent_workers,
            prefetch_factor=loader_prefetch_factor,
        )
        vis_loaders.append(_vis_loader)

        if "network_components" in _vis_dict:
            build_transforms(
                _vis_dict.network_components, registry_category="network_components"
            )({})
        if "network_graph" in _vis_dict:
            network_graph[disp_name] = build_transforms(
                _vis_dict.network_graph, registry_category="network_graph"
            )

    # --------------- Setup optimizer -----------------
    param_groups, stats = build_trainable_param_groups_from_network_components(
        cfg, REGISTRY
    )

    optimizer_name = str(cfg.optimization.get("optimizer", "AdamW8bit")).lower()
    adam_beta1 = float(cfg.optimization.get("adam_beta1", 0.9))
    adam_beta2 = float(cfg.optimization.get("adam_beta2", 0.999))
    adam_eps = float(cfg.optimization.get("adam_eps", 1e-8))
    adam_weight_decay = float(cfg.optimization.get("adam_weight_decay", 1e-4))

    if optimizer_name == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_eps,
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            param_groups,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_eps,
        )
    elif optimizer_name == "adam":
        optimizer = Adam(
            param_groups,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_eps,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{cfg.optimization.get('optimizer')}'. "
            "Use one of: AdamW8bit, AdamW, Adam"
        )

    print(
        f"Loaded optimizer with {stats['num_groups']} groups, "
        f"trainable={stats['total_trainable_numel']:,} / total={stats['total_params_numel']:,} params"
    )

    # Setup scheduler
    sched_cfg = cfg.optimization.get("scheduler", {})
    name = str(sched_cfg.get("name", "three_phase_linear")).lower()

    if name == "three_phase_linear":
        total_steps = int(sched_cfg.get("total_steps", cfg.optimization.max_iter))

        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        start_factor = float(sched_cfg.get("start_factor", 0.0))
        decay_steps = int(sched_cfg.get("decay_steps", 0))
        final_factor = float(sched_cfg.get("final_factor", 0.1))

        constant_steps = max(0, total_steps - warmup_steps - decay_steps)

        assert warmup_steps >= 0 and decay_steps >= 0 and constant_steps >= 0
        assert 0.0 <= start_factor <= 1.0
        assert final_factor > 0.0

        phases = []

        if warmup_steps > 0:
            phases.append(
                (
                    warmup_steps,
                    LinearLR(
                        optimizer,
                        start_factor=start_factor,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                )
            )

        if decay_steps > 0:
            phases.append(
                (
                    decay_steps,
                    LinearLR(
                        optimizer,
                        start_factor=1.0,
                        end_factor=final_factor,
                        total_iters=decay_steps,
                    ),
                )
            )

        if constant_steps > 0:
            phases.append(
                (
                    constant_steps,
                    LinearLR(
                        optimizer,
                        start_factor=final_factor,
                        end_factor=final_factor,
                        total_iters=constant_steps,
                    ),
                )
            )

        if not phases:
            phases = [
                (
                    1,
                    LinearLR(
                        optimizer, start_factor=1.0, end_factor=1.0, total_iters=1
                    ),
                )
            ]

        schedulers = [sch for _, sch in phases]
        cum = 0
        milestones = []
        for i in range(len(phases) - 1):
            cum += phases[i][0]
            milestones.append(cum)

        lr_scheduler = SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones
        )
    elif name == "itercosine":
        total_steps = int(sched_cfg.get("total_iter", cfg.optimization.max_iter))
        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        final_ratio = float(sched_cfg.get("final_ratio", 0.01))

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))

            progress_denom = max(1, total_steps - warmup_steps)
            progress = (
                min(max(current_step - warmup_steps, 0), progress_denom)
                / progress_denom
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler '{name}'")

    # --- collect modules to prepare ---
    nc_items = list(REGISTRY["network_components"].items())

    module_names = []
    modules = []
    for name, obj in nc_items:
        if isinstance(obj, nn.Module):
            module_names.append(name)
            modules.append(obj)

    prepared = accelerator.prepare(
        *modules, optimizer, train_loader, *val_loaders, *vis_loaders, lr_scheduler
    )

    n = len(modules)
    n_val = len(val_loaders)
    n_vis = len(vis_loaders)

    prepared_modules = prepared[:n]
    cursor = n
    optimizer = prepared[cursor]
    cursor += 1
    train_loader = prepared[cursor]
    cursor += 1
    val_loaders = list(prepared[cursor : cursor + n_val])
    cursor += n_val
    vis_loaders = list(prepared[cursor : cursor + n_vis])
    cursor += n_vis
    lr_scheduler = prepared[cursor]

    for name, mod in zip(module_names, prepared_modules):
        REGISTRY["network_components"][name] = mod

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Register hooks
    save_hook = make_save_trainables_hook(REGISTRY, accelerator)
    checkpoint_cfg = cfg.optimization.get("checkpoint", {})
    allow_missing_components = checkpoint_cfg.get("allow_missing_components", [])
    exclude_components = checkpoint_cfg.get("exclude_components", [])
    load_hook = make_load_trainables_hook(
        REGISTRY,
        accelerator,
        allow_missing_components=allow_missing_components,
        exclude_components=exclude_components,
    )
    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # Load checkpoint if specified
    checkpoint_path = cfg.paths.get("checkpoint", None)
    if checkpoint_path is not None:
        if accelerator.is_main_process:
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
        load_hook([], checkpoint_path)
    else:
        if accelerator.is_main_process:
            logging.info("No checkpoint path specified, starting from scratch")

    if accelerator.is_main_process:
        tracker_init_kwargs = {}
        if enable_wandb:
            tracker_init_kwargs["wandb"] = {
                "name": os.path.basename(str(cfg.out_dir_run)),
                "dir": str(cfg.out_dir_run),
            }
        accelerator.init_trackers("marigold-v2", init_kwargs=tracker_init_kwargs)

    # -------------------- Trainer --------------------
    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    trainer = trainer_cls(
        network_graph=network_graph,
        accelerator=accelerator,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loaders=val_loaders,
        vis_loaders=vis_loaders,
        loss_graph=loss_graph,
        lr_scheduler=lr_scheduler,
    )

    # -------------------- Training & Evaluation Loop --------------------
    print("Starting training...")

    trainer.train()


if __name__ == "__main__":
    main()
