import torch
import torch.nn as nn
import numpy as np

import os
from time import time
from torch.utils.data import DataLoader
from PIL import Image
from accelerate import DistributedType
from marigoldv2.core.registry import REGISTRY
from marigoldv2.validation.validate import validate_dataloader
import logging
from marigoldv2.util.logging_util import eval_dict_to_text
from marigoldv2.validation.metric import MetricTracker
from marigoldv2.validation import metric
from marigoldv2.network.change_network_mode import set_to_eval, set_to_train
from marigoldv2.trainer.util import save_step_info_txt, AccelerateRandomContext


class GenericTrainer:
    def __init__(
        self,
        network_graph,
        accelerator,
        loss_graph,
        optimizer,
        train_loader,
        val_loaders,
        vis_loaders,
        lr_scheduler=None,
    ):
        self.network_graph = network_graph
        self.accelerator = accelerator
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.vis_loaders = vis_loaders
        self.loss_graph = loss_graph
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_info = {}
        self.train_vis_loader = None
        self.train_vis_iter = None

    def _aggregate_metric_tracker(self, metric_tracker: MetricTracker):
        keys = list(metric_tracker._data.index)
        if not keys:
            return {}

        totals_local = torch.tensor(
            [float(metric_tracker._data.loc[k, "total"]) for k in keys],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        counts_local = torch.tensor(
            [float(metric_tracker._data.loc[k, "counts"]) for k in keys],
            device=self.accelerator.device,
            dtype=torch.float64,
        )

        totals = self.accelerator.reduce(totals_local, reduction="sum")
        counts = self.accelerator.reduce(counts_local, reduction="sum")
        avgs = totals / counts.clamp_min(1.0)

        return {k: float(avgs[i].item()) for i, k in enumerate(keys)}

    def _normalize_metric_name_for_logging(self, metric_name: str) -> str:
        """Optionally normalize metric names for logger backends."""
        normalized = metric_name

        if getattr(self, "strip_log_from_metric_names_for_logging", False):
            normalized = normalized.replace("_log_", "_", 1)

        if getattr(self, "strip_disp_from_metric_names_for_logging", False):
            normalized = normalized.replace("_disp_", "_", 1)

        if normalized in self.metric_names and normalized != metric_name:
            collision_key = (metric_name, normalized)
            if collision_key not in self._metric_name_collision_warned:
                logging.warning(
                    "Metric name normalization collision for '%s' -> '%s'; "
                    "keeping original metric name in logs.",
                    metric_name,
                    normalized,
                )
                self._metric_name_collision_warned.add(collision_key)
            return metric_name

        return normalized

    def _get_tensorboard_writer(self):
        """Return a TensorBoard SummaryWriter (or compatible object) if available."""
        try:
            tb_obj = self.accelerator.get_tracker("tensorboard", unwrap=True)
            # In newer accelerate versions this is already a SummaryWriter.
            if hasattr(tb_obj, "add_image"):
                return tb_obj
            # In older wrappers, writer is nested.
            if hasattr(tb_obj, "writer") and hasattr(tb_obj.writer, "add_image"):
                return tb_obj.writer
        except Exception:
            pass

        # Fallback path for tracker wrapper variations.
        for tracker in getattr(self.accelerator, "trackers", []):
            writer = getattr(tracker, "writer", None)
            if writer is not None and hasattr(writer, "add_image"):
                return writer

        return None

    def _get_wandb_run(self):
        """Return a wandb run/tracker object if available."""
        try:
            wb_obj = self.accelerator.get_tracker("wandb", unwrap=True)
            if hasattr(wb_obj, "log"):
                return wb_obj
        except Exception:
            pass
        return None

    def _compute_grad_norm(self, params, norm_type: float = 2.0):
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            try:
                param_norm = float(p.grad.detach().norm(norm_type).cpu().item())
            except Exception:
                # Fallback if grad is on CPU or other device
                param_norm = float(p.grad.detach().norm(norm_type).item())
            total += param_norm**norm_type
        if total == 0.0:
            return 0.0
        return total ** (1.0 / norm_type)

    def _log_saved_visualizations_to_tensorboard(
        self,
        saved_output_paths,
        dataset_name: str,
        phase: str,
        max_images_per_folder: int = 8,
    ):
        if not saved_output_paths:
            return
        if not self.accelerator.is_main_process:
            return

        writer = self._get_tensorboard_writer()
        wandb_run = self._get_wandb_run()
        wandb_mod = None
        if wandb_run is not None:
            try:
                import wandb as wandb_mod
            except Exception as e:
                logging.warning(
                    f"W&B tracker found but wandb import failed; skipping W&B image logging ({e})"
                )
                wandb_run = None

        if writer is None and wandb_run is None:
            logging.warning(
                "No TensorBoard/W&B image logger available; skipping image logging for this step."
            )
            return

        def _extract_sample_id(entry, folder_name: str):
            image_path = entry.get("path") if isinstance(entry, dict) else None
            if image_path:
                stem = os.path.splitext(os.path.basename(image_path))[0]
                suffix = f"_{folder_name}"
                if stem.endswith(suffix):
                    return stem[: -len(suffix)]
                return stem
            batch_index = entry.get("batch_index") if isinstance(entry, dict) else None
            if batch_index is None:
                return None
            return f"batch_{int(batch_index):06d}"

        folder_to_entries_by_sample = {}
        common_sample_ids = None
        for folder_name, entries in saved_output_paths.items():
            sample_to_entry = {}
            for entry in entries:
                sample_id = _extract_sample_id(entry, folder_name)
                if sample_id is None:
                    continue
                sample_to_entry[sample_id] = entry
            if not sample_to_entry:
                continue
            folder_to_entries_by_sample[folder_name] = sample_to_entry
            sample_ids = set(sample_to_entry.keys())
            if common_sample_ids is None:
                common_sample_ids = sample_ids
            else:
                common_sample_ids &= sample_ids

        selected_common_ids = []
        if common_sample_ids:
            ordered_common_ids = sorted(common_sample_ids)
            num_to_log = min(max_images_per_folder, len(ordered_common_ids))
            if len(ordered_common_ids) > num_to_log:
                selected_indices = np.random.choice(
                    len(ordered_common_ids), size=num_to_log, replace=False
                )
                selected_common_ids = [
                    ordered_common_ids[idx] for idx in sorted(selected_indices.tolist())
                ]
            else:
                selected_common_ids = ordered_common_ids

        num_logged = 0
        from collections import defaultdict

        wandb_images_by_dataset = defaultdict(list)
        for folder_name, entries in saved_output_paths.items():
            if not entries:
                continue

            if selected_common_ids and folder_name in folder_to_entries_by_sample:
                selected_entries = [
                    folder_to_entries_by_sample[folder_name][sample_id]
                    for sample_id in selected_common_ids
                ]
            else:
                num_to_log = min(max_images_per_folder, len(entries))
                if len(entries) > num_to_log:
                    selected_indices = np.random.choice(
                        len(entries), size=num_to_log, replace=False
                    )
                    selected_entries = [entries[idx] for idx in selected_indices]
                else:
                    selected_entries = entries

            for i, entry in enumerate(selected_entries):
                image_path = entry.get("path") if isinstance(entry, dict) else None
                if not image_path or not os.path.exists(image_path):
                    continue
                try:
                    with Image.open(image_path) as img:
                        img_np = np.asarray(img.convert("RGB"), dtype=np.uint8)
                    tag = f"{phase}/{dataset_name}/{folder_name}/{i:03d}"
                    if writer is not None:
                        img_ts = torch.from_numpy(img_np).permute(2, 0, 1)
                        writer.add_image(
                            tag, img_ts, global_step=self.global_step, dataformats="CHW"
                        )
                    if wandb_run is not None and wandb_mod is not None:
                        dataset_key = f"{phase}/{dataset_name}/images"
                        wandb_images_by_dataset[dataset_key].append(
                            wandb_mod.Image(
                                img_np, caption=os.path.basename(image_path)
                            )
                        )
                    num_logged += 1
                except Exception as e:
                    logging.warning(f"Failed to log visualization: {image_path} ({e})")

        if num_logged > 0 and hasattr(writer, "flush"):
            writer.flush()
        if wandb_run is not None and wandb_images_by_dataset:
            # log per-dataset image lists so W&B groups don't mix datasets
            wandb_run.log(
                {k: v for k, v in wandb_images_by_dataset.items()},
                step=self.global_step,
            )

    def train(self):
        self.cfg = REGISTRY["cfg"]
        accelerator = self.accelerator
        vis_period = self.cfg.optimization.vis_period
        vis_points = self.cfg.optimization.vis_points
        ckpt_points = self.cfg.optimization.ckpt_points
        val_period = self.cfg.optimization.val_period
        val_points = self.cfg.optimization.val_points
        out_dir_run = self.cfg.out_dir_run
        # Initialize max gradient norm from config once per run (fallback 1.0)
        opt_cfg = getattr(self.cfg, "optimization", None)
        if opt_cfg is not None:
            if hasattr(opt_cfg, "get"):
                self.max_grad_norm = float(opt_cfg.get("max_grad_norm", 1.0))
            else:
                self.max_grad_norm = float(getattr(opt_cfg, "max_grad_norm", 1.0))
        else:
            self.max_grad_norm = 1.0
        console_log_every = int(
            self.cfg.get("logging", {}).get(
                "console_log_every",
                self.cfg.get("logging", {}).get(
                    "consol_level", self.cfg.get("logging", {}).get("console_level", 1)
                ),
            )
        )
        console_log_every = max(1, console_log_every)
        self.metric_funcs = [
            getattr(metric, _met) for _met in self.cfg.eval.eval_metrics
        ]
        self.metric_names = [m.__name__ for m in self.metric_funcs]
        self.train_metrics = MetricTracker(
            *(["loss"] + [k for k, v in self.cfg.loss_graph.items() if "weight" in v])
        )
        self.val_metrics = MetricTracker(*self.metric_names)
        val_cfg = self.cfg.validation
        self.main_val_metric = val_cfg.main_val_metric
        self.main_val_metric_goal = val_cfg.main_val_metric_goal  # minimize | maximize
        self.strip_log_from_metric_names_for_logging = bool(
            getattr(val_cfg, "strip_log_from_metric_names_for_logging", False)
        )
        self.strip_disp_from_metric_names_for_logging = bool(
            getattr(val_cfg, "strip_disp_from_metric_names_for_logging", False)
        )
        self._metric_name_collision_warned = set()

        if self.main_val_metric_goal == "minimize":
            self.best_val_score = float("inf")
        else:
            self.best_val_score = float("-inf")

        self.global_step = 0
        self.max_global_step = self.cfg.optimization.max_iter
        # An eval/train round-trip before the loop lowers peak VAE memory.
        set_to_eval()
        set_to_train()

        for epoch in range(self.cfg.optimization.max_epoch):
            start_data_load = time()
            # accumulator for logging the effective (averaged over micro-batches) loss
            running_weighted_loss = torch.tensor(0.0, device=accelerator.device)
            for step, batch in enumerate(self.train_loader):
                if time() - start_data_load > 0.1:
                    print(
                        f"Data loading took more than 0.1s: {time() - start_data_load}s"
                    )
                batch["out"] = {}  # outputs and intermediate results
                with accelerator.accumulate(), accelerator.autocast():
                    self.network_graph["train"](batch)

                # Loss
                batch["loss"] = {}
                batch["weighted_loss"] = 0
                batch["global_step"] = self.global_step
                self.loss_graph(batch)
                # make sure we have a tensor on the accelerator device to accumulate
                cur_loss = batch["weighted_loss"]
                if not isinstance(cur_loss, torch.Tensor):
                    cur_loss = torch.as_tensor(
                        cur_loss or 0.0, device=accelerator.device, dtype=torch.float32
                    )
                else:
                    cur_loss = cur_loss.detach().to(accelerator.device)

                running_weighted_loss = running_weighted_loss + cur_loss

                loss = batch["weighted_loss"]
                if not isinstance(loss, torch.Tensor):
                    loss = torch.as_tensor(
                        loss or 0.0, device=accelerator.device, dtype=torch.float32
                    )
                if not torch.isfinite(loss.detach()).all():
                    if accelerator.is_main_process:
                        logging.warning(
                            "Skipping optimizer step at global_step=%s because weighted_loss is non-finite.",
                            self.global_step,
                        )
                    self.optimizer.zero_grad()
                    running_weighted_loss = torch.tensor(0.0, device=accelerator.device)
                    continue
                accelerator.backward(loss)
                # Optimization Step
                if accelerator.sync_gradients:
                    self.global_step += 1

                    # Checkpointing, visualization, and validation
                    is_main_process = (
                        accelerator.is_main_process
                        or accelerator.distributed_type == DistributedType.DEEPSPEED
                    )
                    do_visualization = (
                        self.global_step % vis_period == 0
                        or self.global_step in vis_points
                    )
                    do_validation = self.global_step % val_period == 0 or (
                        val_points and self.global_step in val_points
                    )
                    do_ckpt = ckpt_points and self.global_step in ckpt_points

                    if do_validation:
                        set_to_eval()
                        with AccelerateRandomContext(accelerator, seed=1234):
                            is_best_checkpoint = self.validate()
                            # After each validation run, optionally visualize a few training samples
                            self.visualize_training_samples_after_validation()
                        if is_best_checkpoint:
                            save_path = os.path.join(
                                out_dir_run, "checkpoint", "checkpoint-best"
                            )
                            accelerator.save_state(save_path)
                            if is_main_process:
                                save_step_info_txt(
                                    save_path, self.global_step, self.val_info
                                )
                        save_path = os.path.join(
                            out_dir_run, "checkpoint", "checkpoint-latest"
                        )
                        accelerator.save_state(save_path)
                        set_to_train()
                    if do_ckpt:
                        set_to_eval()
                        save_path = os.path.join(
                            out_dir_run, "checkpoint", f"checkpoint-{self.global_step}"
                        )
                        accelerator.save_state(save_path)
                        if is_main_process:
                            save_step_info_txt(
                                save_path, self.global_step, self.val_info
                            )
                        set_to_train()
                    if do_visualization:
                        set_to_eval()
                        with AccelerateRandomContext(accelerator, seed=1234):
                            self.visualize()
                        save_path = os.path.join(
                            out_dir_run, "checkpoint", "checkpoint-latest-vis"
                        )
                        accelerator.save_state(save_path)
                        if is_main_process:
                            save_step_info_txt(
                                save_path, self.global_step, self.val_info
                            )

                        set_to_train()

                    seen = set()
                    params = []

                    for name, module in REGISTRY["network_components"].items():
                        if not isinstance(module, nn.Module):
                            continue
                        for p in module.parameters():
                            if not p.requires_grad:
                                continue
                            pid = id(p)
                            if pid in seen:
                                continue
                            seen.add(pid)
                            params.append(p)

                    # compute pre-clip gradient norm for logging
                    pre_grad_norm = self._compute_grad_norm(params)

                    accelerator.clip_grad_norm_(params, self.max_grad_norm)

                    # compute post-clip gradient norm for logging
                    post_grad_norm = self._compute_grad_norm(params)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    effective_loss = running_weighted_loss.detach()
                    # reset accumulator for next optimizer step
                    running_weighted_loss = torch.tensor(0.0, device=accelerator.device)

                    logs = {
                        "train/loss": float(effective_loss.item()),
                        **{
                            f"unweighted train/{key}": val.detach().item()
                            for key, val in batch["loss"].items()
                        },
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "grad/pre_norm": float(pre_grad_norm),
                        "grad/post_norm": float(post_grad_norm),
                    }

                    accelerator.log(logs, step=self.global_step)
                    if (
                        self.global_step % console_log_every == 0
                        and accelerator.is_main_process
                    ):
                        print("The logs: ", logs)

                        # Peak memory in MB (peak since last reset)
                        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                        print(
                            f"Step {self.global_step}: Peak memory usage: {peak_mem:.2f} MB"
                        )
                        torch.cuda.reset_peak_memory_stats()

                start_data_load = time()

                if self.max_global_step <= self.global_step:
                    break

            if self.max_global_step <= self.global_step:
                break

        accelerator.end_training()

    def validate(self):
        ckpt_name = f"iter_{self.global_step:06d}"
        main_metric_name = self.main_val_metric  # from cfg.validation
        per_dataset_scores = []
        self.val_info = {}
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.cfg.out_dir_vis, val_dataset_name, ckpt_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            REGISTRY["cfg"]["vis_out_dir"] = vis_out_dir

            # Check if dataset has its own val_period and val_points
            dataset_cfg = val_loader.dataset.kwargs
            dataset_val_period = dataset_cfg.get("val_period", None)
            dataset_val_points = dataset_cfg.get("val_points", None)
            # Use dataset-specific validation schedule if provided, otherwise use global
            if dataset_val_period is not None or dataset_val_points is not None:
                # Dataset has its own validation schedule
                should_run = False
                if (
                    dataset_val_period is not None
                    and self.global_step % dataset_val_period == 0
                ):
                    should_run = True
                if (
                    dataset_val_points is not None
                    and self.global_step in dataset_val_points
                ):
                    should_run = True

                if not should_run:
                    logging.info(
                        f"Skipping validation on `{val_dataset_name}` (dataset-specific schedule, current step: {self.global_step})"
                    )
                    continue

            saved_output_paths = validate_dataloader(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                enable_visualization=self.accelerator.is_main_process,
            )
            if self.accelerator.is_main_process:
                self._log_saved_visualizations_to_tensorboard(
                    saved_output_paths,
                    dataset_name=val_dataset_name,
                    phase="val_vis",
                )
            val_metric_dict = self._aggregate_metric_tracker(self.val_metrics)

            # Print metric tracker results to console for immediate visibility
            if self.accelerator.is_main_process:
                print(f"=== Metric Tracker Results for {val_dataset_name} ===")
                for metric_name, metric_value in val_metric_dict.items():
                    print(f"  {metric_name}: {metric_value:.6f}")
                print("=" * 50)

            # Reset metric tracker for next dataset
            self.val_metrics.reset()

            if self.accelerator.is_main_process:
                logging.info(
                    f"Iter {self.global_step}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
                )
                scalars = {
                    f"val/{val_dataset_name}/{self._normalize_metric_name_for_logging(k)}": float(
                        v
                    )
                    for k, v in val_metric_dict.items()
                }
                self.accelerator.log(scalars, step=self.global_step)

            # save to file
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                num_samples=val_loader.dataset.__len__(),
            )
            _save_to = os.path.join(
                self.cfg.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.global_step:06d}.txt",
            )
            if self.accelerator.is_main_process:
                with open(_save_to, "w+") as f:
                    f.write(eval_text)

            # collect value for main metric (if present)
            if main_metric_name in val_metric_dict:
                per_dataset_scores.append(float(val_metric_dict[main_metric_name]))
            else:
                logging.warning(
                    f"Main metric '{main_metric_name}' not found for dataset '{val_dataset_name}'. "
                    f"Available keys: {list(val_metric_dict.keys())}"
                )

            self.val_info[val_dataset_name] = val_metric_dict

        # no valid scores -> nothing to compare
        if not per_dataset_scores:
            logging.warning(
                f"No valid scores collected for main metric '{main_metric_name}' at step {self.global_step}."
            )
            return False

        current_score = sum(per_dataset_scores) / len(per_dataset_scores)

        # decide better based on goal
        if self.main_val_metric_goal == "minimize":
            is_best = current_score < self.best_val_score
            better_str = "<"
        else:  # maximize
            is_best = current_score > self.best_val_score
            better_str = ">"

        if is_best:
            logging.info(
                f"New BEST checkpoint at step {self.global_step}: "
                f"{main_metric_name}={current_score:.6f} {better_str} "
                f"{self.best_val_score:.6f} (old best)"
            )
            self.best_val_score = current_score
        else:
            logging.info(
                f"Checkpoint at step {self.global_step}: {main_metric_name}={current_score:.6f} "
                f"did not beat best {self.best_val_score:.6f}"
            )

        return is_best

    def visualize(self):
        ckpt_name = f"iter_{self.global_step:06d}"
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.cfg.out_dir_vis, vis_dataset_name, ckpt_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            REGISTRY["cfg"]["vis_out_dir"] = vis_out_dir

            saved_output_paths = validate_dataloader(
                data_loader=val_loader,
                enable_visualization=self.accelerator.is_main_process,
            )
            self._log_saved_visualizations_to_tensorboard(
                saved_output_paths,
                dataset_name=vis_dataset_name,
                phase="vis",
            )

    def _get_train_vis_loader(self):
        """
        Lazily build a simple sequential DataLoader over the training dataset
        for visualization purposes (no sampler / shuffling).
        Runs only on the main process.
        """
        if self.train_vis_loader is None:
            base_loader = self.train_loader
            self.train_vis_loader = DataLoader(
                dataset=base_loader.dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                collate_fn=base_loader.collate_fn,
            )
        return self.train_vis_loader

    def _next_train_vis_batch(self):
        """
        Return the next batch from the visualization DataLoader.
        Cycles when reaching the end.
        """
        loader = self._get_train_vis_loader()
        if self.train_vis_iter is None:
            self.train_vis_iter = iter(loader)
        try:
            batch = next(self.train_vis_iter)
        except StopIteration:
            self.train_vis_iter = iter(loader)
            batch = next(self.train_vis_iter)
        return batch

    def visualize_training_samples_after_validation(self):
        pass
