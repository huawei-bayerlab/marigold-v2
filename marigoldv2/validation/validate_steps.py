import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import cm
from pathlib import Path

from marigoldv2.core.registry import register, REGISTRY
from marigoldv2.validation.util import get_nested_key
from marigoldv2.validation import metric


def _record_saved_output_path(
    batch, folder_name: str, save_path: str, batch_index=None
):
    saved = batch.setdefault("_saved_output_paths", {})
    per_folder = saved.setdefault(folder_name, [])
    per_folder.append(
        {
            "batch_index": int(batch_index) if batch_index is not None else None,
            "path": str(save_path),
        }
    )


def _get_pred_name(
    rgb_basename: str, name_mode: str = "id", suffix: str = ".npy"
) -> str:
    if name_mode == "rgb_id":
        parts = rgb_basename.split("_", 1)
        pred_basename = "pred_" + (parts[1] if len(parts) > 1 else rgb_basename)
    elif name_mode == "i_d_rgb":
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif name_mode == "id":
        pred_basename = "pred_" + rgb_basename
    elif name_mode == "rgb_i_d":
        parts = rgb_basename.split("_")
        pred_basename = "pred_" + (
            "_".join(parts[1:]) if len(parts) > 1 else rgb_basename
        )
    elif name_mode in {"basename", "none"}:
        pred_basename = rgb_basename
    else:
        raise ValueError(f"Unknown name_mode for SaveDepthNpy: {name_mode}")

    pred_basename = os.path.splitext(pred_basename)[0] + suffix
    return pred_basename


def _get_orig_hw_for_index(batch, index: int):
    meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
    if not isinstance(meta, dict):
        return None

    orig_res = meta.get("orig_res", None)
    if orig_res is None:
        return None

    hw = None
    if torch.is_tensor(orig_res):
        if orig_res.ndim == 2 and orig_res.shape[0] > index and orig_res.shape[1] >= 2:
            hw = [orig_res[index, 0].item(), orig_res[index, 1].item()]
        elif orig_res.ndim == 1 and orig_res.shape[0] >= 2 and index == 0:
            hw = [orig_res[0].item(), orig_res[1].item()]
    elif isinstance(orig_res, (list, tuple)):
        if (
            len(orig_res) > index
            and isinstance(orig_res[index], (list, tuple))
            and len(orig_res[index]) >= 2
        ):
            hw = [orig_res[index][0], orig_res[index][1]]
        elif (
            len(orig_res) >= 2
            and index == 0
            and not isinstance(orig_res[0], (list, tuple))
        ):
            hw = [orig_res[0], orig_res[1]]

    if hw is None:
        return None

    try:
        h, w = int(hw[0]), int(hw[1])
    except Exception:
        return None

    if h <= 0 or w <= 0:
        return None
    return h, w


@register("validation_steps")
class RunInference:
    def __init__(self, **kwargs):
        pass

    def __call__(self, batch):
        dataset_disp_name = batch["dataset_disp_name"]
        network_graph = REGISTRY["built_network_graphs"]
        cfg = REGISTRY["cfg"]
        if dataset_disp_name in network_graph:
            model = network_graph[dataset_disp_name]
        else:
            model = network_graph["train"]

        batch["out"] = {}
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=eval(cfg.trainer.dtype)):
            model(batch)


@register("validation_steps")
class VisualizeRGB:
    def __init__(self, **kwargs):
        self.batch_key = kwargs.get("batch_key", "out/pixel_pred")
        self.folder_name = kwargs.get("folder_name", "rgb_norm")
        self.file_path_key = kwargs.get("file_path_key", None)
        self.value_range = kwargs.get("value_range", [-1, 1])
        self.add_file_counter = kwargs.get("add_file_counter", True)

    def __call__(self, batch):
        cfg = REGISTRY["cfg"]
        dataset_disp_name = batch["dataset_disp_name"]

        if "override_vis_dir" in cfg.get("paths", {}):
            output_dir = os.path.join(
                cfg.paths.override_vis_dir, dataset_disp_name, self.folder_name
            )
        else:
            output_dir = os.path.join(cfg["vis_out_dir"], self.folder_name)

        os.makedirs(output_dir, exist_ok=True)

        existing_files = set()
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if os.path.isfile(os.path.join(output_dir, f)):
                    existing_files.add(f)
        file_counter = len(existing_files)

        if self.file_path_key is None:
            file_path_key = cfg.validation.get("file_path_key", "rgb_path")
        else:
            file_path_key = self.file_path_key

        keys = self.batch_key.split("/")
        tensor_data = batch
        for key in keys:
            if key not in tensor_data:
                return
            tensor_data = tensor_data[key]

        batch_size = len(batch["idx"]) if "idx" in batch else 1

        for b in range(batch_size):
            if file_path_key in batch.get("annotation", {}):
                file_path = batch["annotation"][file_path_key][b]
                base_name = file_path.split("/")[-1].rsplit(".", 1)[0]
            else:
                base_name = f"sample_{b:04d}"

            if self.add_file_counter:
                scene_name = f"{file_counter:04d}_{base_name}"
            else:
                scene_name = f"{base_name}"
            file_counter += 1

            if isinstance(tensor_data, torch.Tensor):
                if tensor_data.ndim == 4:
                    tensor = tensor_data[b]
                elif tensor_data.ndim == 3:
                    tensor = tensor_data if b == 0 else None
                else:
                    tensor = None
            elif isinstance(tensor_data, list):
                tensor = tensor_data[b] if b < len(tensor_data) else None
            else:
                tensor = None

            if tensor is None:
                continue

            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().float()

            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                    tensor_np = np.transpose(tensor.numpy(), (1, 2, 0))
                else:
                    tensor_np = tensor.numpy()
            else:
                tensor_np = np.asarray(tensor)
                if tensor_np.ndim == 3 and tensor_np.shape[0] in (1, 3):
                    tensor_np = np.transpose(tensor_np, (1, 2, 0))

            min_val, max_val = self.value_range[0], self.value_range[1]

            if min_val == -1 and max_val == 1:
                img_array = ((tensor_np + 1.0) / 2.0).clip(0.0, 1.0) * 255
            elif min_val == 0 and max_val == 1:
                img_array = tensor_np.clip(0.0, 1.0) * 255
            elif min_val == 0 and max_val == 255:
                img_array = tensor_np.clip(0.0, 255.0)
            else:
                img_array = ((tensor_np - min_val) / (max_val - min_val)).clip(
                    0.0, 1.0
                ) * 255

            img_array = img_array.round().astype(np.uint8)

            save_path = os.path.join(output_dir, f"{scene_name}_{self.folder_name}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            _record_saved_output_path(batch, self.folder_name, save_path, batch_index=b)


@register("validation_steps")
class VisualizeDepth:
    """Visualize depth map using OpenCV colormap."""

    def __init__(self, **kwargs):
        self.batch_key = kwargs.get("batch_key", "depth_m")
        self.folder_name = kwargs.get("folder_name", "depth")
        self.file_path_key = kwargs.get("file_path_key", None)
        self.colormap = kwargs.get("colormap", "TURBO")
        self.normalize = kwargs.get("normalize", True)
        self.depth_min = kwargs.get("depth_min", None)
        self.depth_max = kwargs.get("depth_max", None)
        self.inverse_depth = kwargs.get("inverse_depth", False)
        self.epsilon = kwargs.get("epsilon", 1e-6)
        self.show_legend = kwargs.get("show_legend", True)
        self.resize_to_orig_res = bool(kwargs.get("resize_to_orig_res", False))
        self.legend_num_ticks = int(kwargs.get("legend_num_ticks", 6))
        self.legend_title = kwargs.get("legend_title", "Depth (m)")

    def _append_depth_legend(
        self, rgb_bgr, colormap_cv, vis_min, vis_max, mpl_cmap_name=None
    ):
        h, w = rgb_bgr.shape[:2]
        if not np.isfinite(vis_min) or not np.isfinite(vis_max):
            return rgb_bgr

        pad = 8
        bar_w = max(28, int(round(h * 0.04)))
        text_w = 140
        title_h = 24
        bottom_h = 16
        bar_h = max(40, h - title_h - bottom_h)
        y0 = title_h
        y1 = y0 + bar_h

        canvas = np.full((h, w + pad + bar_w + text_w, 3), 255, dtype=np.uint8)
        canvas[:, :w] = rgb_bgr
        x0 = w + pad

        if mpl_cmap_name is not None:
            grad01 = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(-1, 1)
            grad01 = np.repeat(grad01, bar_w, axis=1)
            colorbar_rgb = (cm.get_cmap(mpl_cmap_name)(grad01)[..., :3] * 255.0).astype(
                np.uint8
            )
            colorbar = cv2.cvtColor(colorbar_rgb, cv2.COLOR_RGB2BGR)
        else:
            grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
            grad = np.repeat(grad, bar_w, axis=1)
            colorbar = cv2.applyColorMap(grad, colormap_cv)
        canvas[y0:y1, x0 : x0 + bar_w] = colorbar
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y1 - 1), (0, 0, 0), 1)
        cv2.putText(
            canvas,
            self.legend_title,
            (x0, max(14, title_h - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        n_ticks = max(2, self.legend_num_ticks)
        for i in range(n_ticks):
            t = i / (n_ticks - 1)
            y = int(round(y0 + t * (bar_h - 1)))
            cv2.line(canvas, (x0 + bar_w, y), (x0 + bar_w + 6, y), (0, 0, 0), 1)
            vis_val = vis_max - t * (vis_max - vis_min)
            if self.inverse_depth:
                depth_val = 1.0 / max(float(vis_val), self.epsilon)
            else:
                depth_val = float(vis_val)
            label = f"{depth_val:.2f}"
            cv2.putText(
                canvas,
                label,
                (x0 + bar_w + 10, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def __call__(self, batch):
        cfg = REGISTRY["cfg"]
        dataset_disp_name = batch["dataset_disp_name"]

        if "override_vis_dir" in cfg.get("paths", {}):
            output_dir = os.path.join(
                cfg.paths.override_vis_dir, dataset_disp_name, self.folder_name
            )
        else:
            output_dir = os.path.join(cfg["vis_out_dir"], self.folder_name)

        os.makedirs(output_dir, exist_ok=True)

        existing_files = set()
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if os.path.isfile(os.path.join(output_dir, f)):
                    existing_files.add(f)
        file_counter = len(existing_files)

        if self.file_path_key is None:
            file_path_key = cfg.validation.get("file_path_key", "rgb_path")
        else:
            file_path_key = self.file_path_key

        keys = self.batch_key.split("/")
        tensor_data = batch
        for key in keys:
            if key not in tensor_data:
                return
            tensor_data = tensor_data[key]

        batch_size = len(batch["idx"]) if "idx" in batch else 1

        for b in range(batch_size):
            if file_path_key in batch.get("annotation", {}):
                file_path = batch["annotation"][file_path_key][b]
                base_name = file_path.split("/")[-1].rsplit(".", 1)[0]
            else:
                base_name = f"sample_{b:04d}"

            scene_name = f"{file_counter:04d}_{base_name}"
            file_counter += 1

            if isinstance(tensor_data, torch.Tensor):
                if tensor_data.ndim == 4:
                    depth = tensor_data[b]
                elif tensor_data.ndim == 3:
                    depth = tensor_data if b == 0 else None
                elif tensor_data.ndim == 2:
                    depth = tensor_data if b == 0 else None
                else:
                    depth = None
            elif isinstance(tensor_data, list):
                depth = tensor_data[b] if b < len(tensor_data) else None
            else:
                depth = None

            if depth is None:
                continue

            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().float()

            if isinstance(depth, torch.Tensor):
                depth_np = depth.numpy()
            else:
                depth_np = np.asarray(depth)

            if depth_np.ndim == 3:
                if depth_np.shape[0] == 1:
                    depth_np = depth_np[0]
                elif depth_np.shape[2] == 1:
                    depth_np = depth_np[:, :, 0]
                else:
                    depth_np = depth_np[0]
            elif depth_np.ndim != 2:
                continue

            if self.resize_to_orig_res:
                orig_hw = _get_orig_hw_for_index(batch, b)
                if orig_hw is not None and depth_np.shape != orig_hw:
                    depth_np = cv2.resize(
                        depth_np.astype(np.float32),
                        (orig_hw[1], orig_hw[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

            if self.inverse_depth:
                depth_np = 1.0 / (depth_np + self.epsilon)

            finite = np.isfinite(depth_np)
            if np.any(finite):
                vis_min_data = float(np.nanmin(depth_np[finite]))
                vis_max_data = float(np.nanmax(depth_np[finite]))
            else:
                vis_min_data = 0.0
                vis_max_data = 1.0

            if self.normalize:
                if self.depth_min is not None and self.depth_max is not None:
                    vis_min = float(self.depth_min)
                    vis_max = float(self.depth_max)
                else:
                    vis_min = vis_min_data
                    vis_max = vis_max_data

                if vis_max > vis_min:
                    depth_normalized = (depth_np - vis_min) / (vis_max - vis_min)
                else:
                    depth_normalized = np.zeros_like(depth_np)
            else:
                depth_normalized = depth_np
                vis_min = vis_min_data
                vis_max = vis_max_data

            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

            colormap_map = {
                "JET": cv2.COLORMAP_JET,
                "VIRIDIS": cv2.COLORMAP_VIRIDIS,
                "PLASMA": cv2.COLORMAP_PLASMA,
                "INFERNO": cv2.COLORMAP_INFERNO,
                "MAGMA": cv2.COLORMAP_MAGMA,
                "TURBO": cv2.COLORMAP_TURBO,
            }

            if self.colormap.upper() in {"SPECTRAL", "SPECTRAL_R"}:
                mpl_cmap_name = (
                    "Spectral_r"
                    if self.colormap.upper() == "SPECTRAL_R"
                    else "Spectral"
                )
                depth_rgb = (
                    cm.get_cmap(mpl_cmap_name)(depth_normalized.clip(0.0, 1.0))[..., :3]
                    * 255.0
                ).astype(np.uint8)
                rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
                colormap_cv = cv2.COLORMAP_TURBO
            elif self.colormap.upper() == "GRAY":
                rgb = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
                mpl_cmap_name = None
            else:
                colormap_cv = colormap_map.get(
                    self.colormap.upper(), cv2.COLORMAP_TURBO
                )
                rgb = cv2.applyColorMap(depth_uint8, colormap_cv)
                mpl_cmap_name = None
            if self.show_legend:
                rgb = self._append_depth_legend(
                    rgb, colormap_cv, vis_min, vis_max, mpl_cmap_name=mpl_cmap_name
                )

            save_path = os.path.join(output_dir, f"{scene_name}_{self.folder_name}.png")
            cv2.imwrite(save_path, rgb)
            _record_saved_output_path(batch, self.folder_name, save_path, batch_index=b)


@register("validation_steps")
class SaveDepthNpy:
    def __init__(self, **kwargs):
        self.batch_key = kwargs.get("batch_key", "out/depth_rel_pred_m11")
        self.folder_name = kwargs.get("folder_name", "")
        self.file_path_key = kwargs.get("file_path_key", None)
        self.base_dir = kwargs.get("base_dir", None)
        self.name_mode = kwargs.get("name_mode", "id")
        channel = kwargs.get("channel", None)
        self.channel = int(channel) if channel is not None else None
        self.resize_to_orig_res = bool(kwargs.get("resize_to_orig_res", False))

    def __call__(self, batch):
        cfg = REGISTRY["cfg"]
        dataset_disp_name = batch["dataset_disp_name"]

        if "override_vis_dir" in cfg.get("paths", {}):
            output_root = Path(cfg.paths.override_vis_dir) / dataset_disp_name
        else:
            output_root = Path(cfg["vis_out_dir"])

        if self.folder_name:
            output_root = output_root / self.folder_name
        output_root.mkdir(parents=True, exist_ok=True)

        if self.file_path_key is None:
            file_path_key = cfg.validation.get("file_path_key", "rgb_path")
        else:
            file_path_key = self.file_path_key

        keys = self.batch_key.split("/")
        tensor_data = batch
        for key in keys:
            if key not in tensor_data:
                logging.warning(
                    f"SaveDepthNpy: missing batch key '{self.batch_key}' in batch for dataset {dataset_disp_name}"
                )
                return
            tensor_data = tensor_data[key]

        batch_size = len(batch["idx"]) if "idx" in batch else 1
        # absolute() rather than resolve() keeps symlinked scene folders relative.
        base_dir_path = Path(self.base_dir).absolute() if self.base_dir else None

        for b in range(batch_size):
            if file_path_key in batch.get("annotation", {}):
                file_path = str(batch["annotation"][file_path_key][b])
                file_path_obj = Path(file_path)
                rgb_basename = file_path_obj.name
            else:
                file_path_obj = None
                rgb_basename = f"sample_{b:04d}.png"

            if isinstance(tensor_data, torch.Tensor):
                if tensor_data.ndim == 4:
                    depth = tensor_data[b]
                elif tensor_data.ndim in (2, 3):
                    depth = tensor_data if b == 0 else None
                else:
                    depth = None
            elif isinstance(tensor_data, list):
                depth = tensor_data[b] if b < len(tensor_data) else None
            else:
                depth = None

            if depth is None:
                logging.debug(
                    f"SaveDepthNpy: no depth tensor for batch index {b} (dataset {dataset_disp_name})"
                )
                continue

            if isinstance(depth, torch.Tensor):
                depth = depth.detach().cpu().float().numpy()
            else:
                depth = np.asarray(depth)

            depth_np = np.squeeze(depth)

            if depth_np.ndim == 3 and self.channel is not None:
                if self.channel < 0 or self.channel >= depth_np.shape[0]:
                    logging.warning(
                        "SaveDepthNpy: channel=%s out of bounds for tensor shape=%s "
                        "(dataset=%s, basename=%s)",
                        self.channel,
                        tuple(depth_np.shape),
                        dataset_disp_name,
                        rgb_basename,
                    )
                    continue
                depth_np = depth_np[self.channel]

            if depth_np.ndim not in (2, 3):
                logging.warning(
                    f"SaveDepthNpy: skipping save because tensor has unsupported ndim={getattr(depth_np, 'ndim', None)} "
                    f"for dataset {dataset_disp_name} (basename={rgb_basename})"
                )
                continue

            if self.resize_to_orig_res:
                orig_hw = _get_orig_hw_for_index(batch, b)
                if orig_hw is not None and depth_np.shape[-2:] != orig_hw:
                    depth_ts = torch.as_tensor(depth_np, dtype=torch.float32)
                    if depth_ts.ndim == 2:
                        depth_ts = depth_ts.unsqueeze(0).unsqueeze(0)
                        depth_ts = F.interpolate(
                            depth_ts, size=orig_hw, mode="bilinear", align_corners=False
                        )
                        depth_np = depth_ts.squeeze(0).squeeze(0).numpy()
                    elif depth_ts.ndim == 3:
                        depth_ts = depth_ts.unsqueeze(0)
                        depth_ts = F.interpolate(
                            depth_ts, size=orig_hw, mode="bilinear", align_corners=False
                        )
                        depth_np = depth_ts.squeeze(0).numpy()

            pred_basename = _get_pred_name(
                rgb_basename, name_mode=self.name_mode, suffix=".npy"
            )

            rel_parent = Path("")
            if file_path_obj is not None and base_dir_path is not None:
                try:
                    rel_parent = file_path_obj.absolute().parent.relative_to(
                        base_dir_path
                    )
                except ValueError:
                    rel_parent = Path("")

            save_dir = output_root / rel_parent
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / pred_basename
            try:
                np.save(str(save_path), depth_np.astype(np.float32, copy=False))
                logging.info(f"Saved prediction: {save_path}")
                _record_saved_output_path(
                    batch, self.folder_name or "pred_npy", str(save_path), batch_index=b
                )
            except Exception as e:
                logging.exception(
                    f"SaveDepthNpy: failed saving prediction to {save_path}: {e}"
                )


@register("validation_steps")
class ComputeMetrics:
    """Accumulate metrics from ``marigoldv2.validation.metric`` into the tracker.

    ``gt_range``/``pred_range`` describe the value range of the tensors
    (``0_255``, ``0_1``, ``neg1_1`` or ``native``). Aligned metrics compare
    against ``alignment_gt_key`` (e.g. log depth) and, when given,
    ``metric_depth_key`` in metric units.
    """

    def __init__(self, **kwargs):
        self.gt_key = kwargs.get("gt_key", "depth_rel_m11")
        self.pred_key = kwargs.get("pred_key", "out/depth_rel_pred_m11")
        self.mask_key = kwargs.get("mask_key", None)
        self.alignment_gt_key = kwargs.get("alignment_gt_key", None)
        self.metric_depth_key = kwargs.get("metric_depth_key", None)
        self.depth_min = kwargs.get("depth_min", 0.0)
        if self.depth_min is not None:
            self.depth_min = float(self.depth_min)
        self.depth_max = kwargs.get("depth_max", None)
        if self.depth_max is not None:
            self.depth_max = float(self.depth_max)
        self.gt_range = kwargs.get("gt_range", "0_255")
        self.pred_range = kwargs.get("pred_range", "neg1_1")
        self.alignment_gt_range = kwargs.get("alignment_gt_range", "native")
        self.eval_metrics = kwargs.get("eval_metrics", [])

    def __call__(self, batch):
        cfg = REGISTRY["cfg"]
        metric_tracker = batch["metric_tracker"]
        eval_metrics = self.eval_metrics
        metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

        def normalize_range_tag(tag):
            if isinstance(tag, str):
                t = tag.strip().lower().replace("-", "_")
                if t in {"native", "raw", "as_is", "asis"}:
                    return "native"
            if isinstance(tag, (int, float)):
                if float(tag) == 255.0:
                    return "0_255"
                if float(tag) == 1.0:
                    return "0_1"
                if float(tag) == -1.0:
                    return "neg1_1"
            if isinstance(tag, str):
                t = tag.strip().lower().replace("-", "_")
                alias = {
                    "0_255": "0_255",
                    "255": "0_255",
                    "0_1": "0_1",
                    "01": "0_1",
                    "1": "0_1",
                    "neg1_1": "neg1_1",
                    "-1_1": "neg1_1",
                    "minus1_1": "neg1_1",
                }
                if t in alias:
                    return alias[t]
            raise ValueError(f"Unknown range tag: {tag}")

        gt_range = normalize_range_tag(self.gt_range)
        pred_range = normalize_range_tag(self.pred_range)
        alignment_gt_range = normalize_range_tag(self.alignment_gt_range)

        def convert_to_range(tensor, current_range, target_range):
            if current_range == "native" and target_range == "native":
                return tensor.float()
            if current_range == "native" or target_range == "native":
                raise ValueError(
                    f"Cannot convert between 'native' and '{target_range if current_range == 'native' else current_range}'. "
                    "Use matching units for native tensors."
                )
            if current_range == target_range:
                return tensor.float()
            if current_range == "0_255":
                normalized = tensor.float() / 255.0
            elif current_range == "0_1":
                normalized = tensor.float()
            elif current_range == "neg1_1":
                normalized = (tensor.float() + 1.0) / 2.0
            else:
                raise ValueError(f"Unknown current_range: {current_range}")
            if target_range == "0_255":
                return (normalized * 255.0).clamp(0, 255)
            elif target_range == "0_1":
                return normalized.clamp(0, 1)
            elif target_range == "neg1_1":
                return (normalized * 2.0 - 1.0).clamp(-1, 1)
            else:
                raise ValueError(f"Unknown target_range: {target_range}")

        gt_raw = batch[self.gt_key].to(cfg.device)
        pred_ts = get_nested_key(batch, self.pred_key)
        alignment_gt_raw = None
        if self.alignment_gt_key is not None:
            alignment_gt_raw = get_nested_key(batch, self.alignment_gt_key)
            if torch.is_tensor(alignment_gt_raw):
                alignment_gt_raw = alignment_gt_raw.to(cfg.device)
        mask_raw = None
        if self.mask_key is not None:
            mask_raw = get_nested_key(batch, self.mask_key)
        metric_depth_raw = None
        if self.metric_depth_key is not None:
            metric_depth_raw = get_nested_key(batch, self.metric_depth_key)
            if torch.is_tensor(metric_depth_raw):
                metric_depth_raw = metric_depth_raw.to(cfg.device)

        def prepare_mask(mask):
            if mask is None:
                return None
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            return mask.to(cfg.device).bool()

        def upsample_to_gt_resolution(pred, gt):
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(
                    pred.unsqueeze(0) if pred.ndim == 3 else pred,
                    size=gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if pred.ndim == 4:
                    pred = pred.squeeze(0)
            return pred

        aligned_depth_metrics = {
            "aligned_abs_relative_difference",
            "aligned_delta1_acc",
            "aligned_delta2_acc",
            "aligned_delta3_acc",
            "aligned_log_abs_relative_difference",
            "aligned_log_delta1_acc",
            "aligned_log_delta2_acc",
            "aligned_log_delta3_acc",
            "aligned_log_soft_edge_error",
            "aligned_disp_abs_relative_difference",
            "aligned_disp_delta1_acc",
            "aligned_disp_delta2_acc",
            "aligned_disp_delta3_acc",
            "aligned_disp_soft_edge_error",
        }
        metric_gt_metrics = {
            m for m in aligned_depth_metrics if "_disp_" in m or "_log_" in m
        }
        positive_depth_metrics = {
            "delta1_acc",
            "delta2_acc",
            "delta3_acc",
        }
        eps = 1e-6

        batch_size = len(batch["idx"]) if "idx" in batch else 1
        for b in range(batch_size):
            for met_func in metric_funcs:
                _metric_name = met_func.__name__
                if isinstance(pred_ts, np.ndarray):
                    pred = (
                        torch.from_numpy(pred_ts[b] if pred_ts.ndim > 0 else pred_ts)
                        .to(cfg.device)
                        .float()
                    )
                else:
                    pred = pred_ts[b].to(cfg.device).float()

                if _metric_name in aligned_depth_metrics:
                    metric_gt = gt_raw[b]
                    if alignment_gt_raw is not None:
                        metric_gt = alignment_gt_raw[b]
                        if alignment_gt_range != "native":
                            metric_gt = convert_to_range(
                                metric_gt, alignment_gt_range, "0_1"
                            )

                    metric_pred = pred
                    metric_pred = upsample_to_gt_resolution(metric_pred, metric_gt)
                else:
                    metric_gt = convert_to_range(gt_raw[b], gt_range, "0_1")
                    metric_pred = convert_to_range(pred, pred_range, "0_1")
                    metric_pred = upsample_to_gt_resolution(metric_pred, metric_gt)

                valid_mask = prepare_mask(mask_raw[b] if mask_raw is not None else None)
                if valid_mask is None:
                    valid_mask = torch.ones_like(metric_gt, dtype=torch.bool)
                elif valid_mask.shape != metric_gt.shape:
                    try:
                        valid_mask = valid_mask.expand_as(metric_gt)
                    except RuntimeError:
                        if valid_mask.ndim == metric_gt.ndim - 1:
                            valid_mask = valid_mask.unsqueeze(0).expand_as(metric_gt)
                        else:
                            raise

                finite_mask = torch.isfinite(metric_gt) & torch.isfinite(metric_pred)
                metric_mask = valid_mask & finite_mask

                if _metric_name in positive_depth_metrics:
                    metric_mask = metric_mask & (metric_gt > eps)
                    if _metric_name not in aligned_depth_metrics:
                        metric_mask = metric_mask & (metric_pred > eps)

                if not torch.any(metric_mask):
                    continue

                if _metric_name in metric_gt_metrics:
                    metric_depth_gt = None
                    if metric_depth_raw is not None:
                        metric_depth_gt = upsample_to_gt_resolution(
                            metric_depth_raw[b], metric_gt
                        )
                    _metric = met_func(
                        metric_pred,
                        metric_gt,
                        metric_mask,
                        metric_depth_gt=metric_depth_gt,
                        depth_min=self.depth_min,
                        depth_max=self.depth_max,
                    )
                elif _metric_name in aligned_depth_metrics:
                    _metric = met_func(
                        metric_pred,
                        metric_gt,
                        metric_mask,
                        depth_min=self.depth_min,
                        depth_max=self.depth_max,
                    )
                else:
                    _metric = met_func(metric_pred, metric_gt, metric_mask)
                _metric = _metric.item() if torch.is_tensor(_metric) else float(_metric)
                if np.isfinite(_metric):
                    metric_tracker.update(_metric_name, _metric)
