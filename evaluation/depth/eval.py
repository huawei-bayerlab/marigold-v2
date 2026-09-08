# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# Modifications Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import numpy as np
import os
from scipy import ndimage
import torch
import torch.nn.functional as F
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.src.dataset import (
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from evaluation.src.util import metric
from evaluation.src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
    depth2log_space,
    log_space2depth,
)
from evaluation.src.util.metric import MetricTracker

eval_metrics = [
    "abs_relative_difference",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
]

_PPD_POLY_FEATURES = PolynomialFeatures(degree=1, include_bias=False)
_PPD_RANSAC_SEED = 42
_PPD_RANSAC = RANSACRegressor(max_trials=1000, random_state=_PPD_RANSAC_SEED)
_PPD_RANSAC_MODEL = make_pipeline(_PPD_POLY_FEATURES, _PPD_RANSAC)


def _ppd_depth_bounds_from_disp_name(
    disp_name: str,
) -> tuple[float | None, float | None]:
    name = (disp_name or "").lower()
    if "kitti" in name:
        return 0.1, 80.0
    if "nyu" in name:
        return 0.1, 10.0
    if "scannet" in name:
        return 0.01, 10.0
    if "eth3d" in name:
        return 0.01, None
    if "diode" in name:
        return 0.6, 350.0
    return None, None


def _apply_ppd_mask(
    depth: np.ndarray, valid_mask: np.ndarray, disp_name: str
) -> np.ndarray:
    depth_min, depth_max = _ppd_depth_bounds_from_disp_name(disp_name)
    out = valid_mask.astype(bool)
    if depth_min is not None:
        out = out & (depth > float(depth_min))
    if depth_max is not None and math.isfinite(float(depth_max)):
        out = out & (depth < float(depth_max))
    return out


def _apply_diode_gradient_filter(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    disp_name: str,
    grad_threshold: float = 0.3,
) -> np.ndarray:
    name = (disp_name or "").lower()
    if "diode" not in name:
        return valid_mask

    # DIODE gradient filter from the PPD evaluation protocol.
    depth_f = depth.astype(np.float32, copy=False)
    mask = valid_mask.astype(bool, copy=True)
    dx = ndimage.sobel(depth_f, 0)  # horizontal derivative
    dy = ndimage.sobel(depth_f, 1)  # vertical derivative
    grad = np.abs(dx) + np.abs(dy)
    mask[grad > float(grad_threshold)] = 0
    return mask


def _ppd_target_hw_from_disp_name(disp_name: str) -> tuple[int, int] | None:
    name = (disp_name or "").lower()
    # PPD evaluates ETH3D at 2048x1360.
    if "eth3d" in name:
        return 1360, 2048
    return None


def _resize_2d_float(arr: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    ts = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(ts, size=(height, width), mode="nearest")
    else:
        out = F.interpolate(
            ts, size=(height, width), mode="bilinear", align_corners=False
        )
    return out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _resize_for_ppd_protocol(
    depth_gt: np.ndarray,
    valid_mask: np.ndarray,
    depth_pred: np.ndarray,
    disp_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_hw = _ppd_target_hw_from_disp_name(disp_name)
    if target_hw is None:
        return depth_gt, valid_mask, depth_pred

    h, w = target_hw
    depth_gt_r = _resize_2d_float(depth_gt, h, w, mode="nearest")
    valid_mask_r = (
        _resize_2d_float(valid_mask.astype(np.float32), h, w, mode="nearest") > 0.5
    )
    depth_pred_r = _resize_2d_float(depth_pred, h, w, mode="nearest")
    return depth_gt_r, valid_mask_r, depth_pred_r


def _clip_depth_like_ppd(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    gt_valid = depth_gt[valid_mask].astype(np.float32)
    if gt_valid.size == 0:
        return np.clip(depth_pred, a_min=1e-3, a_max=None)
    return np.clip(depth_pred, a_min=1e-3, a_max=float(np.max(gt_valid)))


def align_log_space_ransac_ppd(
    pred_log: np.ndarray,
    gt_log: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    mask_gt = gt_log[valid_mask].astype(np.float32)
    mask_pred = pred_log[valid_mask].astype(np.float32)

    if mask_gt.size == 0 or mask_pred.size == 0:
        scale, shift = 1.0, 0.0
        aligned_log = pred_log
        return aligned_log, scale, shift

    try:
        _PPD_RANSAC_MODEL.fit(mask_pred[:, None], mask_gt[:, None])
        scale = _PPD_RANSAC_MODEL.named_steps["ransacregressor"].estimator_.coef_.item()
        shift = _PPD_RANSAC_MODEL.named_steps[
            "ransacregressor"
        ].estimator_.intercept_.item()
    except Exception:
        scale, shift = 1.0, 0.0

    if scale > 0:
        aligned_log = scale * pred_log + shift
    else:
        pred_mean = float(np.mean(mask_pred))
        gt_mean = float(np.mean(mask_gt))
        scale = (gt_mean / pred_mean) if abs(pred_mean) > 1e-8 else 1.0
        shift = 0.0
        aligned_log = scale * pred_log

    return aligned_log, scale, shift


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold V1 depth metrics (AbsRel, delta) with optional PPD protocol"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory with predictions obtained from inference.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the config file of the evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Base path to the datasets.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--alignment",
        choices=[
            None,
            "least_square",
            "least_square_disparity",
            "least_square_log",
            "ppd_ransac_log",
        ],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda.")
    parser.add_argument(
        "--ppd_eval_protocol",
        action="store_true",
        help="Apply PPD-style validity mask thresholds and clipping behavior.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(args.dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metrics file --------------------
    per_sample_filename = os.path.join(args.output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        # Load predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".npy")
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)
        pred_path = os.path.join(args.prediction_dir, pred_name)
        if not os.path.exists(pred_path):
            logging.warning(f"Can't find prediction: {pred_path}")
            continue
        depth_pred = np.load(pred_path).astype(np.float32)

        if args.ppd_eval_protocol:
            depth_raw, valid_mask, depth_pred = _resize_for_ppd_protocol(
                depth_gt=depth_raw,
                valid_mask=valid_mask,
                depth_pred=depth_pred,
                disp_name=dataset.disp_name,
            )
            valid_mask = _apply_ppd_mask(
                depth=depth_raw,
                valid_mask=valid_mask,
                disp_name=dataset.disp_name,
            )

        # Apply DIODE gradient filtering in both protocol modes.
        valid_mask = _apply_diode_gradient_filter(
            depth=depth_raw,
            valid_mask=valid_mask,
            disp_name=dataset.disp_name,
        )

        depth_raw_ts = torch.from_numpy(depth_raw).to(device)
        valid_mask_ts = torch.from_numpy(valid_mask).to(device)

        # Align with GT using least square
        if "least_square" == args.alignment:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=args.alignment_max_res,
            )
        elif "least_square_disparity" == args.alignment:
            # convert GT depth -> GT disparity
            gt_disparity, gt_non_neg_mask = depth2disparity(
                depth=depth_raw, return_mask=True
            )
            # LS alignment in disparity space
            pred_non_neg_mask = depth_pred > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=args.alignment_max_res,
            )
            # convert to depth
            disparity_pred = np.clip(
                disparity_pred, a_min=1e-3, a_max=None
            )  # avoid 0 disparity
            depth_pred = disparity2depth(disparity_pred)
        elif "least_square_log" == args.alignment:
            gt_log, gt_non_neg_mask = depth2log_space(depth=depth_raw, return_mask=True)
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask
            if isinstance(gt_log, torch.Tensor):
                gt_log = gt_log.cpu().numpy()

            log_space_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_log,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=args.alignment_max_res,
            )
            depth_pred = log_space2depth(log_space_pred)
        elif "ppd_ransac_log" == args.alignment:
            gt_log, gt_non_neg_mask = depth2log_space(depth=depth_raw, return_mask=True)
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask
            if isinstance(gt_log, torch.Tensor):
                gt_log = gt_log.cpu().numpy()
            log_space_pred, scale, shift = align_log_space_ransac_ppd(
                pred_log=depth_pred,
                gt_log=gt_log,
                valid_mask=valid_nonnegative_mask,
            )
            depth_pred = log_space2depth(log_space_pred)

        # Clip to dataset min max
        if not args.ppd_eval_protocol:
            depth_pred = np.clip(
                depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
            )
        else:
            depth_pred = _clip_depth_like_ppd(
                depth_pred=depth_pred,
                depth_gt=depth_raw,
                valid_mask=valid_mask,
            )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate (using CUDA if available)
        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(pred_name + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {args.prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"
    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    if args.alignment:
        metrics_filename += f"-{args.alignment}"
    metrics_filename += ".txt"

    _save_to = os.path.join(args.output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
