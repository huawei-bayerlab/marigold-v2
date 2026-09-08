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

import numpy as np
import torch


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert gt.shape == pred.shape == valid_mask.shape, (
        f"{gt.shape}, {pred.shape}, {valid_mask.shape}"
    )

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver - weighted least squares with w = 1/gt^2
    # sqrt(w) = 1/|gt|; multiply both sides to convert to unweighted problem
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    # sqrt_w = 1.0 / np.maximum(np.abs(gt_masked), 1e-6)
    # X = np.linalg.lstsq(A * sqrt_w, gt_masked * sqrt_w, rcond=None)[0]
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)


# ******************** log space ********************
def depth2log_space(depth, eps=1e-6, **kwargs):
    if isinstance(depth, torch.Tensor):
        log_space = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        log_space = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    log_space[non_negtive_mask] = np.log(depth[non_negtive_mask] + eps)
    return log_space, non_negtive_mask


def log_space2depth(log_space, eps=1e-6, **kwargs):
    depth = np.exp(log_space) - eps
    return depth
