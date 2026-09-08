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

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        if key not in self._data.index:
            # Some validation steps add dataset-specific metrics that are not known
            # when the tracker is first constructed.
            self._data.loc[key] = [0, 0, 0]
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# -------------------- Depth Metrics --------------------


def _abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


# Adapted from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    bit_mat = (max_d1_d2 < threshold_val).to(output.dtype)
    if valid_mask is not None:
        vm = valid_mask.bool()
        if vm.shape != bit_mat.shape:
            if vm.ndim == bit_mat.ndim and vm.shape[1] == 1 and bit_mat.shape[1] != 1:
                vm = vm.repeat(1, bit_mat.shape[1], 1, 1)
            else:
                vm = vm.expand_as(bit_mat)
        bit_mat = bit_mat * vm.to(bit_mat.dtype)
        n = vm.sum((-1, -2)).clamp_min(1).to(bit_mat.dtype)
    else:
        n = torch.full(
            bit_mat.sum((-1, -2)).shape,
            float(output.shape[-1] * output.shape[-2]),
            device=bit_mat.device,
            dtype=bit_mat.dtype,
        )
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def _depth_boundary_mask_from_disparity_jump(
    depth_2d,
    valid_mask_2d,
    disparity_jump_threshold=1.0,
    dilate_radius=0,
    eps=1e-6,
):
    """Build a boundary mask where 4-neighbors differ by > threshold in disparity.

    This follows the boundary definition used by SMD-Nets (arXiv:2104.03866),
    adapted to depth by converting depth to inverse-depth disparity first.
    """
    vm = valid_mask_2d.bool()
    if not torch.any(vm):
        return torch.zeros_like(vm)

    disp = torch.zeros_like(depth_2d, dtype=torch.float32)
    safe = vm & (depth_2d > eps) & torch.isfinite(depth_2d)
    if not torch.any(safe):
        return torch.zeros_like(vm)
    disp[safe] = 1.0 / torch.clamp(depth_2d[safe], min=eps)

    boundary = torch.zeros_like(vm)
    thr = float(disparity_jump_threshold)

    # Horizontal neighbors.
    pair_valid = vm[:, :-1] & vm[:, 1:]
    pair_jump = torch.abs(disp[:, :-1] - disp[:, 1:]) > thr
    edge = pair_valid & pair_jump
    boundary[:, :-1] |= edge
    boundary[:, 1:] |= edge

    # Vertical neighbors.
    pair_valid = vm[:-1, :] & vm[1:, :]
    pair_jump = torch.abs(disp[:-1, :] - disp[1:, :]) > thr
    edge = pair_valid & pair_jump
    boundary[:-1, :] |= edge
    boundary[1:, :] |= edge

    boundary &= vm
    if dilate_radius > 0:
        boundary = _dilate_mask(boundary, radius=int(dilate_radius)) & vm
    return boundary


def _soft_edge_error_depth_2d(
    pred_depth_2d,
    gt_depth_2d,
    valid_mask_2d,
    patch_size=5,
    disparity_jump_threshold=0.05,
    boundary_dilate_radius=0,
    eps=1e-6,
):
    """Compute Soft Edge Error (SEE_k) for one depth map pair.

    SEE_k at boundary pixels is the minimum absolute error between each predicted
    center value and all GT values in its local kxk neighborhood.
    """
    if patch_size % 2 == 0 or patch_size < 1:
        raise ValueError(f"patch_size must be odd and >= 1, got {patch_size}")

    vm = valid_mask_2d.bool()
    finite = torch.isfinite(pred_depth_2d) & torch.isfinite(gt_depth_2d)
    positive = (pred_depth_2d > eps) & (gt_depth_2d > eps)
    vm = vm & finite & positive
    if not torch.any(vm):
        return torch.tensor(
            float("nan"), device=gt_depth_2d.device, dtype=torch.float32
        )

    boundary = _depth_boundary_mask_from_disparity_jump(
        gt_depth_2d,
        vm,
        disparity_jump_threshold=disparity_jump_threshold,
        dilate_radius=boundary_dilate_radius,
        eps=eps,
    )
    if not torch.any(boundary):
        return torch.tensor(
            float("nan"), device=gt_depth_2d.device, dtype=torch.float32
        )

    radius = patch_size // 2
    gt4 = gt_depth_2d.unsqueeze(0).unsqueeze(0)
    vm4 = vm.float().unsqueeze(0).unsqueeze(0)

    # F.unfold returns [B, k*k, H*W]; here B=1.
    gt_patches = F.unfold(gt4, kernel_size=patch_size, padding=radius)[0]  # [k*k, H*W]
    vm_patches = (
        F.unfold(vm4, kernel_size=patch_size, padding=radius)[0] > 0.5
    )  # [k*k, H*W]

    pred_flat = pred_depth_2d.reshape(1, -1)
    abs_err = torch.abs(gt_patches - pred_flat)
    inf = torch.full_like(abs_err, float("inf"))
    abs_err = torch.where(vm_patches, abs_err, inf)
    min_local_err = abs_err.min(dim=0).values.reshape_as(pred_depth_2d)

    score_mask = boundary & vm
    if not torch.any(score_mask):
        return torch.tensor(
            float("nan"), device=gt_depth_2d.device, dtype=torch.float32
        )
    return min_local_err[score_mask].mean().float()


def soft_edge_error(
    pred,
    gt,
    valid_mask=None,
    patch_size=5,
    disparity_jump_threshold=0.05,
    boundary_dilate_radius=0,
    eps=1e-6,
):
    """Soft Edge Error (SEE_k) adapted from arXiv:2104.03866.

    Boundary pixels are extracted from GT depth by thresholding 4-neighbor
    inverse-depth (disparity) jumps, then SEE_k is averaged over those pixels.
    """
    pred_2d = _to_depth_2d(pred.float())
    gt_2d = _to_depth_2d(gt.float())
    if pred_2d.shape != gt_2d.shape:
        raise ValueError(f"Pred/GT shape mismatch: {pred_2d.shape} vs {gt_2d.shape}")
    vm_2d = _to_mask_2d(valid_mask, gt_2d)
    return _soft_edge_error_depth_2d(
        pred_2d,
        gt_2d,
        vm_2d,
        patch_size=patch_size,
        disparity_jump_threshold=disparity_jump_threshold,
        boundary_dilate_radius=boundary_dilate_radius,
        eps=eps,
    )


def _to_depth_2d(depth):
    """Convert depth tensor to 2D shape [H, W]."""
    if depth.ndim == 2:
        return depth
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            return depth[0]
        raise ValueError(
            f"Expected single-channel depth for 3D tensor, got shape {depth.shape}"
        )
    if depth.ndim == 4:
        if depth.shape[0] == 1 and depth.shape[1] == 1:
            return depth[0, 0]
        raise ValueError(
            f"Expected [1,1,H,W] depth for 4D tensor, got shape {depth.shape}"
        )
    raise ValueError(f"Unsupported depth ndim: {depth.ndim}")


def _to_mask_2d(valid_mask, ref_2d):
    if valid_mask is None:
        return torch.ones_like(ref_2d, dtype=torch.bool)

    vm = valid_mask.bool()
    if vm.ndim == 2:
        return vm
    if vm.ndim == 3:
        if vm.shape[0] == 1:
            return vm[0]
        return vm.any(dim=0)
    if vm.ndim == 4:
        if vm.shape[0] == 1 and vm.shape[1] == 1:
            return vm[0, 0]
        if vm.shape[0] == 1:
            return vm[0].any(dim=0)
    raise ValueError(f"Unsupported valid_mask ndim: {vm.ndim}")


def _dilate_mask(mask_2d, radius=1):
    if radius <= 0:
        return mask_2d
    k = 2 * int(radius) + 1
    m = mask_2d.float().unsqueeze(0).unsqueeze(0)
    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return m[0, 0] > 0


def align_scale_shift(output, target, valid_mask=None, eps=1e-6):
    """Align output to target with a per-sample affine model: a * output + b."""
    out = output.float()
    tgt = target.float()

    if valid_mask is None:
        vm = torch.ones_like(tgt, dtype=torch.bool)
    else:
        vm = valid_mask.bool()
        if vm.shape != tgt.shape:
            if vm.ndim == tgt.ndim - 1:
                vm = vm.unsqueeze(0)
            vm = vm.expand_as(tgt)

    if out.shape != tgt.shape:
        out = out.expand_as(tgt)

    aligned = out.clone()
    if out.ndim == 2:
        out_b = out.unsqueeze(0)
        tgt_b = tgt.unsqueeze(0)
        vm_b = vm.unsqueeze(0)
        squeeze_back = True
    elif out.ndim == 3:
        out_b = out
        tgt_b = tgt
        vm_b = vm
        squeeze_back = False
    elif out.ndim == 4:
        out_b = out
        tgt_b = tgt
        vm_b = vm
        squeeze_back = False
    else:
        raise ValueError(f"Unsupported tensor ndim for alignment: {out.ndim}")

    if out_b.ndim == 3:
        out_b = out_b.unsqueeze(0)
        tgt_b = tgt_b.unsqueeze(0)
        vm_b = vm_b.unsqueeze(0)
        squeeze_b = True
    else:
        squeeze_b = False

    aligned_b = out_b.clone()
    batch_dim = out_b.shape[0]
    for i in range(batch_dim):
        xi = out_b[i][vm_b[i]]
        yi = tgt_b[i][vm_b[i]]

        if xi.numel() < 2:
            continue

        A = torch.stack([xi, torch.ones_like(xi)], dim=1)
        try:
            sol = torch.linalg.lstsq(A, yi.unsqueeze(1)).solution
        except RuntimeError:
            # cuSolver can fail with an internal error on some CUDA setups.
            # Fall back to running the least-squares solve on CPU and move
            # the solution back to the original device.
            cpu_A = A.cpu()
            cpu_y = yi.unsqueeze(1).cpu()
            sol = torch.linalg.lstsq(cpu_A, cpu_y).solution.to(A.device)
        a = sol[0, 0]
        b = sol[1, 0]
        if not torch.isfinite(a) or not torch.isfinite(b):
            continue
        aligned_b[i] = a * out_b[i] + b

    if squeeze_b:
        aligned = aligned_b.squeeze(0)
    else:
        aligned = aligned_b
    if squeeze_back:
        aligned = aligned.squeeze(0)

    # Allow callers to disable clamping (e.g., log-space alignment where negatives are valid).
    if eps is None:
        return aligned
    return torch.clamp(aligned, min=float(eps))


def _clip_aligned_depth_bounds(depth, depth_min=None, depth_max=None, eps=1e-6):
    min_val = float(eps)
    if depth_min is not None:
        min_val = max(min_val, float(depth_min))

    if depth_max is not None:
        max_val = float(depth_max)
        if max_val < min_val:
            max_val = min_val
        return torch.clamp(depth, min=min_val, max=max_val)

    return torch.clamp(depth, min=min_val)


def aligned_abs_relative_difference(
    output, target, valid_mask=None, depth_min=None, depth_max=None
):
    aligned_output = align_scale_shift(output, target, valid_mask)
    aligned_output = _clip_aligned_depth_bounds(
        aligned_output, depth_min=depth_min, depth_max=depth_max
    )
    return _abs_relative_difference(aligned_output, target, valid_mask)


def aligned_delta1_acc(pred, gt, valid_mask, depth_min=None, depth_max=None):
    aligned_pred = align_scale_shift(pred, gt, valid_mask)
    aligned_pred = _clip_aligned_depth_bounds(
        aligned_pred, depth_min=depth_min, depth_max=depth_max
    )
    return delta1_acc(aligned_pred, gt, valid_mask)


def aligned_delta2_acc(pred, gt, valid_mask, depth_min=None, depth_max=None):
    aligned_pred = align_scale_shift(pred, gt, valid_mask)
    aligned_pred = _clip_aligned_depth_bounds(
        aligned_pred, depth_min=depth_min, depth_max=depth_max
    )
    return delta2_acc(aligned_pred, gt, valid_mask)


def aligned_delta3_acc(pred, gt, valid_mask, depth_min=None, depth_max=None):
    aligned_pred = align_scale_shift(pred, gt, valid_mask)
    aligned_pred = _clip_aligned_depth_bounds(
        aligned_pred, depth_min=depth_min, depth_max=depth_max
    )
    return delta3_acc(aligned_pred, gt, valid_mask)


def align_scale_shift_log_to_depth(
    pred_log,
    gt_log,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
    log_eps=1e-6,
    depth_eps=1e-6,
):
    """Align prediction to log-depth target, then convert both to metric depth.

    pred_log: predicted log-depth (or a proxy representation that can be affine-aligned)
    gt_log: ground-truth log-depth, e.g. log(depth + log_eps)
    metric_depth_gt: optional metric-depth ground truth. If provided, this is used
      for final metric computation instead of inverting gt_log.
    Returns: (aligned_pred_depth, gt_depth)
    """
    # If model outputs pred_log in [-1,1], map to [-4,6] before alignment: y = 5*x + 1
    pred_log_mapped = pred_log.float() * 5.0 + 1.0
    aligned_pred_log = align_scale_shift(pred_log_mapped, gt_log, valid_mask, eps=None)
    gt_log = gt_log.float()

    aligned_pred_depth = torch.clamp(
        torch.exp(aligned_pred_log) - float(log_eps), min=depth_eps
    )
    aligned_pred_depth = _clip_aligned_depth_bounds(
        aligned_pred_depth, depth_min=depth_min, depth_max=depth_max, eps=depth_eps
    )
    if metric_depth_gt is not None:
        gt_depth = torch.clamp(metric_depth_gt.float(), min=depth_eps)
    else:
        gt_depth = torch.clamp(torch.exp(gt_log) - float(log_eps), min=depth_eps)
    return aligned_pred_depth, gt_depth


def aligned_log_abs_relative_difference(
    pred_log,
    gt_log,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_pred_depth, gt_depth = align_scale_shift_log_to_depth(
        pred_log,
        gt_log,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return _abs_relative_difference(aligned_pred_depth, gt_depth, valid_mask)


def aligned_log_delta1_acc(
    pred_log,
    gt_log,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_pred_depth, gt_depth = align_scale_shift_log_to_depth(
        pred_log,
        gt_log,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return delta1_acc(aligned_pred_depth, gt_depth, valid_mask)


def aligned_log_delta2_acc(
    pred_log,
    gt_log,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_pred_depth, gt_depth = align_scale_shift_log_to_depth(
        pred_log,
        gt_log,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return delta2_acc(aligned_pred_depth, gt_depth, valid_mask)


def aligned_log_delta3_acc(
    pred_log,
    gt_log,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_pred_depth, gt_depth = align_scale_shift_log_to_depth(
        pred_log,
        gt_log,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return delta3_acc(aligned_pred_depth, gt_depth, valid_mask)


def aligned_log_soft_edge_error(
    pred_log,
    gt_log,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
    patch_size=5,
    disparity_jump_threshold=1.0,
    boundary_dilate_radius=0,
    eps=1e-6,
):
    """Align in log-space, convert to metric depth, then compute SEE_k."""
    aligned_pred_depth, gt_depth = align_scale_shift_log_to_depth(
        pred_log,
        gt_log,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
        depth_eps=eps,
    )
    return soft_edge_error(
        aligned_pred_depth,
        gt_depth,
        valid_mask=valid_mask,
        patch_size=patch_size,
        disparity_jump_threshold=disparity_jump_threshold,
        boundary_dilate_radius=boundary_dilate_radius,
        eps=eps,
    )


def aligned_disp_soft_edge_error(
    pred_disp,
    gt_disp,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
    patch_size=5,
    disparity_jump_threshold=1.0,
    boundary_dilate_radius=0,
    eps=1e-6,
):
    """Align disparity with weighted least squares, convert to depth, then compute SEE_k."""
    aligned_pred_depth, gt_depth = align_disparity_scale_shift(
        pred_disp,
        gt_disp,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        eps=eps,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return soft_edge_error(
        aligned_pred_depth,
        gt_depth,
        valid_mask=valid_mask,
        patch_size=patch_size,
        disparity_jump_threshold=disparity_jump_threshold,
        boundary_dilate_radius=boundary_dilate_radius,
        eps=eps,
    )


def align_disparity_scale_shift(
    pred_disp,
    gt_disp,
    valid_mask=None,
    metric_depth_gt=None,
    eps=1e-4,
    depth_min=None,
    depth_max=None,
):
    """Align predicted disparity to gt disparity with a weighted per-sample affine model,
    then convert both to depth by inversion.

    The affine parameters are estimated with weighted L2, where each valid pixel uses
    weight w = 1 / gt_disp. This penalizes low-disparity regions more.

    pred_disp: predicted disparity (same shape as gt_depth or broadcastable)
    gt_disp: ground-truth disparity
    metric_depth_gt: optional metric-depth target. If provided, it is used for the
        returned ground-truth depth instead of inverting `gt_disp`.
    Returns: (aligned_pred_depth, gt_depth)
    """
    pred = pred_disp.float()
    tgt = gt_disp.float()

    if valid_mask is None:
        vm = torch.ones_like(tgt, dtype=torch.bool)
    else:
        vm = valid_mask.bool()
        if vm.shape != tgt.shape:
            if vm.ndim == tgt.ndim - 1:
                vm = vm.unsqueeze(0)
            vm = vm.expand_as(tgt)

    if pred.shape != tgt.shape:
        pred = pred.expand_as(tgt)

    if pred.ndim == 2:
        pred_b = pred.unsqueeze(0)
        tgt_b = tgt.unsqueeze(0)
        vm_b = vm.unsqueeze(0)
        squeeze_back = True
    elif pred.ndim == 3:
        pred_b = pred
        tgt_b = tgt
        vm_b = vm
        squeeze_back = False
    elif pred.ndim == 4:
        pred_b = pred
        tgt_b = tgt
        vm_b = vm
        squeeze_back = False
    else:
        raise ValueError(f"Unsupported tensor ndim for alignment: {pred.ndim}")

    if pred_b.ndim == 3:
        pred_b = pred_b.unsqueeze(0)
        tgt_b = tgt_b.unsqueeze(0)
        vm_b = vm_b.unsqueeze(0)
        squeeze_b = True
    else:
        squeeze_b = False

    aligned_disp_b = pred_b.clone()
    for i in range(pred_b.shape[0]):
        xi = pred_b[i][vm_b[i]]
        yi = tgt_b[i][vm_b[i]]

        if xi.numel() < 2:
            continue

        # Weighted L2: minimize sum_i w_i (a*x_i + b - y_i)^2 with w_i = 1 / y_i.
        yi_safe = yi.clamp(min=eps)
        w = (1.0 / yi_safe).float()
        sqrt_w = torch.sqrt(w)
        A = torch.stack([xi, torch.ones_like(xi)], dim=1)
        Aw = A * sqrt_w.unsqueeze(1)
        yw = yi.unsqueeze(1) * sqrt_w.unsqueeze(1)

        try:
            sol = torch.linalg.lstsq(Aw, yw).solution
        except RuntimeError:
            cpu_Aw = Aw.cpu()
            cpu_yw = yw.cpu()
            sol = torch.linalg.lstsq(cpu_Aw, cpu_yw).solution.to(Aw.device)

        a = sol[0, 0]
        b = sol[1, 0]
        if not torch.isfinite(a) or not torch.isfinite(b):
            continue
        aligned_disp_b[i] = a * pred_b[i] + b

    if squeeze_b:
        aligned_disp = aligned_disp_b.squeeze(0)
    else:
        aligned_disp = aligned_disp_b
    if squeeze_back:
        aligned_disp = aligned_disp.squeeze(0)

    aligned_disp = torch.clamp(aligned_disp, min=eps)
    if metric_depth_gt is not None:
        tgt_depth = torch.clamp(metric_depth_gt.float(), min=eps)
    else:
        tgt_depth = (1.0 / tgt.clamp(min=eps)).float()

    # invert back to depth, clamp to avoid div by zero
    aligned_depth = 1.0 / aligned_disp.clamp(min=eps).float()
    aligned_depth = _clip_aligned_depth_bounds(
        aligned_depth, depth_min=depth_min, depth_max=depth_max, eps=eps
    )
    return aligned_depth, tgt_depth


def aligned_disp_abs_relative_difference(
    pred_disp,
    gt,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_depth, tgt_depth = align_disparity_scale_shift(
        pred_disp,
        gt,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return _abs_relative_difference(aligned_depth, tgt_depth, valid_mask)


def aligned_disp_delta1_acc(
    pred_disp,
    gt,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_depth, tgt_depth = align_disparity_scale_shift(
        pred_disp,
        gt,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return delta1_acc(aligned_depth, tgt_depth, valid_mask)


def aligned_disp_delta2_acc(
    pred_disp,
    gt,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_depth, tgt_depth = align_disparity_scale_shift(
        pred_disp,
        gt,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return delta2_acc(aligned_depth, tgt_depth, valid_mask)


def aligned_disp_delta3_acc(
    pred_disp,
    gt,
    valid_mask=None,
    metric_depth_gt=None,
    depth_min=None,
    depth_max=None,
):
    aligned_depth, tgt_depth = align_disparity_scale_shift(
        pred_disp,
        gt,
        valid_mask,
        metric_depth_gt=metric_depth_gt,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    return delta3_acc(aligned_depth, tgt_depth, valid_mask)
