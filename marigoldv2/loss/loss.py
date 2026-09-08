from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    AutoImageProcessor = None
    AutoModel = None

from marigoldv2.core.registry import REGISTRY, get, register

torch.backends.mkldnn.enabled = False


def _sinkhorn_log(log_K: torch.Tensor, n_iter: int) -> torch.Tensor:
    """Sinkhorn-Knopp normalization in log domain on a square cost matrix."""
    log_M = log_K
    for _ in range(int(n_iter)):
        log_M = log_M - torch.logsumexp(log_M, dim=-1, keepdim=True)
        log_M = log_M - torch.logsumexp(log_M, dim=-2, keepdim=True)
    return log_M


@register("loss")
class WindowMatchedL1Loss:
    """Windowed L1 with K x K Sinkhorn 1-to-1 matching."""

    def __init__(
        self,
        weight: float,
        pred_key: str,
        gt_key: str,
        mask_key: str = None,
        window_size: int = 5,
        sinkhorn_iter: int = 5,
        sinkhorn_tau: float = 0.1,
        masked_cost: float = 1.0e6,
        loss_name: str = "window_matched_l1_loss",
        use_3ch_l1: bool = False,
        dataset_names=None,
        **kwargs,
    ):
        self.weight = float(weight)
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.mask_key = mask_key
        self.window_size = int(window_size)
        if self.window_size < 2:
            raise ValueError(
                f"WindowMatchedL1Loss: window_size must be >= 2, got {self.window_size}"
            )
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.sinkhorn_tau = float(sinkhorn_tau)
        if self.sinkhorn_tau <= 0:
            raise ValueError(
                f"WindowMatchedL1Loss: sinkhorn_tau must be > 0, got {self.sinkhorn_tau}"
            )
        self.masked_cost = float(masked_cost)
        self.loss_name = loss_name
        self.use_3ch_l1 = bool(use_3ch_l1)

        if dataset_names is None:
            self.dataset_names = None
        elif not isinstance(dataset_names, (str, bytes)):
            try:
                self.dataset_names = {str(name) for name in dataset_names}
            except TypeError:
                self.dataset_names = {str(dataset_names)}
        else:
            self.dataset_names = {str(dataset_names)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_key(self, batch, key):
        d = batch
        for p in key.split("/"):
            d = d[p]
        return d

    def __call__(self, batch):
        pred = self._resolve_key(batch, self.pred_key).to(self.device).float()
        gt = self._resolve_key(batch, self.gt_key).to(self.device).float()

        sel_idx = None
        if self.dataset_names is not None and "dataset_name" in batch:
            names = batch["dataset_name"]
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, (list, tuple)):
                try:
                    names = list(names)
                except Exception:
                    names = [str(names)]
            sel_idx = [i for i, n in enumerate(names) if str(n) in self.dataset_names]
            if len(sel_idx) == 0:
                batch.setdefault("loss", {})
                batch.setdefault("weighted_loss", 0.0)
                zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                batch["loss"][self.loss_name] = zero
                return
            pred = pred[sel_idx]
            gt = gt[sel_idx]

        expected_channels = 3 if self.use_3ch_l1 else 1
        if pred.ndim != 4 or pred.shape[1] != expected_channels:
            raise ValueError(
                f"WindowMatchedL1Loss expects pred [B, {expected_channels}, H, W], "
                f"got {tuple(pred.shape)}"
            )
        if gt.ndim != 4 or gt.shape[1] != expected_channels:
            raise ValueError(
                f"WindowMatchedL1Loss expects gt [B, {expected_channels}, H, W], "
                f"got {tuple(gt.shape)}"
            )

        if self.mask_key is not None:
            mask = self._resolve_key(batch, self.mask_key).to(self.device).float()
            if sel_idx is not None:
                mask = mask[sel_idx]
            if mask.ndim == 4 and mask.shape[1] != 1:
                mask = mask[:, :1]
        else:
            mask = torch.ones_like(gt)

        K = self.window_size
        bsz, _, height, width = pred.shape
        pad_h = (K - height % K) % K
        pad_w = (K - width % K) % K
        if pad_h or pad_w:
            pred = F.pad(pred, (0, pad_w, 0, pad_h))
            gt = F.pad(gt, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h))
        padded_h, padded_w = pred.shape[-2:]
        h_blocks, w_blocks = padded_h // K, padded_w // K
        n_block = K * K
        n_blocks = h_blocks * w_blocks

        def to_blocks(x):
            return (
                x.reshape(bsz, x.shape[1], h_blocks, K, w_blocks, K)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(bsz, n_blocks, n_block, x.shape[1])
            )

        pred_blocks = to_blocks(pred)
        gt_blocks = to_blocks(gt)
        mask_blocks = to_blocks(mask)

        pred_i = pred_blocks.unsqueeze(3)
        gt_j = gt_blocks.unsqueeze(2)
        diff = (gt_j - pred_i).abs()
        real_cost = diff.mean(dim=-1)

        mask_i = mask_blocks[..., 0].unsqueeze(-1)
        mask_j = mask_blocks[..., 0].unsqueeze(-2)
        valid_pair = mask_i * mask_j
        cost_for_sinkhorn = real_cost * valid_pair + self.masked_cost * (
            1.0 - valid_pair
        )

        log_K = -cost_for_sinkhorn / self.sinkhorn_tau
        log_M = _sinkhorn_log(log_K, n_iter=self.sinkhorn_iter)
        M = log_M.exp()

        loss_pair = M * real_cost * valid_pair
        block_loss = loss_pair.sum(dim=(-2, -1))
        block_M_valid = (M * valid_pair).sum(dim=(-2, -1))
        block_valid = (block_M_valid > 1.0e-3).float()
        denom = block_valid.sum().clamp_min(1.0)
        block_avg_cost = block_loss / block_M_valid.clamp_min(1.0e-3)
        loss = (block_avg_cost * block_valid).sum() / denom

        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss


@register("loss")
class MSELoss:
    def __init__(
        self,
        weight: float,
        input_key: str,
        gt_key: str,
        mask_key: str = None,
        require_full_patch_valid: bool = False,
        loss_name: str = "MSELoss",
    ):
        self.weight = weight
        self.loss_name = loss_name
        self.input_key = input_key
        self.gt_key = gt_key
        self.mask_key = mask_key
        self.require_full_patch_valid = require_full_patch_valid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_key(self, batch, key):
        d = batch
        for p in key.split("/"):
            d = d[p]
        return d

    def _build_latent_mask(self, mask, pred):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(
                f"Expected mask to be 4D [B,1,H,W] or [B,C,H,W], got {mask.shape}."
            )

        mask = mask[:, :1].float()
        pred_h, pred_w = pred.shape[-2], pred.shape[-1]

        if mask.shape[-2:] == (pred_h, pred_w):
            latent_valid = mask > 0.5
        elif self.require_full_patch_valid:
            invalid = (mask <= 0.5).float()
            pooled_invalid = F.adaptive_max_pool2d(
                invalid, output_size=(pred_h, pred_w)
            )
            latent_valid = pooled_invalid <= 0
        else:
            pooled_valid = F.adaptive_avg_pool2d(mask, output_size=(pred_h, pred_w))
            latent_valid = pooled_valid > 0.5

        latent_valid = latent_valid.float()
        if pred.ndim == latent_valid.ndim and pred.shape[1] != latent_valid.shape[1]:
            if latent_valid.shape[1] == 1 and pred.shape[1] > 1:
                latent_valid = latent_valid.expand(-1, pred.shape[1], -1, -1)
        return latent_valid

    def __call__(self, batch):
        pred = batch["out"][self.input_key].to(self.device).float()
        tgt = batch["out"][self.gt_key].to(self.device).float()

        if self.mask_key is not None:
            mask = self._resolve_key(batch, self.mask_key).to(self.device)
            latent_mask = self._build_latent_mask(mask, pred)
            diff = (pred - tgt) ** 2
            loss = (diff * latent_mask).sum() / latent_mask.sum().clamp(min=1.0)
        else:
            loss = F.mse_loss(pred, tgt, reduction="mean")

        if "loss" not in batch:
            batch["loss"] = {}
        if "weighted_loss" not in batch:
            batch["weighted_loss"] = 0.0

        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] += self.weight * loss


@register("loss")
class MaskedL1Loss:
    """L1 loss with optional valid mask."""

    def __init__(
        self,
        weight: float,
        pred_key: str,
        gt_key: str,
        mask_key: str = None,
        loss_name: str = "masked_l1_loss",
        dataset_names=None,
        **kwargs,
    ):
        self.weight = weight
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.mask_key = mask_key
        self.loss_name = loss_name
        if dataset_names is None:
            self.dataset_names = None
        elif isinstance(dataset_names, (list, tuple)):
            self.dataset_names = set(dataset_names)
        else:
            self.dataset_names = {str(dataset_names)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_key(self, batch, key):
        d = batch
        for p in key.split("/"):
            d = d[p]
        return d

    def __call__(self, batch):
        pred = self._resolve_key(batch, self.pred_key).to(self.device).float()
        gt = self._resolve_key(batch, self.gt_key).to(self.device).float()

        if self.dataset_names is not None and "dataset_name" in batch:
            names = batch["dataset_name"]
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, (list, tuple)):
                try:
                    names = list(names)
                except Exception:
                    names = [str(names)]

            sel_idx = [i for i, n in enumerate(names) if str(n) in self.dataset_names]
            if len(sel_idx) == 0:
                batch.setdefault("loss", {})
                batch.setdefault("weighted_loss", 0.0)
                zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                batch["loss"][self.loss_name] = zero
                return

            if torch.is_tensor(pred) and torch.is_tensor(gt):
                pred = pred[sel_idx]
                gt = gt[sel_idx]
            else:
                raise RuntimeError(
                    "MaskedL1Loss: dataset filtering requires tensorized batched preds/gt"
                )

        if pred.ndim == gt.ndim == 4 and pred.shape[1] != gt.shape[1]:
            if gt.shape[1] == 1 and pred.shape[1] > 1:
                gt = gt.expand(-1, pred.shape[1], -1, -1)
            elif pred.shape[1] == 1 and gt.shape[1] > 1:
                pred = pred.expand(-1, gt.shape[1], -1, -1)

        if self.mask_key is not None:
            mask = self._resolve_key(batch, self.mask_key).to(self.device).float()
            if mask.ndim == pred.ndim and mask.shape[1] != pred.shape[1]:
                if mask.shape[1] == 1 and pred.shape[1] > 1:
                    mask = mask.expand(-1, pred.shape[1], -1, -1)
            diff = (pred - gt).abs()
            loss = (diff * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = F.l1_loss(pred, gt, reduction="mean")

        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss


@register("loss")
class MaskedL1GradientLoss:
    """
    Masked L1 loss on spatial gradients.

    Computes L1 error between finite differences of prediction and target
    along height and width, with optional masking over valid gradient pairs.
    """

    def __init__(
        self,
        weight: float,
        pred_key: str,
        gt_key: str,
        mask_key: str = None,
        loss_name: str = "masked_l1_gradient_loss",
        eps: float = 1e-6,
        **kwargs,
    ):
        self.weight = float(weight)
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.mask_key = mask_key
        self.loss_name = loss_name
        self.eps = float(eps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_key(self, batch, key):
        out = batch
        for part in key.split("/"):
            out = out[part]
        return out

    @staticmethod
    def _match_channels(a: torch.Tensor, b: torch.Tensor):
        if a.ndim == b.ndim == 4 and a.shape[1] != b.shape[1]:
            if a.shape[1] == 1 and b.shape[1] > 1:
                a = a.expand(-1, b.shape[1], -1, -1)
            elif b.shape[1] == 1 and a.shape[1] > 1:
                b = b.expand(-1, a.shape[1], -1, -1)
        return a, b

    @staticmethod
    def _finite_diffs(x: torch.Tensor):
        dh = x[:, :, 1:, :] - x[:, :, :-1, :]
        dw = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dh, dw

    def __call__(self, batch):
        pred = self._resolve_key(batch, self.pred_key).to(self.device).float()
        gt = self._resolve_key(batch, self.gt_key).to(self.device).float()
        pred, gt = self._match_channels(pred, gt)

        pred_dh, pred_dw = self._finite_diffs(pred)
        gt_dh, gt_dw = self._finite_diffs(gt)

        err_h = (pred_dh - gt_dh).abs()
        err_w = (pred_dw - gt_dw).abs()

        if self.mask_key is not None:
            mask = self._resolve_key(batch, self.mask_key).to(self.device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 4 and mask.shape[1] != pred.shape[1]:
                if mask.shape[1] == 1 and pred.shape[1] > 1:
                    mask = mask.expand(-1, pred.shape[1], -1, -1)

            mask_h = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            mask_w = mask[:, :, :, 1:] * mask[:, :, :, :-1]

            num = (err_h * mask_h).sum() + (err_w * mask_w).sum()
            den = mask_h.sum() + mask_w.sum()
            loss = num / den.clamp(min=self.eps)
        else:
            loss = 0.5 * (err_h.mean() + err_w.mean())

        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss


@register("loss")
class IREPADinoV3SpatialLoss:
    """
    iREPA-inspired spatial alignment loss using frozen DINOv3 features.

    This loss aligns predicted and target feature tokens while accentuating
    spatial structure via (1) spatial normalization and (2) token-pair
    similarity matching.
    """

    def __init__(
        self,
        weight: float,
        pred_key: str,
        gt_key: str,
        teacher_input_key: str = None,
        mask_key: str = None,
        loss_name: str = "irepa_dinov3_spatial_loss",
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        input_size: int = 224,
        gamma: float = 0.7,
        eps: float = 1e-6,
        token_l1_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        max_tokens_for_pairwise: int = 256,
        student_feature_keys=None,
        student_feature_weights=None,
        use_student_projection: bool = True,
        projection_type: str = "conv3x3",
        student_feature_dim: int = 64,
        teacher_feature_dim: int = None,
        projection_component_name: str = None,
        dataset_names=None,
        **kwargs,
    ):
        self.weight = float(weight)
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.teacher_input_key = (
            str(teacher_input_key) if teacher_input_key is not None else str(gt_key)
        )
        self.mask_key = mask_key
        self.loss_name = loss_name
        self.model_name = model_name
        self.input_size = int(input_size)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.token_l1_weight = float(token_l1_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.max_tokens_for_pairwise = int(max_tokens_for_pairwise)

        if student_feature_keys is None:
            self.student_feature_keys = []
        elif isinstance(student_feature_keys, Sequence) and not isinstance(
            student_feature_keys, (str, bytes)
        ):
            self.student_feature_keys = [str(k) for k in student_feature_keys]
        else:
            self.student_feature_keys = [str(student_feature_keys)]

        if student_feature_weights is None:
            self.student_feature_weights = [1.0] * max(
                1, len(self.student_feature_keys)
            )
        elif isinstance(student_feature_weights, Sequence) and not isinstance(
            student_feature_weights, (str, bytes)
        ):
            self.student_feature_weights = [float(x) for x in student_feature_weights]
        else:
            self.student_feature_weights = [float(student_feature_weights)]

        if len(self.student_feature_keys) > 0 and len(
            self.student_feature_weights
        ) != len(self.student_feature_keys):
            if len(self.student_feature_weights) == 1:
                self.student_feature_weights = self.student_feature_weights * len(
                    self.student_feature_keys
                )
            else:
                raise ValueError(
                    "IREPADinoV3SpatialLoss: student_feature_weights must match student_feature_keys length."
                )
        self.use_student_projection = bool(use_student_projection)
        self.projection_type = str(projection_type).lower()
        self.student_feature_dim = int(student_feature_dim)
        self.teacher_feature_dim = (
            int(teacher_feature_dim) if teacher_feature_dim is not None else None
        )

        if dataset_names is None:
            self.dataset_names = None
        elif isinstance(dataset_names, Sequence) and not isinstance(
            dataset_names, (str, bytes)
        ):
            self.dataset_names = set(dataset_names)
        else:
            self.dataset_names = {str(dataset_names)}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_model = None
        self._image_processor = None
        self._student_projector = None

        if projection_component_name is None:
            self.projection_component_name = f"{self.loss_name}_StudentProjector"
        else:
            self.projection_component_name = str(projection_component_name)

        if self.use_student_projection and len(self.student_feature_keys) > 0:
            # Default teacher dim from model family if not explicitly provided.
            if self.teacher_feature_dim is None:
                m = str(self.model_name).lower()
                if "vitl" in m or "h16" in m or "7b" in m:
                    self.teacher_feature_dim = 1024
                elif "vitb" in m:
                    self.teacher_feature_dim = 768
                elif "vits" in m:
                    self.teacher_feature_dim = 384
                else:
                    self.teacher_feature_dim = 768

            if self.projection_type not in {"linear", "conv1x1", "conv3x3"}:
                raise ValueError(
                    f"Unsupported projection_type '{self.projection_type}'. Expected one of: linear, conv1x1, conv3x3"
                )

            proj = nn.ModuleDict()
            for k in self.student_feature_keys:
                key_name = k.replace("/", "__")
                if self.projection_type == "linear":
                    layer = nn.Linear(
                        self.student_feature_dim, self.teacher_feature_dim, bias=False
                    )
                elif self.projection_type == "conv1x1":
                    layer = nn.Conv2d(
                        self.student_feature_dim,
                        self.teacher_feature_dim,
                        kernel_size=1,
                        padding=0,
                        bias=False,
                    )
                else:
                    layer = nn.Conv2d(
                        self.student_feature_dim,
                        self.teacher_feature_dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                nn.init.xavier_uniform_(layer.weight)
                proj[key_name] = layer

            REGISTRY.setdefault("network_components", {})
            REGISTRY["network_components"][self.projection_component_name] = proj
            self._student_projector = proj

    def _resolve_key(self, batch, key):
        out = batch
        for part in key.split("/"):
            out = out[part]
        return out

    def _lazy_init_backbone(self):
        if self._feature_model is not None and self._image_processor is not None:
            return
        if AutoModel is None or AutoImageProcessor is None:
            raise RuntimeError(
                "IREPADinoV3SpatialLoss requires 'transformers' with AutoModel and AutoImageProcessor support."
            )

        self._image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._feature_model = AutoModel.from_pretrained(self.model_name)
        self._feature_model.eval().to(self.device)
        for p in self._feature_model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor [B,C,H,W], got shape: {tuple(x.shape)}"
            )
        return x

    @staticmethod
    def _m11_to_01(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

    def _to_dino_inputs(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_4d(x).float()
        x = self._m11_to_01(x)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]

        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        mean = torch.tensor(
            self._image_processor.image_mean, device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self._image_processor.image_std, device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        return (x - mean) / std.clamp_min(self.eps)

    @staticmethod
    def _extract_patch_tokens(model_out) -> torch.Tensor:
        # Prefer patch tokens from last_hidden_state. If CLS exists, drop it.
        if (
            hasattr(model_out, "last_hidden_state")
            and model_out.last_hidden_state is not None
        ):
            tokens = model_out.last_hidden_state
        elif hasattr(model_out, "hidden_states") and model_out.hidden_states:
            tokens = model_out.hidden_states[-1]
        else:
            raise RuntimeError("DINOv3 output does not contain hidden states.")

        # Typical ViT shape is [B, T+1, D] with CLS at index 0.
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected token tensor [B,T,D], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        return tokens

    def _spatial_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], normalize over spatial token axis (dim=1).
        x = x - self.gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    def _build_token_mask(
        self, mask: torch.Tensor, num_tokens: int
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = self._ensure_4d(mask).float()
        if mask.shape[1] != 1:
            mask = mask[:, :1]

        token_hw = int(num_tokens**0.5)
        if token_hw * token_hw != num_tokens:
            return None

        m = F.interpolate(mask, size=(token_hw, token_hw), mode="nearest")
        m = (m > 0.5).flatten(1)  # [B, T]
        return m

    def _token_l1(
        self,
        pred_tokens: torch.Tensor,
        gt_tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        diff = (pred_tokens - gt_tokens).abs().mean(dim=-1)  # [B, T]
        if token_mask is None:
            return diff.mean()
        w = token_mask.float()
        return (diff * w).sum() / w.sum().clamp_min(1.0)

    def _pairwise_structure_loss(
        self,
        pred_tokens: torch.Tensor,
        gt_tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Pairwise cosine structure: S = normalize(X) @ normalize(X)^T
        bsz, num_tokens, _ = pred_tokens.shape
        if num_tokens > self.max_tokens_for_pairwise:
            pred_tokens = pred_tokens[:, : self.max_tokens_for_pairwise, :]
            gt_tokens = gt_tokens[:, : self.max_tokens_for_pairwise, :]
            if token_mask is not None:
                token_mask = token_mask[:, : self.max_tokens_for_pairwise]
            num_tokens = pred_tokens.shape[1]

        pred_norm = F.normalize(pred_tokens, dim=-1)
        gt_norm = F.normalize(gt_tokens, dim=-1)
        sim_pred = torch.matmul(pred_norm, pred_norm.transpose(1, 2))
        sim_gt = torch.matmul(gt_norm, gt_norm.transpose(1, 2))
        diff = (sim_pred - sim_gt).abs()  # [B, T, T]

        if token_mask is None:
            return diff.mean()

        m = token_mask.float()
        pair_mask = m.unsqueeze(2) * m.unsqueeze(1)
        return (diff * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    @staticmethod
    def _as_tokens(x: torch.Tensor) -> torch.Tensor:
        # Convert [B, C, H, W] -> [B, H*W, C], keep [B, T, D] as-is.
        if x.ndim == 5 and x.shape[2] == 1:
            x = x[:, :, 0]
        if x.ndim == 4:
            b, c, h, w = x.shape
            return x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if x.ndim == 3:
            return x
        raise ValueError(
            f"Unsupported student feature shape for token conversion: {tuple(x.shape)}"
        )

    @staticmethod
    def _as_feature_map(x: torch.Tensor) -> torch.Tensor:
        # Convert [B, T, D] token sequences to [B, D, H, W] when T is square.
        if x.ndim == 5 and x.shape[2] == 1:
            x = x[:, :, 0]
        if x.ndim == 4:
            return x
        if x.ndim == 3:
            b, t, d = x.shape
            hw = int(t**0.5)
            if hw * hw != t:
                raise ValueError(
                    f"Cannot apply convolutional projection to non-square token sequence shape {tuple(x.shape)}."
                )
            return x.transpose(1, 2).reshape(b, d, hw, hw)
        raise ValueError(
            f"Unsupported student feature shape for map conversion: {tuple(x.shape)}"
        )

    @staticmethod
    def _match_token_count(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        # Uniform resampling over token axis for mismatched token counts.
        t = tokens.shape[1]
        if t == target_tokens:
            return tokens
        if target_tokens <= 0:
            return tokens
        idx = torch.linspace(0, max(t - 1, 0), target_tokens, device=tokens.device)
        idx = idx.round().long().clamp(0, max(t - 1, 0))
        return tokens.index_select(1, idx)

    def __call__(self, batch):
        self._lazy_init_backbone()

        pred = self._resolve_key(batch, self.pred_key).to(self.device)
        gt = self._resolve_key(batch, self.gt_key).to(self.device)

        if self.dataset_names is not None and "dataset_name" in batch:
            names = batch["dataset_name"]
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, (list, tuple)):
                try:
                    names = list(names)
                except Exception:
                    names = [str(names)]

            sel_idx = [i for i, n in enumerate(names) if str(n) in self.dataset_names]
            if len(sel_idx) == 0:
                batch.setdefault("loss", {})
                batch.setdefault("weighted_loss", 0.0)
                zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                batch["loss"][self.loss_name] = zero
                return
            pred = pred[sel_idx]
            gt = gt[sel_idx]

        mask = None
        if self.mask_key is not None:
            mask = self._resolve_key(batch, self.mask_key).to(self.device)

        gt_in = self._to_dino_inputs(gt)
        with torch.no_grad():
            gt_out = self._feature_model(pixel_values=gt_in)
        gt_tokens = self._extract_patch_tokens(gt_out)

        projector = None
        if self.use_student_projection and len(self.student_feature_keys) > 0:
            projector = get("network_components", self.projection_component_name)

        # Align internal DiT features when student_feature_keys are given,
        # otherwise the DINOv3 features of the prediction itself.
        student_tokens_list = []
        student_weights = []
        if len(self.student_feature_keys) > 0:
            for idx_layer, (k, w) in enumerate(
                zip(self.student_feature_keys, self.student_feature_weights)
            ):
                try:
                    feat = self._resolve_key(batch, k)
                except Exception:
                    continue
                if not torch.is_tensor(feat):
                    continue
                feat = feat.to(self.device).float()

                if projector is not None:
                    proj_key = self.student_feature_keys[idx_layer].replace("/", "__")
                    if proj_key in projector:
                        layer = projector[proj_key]
                        if self.projection_type == "linear":
                            feat_tok = self._as_tokens(feat)
                            if feat_tok.shape[-1] != self.student_feature_dim:
                                raise ValueError(
                                    f"{self.loss_name}: student feature dim mismatch for {self.student_feature_keys[idx_layer]}: "
                                    f"expected {self.student_feature_dim}, got {feat_tok.shape[-1]}"
                                )
                            feat = layer(feat_tok)
                        else:
                            feat_map = self._as_feature_map(feat)
                            if feat_map.shape[1] != self.student_feature_dim:
                                raise ValueError(
                                    f"{self.loss_name}: student feature dim mismatch for {self.student_feature_keys[idx_layer]}: "
                                    f"expected {self.student_feature_dim}, got {feat_map.shape[1]}"
                                )
                            feat = layer(feat_map)

                feat = self._as_tokens(feat)
                feat = self._match_token_count(feat, gt_tokens.shape[1])
                student_tokens_list.append(feat)
                student_weights.append(float(w))

        if len(student_tokens_list) == 0:
            pred_in = self._to_dino_inputs(pred)
            pred_out = self._feature_model(pixel_values=pred_in)
            pred_tokens = self._extract_patch_tokens(pred_out)
            student_tokens_list = [pred_tokens]
            student_weights = [1.0]

        gt_tokens = self._spatial_normalize(gt_tokens)

        total_w = sum(student_weights) if len(student_weights) > 0 else 1.0
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        for idx_layer, (s_tok, w) in enumerate(
            zip(student_tokens_list, student_weights)
        ):
            s_tok = self._spatial_normalize(s_tok)
            token_mask = (
                self._build_token_mask(mask, num_tokens=s_tok.shape[1])
                if mask is not None
                else None
            )

            # Token L1 is only well-defined when feature dims match.
            if s_tok.shape[-1] == gt_tokens.shape[-1]:
                l_tok = self._token_l1(s_tok, gt_tokens, token_mask)
            else:
                l_tok = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            l_pair = self._pairwise_structure_loss(s_tok, gt_tokens, token_mask)
            l_layer = self.token_l1_weight * l_tok + self.pairwise_weight * l_pair
            loss = loss + (float(w) / max(total_w, 1e-12)) * l_layer

        batch.setdefault("loss", {})
        batch.setdefault("weighted_loss", 0.0)
        batch["loss"][self.loss_name] = loss
        batch["weighted_loss"] = batch["weighted_loss"] + self.weight * loss
