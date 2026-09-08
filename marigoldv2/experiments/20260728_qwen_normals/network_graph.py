import torch

from marigoldv2.core.registry import register
from marigoldv2.validation.util import get_nested_key


@register("network_graph")
class NormalizeSurfaceNormals:
    """Convert a decoded three-channel image into unit surface normals."""

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = str(kwargs.get("input_key", "out/pixel_pred"))
        self.output_key = str(kwargs.get("output_key", "normal_pred"))
        self.eps = float(kwargs.get("eps", 1e-6))

    def __call__(self, batch):
        decoded = get_nested_key(batch, self.input_key)
        if decoded is None:
            raise KeyError(f"NormalizeSurfaceNormals: missing '{self.input_key}'")
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                f"NormalizeSurfaceNormals expects [B,3,H,W], got {tuple(decoded.shape)}"
            )

        dtype = decoded.dtype
        decoded_float = decoded.float()
        magnitude = torch.linalg.vector_norm(decoded_float, dim=1, keepdim=True)
        normal = decoded_float / magnitude.clamp_min(self.eps)
        normal = torch.where(magnitude > self.eps, normal, torch.zeros_like(normal))
        batch.setdefault("out", {})
        batch["out"][self.output_key] = normal.to(dtype=dtype)
        return batch
