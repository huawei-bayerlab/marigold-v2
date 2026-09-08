from marigoldv2.core.registry import register
from marigoldv2.validation.util import get_nested_key


@register("network_graph")
class RGBAlbedoPrediction:
    """Map decoded Qwen RGB from [-1, 1] to linear RGB albedo in [0, 1]."""

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = str(kwargs.get("input_key", "out/pixel_pred"))
        self.output_key = str(kwargs.get("output_key", "albedo_pred"))

    def __call__(self, batch):
        decoded = get_nested_key(batch, self.input_key)
        if decoded is None:
            raise KeyError(f"RGBAlbedoPrediction: missing '{self.input_key}'")
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                f"RGBAlbedoPrediction expects [B,3,H,W], got {tuple(decoded.shape)}"
            )
        batch.setdefault("out", {})
        batch["out"][self.output_key] = ((decoded.float() + 1.0) * 0.5).to(
            decoded.dtype
        )
        return batch


@register("network_graph")
class GrayscaleAlbedoPrediction:
    """Map decoded Qwen RGB from [-1, 1] to repeated linear grayscale."""

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = str(kwargs.get("input_key", "out/pixel_pred"))
        self.output_key = str(kwargs.get("output_key", "albedo_pred"))

    def __call__(self, batch):
        decoded = get_nested_key(batch, self.input_key)
        if decoded is None:
            raise KeyError(f"GrayscaleAlbedoPrediction: missing '{self.input_key}'")
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                f"GrayscaleAlbedoPrediction expects [B,3,H,W], got {tuple(decoded.shape)}"
            )
        dtype = decoded.dtype
        rgb = (decoded.float() + 1.0) * 0.5
        gray = 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
        batch.setdefault("out", {})
        batch["out"][self.output_key] = gray.repeat(1, 3, 1, 1).to(dtype)
        return batch
