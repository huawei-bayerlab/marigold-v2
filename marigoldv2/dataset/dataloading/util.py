import cv2
import numpy as np
import torch
from PIL import Image


def load_rgb_data(rgb_path, key_prefix="rgb"):
    rgb = read_rgb_file(rgb_path)
    rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

    outputs = {
        f"{key_prefix}_int": torch.from_numpy(rgb).int(),
        f"{key_prefix}_norm": torch.from_numpy(rgb_norm),
    }
    return outputs


def read_rgb_file(rgb_path) -> np.ndarray:
    """Load a JPEG or PNG from disk and return a CHW uint8 array with 3 channels."""
    img = Image.open(rgb_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # shape (H, W, 3)
    return arr.transpose(2, 0, 1)  # shape (3, H, W)


def _lanczos_resize_chw(x, out_hw):
    """Lanczos4 resize for CHW images on CPU via OpenCV, preserving float32."""
    H_out, W_out = map(int, out_hw)

    if isinstance(x, np.ndarray):
        assert x.ndim == 3, "expect CHW"
        c = x.shape[0]
        hwc = np.transpose(x, (1, 2, 0)).astype(np.float32, copy=False)
        out = cv2.resize(hwc, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
        if c == 1 and out.ndim == 2:
            out = out[:, :, None]
        return np.transpose(out, (2, 0, 1)).astype(np.float32, copy=False)

    assert isinstance(x, torch.Tensor) and x.ndim == 3, "expect torch CHW"
    dev = x.device
    c = x.shape[0]
    hwc = x.detach().to("cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
    out = cv2.resize(hwc, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
    if c == 1 and out.ndim == 2:
        out = out[:, :, None]
    out_chw = torch.from_numpy(out).permute(2, 0, 1).contiguous()
    return out_chw.to(dev)
