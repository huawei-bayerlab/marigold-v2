from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def _is_img_like(x) -> bool:
    return torch.is_tensor(x) and x.ndim >= 2


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate dict samples; ragged image tensors stay lists instead of failing.

    Image-like tensors of equal shape are stacked, otherwise kept as a list.
    Other tensors go through ``default_collate`` with a list fallback. Numbers
    become tensors; strings, bytes, None, and lists are kept as lists; nested
    dicts are collated recursively. Keys missing from some samples are kept
    ragged (optional manifest metadata).
    """
    out: Dict[str, Any] = {}
    first_keys = list(batch[0].keys())
    extra_keys = sorted({k for b in batch for k in b.keys()} - set(first_keys))
    for k in first_keys + extra_keys:
        vals = [b.get(k, None) for b in batch]

        if any(v is None for v in vals):
            present_vals = [v for v in vals if v is not None]
            if present_vals and all(isinstance(v, dict) for v in present_vals):
                vals_dict = [v if isinstance(v, dict) else {} for v in vals]
                out[k] = custom_collate(vals_dict)  # type: ignore[arg-type]
            else:
                out[k] = vals
            continue

        if all(isinstance(v, np.ndarray) for v in vals):
            vals = [torch.from_numpy(v) for v in vals]

        if all(torch.is_tensor(v) for v in vals) and all(_is_img_like(v) for v in vals):
            shapes = [tuple(v.shape) for v in vals]
            out[k] = torch.stack(vals, dim=0) if len(set(shapes)) == 1 else vals
        elif all(torch.is_tensor(v) for v in vals):
            try:
                out[k] = default_collate(vals)
            except Exception:
                out[k] = vals
        elif all(isinstance(v, (int, float, bool)) for v in vals):
            out[k] = torch.tensor(vals)
        elif all(isinstance(v, str) for v in vals):
            out[k] = vals
        elif all(isinstance(v, (bytes, bytearray)) for v in vals):
            out[k] = list(vals)
        elif all(isinstance(v, dict) for v in vals):
            out[k] = custom_collate(vals)  # type: ignore[arg-type]
        elif all(isinstance(v, (list, tuple)) for v in vals):
            out[k] = [list(v) for v in vals]
        else:
            types = [type(v).__name__ for v in vals]
            raise TypeError(f"Don't know how to collate key '{k}' with types: {types}")
    return out
