import os
import random
import numpy as np
import torch
from accelerate import Accelerator


def save_step_info_txt(save_dir: str, step: int, val_info: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "step_info.txt")

    lines = []
    lines.append(f"global_step: {step}")
    lines.append("")

    for dataset_name, metrics in (val_info or {}).items():
        lines.append(f"[{dataset_name}]")
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()
            lines.append(f"{k}: {v}")
        lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class AccelerateRandomContext:
    """
    Snapshot current RNG (Python/NumPy/Torch CPU/CUDA) per process,
    set a fixed seed during the context, then restore exactly.
    """

    def __init__(
        self, accelerator: Accelerator, seed: int, device_specific: bool = True
    ):
        self.accelerator = accelerator
        self.base_seed = seed
        self.device_specific = device_specific

        self._py = None
        self._np = None
        self._torch_cpu = None
        self._torch_cuda_all = None

    def __enter__(self):
        self._py = random.getstate()
        self._np = np.random.get_state()
        self._torch_cpu = torch.get_rng_state()
        if torch.cuda.is_available():
            self._torch_cuda_all = torch.cuda.get_rng_state_all()

        self.accelerator.wait_for_everyone()

        seed = self.base_seed + (
            self.accelerator.process_index if self.device_specific else 0
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self._py)
        np.random.set_state(self._np)
        torch.set_rng_state(self._torch_cpu)
        if torch.cuda.is_available() and self._torch_cuda_all is not None:
            torch.cuda.set_rng_state_all(self._torch_cuda_all)

        self.accelerator.wait_for_everyone()
