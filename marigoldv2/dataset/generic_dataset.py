from torch.utils.data import Dataset
from PIL import UnidentifiedImageError

from marigoldv2.core.registry import register


@register("dataset")
class GenericDataset(Dataset):
    """Manifest entries run through a transform graph; unreadable files are skipped."""

    def __init__(self, manifest, transform_graph, disp_name, **kwargs) -> None:
        super().__init__()
        self.disp_name = disp_name
        self.transform_graph = transform_graph
        self.kwargs = dict(kwargs)
        self.max_io_retries = int(self.kwargs.get("max_io_retries", 5))
        self.annotations = [dict(v) for v in manifest.values()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        cur_index = index
        for _ in range(self.max_io_retries + 1):
            annotation = self.annotations[cur_index]
            sample = {
                "idx": cur_index,
                "annotation": annotation,
                "dataset_name": self.disp_name,
            }
            try:
                result = self.transform_graph(sample)
            except (UnidentifiedImageError, OSError) as exc:
                rgb_path = annotation.get("rgb_path", "<unknown>")
                print(
                    f"[GenericDataset] Skipping unreadable sample: {rgb_path} "
                    f"({type(exc).__name__}: {exc})"
                )
                if len(self) > 1:
                    cur_index = (cur_index + 1) % len(self)
                continue
            return result if isinstance(result, (dict, list)) else sample

        raise RuntimeError(
            f"Failed to fetch a readable sample after {self.max_io_retries} retries "
            f"starting from index {index} in dataset '{self.disp_name}'."
        )
