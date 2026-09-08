from torch.utils.data import Dataset

from marigoldv2.core.registry import register


@register("manifest")
class GenericManifest(Dataset):
    def __init__(
        self,
        transform_graph,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transform_graph = transform_graph

    def __call__(self):
        manifest = {}
        self.transform_graph(manifest)
        return manifest
