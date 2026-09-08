from tqdm import tqdm
from time import time

from marigoldv2.core.builder import build_transforms


def validate_dataloader(data_loader, metric_tracker=None, enable_visualization=True):
    dataset_disp_name = data_loader.dataset.disp_name
    dataset_cfg = data_loader.dataset.kwargs

    step_cfg = dataset_cfg["validation_steps"]
    if not enable_visualization:
        filtered_steps = []
        for s in step_cfg:
            if isinstance(s, dict) and len(s) == 1:
                step_name = next(iter(s.keys()))
                if isinstance(step_name, str) and step_name.startswith("Visualize"):
                    continue
            filtered_steps.append(s)
        step_cfg = filtered_steps

    validation_steps = build_transforms(step_cfg, registry_category="validation_steps")
    saved_output_paths = {}

    def _merge_saved_paths(batch_saved):
        if not isinstance(batch_saved, dict):
            return
        for folder_name, entries in batch_saved.items():
            saved_output_paths.setdefault(folder_name, []).extend(entries)

    start_loading = time()
    for i, batch in enumerate(
        tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
        start=1,
    ):
        end_loading = time()
        if end_loading - start_loading > 0.1:
            print(
                f"Validation data loading took longer than 0.1s: {end_loading - start_loading}"
            )

        batch["metric_tracker"] = metric_tracker
        batch["dataset_disp_name"] = dataset_disp_name
        batch["dataset_cfg"] = dataset_cfg
        validation_steps(batch)
        _merge_saved_paths(batch.get("_saved_output_paths", {}))

        start_loading = time()

    return saved_output_paths
