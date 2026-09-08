# builder.py
from typing import Any, Dict
from .registry import get
import importlib
import omegaconf


def register_experiment_modules(cfg):
    for m in cfg.register_modules:
        importlib.import_module(m)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class ComposeDict:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, data: Dict[str, Any]):
        for k, step in self.steps.items():
            result = step(data)
            if result is not None:
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict):
                    data = result
        return data


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_transforms(cfg, registry_category):
    step_dict = {}
    if isinstance(cfg, (dict, omegaconf.dictconfig.DictConfig)):
        for k, v in cfg.items():
            cls_key = (
                v.get("_target_", k)
                if isinstance(v, (dict, omegaconf.dictconfig.DictConfig))
                else k
            )
            kwargs = (
                dict(v) if isinstance(v, (dict, omegaconf.dictconfig.DictConfig)) else v
            )
            if isinstance(kwargs, dict):
                kwargs.pop("_target_", None)
            cls = get(registry_category, cls_key)
            step_dict[k] = cls(**kwargs)
    else:
        for i, transform_cfg in enumerate(cfg):
            for k, v in transform_cfg.items():
                cls = get(registry_category, k)
                step_dict[k + "_" + str(i)] = cls(**v)
    return ComposeDict(step_dict)


def build_dataset(cfg):
    ds_transform = build_transforms(
        cfg.transform, registry_category="dataset_transform"
    )
    manifest_transform = build_transforms(
        cfg.manifest_cfg.manifest_graph, registry_category="manifest_transform"
    )

    cfg["manifest"] = get("manifest", cfg.manifest_cfg.name)(
        transform_graph=manifest_transform, **cfg.manifest_cfg
    )()

    ds = get("dataset", cfg.name)(
        transform_graph=ds_transform,
        **cfg,
    )
    return ds
