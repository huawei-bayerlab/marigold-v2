# registry.py

from typing import Callable, Dict, Any, Optional

# One dictionary holding sub-registries by category
REGISTRY: Dict[str, Dict[str, Any]] = {
    "dataset": {},
    "dataset_transform": {},
    "dataset_preparation": {},
    "manifest": {},
    "manifest_transform": {},
    "validation_steps": {},
    "network_components": {},
    "additional_components": {},
    "network_graph": {},
    "loss": {},
    "sampler": {},
    "collate": {},
    "optimizer": {},
    "scheduler": {},
    "cfg": {},
    "other": {},
}


class CategoryError(Exception):
    """Raised when a registry category does not exist."""

    pass


class RegistrationError(Exception):
    """Raised when an object cannot be registered or retrieved."""

    pass


def register(category: str, name: Optional[str] = None) -> Callable:
    """
    Usage:
        @register("transform")
        class ToTensor: ...

        @register("dataset", name="ImagePair")
        class MyDataset: ...
    """
    if category not in REGISTRY:
        raise CategoryError(f"Unknown category '{category}'. Known: {list(REGISTRY)}")

    def deco(obj: Any) -> Any:
        key = name or getattr(obj, "__name__", None)
        if key is None:
            raise ValueError(
                "Could not infer a name from object; provide 'name' explicitly."
            )
        if key in REGISTRY[category]:
            raise RegistrationError(f"{category}.{key} already registered")
        REGISTRY[category][key] = obj
        return obj

    return deco


def get(category: str, name: str):
    if category not in REGISTRY:
        raise CategoryError(f"Unknown category '{category}'. Known: {list(REGISTRY)}")
    try:
        return REGISTRY[category][name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY[category].keys()))
        raise RegistrationError(f"{category}.{name} not found. Known: [{known}]")
