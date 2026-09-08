import os
from urllib.parse import urlparse
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def build_trainable_param_groups_from_network_components(cfg, registry_or_cfg_dict):
    network_components = registry_or_cfg_dict["network_components"]

    lr = cfg.optimization.scheduler.base_lr

    param_groups = []
    seen_param_ids = set()
    total_params = 0
    total_trainable = 0

    for name, module in network_components.items():
        if module is None:
            continue
        if not hasattr(module, "parameters"):
            continue

        for p in module.parameters(recurse=True):
            total_params += p.numel()
            if p.requires_grad:
                total_trainable += p.numel()

        trainable = []
        for p in module.parameters(recurse=True):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            trainable.append(p)

        if len(trainable) > 0:
            param_groups.append({"params": trainable, "lr": lr, "name": name})

    if len(param_groups) == 0:
        raise ValueError(
            "No trainable parameters found in network_components (requires_grad=True)."
        )

    return param_groups, {
        "num_groups": len(param_groups),
        "total_params_numel": total_params,
        "total_trainable_numel": total_trainable,
    }


def trainable_state_dict(module: torch.nn.Module) -> dict:
    named_params = dict(module.named_parameters(recurse=True))
    out = {}
    for k, v in module.state_dict().items():
        p = named_params.get(k, None)
        if p is not None and p.requires_grad:
            out[k] = v.detach().cpu()
        elif k.endswith(".weight_u") or k.endswith(".weight_v"):
            out[k] = v.detach().cpu()
    return out


def make_save_trainables_hook(
    registry, accelerator, *, filename="trainables.safetensors"
):
    def save_trainables_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

            merged = {}
            for name, module in registry["network_components"].items():
                if not isinstance(module, nn.Module):
                    continue

                module = accelerator.unwrap_model(module)
                sd = trainable_state_dict(module)

                for k, v in sd.items():
                    merged[f"{name}.{k}"] = v

            out_path = os.path.join(output_dir, filename)
            save_file(merged, out_path)
            print(f"[save] trainables -> {out_path} ({len(merged)} tensors)")

        models.clear()
        weights.clear()

    return save_trainables_hook


def make_load_trainables_hook(
    registry,
    accelerator,
    *,
    filename="trainables.safetensors",
    exclude_components=None,
    allow_missing_components=None,
):
    if exclude_components is None:
        exclude_components = []
    exclude_components = set(exclude_components)
    if allow_missing_components is None:
        allow_missing_components = []
    allow_missing_components = set(allow_missing_components)

    is_main = accelerator.is_main_process if accelerator else True

    def _resolve_checkpoint_dir(input_dir):
        if os.path.isdir(input_dir):
            return input_dir

        candidate = str(input_dir).rstrip("/")
        repo_id = None
        if candidate.startswith("https://huggingface.co/"):
            parsed = urlparse(candidate)
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                repo_id = "/".join(parts[:2])
        elif "/" in candidate and not candidate.startswith("/"):
            repo_id = candidate

        if repo_id is None:
            return input_dir

        if is_main:
            print(f"[load] Resolving Hugging Face checkpoint repo: {repo_id}")
        return snapshot_download(repo_id=repo_id, repo_type="model")

    def load_trainables_hook(models, input_dir):
        loaded_components = set()
        path = os.path.join(_resolve_checkpoint_dir(input_dir), filename)
        if not os.path.exists(path):
            if is_main:
                print(f"[load] No trainable checkpoint found at {path}")
            models.clear()
            return
        if is_main:
            print(f"[load] Loading trainables from {path}")
        sd = load_file(path, device="cpu")

        by_comp = {}
        for k, v in sd.items():
            if "." not in k:
                continue
            comp, rest = k.split(".", 1)
            by_comp.setdefault(comp, {})[rest] = v

        for name, module in registry["network_components"].items():
            if name in exclude_components or name not in by_comp:
                continue
            if not isinstance(module, nn.Module):
                continue
            module = accelerator.unwrap_model(module) if accelerator else module
            missing, unexpected = module.load_state_dict(by_comp[name], strict=False)
            if is_main and (missing or unexpected):
                print(
                    f"[load] {name}: missing={len(missing)} unexpected={len(unexpected)}"
                )
            loaded_components.add(name)

        if is_main:
            _report_unloaded_trainables(
                registry,
                loaded_components,
                allow_missing_components,
                exclude_components,
            )
        models.clear()

    return load_trainables_hook


def _report_unloaded_trainables(
    registry, loaded_components, allow_missing_components, exclude_components
):
    missing_trainable_components = []
    allowed_missing = []

    for name, module in registry["network_components"].items():
        if name in exclude_components or name in loaded_components:
            continue
        if not isinstance(module, nn.Module):
            continue
        has_trainable = any(p.requires_grad for p in module.parameters(recurse=True))
        if not has_trainable:
            continue
        if name in allow_missing_components:
            allowed_missing.append(name)
        else:
            missing_trainable_components.append(name)

    for name in allowed_missing:
        print(f"[load] {name}: no checkpoint weights found; initialized from scratch")

    if missing_trainable_components:
        names = ", ".join(sorted(missing_trainable_components))
        print(
            f"[load] Warning: no checkpoint weights found for trainable component(s): {names}"
        )
