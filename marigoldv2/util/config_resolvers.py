import os
from typing import Any

import omegaconf
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _to_number(x: Any):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        try:
            return float(x) if "." in x else int(x)
        except ValueError:
            pass
    raise ValueError(f"Cannot convert {x!r} to number")


def _maybe_int(orig, val):
    try:
        if int(orig) == orig and int(val) == val:
            return int(val)
    except Exception:
        pass
    return val


def register_omegaconf_resolvers() -> None:
    """Register the ``mul`` and ``scale_list`` resolvers used by the YAML configs.

    ``${mul:${scale},30000}`` multiplies two numbers; ``${scale_list:${scale},5000,10000}``
    multiplies every element of a list. Integer inputs stay integers.
    """
    if OmegaConf.has_resolver("mul"):
        return

    def mul(a, b):
        a_n = _to_number(a)
        return _maybe_int(a_n, a_n * _to_number(b))

    def scale_list(factor, *values):
        f = _to_number(factor)
        return [_maybe_int(_to_number(v), _to_number(v) * f) for v in values]

    OmegaConf.register_new_resolver("mul", mul)
    OmegaConf.register_new_resolver("scale_list", scale_list)


def recursive_load_config(config_path: str) -> OmegaConf:
    """Load a YAML config, merging its ``base_config`` list (repo-relative) first."""
    register_omegaconf_resolvers()
    conf = OmegaConf.load(config_path)
    output_conf = OmegaConf.create({})
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        assert isinstance(base_configs, omegaconf.listconfig.ListConfig)
        for _path in base_configs:
            assert _path != config_path, "base_config must not include itself"
            if not os.path.isabs(_path):
                _path = os.path.join(REPO_ROOT, _path)
            output_conf = OmegaConf.merge(output_conf, recursive_load_config(_path))
    return OmegaConf.merge(output_conf, conf)
