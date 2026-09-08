import fnmatch
import logging

import torch
import torch.nn as nn
from bitsandbytes.functional import dequantize_4bit
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

from marigoldv2.core.registry import REGISTRY, register


def _bnb4bit_to_bf16_linear(bnb_lin: nn.Module) -> nn.Linear:
    """Convert a bitsandbytes Linear4bit to a standard nn.Linear in bfloat16."""
    assert hasattr(bnb_lin, "weight") and hasattr(bnb_lin.weight, "quant_state"), (
        "Not a 4-bit BnB linear"
    )
    device = next(bnb_lin.parameters()).device

    W = dequantize_4bit(bnb_lin.weight.data, bnb_lin.weight.quant_state)
    lin = nn.Linear(
        in_features=bnb_lin.in_features,
        out_features=bnb_lin.out_features,
        bias=bnb_lin.bias is not None,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        lin.weight.copy_(W.to(device=device, dtype=torch.bfloat16))
        if bnb_lin.bias is not None:
            lin.bias.copy_(
                bnb_lin.bias.detach().to(device=device, dtype=torch.bfloat16)
            )
    return lin


def _set_submodule(root: nn.Module, module_name: str, new_module: nn.Module):
    """Replace a nested module by dotted name, e.g. 'transformer_blocks.0.attn.to_out.0'."""
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _expand_module_patterns(model: nn.Module, patterns):
    """Expand wildcard patterns against model module names."""
    all_names = [n for n, _ in model.named_modules()]
    expanded = []
    for pat in patterns:
        if any(ch in pat for ch in ["*", "?", "["]):
            hits = [n for n in all_names if fnmatch.fnmatch(n, pat)]
            expanded.extend(hits)
        else:
            expanded.append(pat)
    # Keep order, remove duplicates.
    seen = set()
    out = []
    for n in expanded:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _dequantize_modules_to_bf16(model: nn.Module, module_patterns):
    """Dequantize selected 4-bit linear modules to bf16 nn.Linear."""
    target_names = _expand_module_patterns(model, module_patterns)
    replaced, skipped, missing = [], [], []

    named_modules = dict(model.named_modules())
    for name in target_names:
        if name not in named_modules:
            missing.append(name)
            continue

        mod = named_modules[name]
        if hasattr(mod, "weight") and hasattr(
            getattr(mod, "weight", None), "quant_state"
        ):
            new_mod = _bnb4bit_to_bf16_linear(mod)
            new_mod.requires_grad_(False)
            _set_submodule(model, name, new_mod)
            replaced.append(name)
        else:
            skipped.append(name)

    if replaced:
        logging.info(f"[Qwen-Image-Edit] dequantized modules to bf16: {replaced}")
    if skipped:
        logging.info(
            f"[Qwen-Image-Edit] dequantize skipped (not 4-bit linear): {skipped}"
        )
    if missing:
        logging.warning(f"[Qwen-Image-Edit] dequantize targets not found: {missing}")


def _normalize_quant_level(level) -> str:
    if level is None:
        return "4bit"
    value = str(level).strip().lower()
    aliases = {
        "4": "4bit",
        "4bit": "4bit",
        "nf4": "4bit",
        "8": "8bit",
        "8bit": "8bit",
        "int8": "8bit",
        "none": "none",
        "no": "none",
        "off": "none",
        "false": "none",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported quantization level '{level}'. Use one of: 4bit, 8bit, none"
        )
    return aliases[value]


@register("network_components")
class LoadQwenImageEditVAE:
    """Load the Qwen-Image-Edit VAE; optionally make the decoder trainable."""

    def __init__(self, name="VAE", train_decoder=False, **kwargs):
        self.name = name
        self.train_decoder = bool(train_decoder)

    def __call__(self, _=None):
        cfg = REGISTRY["cfg"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae_dtype = torch.bfloat16
        vae = AutoencoderKLQwenImage.from_pretrained(
            cfg.paths.ckpt_qwen_image_edit,
            subfolder="vae",
            local_files_only=True,
            torch_dtype=vae_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True,
        )

        vae.requires_grad_(False)
        if self.train_decoder:
            # Encoder and quant_conv stay frozen; post_quant_conv and decoder train.
            for sub in ("post_quant_conv", "decoder"):
                getattr(vae, sub).requires_grad_(True)
            n_train = sum(p.numel() for p in vae.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in vae.parameters())
            logging.info(
                f"[Qwen-Image-Edit VAE] decoder unfrozen; trainable params: {n_train} / {n_total}"
            )
        vae._supports_gradient_checkpointing = True
        vae.enable_gradient_checkpointing()
        vae.to(device, dtype=vae_dtype)

        logging.info("Loaded Qwen-Image-Edit VAE.")
        REGISTRY["network_components"][self.name] = vae


@register("network_components")
class LoadQwenImageEditTransformerFlexible:
    """Load the Qwen-Image-Edit DiT with 4/8-bit quantization and LoRA adapters.

    Reads ``cfg.optimization.quantization`` (level, skip_modules,
    dequantize_modules) and ``cfg.optimization.lora`` (rank, alpha, dropout,
    init, target_modules). Only the LoRA parameters are trainable.
    """

    def __init__(self, name="Diffuser", **kwargs):
        self.name = name

    def __call__(self, _=None):
        cfg = REGISTRY["cfg"]
        model_id = cfg.paths.get(
            "ckpt_qwen_image_edit_transformer", cfg.paths.ckpt_qwen_image_edit
        )
        lora_cfg_in = cfg.optimization.lora

        quant_cfg = cfg.optimization.get("quantization", {})
        quant_level = _normalize_quant_level(quant_cfg.get("level", "4bit"))
        skip_modules = list(
            quant_cfg.get("skip_modules", ["transformer_blocks.0.img_mod"])
        )
        dequantize_modules = list(quant_cfg.get("dequantize_modules", ["proj_out"]))

        quantization_config = None
        if quant_level == "4bit":
            quantization_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip_modules,
            )
        elif quant_level == "8bit":
            quantization_config = DiffusersBitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules,
            )

        transformer_kwargs = dict(
            pretrained_model_name_or_path=model_id,
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = quantization_config

        transformer = QwenImageTransformer2DModel.from_pretrained(**transformer_kwargs)

        if quantization_config is not None:
            transformer = prepare_model_for_kbit_training(
                transformer, use_gradient_checkpointing=False
            )

        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()

        if quant_level == "4bit":
            _dequantize_modules_to_bf16(transformer, dequantize_modules)
        elif dequantize_modules:
            logging.info(
                "[Qwen-Image-Edit] dequantize_modules is only applied for 4-bit mode; skipping for level=%s",
                quant_level,
            )

        target_modules = [t for t in lora_cfg_in.target_modules if "proj_out" not in t]
        lconf = LoraConfig(
            r=int(lora_cfg_in.rank),
            lora_alpha=int(lora_cfg_in.lora_alpha),
            lora_dropout=float(lora_cfg_in.lora_dropout),
            init_lora_weights=str(lora_cfg_in.init_lora_weights),
            target_modules=target_modules,
        )
        transformer.add_adapter(lconf)

        for name, p in transformer.named_parameters():
            if "lora_" in name:
                p.data = p.data.to(torch.bfloat16)
                p.requires_grad_(True)

        trn = transformer.num_parameters(only_trainable=True)
        tot = transformer.num_parameters()
        logging.info(
            f"[Qwen-Image-Edit] quantization={quant_level}; trainable params: {trn} / {tot}"
        )

        REGISTRY["network_components"][self.name] = transformer
