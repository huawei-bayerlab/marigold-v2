import contextlib
import os

import torch
from diffusers import QwenImageEditPipeline

from marigoldv2.core.registry import REGISTRY, get, register


def _torch_load_full(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=map_location)


def _latent_stats(vae, ref):
    """Per-channel mean and inverse std of the Qwen VAE latent space."""
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=ref.device, dtype=ref.dtype)
    std = torch.tensor(vae.config.latents_std, device=ref.device, dtype=ref.dtype)
    return mean.view(shape), (1.0 / std).view(shape)


@register("network_graph")
class QwenImageEncode:
    """Encode an RGB image in [-1, 1] into normalized Qwen VAE latents."""

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = kwargs.get("input_key", "rgb_norm")
        self.output_key = kwargs.get("output_key", "lat_encoding")
        self.vae_component = kwargs.get("vae_component", "VAE")

    @torch.no_grad()
    def __call__(self, batch):
        k = self.input_key
        if k not in batch:
            return
        vae = get("network_components", self.vae_component)
        batch[k] = batch[k].to(device=vae.device, dtype=vae.dtype)
        out = vae.encode(batch[k].unsqueeze(2)).latent_dist.sample()
        mean, std_inv = _latent_stats(vae, out)
        batch["out"][self.output_key] = (out - mean) * std_inv


@register("network_graph")
class QwenDepthEncode:
    """Encode single-channel relative depth into Qwen VAE latent space.

    Depth is expected in [-1, 1] with shape [B, 1, H, W]. Since the Qwen VAE
    is RGB-based, the depth channel is repeated to 3 channels before encoding.
    """

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = kwargs.get("input_key", "depth_rel_m11")
        self.output_key = kwargs.get("output_key", "lat_depth_gt")
        self.vae_component = kwargs.get("vae_component", "VAE")

    @torch.no_grad()
    def __call__(self, batch):
        k = self.input_key
        if k not in batch:
            return

        vae = get("network_components", self.vae_component)

        x = batch[k].to(device=vae.device, dtype=vae.dtype)
        if x.ndim != 4:
            raise ValueError(f"Expected {k} to have shape [B, C, H, W], got {x.shape}")

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected {k} to have 1 or 3 channels, got {x.shape[1]}")

        out = vae.encode(x.unsqueeze(2)).latent_dist.sample()
        mean, std_inv = _latent_stats(vae, out)
        batch["out"][self.output_key] = (out - mean) * std_inv


@register("network_graph")
class QwenImageDecode:
    """Decode normalized Qwen VAE latents into an image in [-1, 1]."""

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = kwargs.get("input_key", "lat_encoding")
        self.output_key = kwargs.get("output_key", "pixel_pred")

    def __call__(self, batch):
        vae = get("network_components", "VAE")
        if self.input_key not in batch["out"]:
            return
        latents = batch["out"][self.input_key]
        mean, std_inv = _latent_stats(vae, latents)
        latents = latents / std_inv + mean
        decoded = vae.decode(latents)
        batch["out"][self.output_key] = decoded.sample[:, :, 0]  # drop temporal dim


@register("network_graph")
class QwenImageEdit2509Step:
    """Single rectified-flow step of the Qwen-Image-Edit-2509 transformer.

    Reads normalized VAE latents from ``batch["out"]["lat_encoding"]`` and the
    precomputed prompt embeddings ``<embed_dir>/{prefix}_prompt_embeds.pt`` and
    ``{prefix}_prompt_mask.pt``. Writes the predicted latents back to
    ``batch["out"]["lat_encoding"]``. With ``capture_hidden_states`` the
    selected transformer hidden states are stored under
    ``batch["out"]["{hidden_state_key_prefix}_{index}"]`` for iREPA.
    """

    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        embeds_dir = REGISTRY["cfg"].paths.embed_dir
        prefix = kwargs.get("prefix", "qwen_edit_2509")
        self.prompt_embeds_cpu = _torch_load_full(
            os.path.join(embeds_dir, f"{prefix}_prompt_embeds.pt")
        ).contiguous()  # [N_ctx, L, D]
        self.prompt_mask_cpu = _torch_load_full(
            os.path.join(embeds_dir, f"{prefix}_prompt_mask.pt")
        ).contiguous()  # [N_ctx, L]
        if self.prompt_mask_cpu.dtype != torch.bool:
            self.prompt_mask_cpu = self.prompt_mask_cpu > 0

        self.predict_eps = kwargs.get("predict_eps", False)
        self.predict_vel = kwargs.get("predict_vel", True)
        self.capture_hidden_states = bool(kwargs.get("capture_hidden_states", False))
        self.hidden_state_indices = list(kwargs.get("hidden_state_indices", [-1]))
        self.hidden_state_key_prefix = str(
            kwargs.get("hidden_state_key_prefix", "qwen_dit_hidden_state")
        )

    @staticmethod
    def _match_batch(t, B):
        if t.size(0) == B:
            return t
        if t.size(0) == 1 and B > 1:
            return t.expand(B, *t.shape[1:])
        if t.size(0) > B:
            return t[:B]
        reps = (B + t.size(0) - 1) // t.size(0)
        return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]

    def __call__(self, batch):
        diffuser = get("network_components", "Diffuser")
        vae = get("network_components", "VAE")

        model_input = batch["out"]["lat_encoding"]
        if model_input.ndim == 5 and model_input.shape[2] == 1:
            model_input_4d = model_input[:, :, 0]
        elif model_input.ndim == 4:
            model_input_4d = model_input
        else:
            raise ValueError(f"Unexpected lat_encoding shape: {model_input.shape}")
        B, C, H, W = model_input_4d.shape
        device = next(diffuser.parameters()).device
        run_dtype = torch.bfloat16

        prompt_embeds = self._match_batch(self.prompt_embeds_cpu, B).to(
            device=device, dtype=run_dtype, non_blocking=True
        )
        prompt_mask = self._match_batch(self.prompt_mask_cpu, B).to(
            device=device, dtype=torch.bool, non_blocking=True
        )

        packed = QwenImageEditPipeline._pack_latents(
            model_input_4d, batch_size=B, num_channels_latents=C, height=H, width=W
        ).to(run_dtype)  # [B, (H/2)*(W/2), C*4]

        # Fixed timestep t = 0.499, computed in bf16 exactly as during training.
        timestep = torch.full((B,), 499.0, device=device, dtype=run_dtype) / 1000.0
        img_shapes = [[(1, H // 2, W // 2)]] * B
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()
        attention_kwargs = getattr(diffuser, "attention_kwargs", None) or {}
        call_kwargs = dict(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
        )
        cache_ctx = (
            diffuser.cache_context("cond")
            if hasattr(diffuser, "cache_context")
            else contextlib.nullcontext()
        )
        with cache_ctx:
            if self.capture_hidden_states:
                try:
                    out_obj = diffuser(
                        **call_kwargs, output_hidden_states=True, return_dict=True
                    )
                    model_pred = out_obj.sample
                    hidden_states = getattr(out_obj, "hidden_states", None)
                    if hidden_states is not None:
                        n = len(hidden_states)
                        for idx in self.hidden_state_indices:
                            resolved = idx if idx >= 0 else n + idx
                            if 0 <= resolved < n:
                                key = f"{self.hidden_state_key_prefix}_{idx}"
                                batch["out"][key] = hidden_states[resolved]
                except TypeError:  # transformer without output_hidden_states
                    model_pred = diffuser(**call_kwargs, return_dict=False)[0]
            else:
                model_pred = diffuser(**call_kwargs, return_dict=False)[0]

        temporal_downsample = vae.config.get("temperal_downsample", None)
        vae_scale_factor = (
            2 ** len(temporal_downsample) if temporal_downsample is not None else 8
        )
        model_pred = QwenImageEditPipeline._unpack_latents(
            model_pred,
            height=H * vae_scale_factor,
            width=W * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )  # [B, C, 1, H, W]
        if self.predict_eps:
            latents = model_input.to(vae.dtype) + model_pred.to(vae.dtype)
        elif self.predict_vel:
            latents = model_input.to(vae.dtype) - model_pred.to(vae.dtype)
        else:
            latents = model_pred.to(vae.dtype)
        batch["out"]["lat_encoding"] = latents


@register("network_graph")
class SelectChannel:
    def __init__(self, kwargs=None):
        kwargs = kwargs or {}
        self.input_key = kwargs.get("input_key", "pixel_pred")
        self.output_key = kwargs.get("output_key", "depth_rel_pred_m11")
        self.channel = kwargs.get("channel", "mean")

    def __call__(self, batch):
        x = batch["out"][self.input_key]
        if self.channel == "mean":
            y = x.mean(dim=1, keepdim=True)
        else:
            idx = int(self.channel)
            y = x[:, idx : idx + 1]
        batch["out"][self.output_key] = y
