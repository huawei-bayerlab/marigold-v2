"""Marigold-V2-DC: zero-shot metric depth completion by test-time LoRA on a frozen depth prior.

The released `depth/Log-stage2` checkpoint is a relative-depth model, so the sparse points have to
supply the metric scale. We freeze the prior *and* its trained depth LoRA, inject a second,
zero-initialised LoRA into the last N transformer blocks, and optimise that adapter together with a
two-parameter log-affine (a, b) against the provided sparse depth. One denoising step at t = 499/1000,
100 Adam steps, L1 + L2 on the sparse pixels. Ground truth is never read during optimisation.

Three configurations make up the table (see evaluation/depth_completion/REPRODUCING_DEPTH_COMPLETION.md):

  baseline  one pass at the per-dataset default resolution
  high-res  the prior runs on an upscaled RGB while the loss stays on the native pixels
  tiled     the whole procedure repeated per overlapping tile, then cross-faded

Run:  python -m evaluation.depth_completion.run_marigold_v2 --dataset ibims1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from peft import LoraConfig
from PIL import Image
from safetensors.torch import load_file

from evaluation.depth_completion.common import (
    ASSETS,
    MIN_VALID_DEPTH_M,
    RESULTS_DIR,
    compute_metrics,
    load_dc_sample,
    load_split,
    open_results,
    sample_id,
    splat_sparse,
    valid_mask,
)
from evaluation.depth_completion.tiling import run_tiled

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # deterministic cuBLAS GEMM, best effort

BASE = ASSETS / "checkpoints" / "Qwen-Image-Edit-2509"
DEPTH_CKPTS = ASSETS / "checkpoints" / "Marigold-V2" / "depth"
EMBEDS = ASSETS / "checkpoints" / "Marigold-V2" / "qwen_text_embeddings"
PREFIX = "qwen_edit_2509_qwen_depth_realimg512"

T_CONST = 499.0 / 1000.0
MAX_DEPTH_M = 1000.0
# The depth LoRA's target modules: the released inference config's list minus "proj_out", which
# component_loader filters out. Rank 128, as trained.
TARGET_MODULES = [
    "img_in",
    "txt_in",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_add_out",
    "norm.linear",
    "a_to_out",
    "b_to_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "norm_out.linear",
]
# Processing long side per dataset; 0 means native. `base` upscales NYUv2 to the resolution the
# prior operates at and downscales DDAD to fit memory. `hires` is the tuned per-dataset resolution
# and additionally implies --loss-at-native; it is what rows 2 and 3 of the table use.
PRESETS = {
    "base": {"ibims1": 0, "nyudepthv2": 768, "kittidc": 0, "ddad": 1024},
    "hires": {"ibims1": 1280, "nyudepthv2": 1152, "kittidc": 1824, "ddad": 0},
}


class QwenPrior:
    """The frozen Marigold-V2 depth prior: Qwen-Image-Edit-2509 plus the trained depth LoRA."""

    def __init__(self, ckpt):
        self.vae = self.tr = self.dev = None
        self.dt = torch.bfloat16
        self.ckpt = Path(ckpt)

    def load(self):
        self.dev = torch.device("cuda")
        vae = AutoencoderKLQwenImage.from_pretrained(
            BASE, subfolder="vae", local_files_only=True, torch_dtype=self.dt
        )
        vae = vae.to(self.dev).eval().requires_grad_(False)
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dt,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
        tr = QwenImageTransformer2DModel.from_pretrained(
            BASE,
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=self.dt,
            quantization_config=qconf,
        )
        tr.add_adapter(
            LoraConfig(
                r=128,
                lora_alpha=128,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=TARGET_MODULES,
            )
        )
        tr = tr.eval().requires_grad_(False)
        tr.enable_gradient_checkpointing()  # the adapter is optimised through the whole DiT

        by_component = {}
        for k, v in load_file(str(self.ckpt)).items():
            component, rest = k.split(".", 1)
            by_component.setdefault(component, {})[rest] = v
        if "Diffuser" not in by_component:
            raise RuntimeError(f"no Diffuser weights in {self.ckpt}")
        self._load_component(tr, by_component["Diffuser"], "Diffuser")
        if "VAE" in by_component:  # only present when the decoder was fine-tuned
            self._load_component(vae, by_component["VAE"], "VAE")

        self.mean = torch.tensor(
            vae.config.latents_mean, device=self.dev, dtype=self.dt
        ).view(1, 16, 1, 1, 1)
        self.std = torch.tensor(
            vae.config.latents_std, device=self.dev, dtype=self.dt
        ).view(1, 16, 1, 1, 1)
        self.emb = torch.load(
            EMBEDS / f"{PREFIX}_prompt_embeds.pt", map_location="cpu"
        ).to(self.dev, self.dt)[0:1]
        self.mask = (
            torch.load(EMBEDS / f"{PREFIX}_prompt_mask.pt", map_location="cpu") > 0
        ).to(self.dev)[0:1]
        self.vae, self.tr = vae, tr

    def _load_component(self, module, state, name):
        """Load a component's trained weights; every other weight stays pretrained."""
        _, unexpected = module.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} {name} keys in {self.ckpt} do not exist in the model, "
                f"e.g. {unexpected[0]}"
            )

    @torch.no_grad()
    def encode(self, rgb_norm):
        z = self.vae.encode(rgb_norm.unsqueeze(2)).latent_dist.mode()
        return (z - self.mean) / self.std

    def transformer_z0(self, z_norm):
        """One velocity step at the fixed training timestep: z0 = z - v(z, t)."""
        b, _, _, h, w = z_norm.shape
        packed = QwenImageEditPipeline._pack_latents(
            z_norm[:, :, 0].to(self.dt), b, 16, h, w
        )
        t = torch.full((b,), T_CONST, device=self.dev, dtype=self.dt)
        v = self.tr(
            hidden_states=packed,
            timestep=t,
            encoder_hidden_states=self.emb,
            encoder_hidden_states_mask=self.mask,
            img_shapes=[[(1, h // 2, w // 2)]] * b,
            txt_seq_lens=[int(self.mask.sum())],
            return_dict=False,
        )[0]
        v = QwenImageEditPipeline._unpack_latents(v, h * 8, w * 8, 8)
        return z_norm - v

    def decode_m11(self, z_norm):
        z = z_norm.to(self.dt) * self.std + self.mean
        d = self.vae.decode(z).sample[:, :, 0]
        return d.mean(1, keepdim=True).float()


def set_determinism(seed):
    """Freeze seeds and ask for deterministic kernels.

    The 4-bit prior's matmul has no deterministic GPU kernel, so per-sample results still move by
    about 1e-3 between runs; the dataset mean is stable to the same order.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resize_for_processing(rgb, sparse, long_side):
    """Put the RGB (and, unless the loss stays native, the sparse) on the processing grid."""
    h, w = rgb.shape[:2]
    current = max(h, w)
    if not long_side or current == long_side:
        return rgb, sparse
    s = long_side / current
    nh, nw = round(h * s), round(w * s)
    interp = Image.LANCZOS if s > 1 else Image.BILINEAR
    rgb_p = np.asarray(Image.fromarray(rgb).resize((nw, nh), interp), np.uint8)
    return rgb_p, splat_sparse(sparse, nh, nw)


def fit_ab(d_flat, log_target):
    """Least-squares (a, b) mapping the prior's relative output onto log metres at the anchors."""
    design = torch.stack([d_flat, torch.ones_like(d_flat)], 1)
    ab = torch.linalg.lstsq(design, log_target[:, None]).solution[:, 0]
    return ab[0].float(), ab[1].float()


def build_adapter(prior, rank, last_blocks, seed):
    """Load the frozen prior and inject the zero-initialised test-time adapter.

    The seed is set before any adapter init so a run is reproducible, and the pristine adapter
    weights are kept so every sample starts from zero delta.
    """
    torch.manual_seed(seed)
    prior.load()  # draws RNG for the depth LoRA's gaussian init
    blocks = {
        int(m.group(1))
        for name, _ in prior.tr.named_modules()
        if (m := re.search(r"transformer_blocks\.(\d+)\b", name))
    }
    n = max(blocks) + 1
    keep = set(range(n - last_blocks, n)) if last_blocks else blocks
    targets = [
        name
        for name, mod in prior.tr.named_modules()
        if hasattr(mod, "lora_A")
        and (m := re.search(r"transformer_blocks\.(\d+)\b", name))
        and int(m.group(1)) in keep
    ]
    prior.tr.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            init_lora_weights=True,
            target_modules=targets,
        ),
        adapter_name="ttt",
    )
    prior.tr.set_adapters(["default", "ttt"], weights=[1.0, 1.0])
    for name, p in prior.tr.named_parameters():
        p.requires_grad_("ttt" in name)
    params = [
        p
        for name, p in prior.tr.named_parameters()
        if "ttt" in name and p.requires_grad
    ]
    print(
        f"[build] rank={rank} last_blocks={last_blocks or 'all'} of {n} "
        f"trainable={sum(p.numel() for p in params) / 1e6:.2f}M",
        flush=True,
    )
    return {"params": params, "init": [p.detach().clone() for p in params]}


def predict_sample(prior, state, rgb, sparse, native_hw, cfg, proc_long_side):
    """Complete one image (or one tile) and return metric depth at `native_hw`, plus (a, b).

    `--loss-at-native` is what separates the high-res row from a plain resolution bump: only the
    RGB goes to the processing grid, the prediction is brought back to the native grid inside the
    optimisation loop, and the sparse points therefore stay on their own pixels instead of being
    splatted onto a coarser or finer grid.
    """
    for p, p0 in zip(state["params"], state["init"]):
        p.data.copy_(p0)  # every sample starts from a zero-delta adapter

    rgb_loss_native = bool(cfg.loss_at_native and proc_long_side)
    if proc_long_side:
        rgb_proc, sparse_proc = resize_for_processing(rgb, sparse, proc_long_side)
        rgb, sparse = rgb_proc, (sparse if rgb_loss_native else sparse_proc)
    h0, w0 = native_hw if rgb_loss_native else rgb.shape[:2]

    hp, wp = rgb.shape[:2]
    hr, wr = (
        (hp + 15) // 16 * 16,
        (wp + 15) // 16 * 16,
    )  # the VAE needs a multiple of 16
    rgb_r = (
        np.asarray(Image.fromarray(rgb).resize((wr, hr), Image.LANCZOS))
        if (hr, wr) != (hp, wp)
        else rgb
    )
    rgb_norm = (
        torch.from_numpy(np.ascontiguousarray(rgb_r)).permute(2, 0, 1)[None].float()
        / 255.0
        * 2
        - 1
    ).to(prior.dev, prior.dt)
    z_in = prior.encode(rgb_norm)

    sp = torch.from_numpy(sparse)[None, None].float().to(prior.dev)
    smask = sp > 0
    sval = sp[smask]
    log_sval = torch.log(sval.clamp(min=MIN_VALID_DEPTH_M)).double()
    lo, hi = np.log(MIN_VALID_DEPTH_M), np.log(MAX_DEPTH_M)

    def relative():
        return F.interpolate(
            prior.decode_m11(prior.transformer_z0(z_in)),
            size=(h0, w0),
            mode="bilinear",
            align_corners=False,
        )

    with torch.no_grad():
        a0, b0 = fit_ab(relative()[smask].double(), log_sval)
    a = torch.nn.Parameter(a0)
    b = torch.nn.Parameter(b0)
    opt = torch.optim.Adam(
        [
            {"params": state["params"], "lr": cfg.lr},
            {"params": [a, b], "lr": cfg.lr_ab},
        ]
    )
    for _ in range(cfg.steps):
        opt.zero_grad()
        pred = torch.exp(torch.clamp(a * relative() + b, lo, hi))[smask]
        loss = F.l1_loss(pred, sval) + F.mse_loss(pred, sval)
        loss.backward()
        opt.step()
    with torch.no_grad():
        depth = torch.exp(torch.clamp(a * relative() + b, lo, hi))

    dpred = depth[0, 0].cpu().numpy().astype(np.float32)
    if dpred.shape != tuple(native_hw):
        dpred = np.asarray(
            Image.fromarray(dpred).resize((native_hw[1], native_hw[0]), Image.BILINEAR),
            np.float32,
        )
    return dpred, {"a": float(a.detach()), "b": float(b.detach())}


def predict_tiled(prior, state, rgb, sparse, native_hw, cfg, proc_long_side):
    """Run the full procedure per overlapping tile and cross-fade the metric results.

    Each tile anchors to the sparse points inside it, at `--tile-scale` times its own long side,
    so a tile is completed at higher pixel density than the whole image would be. A tile with too
    few sparse points is skipped and its pixels come from the untiled prediction.
    """

    def predict(rgb_t, sparse_t):
        hw = rgb_t.shape[:2]
        proc = round(cfg.tile_scale * max(hw)) if cfg.tile_scale else proc_long_side
        return predict_sample(prior, state, rgb_t, sparse_t, hw, cfg, proc)[0]

    def untiled():
        return predict_sample(
            prior, state, rgb, sparse, native_hw, cfg, proc_long_side
        )[0]

    return run_tiled(
        predict,
        rgb,
        sparse,
        native_hw,
        cfg.tiles,
        cfg.tile_overlap,
        fallback=untiled,
        verbose=True,
    )


def shard_bounds(spec, total):
    """`I/N` -> (offset, limit) for a contiguous, near-equal slice of `total` samples.

    Shards are contiguous rather than strided so each one writes a distinct `_o<offset>.jsonl` and
    the summary merges them by sample id with no further bookkeeping.
    """
    try:
        index, count = (int(v) for v in spec.split("/"))
    except ValueError:
        raise SystemExit(f"--shard expects I/N, got {spec!r}") from None
    if not 0 < count or not 0 <= index < count:
        raise SystemExit(f"--shard {spec}: need 0 <= I < N and N > 0")
    start = index * total // count
    return start, (index + 1) * total // count - start


def resolve_checkpoint(spec):
    """A depth-variant name under assets/checkpoints/Marigold-V2/depth, or a path to one."""
    ckpt = Path(spec) if spec else DEPTH_CKPTS / "Log-stage2"
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = DEPTH_CKPTS / spec
    if ckpt.is_dir():
        ckpt = ckpt / "trainables.safetensors"
    return ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset", required=True, choices=["ibims1", "nyudepthv2", "kittidc", "ddad"]
    )
    p.add_argument(
        "--shard",
        default=None,
        metavar="I/N",
        help="run shard I of N (0-based) on this process; every sample is completed "
        "independently, so sharding does not change any result",
    )
    p.add_argument(
        "--offset", type=int, default=0, help="shard start, if not using --shard"
    )
    p.add_argument("--limit", type=int, default=None, help="shard length")
    p.add_argument(
        "--resume", action="store_true", help="append, skipping samples already scored"
    )
    p.add_argument(
        "--proc-preset",
        choices=list(PRESETS),
        default="base",
        help="per-dataset processing resolution; 'hires' also implies --loss-at-native",
    )
    p.add_argument(
        "--proc-long-side",
        type=int,
        default=-1,
        help="override the preset: -1 = use it, 0 = native resolution",
    )
    p.add_argument(
        "--loss-at-native",
        action="store_true",
        help="supervise on the native pixels, so the sparse is never resampled",
    )
    p.add_argument(
        "--tiles", default="1x1", help="RxC tiling, each tile completed on its own"
    )
    p.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="overlap as a fraction of tile size",
    )
    p.add_argument(
        "--tile-scale",
        type=float,
        default=2.0,
        help="per-tile processing long side = scale x the tile's own long side",
    )
    p.add_argument(
        "--steps", type=int, default=100, help="test-time optimisation steps per sample"
    )
    p.add_argument("--rank", type=int, default=16, help="rank of the test-time LoRA")
    p.add_argument(
        "--last-blocks",
        type=int,
        default=12,
        help="adapt the last N transformer blocks",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for the adapter"
    )
    p.add_argument(
        "--lr-ab",
        type=float,
        default=0.03,
        help="learning rate for the log-affine (a, b)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt", default=None, help="depth variant name or path (default: Log-stage2)"
    )
    p.add_argument(
        "--pred-dir",
        default=None,
        help="where pred_<sample>.npy goes (default: <out-dir>/predictions/<tag>_<dataset>)",
    )
    p.add_argument(
        "--no-save-pred",
        action="store_true",
        help="do not write the predicted depth maps",
    )
    p.add_argument("--tag", default="marigold_v2")
    p.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR),
        help="where the per-sample results and the predictions go",
    )
    a = p.parse_args()
    if a.proc_preset != "base":
        a.loss_at_native = True  # the preset names a method, not only a resolution
    set_determinism(a.seed)

    droot, lines, _ = load_split(a.dataset)
    if a.shard:
        # --shard computes its own offset and limit, so accepting these together would silently
        # ignore whichever the user meant.
        if a.offset or a.limit is not None:
            raise SystemExit("--shard sets the slice itself; drop --offset/--limit")
        a.offset, a.limit = shard_bounds(a.shard, len(lines))
    lines = lines[a.offset : (a.offset + a.limit) if a.limit else None]
    if not lines:
        print(f"[skip] {a.dataset}: empty shard", flush=True)
        return
    proc = (
        PRESETS[a.proc_preset][a.dataset] if a.proc_long_side < 0 else a.proc_long_side
    )

    ckpt = resolve_checkpoint(a.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    print(f"[ckpt] {ckpt}", flush=True)
    prior = QwenPrior(ckpt)
    state = build_adapter(prior, a.rank, a.last_blocks, a.seed)
    pred_dir = None
    if not a.no_save_pred:
        pred_dir = Path(
            a.pred_dir or Path(a.out_dir) / "predictions" / f"{a.tag}_{a.dataset}"
        )
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pred] {pred_dir}", flush=True)

    out, handle, done = open_results(a.out_dir, a.tag, a.dataset, a.offset, a.resume)
    maes = []
    with handle as f:
        for line in lines:
            if sample_id(line) in done:
                continue  # before loading: resuming should not read the sample off disk at all
            t = time.time()
            sid, rgb, gt, sparse = load_dc_sample(droot, line)
            if a.tiles == "1x1":
                dpred, info = predict_sample(
                    prior, state, rgb, sparse, gt.shape, a, proc
                )
            else:
                dpred, info = predict_tiled(
                    prior, state, rgb, sparse, gt.shape, a, proc
                )
                info = {"per_tile": info}
            m = compute_metrics(dpred, gt, valid_mask(gt), sid)
            if pred_dir is not None:
                np.save(pred_dir / f"pred_{sid}.npy", dpred)
            row = {
                "tag": a.tag,
                "dataset": a.dataset,
                "ckpt": ckpt.parent.name,
                "preset": a.proc_preset,
                "proc_long_side": proc,
                "loss_at_native": a.loss_at_native,
                "tiles": a.tiles if a.tiles != "1x1" else None,
                "tile_scale": a.tile_scale if a.tiles != "1x1" else None,
                "tile_overlap": a.tile_overlap if a.tiles != "1x1" else None,
                "steps": a.steps,
                "rank": a.rank,
                "last_blocks": a.last_blocks,
                "lr": a.lr,
                "lr_ab": a.lr_ab,
                "seed": a.seed,
                "n_sparse": int((sparse > 0).sum()),
                "secs": round(time.time() - t, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                **m,
                **info,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            maes.append(m["mae"])
            print(
                f"{a.tag} {a.dataset}/{sid}: MAE {m['mae']:.4f} ({row['secs']}s)",
                flush=True,
            )
    print(
        f"[done] {a.dataset} n={len(maes)} meanMAE={np.nanmean(maes) if maes else float('nan'):.4f} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
