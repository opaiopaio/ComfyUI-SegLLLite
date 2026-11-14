"""Control-LoRA support: the ``control_embedder`` (pose signal) half.

A *control-LoRA* file (Anima pose model) is one file with two halves:

* ``diffusion_model.blocks.<n>.*.lora_A/lora_B`` + ``llm_adapter.*`` -- a normal
  rank-16 LoRA, loaded with :func:`apply_lora_deltas`;
* ``diffusion_model.control_embedder.proj.weight/.bias`` -- a *zero-initialised*
  ``Linear(in_channels * p * p * r -> model_channels)`` that turns a VAE-encoded
  skeleton into tokens which are **added to the DiT patch-embed output**.

This module implements the second half. Nothing is copied from the upstream
implementation: the shape of the maths follows from the weight tensors
(``Linear(64, 2048)`` for every Anima model), and the code is written here.

Injection contract (mirrors what the trained weights expect):

    tokens = strength * control_embedder(control_latent)      # (B, T, H/p, W/p, D)
    x_embedder_output += tokens                                # per DiT forward

The addition is installed as a temporary forward hook on ``dit.x_embedder`` and
removed in a ``finally`` so it never leaks onto the shared DiT.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

LOG = "[SegLLLite]"

#: prefix inside the adapter file
EMBEDDER_MARK = "control_embedder."
#: (path, mtime, size) -> (weight, bias)
_WEIGHTS_CACHE: dict = {}


class ControlEmbedderError(RuntimeError):
    """Raised when the file does not carry usable control-embedder weights."""


def read_lora_keys(path: str) -> list:
    """Keys of a safetensors/ckpt file without loading the tensors."""
    if os.path.splitext(path)[1].lower() == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return list(f.keys())
    sd = torch.load(path, map_location="cpu")
    return list(sd.keys())


def load_embedder(path: str):
    """Return ``(weight, bias)`` of the control embedder (cached per file)."""
    stamp = (path, os.path.getmtime(path), os.path.getsize(path))
    cached = _WEIGHTS_CACHE.get(stamp)
    if cached is not None:
        return cached

    if os.path.splitext(path)[1].lower() == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            keys = [k for k in f.keys() if EMBEDDER_MARK in k]
            if not keys:
                raise ControlEmbedderError(
                    f"{os.path.basename(path)} carries no '{EMBEDDER_MARK}' weights "
                    f"(expected e.g. diffusion_model.control_embedder.proj.weight). "
                    "This file is not a control-LoRA."
                )
            weight = bias = None
            for k in keys:
                if k.endswith("proj.weight"):
                    weight = f.get_tensor(k)
                elif k.endswith("proj.bias"):
                    bias = f.get_tensor(k)
    else:
        sd = torch.load(path, map_location="cpu")
        weight = bias = None
        for k, v in sd.items():
            if EMBEDDER_MARK in k and k.endswith("proj.weight"):
                weight = v
            elif EMBEDDER_MARK in k and k.endswith("proj.bias"):
                bias = v

    if weight is None:
        raise ControlEmbedderError(
            f"{os.path.basename(path)} has '{EMBEDDER_MARK}' keys but no "
            "'...control_embedder.proj.weight'"
        )
    result = (weight.float(), None if bias is None else bias.float())
    _WEIGHTS_CACHE[stamp] = result
    return result


def has_embedder(path: str) -> bool:
    try:
        load_embedder(path)
        return True
    except Exception:
        return False


def _patch_sizes(dit):
    """``(spatial, temporal)`` patch size of the DiT, with sane fallbacks."""
    spatial = getattr(dit, "patch_spatial", 2)
    temporal = getattr(dit, "patch_temporal", 1)
    if isinstance(spatial, (tuple, list)):
        spatial = spatial[0]
    if isinstance(temporal, (tuple, list)):
        temporal = temporal[0]
    return int(spatial or 2), int(temporal or 1)


def pack_latent(latent: torch.Tensor, spatial: int = 2, temporal: int = 1):
    """``(B, C, T, H, W)`` -> ``(B, T, ceil(H/p), ceil(W/p), C*p*p*r)``.

    Padding matches ``comfy.ldm.common_dit.pad_to_patch_size``: the DiT pads the
    *sampler* latent up to a patch multiple before ``x_embedder``, so the token
    grid is ``ceil``, never ``floor`` -- our control latent must line up with it.
    """
    if latent.dim() == 4:  # (B, C, H, W)
        latent = latent.unsqueeze(2)
    b, c, t, h, w = latent.shape
    p, r = spatial, temporal

    pad_h = (p - h % p) % p
    pad_w = (p - w % p) % p
    pad_t = (r - t % r) % r
    if pad_h or pad_w or pad_t:
        # F.pad applies from the last dim backwards: (W, H, T)
        latent = F.pad(latent, (0, pad_w, 0, pad_h, 0, pad_t))
    h2, w2 = (h + pad_h) // p, (w + pad_w) // p
    t2 = (t + pad_t) // r

    x = latent.reshape(b, c, t2, r, h2, p, w2, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, T, H2, W2, C, r, p, p)
    return x.reshape(b, t2, h2, w2, c * r * p * p)


def control_tokens(
    latent: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    spatial: int = 2,
    temporal: int = 1,
    strength: float = 1.0,
    target_grid: tuple | None = None,
):
    """Project a control latent into ``x_embedder`` output space.

    ``target_grid`` is ``(H_tokens, W_tokens)`` of the actual ``x_embedder``
    output. When the two disagree (odd crop sizes, refiner passes) the tokens are
    resized instead of raising, because a wrong size is recoverable while a hard
    failure in the middle of sampling is not.
    """
    packed = pack_latent(latent, spatial, temporal)
    in_features = packed.shape[-1]
    if in_features != weight.shape[1]:
        raise ControlEmbedderError(
            f"control latent packs to {in_features} features but the embedder "
            f"expects {weight.shape[1]} (latent channels={latent.shape[1]}, "
            f"patch={spatial}x{spatial}x{temporal}). Wrong VAE or patch size?"
        )

    tokens = F.linear(packed, weight, bias)

    if target_grid is not None:
        th, tw = int(target_grid[0]), int(target_grid[1])
        b, t, h, w, d = tokens.shape
        if (h, w) != (th, tw):
            src = tokens.permute(0, 1, 4, 2, 3).reshape(b * t, d, h, w)
            src = F.interpolate(src, size=(th, tw), mode="nearest")
            tokens = src.reshape(b, t, d, th, tw).permute(0, 1, 3, 4, 2)
            print(
                f"{LOG} control tokens resized {h}x{w} -> {th}x{tw} to match x_embedder"
            )

    return tokens * float(strength)


def compute_from_latent(latent: torch.Tensor, weight, bias, **kwargs):
    """Convenience wrapper that accepts a LATENT dict or a bare tensor."""
    if isinstance(latent, dict):
        latent = latent.get("samples", latent)
    return control_tokens(latent, weight, bias, **kwargs)


def dit_of(model):
    """The diffusion model inside a ModelPatcher, or a clear error."""
    dit = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dit is None:
        raise RuntimeError(f"{LOG} model has no diffusion_model; cannot inject control")
    return dit


def check_channels(model, weight) -> int:
    """Validate the embedder output width against the model's channels."""
    dit = dit_of(model)
    model_channels = getattr(dit, "model_channels", weight.shape[0])
    if int(weight.shape[0]) != int(model_channels):
        raise RuntimeError(
            f"{LOG} control embedder outputs {weight.shape[0]} channels but this model "
            f"has {model_channels}"
        )
    return int(model_channels)


def encode_image(vae, image):
    """IMAGE (B,H,W,3|4) -> control latent (B,C,h,w)."""
    if vae is None:
        raise RuntimeError(f"{LOG} a vae is required to encode the control image")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[-1] == 4:  # drop alpha
        image = image[..., :3]
    out = vae.encode(image)
    return out["samples"] if isinstance(out, dict) else out


def token_provider(latent, weight, bias, *, spatial=2, temporal=1, strength=1.0):
    """``x_embedder`` output -> control tokens, computed once per grid/device/dtype.

    Shared by the SEGS wrappers and the standalone model node so the projection
    maths and its caching live in exactly one place.
    """
    cache = {}

    def provider(output):
        key = (output.shape[2], output.shape[3], str(output.device), str(output.dtype))
        tokens = cache.get(key)
        if tokens is None:
            tokens = compute_from_latent(
                latent,
                weight,
                bias,
                spatial=spatial,
                temporal=temporal,
                strength=strength,
                target_grid=key[:2],
            )
            tokens = tokens.to(output.device, output.dtype)
            cache[key] = tokens
            print(
                f"{LOG} control tokens {tuple(tokens.shape)} "
                f"(latent {tuple(latent.shape)}, strength={float(strength):g})"
            )
        return tokens

    return provider


def attach(model, tokens_provider, key: str = "seglllite_control"):
    """Return a clone with a DIFFUSION_MODEL wrapper adding control tokens.

    ``tokens_provider(output) -> tokens`` is called on every DiT forward with the
    ``x_embedder`` output, so it can build (and cache) tokens on the real device
    and dtype the model ended up on -- ComfyUI loads models lazily, so at apply
    time the DiT may still be on CPU.
    """
    dit = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dit is None:
        raise RuntimeError(f"{LOG} model has no diffusion_model; cannot inject control")
    if getattr(dit, "x_embedder", None) is None:
        raise RuntimeError(
            f"{LOG} this DiT exposes no 'x_embedder' attribute, so control tokens "
            "cannot be injected (unsupported architecture)"
        )

    def wrapper(executor, *args, **kwargs):
        def add_control(_module, _inputs, output):
            return output + tokens_provider(output)

        handle = dit.x_embedder.register_forward_hook(add_control)
        try:
            return executor(*args, **kwargs)
        finally:
            handle.remove()

    patched = model.clone()
    patched.add_wrapper_with_key("diffusion_model", key, wrapper)
    return patched
