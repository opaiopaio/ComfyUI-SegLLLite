"""Standalone Control-LoRA (pose adapter) node: MODEL -> MODEL.

A control-LoRA file (e.g. an Anima pose adapter) is one file doing two jobs:

* a rank-16 LoRA on the DiT blocks -- normally loaded with ``LoraLoaderModelOnly``;
* a zero-init ``control_embedder`` that turns a VAE-encoded skeleton into tokens
  added to the DiT patch-embed output -- that is this node.

Upstream ships this as a third-party node package with no licence declaration.
This implementation is our own (see ``compat/control_embedder.py``); only the
weight key names and the maths dictated by the tensor shapes are shared.

The node accepts either an already encoded ``control_latent`` (drop-in
replacement for the upstream parameter set) or a raw ``image`` + ``vae`` pair,
which saves wiring a separate VAEEncode and guarantees the encoding resolution
matches the control image.
"""

from __future__ import annotations

from ..compat import control_embedder as cembedder
from ..compat import control_lora as clora
from ..i18n import t

LOG = "[SegLLLite]"


def _latent_from(control_latent, image, vae):
    """Pick the control latent from whichever source was provided."""
    if control_latent is not None:
        if image is not None:
            print(f"{LOG} both control_latent and image given; using control_latent")
        latent = control_latent.get("samples") if isinstance(control_latent, dict) else control_latent
        if latent is None:
            raise ValueError(t("err.ctrl_need_source"))
        return latent
    if image is not None:
        if vae is None:
            raise ValueError(t("err.ctrl_image_needs_vae"))
        return cembedder.encode_image(vae, image)
    raise ValueError(t("err.ctrl_need_source"))


class ControlLoRAApplyModel:
    @classmethod
    def INPUT_TYPES(cls):
        from .lllite_segs import _lora_names

        return {
            "required": {
                "model": ("MODEL",),
                "control_lora_name": (_lora_names(), {"tooltip": t("tip.control_lora_name")}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": t("tip.control_lora_strength")},
                ),
            },
            "optional": {
                "control_latent": ("LATENT", {"tooltip": t("tip.ctrl_latent")}),
                "image": ("IMAGE", {"tooltip": t("tip.ctrl_image")}),
                "vae": ("VAE", {"tooltip": t("tip.ctrl_vae")}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = t("category_control")
    DESCRIPTION = t("desc.control_lora_apply_model")

    def run(self, model, control_lora_name, strength, control_latent=None, image=None, vae=None):
        # strength=0 is the documented baseline: hand the model back untouched so
        # there is no clone, no hook and no chance of a numerical difference.
        if float(strength) == 0.0:
            print(f"{LOG} Control-LoRA(Anima): strength=0, model returned unchanged")
            return (model,)

        path = clora.resolve_lora_path(control_lora_name)
        if path is None:
            raise ValueError(t("err.ctrl_bad_file", name=control_lora_name))
        if not cembedder.has_embedder(path):
            keys = []
            try:
                keys = [k for k in cembedder.read_lora_keys(path)][:3]
            except Exception:
                pass
            print(f"{LOG} keys found in {control_lora_name}: {keys}")
            raise ValueError(t("err.ctrl_no_embedder", name=control_lora_name))

        latent = _latent_from(control_latent, image, vae)
        weight, bias = cembedder.load_embedder(path)
        model_channels = cembedder.check_channels(model, weight)

        dit = cembedder.dit_of(model)
        spatial, temporal = cembedder._patch_sizes(dit)
        provider = cembedder.token_provider(
            latent, weight, bias, spatial=spatial, temporal=temporal,
            strength=float(strength),
        )
        patched = cembedder.attach(model, provider)
        print(
            f"{LOG} Control-LoRA(Anima): {control_lora_name} attached "
            f"(channels={model_channels}, patch={spatial}x{spatial}, strength={float(strength):g})"
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"ControlLoRAApplyModel": ControlLoRAApplyModel}
NODE_DISPLAY_NAME_MAPPINGS = {"ControlLoRAApplyModel": t("node.control_lora_apply_model")}
