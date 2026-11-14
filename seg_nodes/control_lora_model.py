"""Standalone Control-LoRA (pose adapter) node: MODEL -> MODEL.

A control-LoRA file (e.g. an Anima pose adapter) is one file doing two jobs:

* a rank-16 LoRA on the DiT blocks -- normally loaded with ``LoraLoaderModelOnly``;
* a zero-init ``control_embedder`` that turns a VAE-encoded skeleton into tokens
  added to the DiT patch-embed output.

This node owns both halves, so one file / one node is enough:

* ``lora_strength > 0`` applies the adapter's LoRA half **with optional
  cross-generation block remapping** (28-layer adapter on a 40-layer model);
* ``strength`` applies the pose embedder (``0`` disables it).

With both set to 0 the model is returned untouched (no clone, no hook).

Upstream ships this as a third-party node package with no licence declaration.
This implementation is our own (see ``compat/control_embedder.py``,
``compat/control_lora.py``); only weight key names and the maths dictated by the
tensor shapes are shared.
"""

from __future__ import annotations

from ..anima_arch import AUTO_LABEL, MODEL_CHOICES
from ..compat import control_embedder as cembedder
from ..compat import control_lora as clora
from ..i18n import t

from .lllite_segs import _control_lora_names, resolve_block_mapping

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
        return {
            "required": {
                "model": ("MODEL",),
                "control_lora_name": (_control_lora_names(), {"tooltip": t("tip.control_lora_name")}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": t("tip.control_lora_strength")},
                ),
                # appended after the published widgets (kept stable for workflows)
                "lora_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": t("tip.ctrl_model_lora_strength")},
                ),
                "enable_remap": ("BOOLEAN", {"default": True}),
                "target_model": (
                    MODEL_CHOICES,
                    {"default": AUTO_LABEL, "tooltip": t("tip.lllite_target")},
                ),
                "source_model": (
                    MODEL_CHOICES,
                    {"default": AUTO_LABEL, "tooltip": t("tip.lllite_source")},
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

    def run(self, model, control_lora_name, strength, lora_strength=0.0,
            enable_remap=True, target_model=AUTO_LABEL, source_model=AUTO_LABEL,
            control_latent=None, image=None, vae=None):
        # Both halves off is the documented baseline: hand the model back as-is.
        if float(strength) == 0.0 and float(lora_strength) == 0.0:
            print(f"{LOG} Control-LoRA(Anima): strength=0 and lora_strength=0, model unchanged")
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

        patched = model

        # ① the adapter's LoRA half (optional), remapped when needed
        if float(lora_strength) > 0.0:
            sd = clora.load_adapter(path)
            mapping = None
            if enable_remap:
                _, _, mapping = resolve_block_mapping(
                    patched, target_model, source_model,
                    clora.block_count_of_keys(sd.keys()),
                )
            patched, applied, candidates = clora.apply_lora_deltas(
                patched, sd, float(lora_strength), mapping
            )
            if applied == 0:
                raise RuntimeError(t("err.ctrl_lora_no_match", n=candidates))

        # ② the pose embedder
        if float(strength) != 0.0:
            latent = _latent_from(control_latent, image, vae)
            weight, bias = cembedder.load_embedder(path)
            channels = cembedder.check_channels(patched, weight)
            dit = cembedder.dit_of(patched)
            spatial, temporal = cembedder._patch_sizes(dit)
            provider = cembedder.token_provider(
                latent, weight, bias, spatial=spatial, temporal=temporal,
                strength=float(strength),
            )
            patched = cembedder.attach(patched, provider)
            print(
                f"{LOG} Control-LoRA(Anima): {control_lora_name} pose attached "
                f"(channels={channels}, patch={spatial}x{spatial}, strength={float(strength):g})"
            )

        return (patched,)


NODE_CLASS_MAPPINGS = {"ControlLoRAApplyModel": ControlLoRAApplyModel}
NODE_DISPLAY_NAME_MAPPINGS = {"ControlLoRAApplyModel": t("node.control_lora_apply_model")}
