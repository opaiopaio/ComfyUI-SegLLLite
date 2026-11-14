"""Standalone Control-LoRA (pose) apply node for SEGS.

Why a separate node
-------------------
The two LLLite apply nodes can already attach a control-LoRA through a switch,
but a pose adapter is useful on its own:

* pose only (no LLLite): redraw the region while the skeleton holds the pose;
* chained: ``LLLite Apply (SEGS)`` then this node -- both controls run, because
  LLLite owns the single ``model_function_wrapper`` slot while the control-LoRA
  appends to the multi-slot ``wrappers["diffusion_model"]``.

Per segment the node runs, in order: the adapter's LoRA half (optional, block
indices remapped when needed), then the pose embedder injected at the DiT
patch-embed output.
"""

from __future__ import annotations

from ..anima_arch import AUTO_LABEL, MODEL_CHOICES
from ..compat import control_embedder as cembedder
from ..compat import control_lora as clora
from ..compat import impact
from ..i18n import t

from .lllite_segs import (
    CONTROL_HINT_MODES,
    _control_lora_names,
    apply_control_lora,
    build_control_lora_config,
    resolve_ctrl_hint,
)

LOG = "[SegLLLite]"


class ControlLoRASEGSWrapper:
    """Per-SEG control-LoRA wrapper (pose only) implementing the Impact protocol."""

    def __init__(
        self,
        cfg,
        preprocessor=None,
        prev_wrapper=None,
        preserve_wrapper=True,
        control_image=None,
        original_size=None,
        crop_region=None,
    ):
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.prev_wrapper = prev_wrapper if preserve_wrapper else None
        self._pending_hint = None

        self._ctrl_image = None
        if cfg.get("hint_source") == "control_image":
            if control_image is None or original_size is None or crop_region is None:
                raise ValueError(t("err.ctrl_control_image"))
            h, w = int(original_size[0]), int(original_size[1])
            full = impact.tensor_resize(control_image, w, h)
            self._ctrl_image = impact.tensor_crop(full, crop_region)

    # -- Impact protocol ---------------------------------------------------- #
    def apply(self, positive, negative, image, mask=None, use_acn=False):
        prev_pils = []
        if self.prev_wrapper is not None:
            positive, negative, prev_pils = self.prev_wrapper.apply(
                positive, negative, image, mask, use_acn=use_acn
            )
        # No LLLite here, so the fallback is the segment's own crop.
        self._pending_hint = resolve_ctrl_hint(
            self.cfg, image, mask, self._ctrl_image, self.preprocessor, image
        )
        return positive, negative, prev_pils

    def doit_ipadapter(self, model):
        prev_pils = []
        if self.prev_wrapper is not None:
            model, prev_pils = self.prev_wrapper.doit_ipadapter(model)

        if self._pending_hint is None:  # apply() was not called
            return model, prev_pils
        return apply_control_lora(model, self.cfg, self._pending_hint), prev_pils


class ControlLoRAApplySEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "control_lora_name": (_control_lora_names(), {"tooltip": t("tip.control_lora_name")}),
                "vae": ("VAE", {"tooltip": t("tip.ctrl_vae")}),
                "control_lora_strength": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": t("tip.control_lora_strength")},
                ),
                "control_lora_lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": t("tip.control_lora_lora_strength")},
                ),
                "control_lora_hint": (
                    CONTROL_HINT_MODES,
                    {"default": "segs_preprocessor", "tooltip": t("tip.control_lora_hint")},
                ),
                "control_lora_start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": t("tip.pose_ramp")},
                ),
                "control_lora_end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": t("tip.pose_ramp")},
                ),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
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
                "segs_preprocessor": ("SEGS_PREPROCESSOR", {"tooltip": t("tip.seg_processor")}),
                "control_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "run"
    CATEGORY = t("category")
    DESCRIPTION = t("desc.control_lora_apply")

    def run(self, segs, control_lora_name, vae, control_lora_strength,
            control_lora_lora_strength, control_lora_hint,
            control_lora_start_percent, control_lora_end_percent, preserve_wrapper,
            enable_remap, target_model, source_model,
            segs_preprocessor=None, control_image=None):
        if control_lora_hint == "segs_preprocessor" and segs_preprocessor is None:
            raise ValueError(t("err.ctrl_needs_preprocessor"))
        if control_lora_hint == "control_image" and control_image is None:
            raise ValueError(t("err.ctrl_control_image"))

        cfg = build_control_lora_config(
            True,
            control_lora_name,
            control_lora_strength,
            control_lora_lora_strength,
            control_lora_hint,
            vae,
            start_percent=control_lora_start_percent,
            end_percent=control_lora_end_percent,
            remap=enable_remap,
            target_choice=target_model,
            source_choice=source_model,
        )

        orig = impact.original_size(segs)

        def factory(seg, prev):
            return ControlLoRASEGSWrapper(
                cfg,
                preprocessor=segs_preprocessor,
                prev_wrapper=prev,
                preserve_wrapper=preserve_wrapper,
                control_image=control_image,
                original_size=orig,
                crop_region=impact.get_seg_field(seg, "crop_region"),
            )

        out = impact.apply_wrapper_to_segs(segs, factory)
        print(
            f"{LOG} Control-LoRA(SEGS): {control_lora_name} -> "
            f"{impact.segs_count(out)} seg(s), hint={control_lora_hint}, "
            f"pose={control_lora_strength:g}, lora={control_lora_lora_strength:g}"
        )
        return (out,)


NODE_CLASS_MAPPINGS = {"ControlLoRAApplySEGS": ControlLoRAApplySEGS}
NODE_DISPLAY_NAME_MAPPINGS = {"ControlLoRAApplySEGS": t("node.control_lora_apply")}
