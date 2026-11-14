"""LLLite ⇄ SEGS bridge.

Why this exists
---------------
Anima-LLLite wraps the model with a closure that captures **one global image**
and rescales it to whatever latent size is being sampled. Impact's detailer
crops each region (e.g. 576x576) and samples it separately, so the full image
gets squeezed into that crop → the hint and the latent are geometrically
misaligned. This is a data-flow mismatch, not a parameter issue.

Impact's ``control_net_wrapper`` protocol is invoked **inside** the per-SEG loop
with the already-cropped, already-upscaled image:

    positive, negative, cnet_pils = wrapper.apply(positive, negative, upscaled_image, noise_mask)
    model, cnet_pils2 = wrapper.doit_ipadapter(model)

So we only need a wrapper that stores that crop and patches the model with an
LLLite whose hint *is* that crop. Hint and latent are then 1:1 by construction.

Licensing: we duck-type the protocol and never import comfyui-impact-pack
(GPL-3.0), which keeps this package MIT.
"""

import torch

from ..compat import anima_lllite as anima
from ..compat import impact
from ..i18n import t

LOG = "[LLLiteSEGS]"

HINT_MODES = ["image", "pose", "control_image"]


# --------------------------------------------------------------------------- #
# wrapper
# --------------------------------------------------------------------------- #
class LLLiteSEGSWrapper:
    """Per-SEG ControlNet-LLLite wrapper implementing the Impact protocol."""

    def __init__(
        self,
        lllite_name,
        strength,
        start_percent,
        end_percent,
        hint_mode,
        preprocessor=None,
        prev_wrapper=None,
        preserve_wrapper=True,
        control_image=None,
        original_size=None,
        crop_region=None,
        need_mask=False,
    ):
        self.lllite_name = lllite_name
        self.strength = float(strength)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.hint_mode = hint_mode
        self.preprocessor = preprocessor
        self.prev_wrapper = prev_wrapper if preserve_wrapper else None
        self.need_mask = bool(need_mask)

        # control_image mode: match the old ControlNetApply(SEGS) behaviour --
        # resize the global control image to the original image size, then cut
        # out this segment's region once, up front.
        self.control_image = None
        if (
            hint_mode == "control_image"
            and control_image is not None
            and original_size is not None
            and crop_region is not None
        ):
            try:
                h, w = int(original_size[0]), int(original_size[1])
                full = impact.tensor_resize(control_image, w, h)
                self.control_image = impact.tensor_crop(full, crop_region)
            except Exception as e:  # pragma: no cover - defensive
                print(f"{LOG} control_image crop failed: {e}")

        self._pending_hint = None
        self._pending_mask = None

    # -- Impact protocol ---------------------------------------------------- #
    def apply(self, positive, negative, image, mask=None, use_acn=False):
        prev_pils = []
        if self.prev_wrapper is not None:
            positive, negative, prev_pils = self.prev_wrapper.apply(
                positive, negative, image, mask, use_acn=use_acn
            )

        if self.control_image is not None:
            hint = self.control_image
        elif self.hint_mode == "pose" and self.preprocessor is not None:
            hint = _run_preprocessor(self.preprocessor, image, mask)
        else:
            hint = image  # the segment's own crop, already at guide_size

        self._pending_hint = hint
        self._pending_mask = mask if self.need_mask else None
        # LLLite produces no cnet PIL images of its own.
        return positive, negative, prev_pils

    def doit_ipadapter(self, model):
        prev_pils = []
        if self.prev_wrapper is not None:
            model, prev_pils = self.prev_wrapper.doit_ipadapter(model)

        hint = self._pending_hint
        if hint is None:  # apply() was not called -> nothing to do
            return model, prev_pils

        patched = anima.apply_lllite(
            model,
            self.lllite_name,
            hint,
            self.strength,
            self.start_percent,
            self.end_percent,
            True,
            self._pending_mask,
        )
        # Keep _pending_hint: some detailers call doit_ipadapter more than once
        # per segment. Re-patching a fresh clone each time is idempotent in
        # effect, because `model` is the pristine one on every call.
        return patched, prev_pils


def _run_preprocessor(preprocessor, image, mask):
    """SEGS_PREPROCESSOR protocol: ``apply(image, mask) -> IMAGE``."""
    try:
        out = preprocessor.apply(image, mask)
    except TypeError:
        # Some providers accept only the image.
        out = preprocessor.apply(image)
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _controlnet_names():
    try:
        import folder_paths

        names = list(folder_paths.get_filename_list("controlnet"))
    except Exception:
        names = []
    return names or ["None"]


def _seg_hint(seg, hint_mode, preprocessor, original_size, control_image):
    """Compute the hint for a single seg (used by the preview node)."""
    image = impact.get_seg_field(seg, "cropped_image")
    mask = impact.get_seg_field(seg, "cropped_mask")
    crop_region = impact.get_seg_field(seg, "crop_region")

    if hint_mode == "control_image" and control_image is not None and original_size:
        h, w = int(original_size[0]), int(original_size[1])
        full = impact.tensor_resize(control_image, w, h)
        return impact.tensor_crop(full, crop_region)
    if hint_mode == "pose" and preprocessor is not None:
        return _run_preprocessor(preprocessor, image, mask)
    return image


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
class LLLiteApplySEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "lllite_name": (_controlnet_names(),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "hint_mode": (HINT_MODES, {"default": "image", "tooltip": t("tip.hint_mode")}),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "segs_preprocessor": (
                    "SEGS_PREPROCESSOR",
                    {"tooltip": t("tip.seg_processor")},
                ),
                "control_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "run"
    CATEGORY = t("category")
    DESCRIPTION = t("desc.lllite_apply")

    def run(
        self,
        segs,
        lllite_name,
        strength,
        start_percent,
        end_percent,
        hint_mode,
        preserve_wrapper,
        segs_preprocessor=None,
        control_image=None,
    ):
        if hint_mode == "pose" and segs_preprocessor is None:
            raise ValueError(t("err.pose_needs_preprocessor"))
        if hint_mode == "control_image" and control_image is None:
            raise ValueError(t("err.control_needs_image"))
        if lllite_name in (None, "", "None"):
            raise ValueError(t("err.bad_lllite", name=lllite_name))

        # Fail early (and clearly) if Anima-LLLite is missing or the file is bad.
        if anima.resolve_weights_path(lllite_name) is None:
            raise ValueError(t("err.bad_lllite", name=lllite_name))
        anima.load_module()  # raises with a readable message when absent

        need_mask = anima.cond_in_channels(lllite_name) == 4
        orig = impact.original_size(segs)

        def factory(seg, prev):
            return LLLiteSEGSWrapper(
                lllite_name=lllite_name,
                strength=strength,
                start_percent=start_percent,
                end_percent=end_percent,
                hint_mode=hint_mode,
                preprocessor=segs_preprocessor,
                prev_wrapper=prev,
                preserve_wrapper=preserve_wrapper,
                control_image=control_image,
                original_size=orig,
                crop_region=impact.get_seg_field(seg, "crop_region"),
                need_mask=need_mask,
            )

        out = impact.apply_wrapper_to_segs(segs, factory)
        print(
            f"{LOG} LLLite(SEGS): {lllite_name} -> {impact.segs_count(out)} seg(s), "
            f"mode={hint_mode}, 4ch_mask={need_mask}"
        )
        return (out,)


class LLLiteClearSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "run"
    CATEGORY = t("category")
    DESCRIPTION = t("desc.lllite_clear")

    def run(self, segs):
        return (impact.clear_wrappers(segs),)


class LLLiteHintPreviewSEGS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "hint_mode": (HINT_MODES, {"default": "image", "tooltip": t("tip.hint_mode")}),
            },
            "optional": {
                "segs_preprocessor": ("SEGS_PREPROCESSOR", {"tooltip": t("tip.seg_processor")}),
                "control_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hints",)
    FUNCTION = "run"
    CATEGORY = t("category")
    DESCRIPTION = t("desc.lllite_preview")
    OUTPUT_NODE = True

    def run(self, segs, hint_mode, segs_preprocessor=None, control_image=None):
        if hint_mode == "pose" and segs_preprocessor is None:
            raise ValueError(t("err.pose_needs_preprocessor"))
        if hint_mode == "control_image" and control_image is None:
            raise ValueError(t("err.control_needs_image"))

        orig = impact.original_size(segs)
        hints = []
        for _, seg in impact.iter_segs(segs):
            h = _seg_hint(seg, hint_mode, segs_preprocessor, orig, control_image)
            if h is None:
                continue
            if h.dim() == 3:
                h = h.unsqueeze(0)
            hints.append(h)

        if not hints:
            raise ValueError(t("hint.no_segs"))

        # A single IMAGE batch needs a common size -> resize every hint to the
        # first one (preview only, never used for sampling).
        base_h, base_w = hints[0].shape[1], hints[0].shape[2]
        norm = [impact.tensor_resize(h, base_w, base_h) for h in hints]
        batch = torch.cat(norm, dim=0)

        text = t("hint.seg_count", n=len(hints))
        return {"ui": {"text": [text]}, "result": (batch,)}


NODE_CLASS_MAPPINGS = {
    "LLLiteApplySEGS": LLLiteApplySEGS,
    "LLLiteClearSEGS": LLLiteClearSEGS,
    "LLLiteHintPreviewSEGS": LLLiteHintPreviewSEGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLLiteApplySEGS": t("node.lllite_apply"),
    "LLLiteClearSEGS": t("node.lllite_clear"),
    "LLLiteHintPreviewSEGS": t("node.lllite_preview"),
}
