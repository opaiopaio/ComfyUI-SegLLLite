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

Optional Control-LoRA (pose) support
------------------------------------
A control-LoRA (e.g. an Anima pose adapter) can be attached from the same node,
because the two mechanisms live in different places and therefore coexist:

* LLLite          -> ``model_options["model_function_wrapper"]`` (single slot)
* control-LoRA    -> ``wrappers["diffusion_model"]`` (multi slot, consumed inside
                     the DiT forward)

One switch then runs "redraw + pose control" on every segment, with the skeleton
coming from the SEGS preprocessor, a global control image, or the LLLite hint.

Licensing: we duck-type the Impact protocol and never import comfyui-impact-pack
(GPL-3.0), which keeps this package MIT. The control-LoRA maths follows from the
weight tensor shapes; no upstream code is copied.
"""

import torch

from ..anima_arch import (
    bucket_block_count,
    describe_count,
    get_mapping,
    resolve_dropdown,
)
from ..compat import anima_lllite as anima
from ..compat import control_embedder as cembedder
from ..compat import control_lora as clora
from ..compat import impact
from ..i18n import t

LOG = "[SegLLLite]"

HINT_MODES = ["image", "pose", "control_image"]

#: where the *control-LoRA* (pose) signal comes from -- independent of hint_mode
CONTROL_HINT_MODES = ["segs_preprocessor", "control_image", "same as lllite"]


# --------------------------------------------------------------------------- #
# shared block-count resolution
# --------------------------------------------------------------------------- #
def resolve_block_mapping(model, target_choice, source_choice, source_blocks):
    """Resolve ``(source, target, mapping)`` for cross-generation remapping.

    ``source_blocks`` is what the *file* says (or None when unreadable). The file
    wins over a manually chosen ``source_choice`` -- a wrong override silently
    skips the remap and then fails deep inside a loader, which is far worse than
    ignoring a stale dropdown.
    """
    chosen = resolve_dropdown(source_choice)
    if source_blocks and chosen and source_blocks != chosen:
        print(
            f"{LOG} source_model={describe_count(chosen)} but the file is "
            f"{describe_count(source_blocks)}; using the file"
        )
    source = source_blocks or chosen
    if source is None:
        raise ValueError(t("err.unknown_source"))

    target = resolve_dropdown(target_choice)
    if target is None:
        target = bucket_block_count(detect_model_blocks(model))
    if target is None:
        print(f"{LOG} could not detect the model's block count; assuming it matches the file")
        target = source

    if source == target:
        return source, target, None
    mapping = get_mapping(source, target)
    if mapping is None:
        raise ValueError(
            t("err.remap_direction", src=describe_count(source), tgt=describe_count(target))
        )
    return source, target, mapping


def detect_model_blocks(model):
    from .lllite_remap import detect_model_block_count

    return detect_model_block_count(model)


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
        apply_fn=None,
        control_lora=None,
    ):
        self.lllite_name = lllite_name
        self.strength = float(strength)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.hint_mode = hint_mode
        self.preprocessor = preprocessor
        self.prev_wrapper = prev_wrapper if preserve_wrapper else None
        self.need_mask = bool(need_mask)
        # Optional hook: LLLiteApplySEGSRemap passes a remapping loader here.
        self.apply_fn = apply_fn

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

        # --- optional control-LoRA (pose) ---
        self.ctrl = dict(control_lora) if control_lora else None
        self._ctrl_image = None
        if self.ctrl and self.ctrl.get("hint_source") == "control_image":
            if control_image is not None and original_size is not None and crop_region is not None:
                try:
                    h, w = int(original_size[0]), int(original_size[1])
                    full = impact.tensor_resize(control_image, w, h)
                    self._ctrl_image = impact.tensor_crop(full, crop_region)
                except Exception as e:  # pragma: no cover - defensive
                    print(f"{LOG} control-LoRA control_image crop failed: {e}")
            else:
                raise ValueError(t("err.ctrl_control_image"))

        self._pending_hint = None
        self._pending_mask = None
        self._pending_ctrl_hint = None

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

        if self.ctrl is not None:
            source = self.ctrl.get("hint_source")
            if source == "control_image":
                ctrl_hint = self._ctrl_image
            elif source == "segs_preprocessor":
                if self.preprocessor is None:
                    raise ValueError(t("err.ctrl_needs_preprocessor"))
                ctrl_hint = _run_preprocessor(self.preprocessor, image, mask)
            else:  # "same as lllite"
                ctrl_hint = hint
            self._pending_ctrl_hint = ctrl_hint

        # LLLite produces no cnet PIL images of its own.
        return positive, negative, prev_pils

    def doit_ipadapter(self, model):
        prev_pils = []
        if self.prev_wrapper is not None:
            model, prev_pils = self.prev_wrapper.doit_ipadapter(model)

        hint = self._pending_hint
        if hint is None:  # apply() was not called -> nothing to do
            return model, prev_pils

        # ① control-LoRA LoRA half (optional, block-remapped when asked)
        if self.ctrl is not None and float(self.ctrl.get("lora_strength", 0.0)) > 0.0:
            model = self._apply_ctrl_lora_deltas(model)

        # ② LLLite (unchanged behaviour, including cross-generation remap)
        apply = self.apply_fn or anima.apply_lllite
        patched = apply(
            model,
            self.lllite_name,
            hint,
            self.strength,
            self.start_percent,
            self.end_percent,
            True,
            self._pending_mask,
        )

        # ③ control-LoRA pose embedder
        if self.ctrl is not None and float(self.ctrl.get("strength", 0.0)) != 0.0:
            patched = self._attach_control(patched)

        # Keep _pending_hint: some detailers call doit_ipadapter more than once
        # per segment. Re-patching a fresh clone each time is idempotent in
        # effect, because `model` is the pristine one on every call.
        return patched, prev_pils

    # -- control-LoRA internals --------------------------------------------- #
    def _apply_ctrl_lora_deltas(self, model):
        return apply_ctrl_lora_deltas(model, self.ctrl)

    def _attach_control(self, model):
        return attach_control_tokens(model, self.ctrl, self._pending_ctrl_hint)


def _vae_encode(vae, image):
    """Backwards-compatible alias for ``control_embedder.encode_image``."""
    return cembedder.encode_image(vae, image)


def apply_ctrl_lora_deltas(model, cfg):
    """Patch the model with the adapter's LoRA half (block-remapped if asked)."""
    sd = clora.load_adapter(cfg["path"])
    mapping = None
    if cfg.get("remap"):
        source_blocks = clora.block_count_of_keys(sd.keys())
        _, _, mapping = resolve_block_mapping(
            model,
            cfg.get("target_choice"),
            cfg.get("source_choice"),
            source_blocks,
        )
    patched, applied, candidates = clora.apply_lora_deltas(
        model, sd, float(cfg["lora_strength"]), mapping
    )
    if applied == 0:
        raise RuntimeError(t("err.ctrl_lora_no_match", n=candidates))
    return patched


def attach_control_tokens(model, cfg, hint):
    """Add ``strength * control_embedder(vae(hint))`` to the patch-embed output."""
    vae = cfg.get("vae")
    if vae is None:
        raise ValueError(t("err.ctrl_needs_vae"))
    if hint is None:
        return model

    weight, bias = cembedder.load_embedder(cfg["path"])
    try:
        cembedder.check_channels(model, weight)
    except RuntimeError as e:
        raise ValueError(t("err.ctrl_dim_mismatch", w=weight.shape[0],
                           m=int(getattr(cembedder.dit_of(model), "model_channels", 0)))) from e

    dit = cembedder.dit_of(model)
    spatial, temporal = cembedder._patch_sizes(dit)
    latent = cembedder.encode_image(vae, hint)
    provider = cembedder.token_provider(
        latent, weight, bias, spatial=spatial, temporal=temporal,
        strength=float(cfg.get("strength", 1.0)),
    )
    return cembedder.attach(model, provider)


def resolve_ctrl_hint(cfg, image, mask, ctrl_image, preprocessor, fallback):
    """Pick the skeleton image for the control-LoRA backend."""
    source = cfg.get("hint_source")
    if source == "control_image":
        if ctrl_image is None:
            raise ValueError(t("err.ctrl_control_image"))
        return ctrl_image
    if source == "segs_preprocessor":
        if preprocessor is None:
            raise ValueError(t("err.ctrl_needs_preprocessor"))
        return _run_preprocessor(preprocessor, image, mask)
    return fallback


def apply_control_lora(model, cfg, hint):
    """LoRA half (optional) + pose embedder, in that order."""
    if float(cfg.get("lora_strength", 0.0)) > 0.0:
        model = apply_ctrl_lora_deltas(model, cfg)
    if float(cfg.get("strength", 0.0)) != 0.0 and hint is not None:
        model = attach_control_tokens(model, cfg, hint)
    return model


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


def _lora_names():
    try:
        import folder_paths

        names = list(folder_paths.get_filename_list("loras"))
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


def _ctrl_lora_inputs():
    """INPUT_TYPES fragment shared by both apply nodes (appended last)."""
    return {
        "enable_control_lora": (
            "BOOLEAN",
            {"default": False, "tooltip": t("tip.enable_control_lora")},
        ),
        "control_lora_name": (_lora_names(), {"tooltip": t("tip.control_lora_name")}),
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
    }


def build_control_lora_config(
    enable,
    control_lora_name,
    control_lora_strength,
    control_lora_lora_strength,
    control_lora_hint,
    vae,
    *,
    remap=False,
    target_choice=None,
    source_choice=None,
):
    """Validate the switch and build the per-wrapper config (or None)."""
    if not enable:
        return None

    if vae is None:
        raise ValueError(t("err.ctrl_needs_vae"))
    if control_lora_name in (None, "", "None"):
        raise ValueError(t("err.ctrl_bad_file", name=control_lora_name))

    path = clora.resolve_lora_path(control_lora_name)
    if path is None:
        raise ValueError(t("err.ctrl_bad_file", name=control_lora_name))
    if not cembedder.has_embedder(path):
        raise ValueError(t("err.ctrl_no_embedder", name=control_lora_name))

    if control_lora_hint == "segs_preprocessor":
        pass  # checked per-wrapper (needs the preprocessor object)
    elif control_lora_hint == "control_image":
        pass  # checked per-wrapper

    if control_lora_lora_strength > 0.0:
        print(f"{LOG} control-LoRA: {control_lora_name} ({clora.describe_file(path)})")

    return {
        "path": path,
        "name": control_lora_name,
        "strength": float(control_lora_strength),
        "lora_strength": float(control_lora_lora_strength),
        "hint_source": control_lora_hint,
        "vae": vae,
        "remap": bool(remap),
        "target_choice": target_choice,
        "source_choice": source_choice,
    }


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
class LLLiteApplySEGS:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
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
        }
        # Appended AFTER the published inputs: ComfyUI stores widget values by
        # position, so inserting anything earlier would shift saved workflows.
        required.update(_ctrl_lora_inputs())
        return {
            "required": required,
            "optional": {
                "segs_preprocessor": (
                    "SEGS_PREPROCESSOR",
                    {"tooltip": t("tip.seg_processor")},
                ),
                "control_image": ("IMAGE",),
                "vae": ("VAE", {"tooltip": t("tip.ctrl_vae")}),
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
        enable_control_lora=False,
        control_lora_name="None",
        control_lora_strength=1.2,
        control_lora_lora_strength=1.0,
        control_lora_hint="segs_preprocessor",
        segs_preprocessor=None,
        control_image=None,
        vae=None,
    ):
        if hint_mode == "pose" and segs_preprocessor is None:
            raise ValueError(t("err.pose_needs_preprocessor"))
        if hint_mode == "control_image" and control_image is None:
            raise ValueError(t("err.control_needs_image"))
        if enable_control_lora and control_lora_hint == "segs_preprocessor" \
                and segs_preprocessor is None:
            raise ValueError(t("err.ctrl_needs_preprocessor"))
        if enable_control_lora and control_lora_hint == "control_image" \
                and control_image is None:
            raise ValueError(t("err.ctrl_control_image"))
        if lllite_name in (None, "", "None"):
            raise ValueError(t("err.bad_lllite", name=lllite_name))

        # Fail early (and clearly) if Anima-LLLite is missing or the file is bad.
        if anima.resolve_weights_path(lllite_name) is None:
            raise ValueError(t("err.bad_lllite", name=lllite_name))
        if not anima.is_lllite_file(lllite_name):
            raise ValueError(t("err.not_lllite", name=lllite_name))
        anima.load_module()  # raises with a readable message when absent

        ctrl = build_control_lora_config(
            enable_control_lora,
            control_lora_name,
            control_lora_strength,
            control_lora_lora_strength,
            control_lora_hint,
            vae,
        )

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
                control_lora=ctrl,
            )

        out = impact.apply_wrapper_to_segs(segs, factory)
        print(
            f"{LOG} LLLite(SEGS): {lllite_name} -> {impact.segs_count(out)} seg(s), "
            f"mode={hint_mode}, 4ch_mask={need_mask}"
            + (f", +control_lora={control_lora_name}" if ctrl else "")
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
