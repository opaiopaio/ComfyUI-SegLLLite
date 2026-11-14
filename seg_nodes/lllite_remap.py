"""LLLite Apply (SEGS) with cross-generation **block remapping**.

Why this exists
---------------
ControlNet-LLLite is, structurally, a LoRA family: per-block modules
(``lllite_dit_blocks_<N>_<atom>.down/.mid/.up`` + a per-module ``depth_embed``)
plus one shared ``lllite_conditioning1.*`` network. The block index lives in the
weight key, exactly like ``lora_unet_blocks_N_``.

Anima ships in 28 / 40 / 52 block variants (base → 2.9B → 3.8B), each a depth
expansion of the previous one where new blocks are spliced into the middle — so
the same index means a *different* block. Anima-LLLite builds its modules from
the **target** model and then loads keys by name:

* files with ``depth_embed`` (the current format) fail loudly with
  ``RuntimeError: depth_embed slices missing for module idx(es) [...]``;
* files without it load silently onto the **wrong** blocks.

Either way a 28-block LLLite cannot be used on a 40-block model. This node
rewrites the block index in the weight keys before the load, the same way
``LoRA Stack Apply (Anima Remap)`` rewrites LoRA keys.

How the load is intercepted
---------------------------
``Anima-LLLite/nodes.py`` binds ``load_lllite_weights`` at import time, so we
swap that name **on their nodes module** for the duration of one ``apply`` call
and restore it in a ``finally`` block. Nothing on disk is modified, and other
users of the package are unaffected.
"""

from __future__ import annotations

import os
import re

from ..anima_arch import (
    AUTO_LABEL,
    MODEL_CHOICES,
    bucket_block_count,
    describe_count,
    get_mapping,
    inserted_sources,
    resolve_dropdown,
)
from ..i18n import t

from ..compat import anima_lllite as anima
from ..compat import impact

LOG = "[SegLLLite]"

# --------------------------------------------------------------------------- #
# Key handling
# --------------------------------------------------------------------------- #
#: per-module weight prefix, e.g. ``lllite_dit_blocks_12_self_attn_q_proj.down.weight``
_BLOCK_RE = re.compile(r"^(lllite_dit_blocks_)(\d+)(_)")
#: the shared conditioning network -- never remapped
_SHARED_PREFIX = "lllite_conditioning1."

#: output-projection tensors scaled by ``extend_strength`` for synthesized blocks
_SCALE_SUFFIXES = (".up.weight", ".up.bias")


def is_shared_key(key: str) -> bool:
    return key.startswith(_SHARED_PREFIX)


def block_index_of(key: str):
    """Block index encoded in a per-module key, or None for shared keys."""
    m = _BLOCK_RE.match(key)
    return int(m.group(2)) if m else None


def block_count_of_keys(keys) -> "int | None":
    """Raw (unbucketed) block count implied by the keys, or None."""
    idxs = [i for i in (block_index_of(k) for k in keys) if i is not None]
    return (max(idxs) + 1) if idxs else None


def rewrite_key(key: str, new_index: int):
    """Return ``key`` with its block index replaced, or None if not block-keyed."""
    m = _BLOCK_RE.match(key)
    if not m:
        return None
    return f"{m.group(1)}{int(new_index)}{m.group(3)}{key[m.end():]}"


def remap_state_dict(sd: dict, mapping: dict) -> dict:
    """Move every per-module key to its mapped block index.

    Keys without a block index (the shared conditioning network) are copied
    through untouched.
    """
    out = {}
    for key, value in sd.items():
        idx = block_index_of(key)
        if idx is None or idx not in mapping:
            out[key] = value
            continue
        out[rewrite_key(key, mapping[idx])] = value
    return out


def fill_missing_depth_embeds(sd: dict, mapping: dict) -> dict:
    """Give every newly inserted block a ``depth_embed`` slice.

    Anima-LLLite's ``_from_saved_state_dict`` refuses to load unless **every**
    module of the target model has a depth slice::

        RuntimeError: depth_embed slices missing for module idx(es) [...]

    A 28-block file remapped onto 40 blocks therefore cannot work in "keys only"
    mode -- without this the load always fails. We copy the slice from the block
    each new block was spliced after (``inserted_sources``), which is the same
    approximation Extend mode uses for the LoRA weights.
    """
    # NOTE: build from the ORIGINAL sd. Its block indices are in the *source*
    # numbering, which is what ``inserted_sources`` refers to -- feeding an
    # already-remapped dict here would look up the wrong blocks and leave holes.
    base = remap_state_dict(sd, mapping)
    inserted = inserted_sources(mapping, max(mapping.values()) + 1)

    by_source = {}
    for key, value in sd.items():
        if not key.endswith(".depth_embed"):
            continue
        idx = block_index_of(key)
        if idx is not None:
            by_source.setdefault(idx, []).append((key, value))

    for target_idx, source_idx in inserted.items():
        for key, value in by_source.get(source_idx, []):
            new_key = rewrite_key(key, target_idx)
            if new_key and new_key not in base:
                base[new_key] = value.clone() if hasattr(value, "clone") else value
    return base


def extend_state_dict(sd: dict, mapping: dict, extend_strength: float = 1.0) -> dict:
    """Also synthesize weights for the newly inserted blocks.

    Each inserted target block copies the weights of the block it was spliced
    after (``inserted_sources``). Those layers did not exist when the LLLite was
    trained, so this is a best-effort approximation; ``extend_strength`` scales
    the output projections (``.up.*``) to dial it down.
    """
    base = remap_state_dict(sd, mapping)
    inserted = inserted_sources(mapping, max(mapping.values()) + 1)
    stems = {}
    for key, value in sd.items():
        idx = block_index_of(key)
        if idx is not None:
            stems.setdefault(idx, []).append((key, value))

    for target_idx, source_idx in inserted.items():
        for key, value in stems.get(source_idx, []):
            new_key = rewrite_key(key, target_idx)
            if new_key is None or new_key in base:
                continue
            try:
                if extend_strength != 1.0 and any(key.endswith(s) for s in _SCALE_SUFFIXES):
                    value = value * float(extend_strength)
                base[new_key] = value.clone() if hasattr(value, "clone") else value
            except Exception:
                base[new_key] = value
    return base


# --------------------------------------------------------------------------- #
# Model / file probing
# --------------------------------------------------------------------------- #
def detect_model_block_count(model_patcher):
    """Block count of the connected MODEL (raw, not bucketed)."""
    if model_patcher is None:
        return None

    inner = getattr(model_patcher, "model", None)
    dm = getattr(inner, "diffusion_model", None)
    blocks = getattr(dm, "blocks", None)
    if blocks is not None:
        try:
            n = len(blocks)
            if n > 0:
                return n
        except Exception:
            pass

    for getter in (
        lambda: model_patcher.model_state_dict(),
        lambda: dm.state_dict() if dm is not None else None,
        lambda: inner.state_dict() if inner is not None else None,
    ):
        try:
            sd = getter()
        except Exception:
            continue
        if not sd or not isinstance(sd, dict):
            continue
        count = block_count_of_keys(sd.keys())
        if count:
            return count
    return None


def read_weight_keys(path: str):
    """Keys of a .safetensors / torch weight file without loading tensors."""
    if os.path.splitext(path)[1].lower() == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return list(f.keys())
    import torch

    sd = torch.load(path, map_location="cpu")
    return list(sd.keys())


def peek_lllite_block_count(lora_path: str):
    """Bucketed generation of a LLLite file (28 / 40 / 52), or None."""
    try:
        return bucket_block_count(block_count_of_keys(read_weight_keys(lora_path)))
    except Exception as e:
        print(f"{LOG} could not read LLLite keys: {e}")
        return None


def is_lllite_file(lllite_name: str) -> bool:
    """True when the file carries ControlNet-LLLite metadata (v2 name format)."""
    return anima.is_lllite_file(lllite_name)


# --------------------------------------------------------------------------- #
# Remapped apply
# --------------------------------------------------------------------------- #
def _control_module():
    """Anima-LLLite's ``control_net_lllite_anima`` module (loader + converter)."""
    import importlib

    nodes_mod = anima.load_module()
    pkg = getattr(nodes_mod, "__package__", None)
    if not pkg:
        raise RuntimeError("cannot resolve the Anima-LLLite package name")
    return importlib.import_module(pkg + ".control_net_lllite_anima")


def _load_weights_file(file: str) -> dict:
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        return load_file(file)
    import torch

    return torch.load(file, map_location="cpu")


def _make_loader(ctrl, mapping, extend: bool, extend_strength: float):
    """Build a ``load_lllite_weights`` replacement applying a key remap."""

    def load_lllite_weights(lllite, file, strict: bool = False):
        sd = _load_weights_file(file)
        if mapping is not None:
            before = block_count_of_keys(sd.keys())
            if extend:
                sd = extend_state_dict(sd, mapping, extend_strength)
            else:
                # depth_embed must exist for EVERY target module, so even
                # "keys only" has to synthesize the slices of inserted blocks.
                sd = fill_missing_depth_embeds(sd, mapping)
            after = block_count_of_keys(sd.keys())
            print(
                f"{LOG} LLLite remap: {before} -> {after} blocks"
                + (" (extend mode)" if extend else " (keys only)")
            )

        converted = ctrl._from_saved_state_dict(lllite, sd)
        info = lllite.load_state_dict(converted, strict=strict)
        missing = len(getattr(info, "missing_keys", []) or [])
        unexpected = len(getattr(info, "unexpected_keys", []) or [])
        print(f"{LOG} LLLite weights loaded: missing={missing}, unexpected={unexpected}")
        if missing or unexpected:
            print(
                f"{LOG} warning: the LLLite file does not fully match this model "
                f"(missing={missing}, unexpected={unexpected}). "
                "Check target_model / source_model."
            )
        return info

    return load_lllite_weights


def apply_lllite_remapped(
    model,
    lllite_name,
    image,
    strength,
    start_percent,
    end_percent,
    preserve_wrapper=True,
    mask=None,
    target_choice=AUTO_LABEL,
    source_choice=AUTO_LABEL,
    remap_mode="Keys only (recommended)",
    extend_strength=1.0,
):
    """Patch ``model`` with an LLLite, remapping block indices when needed."""
    from .lllite_segs import resolve_block_mapping

    # One resolver shared by every caller (this node, both apply nodes, the
    # control-LoRA half): the file's own block count wins over a stale dropdown.
    path = anima.resolve_weights_path(lllite_name)
    file_blocks = peek_lllite_block_count(path) if path else None
    source_blocks, target_blocks, mapping = resolve_block_mapping(
        model, target_choice, source_choice, file_blocks
    )
    print(
        f"{LOG} LLLite: file={describe_count(source_blocks)}, "
        f"model={describe_count(target_blocks)}, mode={remap_mode}"
        + ("" if mapping else ", no remap needed")
    )

    nodes_mod = anima.load_module()
    ctrl = _control_module()
    original = nodes_mod.load_lllite_weights
    nodes_mod.load_lllite_weights = _make_loader(
        ctrl,
        mapping,
        extend=remap_mode.startswith("Extend"),
        extend_strength=float(extend_strength),
    )
    try:
        return anima.apply_lllite(
            model, lllite_name, image, strength, start_percent, end_percent,
            preserve_wrapper, mask,
        )
    finally:
        nodes_mod.load_lllite_weights = original


def make_apply_fn(lllite_name, target_choice, source_choice, remap_mode, extend_strength):
    """Closure matching ``anima.apply_lllite``'s signature, for the SEGS wrapper."""

    def _apply(model, name, image, strength, sp, ep, preserve, mask):
        return apply_lllite_remapped(
            model, name, image, strength, sp, ep, preserve, mask,
            target_choice=target_choice,
            source_choice=source_choice,
            remap_mode=remap_mode,
            extend_strength=extend_strength,
        )

    return _apply


REMAP_MODES = ["Keys only (recommended)", "Extend to new layers (experimental)"]


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
class LLLiteApplySEGSRemap:
    """Same as LLLite Apply (SEGS) plus cross-generation block remapping."""

    @classmethod
    def INPUT_TYPES(cls):
        from .lllite_segs import HINT_MODES, _controlnet_names, _ctrl_lora_inputs

        required = {
                "segs": ("SEGS",),
                "lllite_name": (_controlnet_names(),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hint_mode": (HINT_MODES, {"default": "image", "tooltip": t("tip.hint_mode")}),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
                # --- remap options (mirrors LoRA Stack Apply (Anima Remap)) ---
                "enable_remap": ("BOOLEAN", {"default": True}),
                "target_model": (
                    MODEL_CHOICES,
                    {"default": AUTO_LABEL, "tooltip": t("tip.lllite_target")},
                ),
                "source_model": (
                    MODEL_CHOICES,
                    {"default": AUTO_LABEL, "tooltip": t("tip.lllite_source")},
                ),
                "remap_mode": (REMAP_MODES, {"default": REMAP_MODES[0]}),
                "extend_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                     "tooltip": t("tip.lllite_extend")},
                ),
        }
        # Appended after every published input so saved workflows keep mapping.
        required.update(_ctrl_lora_inputs())
        return {
            "required": required,
            "optional": {
                "segs_preprocessor": ("SEGS_PREPROCESSOR", {"tooltip": t("tip.seg_processor")}),
                "control_image": ("IMAGE",),
                "vae": ("VAE", {"tooltip": t("tip.ctrl_vae")}),
            },
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "run"
    CATEGORY = t("category")
    DESCRIPTION = t("desc.lllite_apply_remap")

    def run(self, segs, lllite_name, strength, start_percent, end_percent, hint_mode,
            preserve_wrapper, enable_remap, target_model, source_model, remap_mode,
            extend_strength, enable_control_lora=False, control_lora_name="None",
            control_lora_strength=1.2, control_lora_lora_strength=1.0,
            control_lora_hint="segs_preprocessor",
            control_lora_start_percent=0.0, control_lora_end_percent=1.0,
            segs_preprocessor=None, control_image=None, vae=None):
        from .lllite_segs import LLLiteSEGSWrapper, build_control_lora_config

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

        anima.load_module()  # first: reading the metadata needs the package
        path = anima.resolve_weights_path(lllite_name)
        if path is None:
            raise ValueError(t("err.bad_lllite", name=lllite_name))
        if not is_lllite_file(lllite_name):
            raise ValueError(t("err.not_lllite", name=lllite_name))

        # Fail early when the direction is unsupported (older -> newer only).
        src = resolve_dropdown(source_model) or peek_lllite_block_count(path)
        tgt = resolve_dropdown(target_model)
        if enable_remap and src and tgt and src > tgt:
            raise ValueError(
                t("err.remap_direction",
                  src=describe_count(src), tgt=describe_count(tgt))
            )

        need_mask = anima.cond_in_channels(lllite_name) == 4
        orig = impact.original_size(segs)
        apply_fn = (
            make_apply_fn(lllite_name, target_model, source_model, remap_mode, extend_strength)
            if enable_remap
            else None
        )
        ctrl = build_control_lora_config(
            enable_control_lora,
            control_lora_name,
            control_lora_strength,
            control_lora_lora_strength,
            control_lora_hint,
            vae,
            start_percent=control_lora_start_percent,
            end_percent=control_lora_end_percent,
            remap=enable_remap,
            extend=remap_mode.startswith("Extend"),
            extend_strength=extend_strength,
            target_choice=target_model,
            source_choice=source_model,
        )

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
                apply_fn=apply_fn,
                control_lora=ctrl,
            )

        out = impact.apply_wrapper_to_segs(segs, factory)
        print(
            f"{LOG} LLLite(SEGS) remap={enable_remap} {lllite_name} -> "
            f"{impact.segs_count(out)} seg(s), mode={hint_mode}"
        )
        return (out,)


NODE_CLASS_MAPPINGS = {"LLLiteApplySEGSRemap": LLLiteApplySEGSRemap}
NODE_DISPLAY_NAME_MAPPINGS = {"LLLiteApplySEGSRemap": t("node.lllite_apply_remap")}
