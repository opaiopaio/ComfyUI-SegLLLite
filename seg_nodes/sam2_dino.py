"""SAM2 + GroundingDINO text-prompted segmentation.

The goal is *not* to re-implement these models. It is to expose them through a
node that keeps working when only the **wrapper** breaks:

  - ``transformers>=5`` removed ``get_extended_attention_mask`` and friends, which
    the vendored GroundingDINO in ``comfyui-sam2`` binds at build time. We apply
    ``compat.transformers_shim`` first, restoring the removed helpers at runtime
    (no file patched, no global version pin).
  - ``comfyui-sam2`` itself is imported **by file path** and only when this node
    actually runs, so a missing/broken pack disables just this node.

If ``comfyui-sam2`` is not installed the node is not registered at all (see
package ``__init__.py``); it never breaks ComfyUI startup.
"""

import importlib.util
import os

import torch

from ..compat.transformers_shim import ensure_legacy_attention_helpers
from ..i18n import t

LOG = "[SegLLLite]"

_MODULE_NAME = "lllitesegs_sam2_impl"
_PACKAGE_DIRS = ("comfyui-sam2", "ComfyUI-SAM2", "comfyui_sam2")
_module = None
_dino_cache = {}
_sam_cache = {}


def custom_nodes_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_package_dir():
    base = custom_nodes_dir()
    for name in _PACKAGE_DIRS:
        path = os.path.join(base, name, "node.py")
        if os.path.isfile(path):
            return path
    return None


def get_impl(force: bool = False):
    """Import comfyui-sam2's ``node.py`` (read-only) with the compat shim applied."""
    global _module
    if _module is not None and not force:
        return _module

    path = find_package_dir()
    if path is None:
        raise RuntimeError(
            "comfyui-sam2 is not installed. Install it to use SAM2 + GroundingDINO "
            "segmentation (this package does not bundle those models)."
        )

    # Must happen before the vendored model is *built*; harmless if already set.
    ensure_legacy_attention_helpers()

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _module = mod
    return mod


def probe():
    """Return the impl module or None — used for defensive registration."""
    try:
        return get_impl()
    except Exception as e:
        print(f"{LOG} SAM2/DINO node disabled: {e}")
        return None


def _dino_names(mod):
    try:
        return list(mod.list_groundingdino_model())
    except Exception:
        return []


def _sam_names(mod):
    try:
        return list(mod.list_sam_model())
    except Exception:
        return []


def _get_dino(mod, name):
    if name not in _dino_cache:
        print(f"{LOG} loading GroundingDINO: {name}")
        _dino_cache[name] = mod.load_groundingdino_model(name)
    return _dino_cache[name]


def _get_sam(mod, name):
    if name not in _sam_cache:
        print(f"{LOG} loading SAM2: {name}")
        _sam_cache[name] = mod.load_sam_model(name)
    return _sam_cache[name]


# --------------------------------------------------------------------------- #
# conversions
# --------------------------------------------------------------------------- #
def _batch_to_pil(image):
    """ComfyUI IMAGE (B,H,W,C in 0..1) -> first frame as PIL.

    Returns **RGBA**: comfyui-sam2's ``create_tensor_output`` paints the outside
    of each mask with a 4-value RGBA colour (``np.array([0,0,0,0])``) and derives
    the mask from the alpha channel, so a 3-channel RGB array crashes there with
    "value array of shape (4,) could not be broadcast to indexing result of shape
    (N,3)". GroundingDINO converts back to RGB internally, so RGBA is safe for it
    too.
    """
    from PIL import Image

    if image.dim() == 4:
        image = image[0]
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0).round().astype("uint8")
    if arr.shape[-1] == 4:  # IMAGE is normally RGB; be tolerant of RGBA input
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr[:, :, :3], "RGB").convert("RGBA")


def _mask_batch(masks):
    """Union a list of (1,H,W) masks into a single (1,H,W) tensor."""
    if not masks:
        return None
    stacked = torch.cat([m if m.dim() == 3 else m.unsqueeze(0) for m in masks], dim=0)
    return stacked.amax(dim=0, keepdim=True).to(torch.float32)


class Sam2DinoSegment:
    """Text-prompted segmentation: GroundingDINO boxes -> SAM2 masks."""

    @classmethod
    def INPUT_TYPES(cls):
        mod = probe()
        dino = _dino_names(mod) if mod else ["unavailable"]
        sam = _sam_names(mod) if mod else ["unavailable"]
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"default": "hand", "multiline": True},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "dino_model": (dino,),
                "sam_model": (sam,),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "run"
    CATEGORY = t("category")
    DESCRIPTION = t("desc.sam2dino")

    def run(self, image, prompt, threshold, dino_model, sam_model):
        mod = get_impl()
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")

        dino = _get_dino(mod, dino_model)
        sam = _get_sam(mod, sam_model)

        pil = _batch_to_pil(image)
        boxes = mod.groundingdino_predict(dino, pil, prompt, threshold)

        if boxes is None or len(boxes) == 0:
            print(f"{LOG} SAM2/DINO: no box matched '{prompt}' @ {threshold}")
            h, w = pil.size[1], pil.size[0]
            return (torch.zeros((1, h, w), dtype=torch.float32), image[:1])

        out_images, out_masks = mod.sam_segment(sam, pil, boxes)
        union = _mask_batch(out_masks)
        if union is None:
            h, w = pil.size[1], pil.size[0]
            union = torch.zeros((1, h, w), dtype=torch.float32)

        preview = torch.cat(out_images, dim=0) if out_images else image[:1]
        print(f"{LOG} SAM2/DINO: '{prompt}' -> {union.shape[0]} mask(s), {union.sum().item():.0f} px")
        return (union, preview)


NODE_CLASS_MAPPINGS = {"Sam2DinoSegment": Sam2DinoSegment}
NODE_DISPLAY_NAME_MAPPINGS = {"Sam2DinoSegment": t("node.sam2dino")}
