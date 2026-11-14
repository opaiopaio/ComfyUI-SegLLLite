"""ComfyUI-SegLLLite — defensive node registration.

Hard rules (from the brand spec):
  * A missing optional dependency must NEVER break ComfyUI startup: each node
    module is imported in its own try/except and simply skipped on failure.
  * Node keys are a permanent contract (they live inside workflow JSON).

Core has no hard Python dependencies; ``transformers`` / ``torch`` / Impact Pack
/ Anima-LLLite / comfyui-sam2 are all optional and imported lazily.
"""

# NOTE: no sys.path manipulation and no absolute intra-package imports here on
# purpose. ComfyUI already has top-level modules named `nodes`, `compat`, `i18n`
# (its own nodes.py!) in sys.modules, so only relative imports are safe.
from .i18n import LANG, language_name, t  # noqa: E402

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ---- LLLite (SEGS) nodes: need torch only, which ships with ComfyUI -------- #
try:
    from .seg_nodes.lllite_segs import (
        NODE_CLASS_MAPPINGS as _A,
        NODE_DISPLAY_NAME_MAPPINGS as _AD,
    )

    NODE_CLASS_MAPPINGS.update(_A)
    NODE_DISPLAY_NAME_MAPPINGS.update(_AD)
except Exception as e:  # pragma: no cover
    print(f"[SegLLLite] LLLite SEGS nodes disabled: {e}")

# ---- LLLite remap node (needs anima_arch tables + Anima-LLLite at runtime) -- #
try:
    from .seg_nodes.lllite_remap import (
        NODE_CLASS_MAPPINGS as _R,
        NODE_DISPLAY_NAME_MAPPINGS as _RD,
    )

    NODE_CLASS_MAPPINGS.update(_R)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RD)
except Exception as e:  # pragma: no cover
    print(f"[SegLLLite] LLLite remap node disabled: {e}")

# ---- SAM2 + GroundingDINO: only if comfyui-sam2 is installed -------------- #
try:
    from .seg_nodes import sam2_dino as _sd

    if _sd.probe() is not None:
        NODE_CLASS_MAPPINGS.update(_sd.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_sd.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:  # pragma: no cover
    print(f"[SegLLLite] SAM2/DINO nodes disabled: {e}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(
    "[SegLLLite] loaded nodes: "
    + (", ".join(sorted(NODE_CLASS_MAPPINGS)) or "(none — check optional dependencies)")
    + f"   (language: {language_name()} [{LANG}])"
)
