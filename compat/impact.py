"""Impact-Pack protocol helpers — **zero `import impact`**.

comfyui-impact-pack is GPL-3.0. Importing its modules would make this package a
derivative work. Instead we rely only on its *protocol* (method/attribute names,
which are not copyrightable) and on the SEGS data shape:

    SEGS = (original_size, [SEG, ...])          # original_size = (H, W, ...)
    SEG  = namedtuple with fields:
           cropped_image, cropped_mask, confidence, crop_region,
           bbox, label, control_net_wrapper

`collections.namedtuple` instances all support ``_replace``, so we can rewrite
``control_net_wrapper`` without knowing the concrete SEG class.

The two tensor helpers below are ~5 lines of generic tensor maths, written from
scratch (``torch.nn.functional.interpolate`` / plain slicing).
"""

import torch
import torch.nn.functional as F

_LOG = "[SegLLLite]"


# --------------------------------------------------------------------------- #
# SEGS shape helpers
# --------------------------------------------------------------------------- #
def is_segs(obj) -> bool:
    """True for an Impact-style SEGS tuple ``(original_size, [SEG, ...])``."""
    return (
        isinstance(obj, (tuple, list))
        and len(obj) == 2
        and isinstance(obj[1], (list, tuple))
    )


def segs_count(segs) -> int:
    if not is_segs(segs):
        return 0
    return len(segs[1])


def original_size(segs):
    return segs[0] if is_segs(segs) else None


def iter_segs(segs):
    """Yield ``(index, seg)``; tolerates a malformed / empty SEGS."""
    if not is_segs(segs):
        return
    for i, seg in enumerate(segs[1]):
        yield i, seg


def get_seg_field(seg, name, default=None):
    return getattr(seg, name, default)


def replace_seg(seg, **fields):
    """``namedtuple._replace`` — works for any Impact/newer SEG variant."""
    try:
        return seg._replace(**fields)
    except AttributeError:
        raise TypeError(
            f"SEG object of type {type(seg).__name__} has no _replace(); "
            "unexpected SEGS implementation"
        ) from None


def apply_wrapper_to_segs(segs, wrapper_factory):
    """Return a new SEGS with ``control_net_wrapper`` set per segment.

    ``wrapper_factory(seg, prev_wrapper) -> wrapper``. The incoming SEGS is not
    mutated (SEG is a namedtuple; ``_replace`` returns a copy).
    """
    orig, seg_list = segs[0], segs[1]
    out = []
    for seg in seg_list:
        prev = get_seg_field(seg, "control_net_wrapper")
        out.append(replace_seg(seg, control_net_wrapper=wrapper_factory(seg, prev)))
    return (orig, out)


def clear_wrappers(segs):
    if not is_segs(segs):
        return segs
    return apply_wrapper_to_segs(segs, lambda seg, prev: None)


# --------------------------------------------------------------------------- #
# tensor helpers (IMAGE = B,H,W,C float tensor)
# --------------------------------------------------------------------------- #
def tensor_resize(t, w: int, h: int):
    """Bilinear resize of a B,H,W,C image tensor to width ``w`` / height ``h``."""
    if t is None:
        return None
    if t.dim() == 3:
        t = t.unsqueeze(0)
    b, ch, th, tw = t.shape[0], t.shape[-1], t.shape[1], t.shape[2]
    if (th, tw) == (h, w):
        return t
    x = t.permute(0, 3, 1, 2)  # B,C,H,W
    x = F.interpolate(x, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)  # B,H,W,C


def tensor_crop(t, region):
    """Crop a B,H,W,C tensor by Impact's ``crop_region = (x1, y1, x2, y2)``.

    Impact stores **absolute corners**, not ``(x, y, w, h)``: ``core.crop_ndarray4``
    slices ``npimg[:, y1:y2, x1:x2, :]``. Reading it as width/height silently
    shifts and stretches every per-segment hint, so the convention is spelled out
    here and pinned by a regression test.
    """
    if t is None or region is None:
        return t
    if t.dim() == 3:
        t = t.unsqueeze(0)
    x1, y1, x2, y2 = (int(v) for v in region)
    H, W = t.shape[1], t.shape[2]
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return t[:, y1:y2, x1:x2, :]


def mask_crop(mask, region):
    """Crop a B,H,W (or H,W) mask by Impact's ``crop_region = (x1, y1, x2, y2)``."""
    if mask is None or region is None:
        return mask
    squeeze = mask.dim() == 2
    if squeeze:
        mask = mask.unsqueeze(0)
    x1, y1, x2, y2 = (int(v) for v in region)
    H, W = mask.shape[1], mask.shape[2]
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    out = mask[:, y1:y2, x1:x2]
    return out.squeeze(0) if squeeze else out


__all__ = [
    "is_segs",
    "segs_count",
    "original_size",
    "iter_segs",
    "get_seg_field",
    "replace_seg",
    "apply_wrapper_to_segs",
    "clear_wrappers",
    "tensor_resize",
    "tensor_crop",
    "mask_crop",
]
