"""Restore BERT helpers that ``transformers>=5`` removed.

transformers 5 dropped ``get_extended_attention_mask`` / ``invert_attention_mask``
/ ``get_head_mask`` from ``PreTrainedModel`` (previously provided by
``ModuleUtilsMixin``). Vendored model code — notably GroundingDINO as shipped in
``comfyui-sam2`` — still binds those attributes and dies with::

    AttributeError: 'BertModel' object has no attribute 'get_extended_attention_mask'

We add them back **at runtime** (no file is modified, no global version pin):

  - If the methods already exist (transformers<5) we do nothing.
  - If they are missing we attach mathematically equivalent implementations to
    ``PreTrainedModel``, so every subclass inherits them.

This is a *compatibility* shim: it only restores behaviour that used to exist.
"""

import torch

_LOG = "[LLLiteSEGS]"
_STATUS = None  # None = not checked yet


def _get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for attention_mask (shape {tuple(attention_mask.shape)})")
    extended = extended.to(dtype=dtype or torch.float32)
    return (1.0 - extended) * torch.finfo(extended.dtype).min


def _invert_attention_mask(self, encoder_attention_mask):
    if encoder_attention_mask.dim() == 3:
        extended = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        extended = encoder_attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for encoder_attention_mask (shape {tuple(encoder_attention_mask.shape)})"
        )
    extended = extended.to(dtype=torch.float32)
    return (1.0 - extended) * torch.finfo(extended.dtype).min


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"head_mask must be 1D or 2D, got {head_mask.dim()}D")
    return head_mask


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


_SHIM = {
    "get_extended_attention_mask": _get_extended_attention_mask,
    "invert_attention_mask": _invert_attention_mask,
    "get_head_mask": _get_head_mask,
}


def ensure_legacy_attention_helpers(force: bool = False) -> str:
    """Attach the missing helpers; returns a short status string.

    ``"already-present"`` | ``"patched"`` | ``"unavailable: <reason>"``
    """
    global _STATUS
    if _STATUS is not None and not force:
        return _STATUS

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as e:  # transformers not installed at all
        _STATUS = f"unavailable: {e}"
        return _STATUS

    added = []
    for name, fn in _SHIM.items():
        if not hasattr(PreTrainedModel, name):
            setattr(PreTrainedModel, name, fn)
            added.append(name)

    if added:
        _STATUS = "patched"
        print(f"{_LOG} transformers compatibility shim applied: {', '.join(added)}")
    else:
        _STATUS = "already-present"
    return _STATUS


def transformers_major() -> int:
    try:
        import transformers

        return int(str(transformers.__version__).split(".")[0])
    except Exception:
        return -1


__all__ = ["ensure_legacy_attention_helpers", "transformers_major"]
