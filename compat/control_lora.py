"""Control-LoRA support: the LoRA-delta half (plus block-index remapping).

The same adapter file that carries ``control_embedder.proj.*`` also carries a
normal rank-16 LoRA (``diffusion_model.blocks.<n>.*.lora_A/lora_B`` and
``llm_adapter.*``). Upstream applies that half with ``LoraLoaderModelOnly``; we
fuse it here so one switch does both.

On Anima 2.9B / 3.8B the block indices have to be rewritten first -- the LoRA was
trained on the 28-layer layout, and layers are spliced into the middle, so block
``n`` is not block ``n`` any more. That remap reuses the same tables as the
LLLite remap (``anima_arch``).
"""

from __future__ import annotations

import os
import re

import torch

from ..anima_arch import describe_count
from . import control_embedder

LOG = "[SegLLLite]"

#: ``diffusion_model.blocks.<n>.`` / ``blocks.<n>.`` (peft and kohya styles)
_BLOCK_DOT = re.compile(r"(\.blocks\.)(\d+)(\.)")
#: ``lora_unet_blocks_<n>_`` (kohya/A1111 flattened style)
_BLOCK_FLAT = re.compile(r"(_blocks_)(\d+)(_)")
#: kept untouched: text-encoder side and non-block adapters
_EXCLUDED = ("llm_adapter", "input_blocks", "output_blocks", "lora_te")


def lora_part(sd: dict) -> dict:
    """Drop the control-embedder tensors, keep the LoRA ones."""
    return {k: v for k, v in sd.items() if control_embedder.EMBEDDER_MARK not in k}


def block_index_of(key: str):
    if any(x in key for x in _EXCLUDED):
        return None
    m = _BLOCK_DOT.search(key) or _BLOCK_FLAT.search(key)
    return int(m.group(2)) if m else None


def block_count_of_keys(keys) -> "int | None":
    idx = [i for i in (block_index_of(k) for k in keys) if i is not None]
    return (max(idx) + 1) if idx else None


def remap_key(key: str, new_index: int):
    if any(x in key for x in _EXCLUDED):
        return None
    for pattern in (_BLOCK_DOT, _BLOCK_FLAT):
        if pattern.search(key):
            return pattern.sub(lambda m: f"{m.group(1)}{int(new_index)}{m.group(3)}", key, count=1)
    return None


def remap_lora_state_dict(sd: dict, mapping: dict) -> tuple:
    """Rewrite block indices; returns ``(new_sd, moved_key_count)``."""
    out = {}
    moved = 0
    for key, value in sd.items():
        idx = block_index_of(key)
        if idx is not None and idx in mapping:
            new_key = remap_key(key, mapping[idx])
            if new_key and new_key != key:
                moved += 1
                out[new_key] = value
                continue
        out[key] = value
    return out, moved


#: (path, mtime, size) -> state dict
_ADAPTER_CACHE: dict = {}


def load_adapter(path: str) -> dict:
    """Full adapter state dict (LoRA half + control embedder), cached per file.

    Per-SEG detailers call this once per segment; without the cache a 6-segment
    run would re-read the same 70 MB file six times.
    """
    stamp = (path, os.path.getmtime(path), os.path.getsize(path))
    sd = _ADAPTER_CACHE.get(stamp)
    if sd is None:
        sd = load_lora_file(path)
        _ADAPTER_CACHE[stamp] = sd
    return sd


def load_lora_file(path: str) -> dict:
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu")


def resolve_lora_path(name: str):
    """Resolve a combo value against ``models/loras`` (bare name or subpath)."""
    try:
        import folder_paths

        full = folder_paths.get_full_path("loras", name)
        if full:
            return full
        # be forgiving: some users keep the adapter next to their other ControlNets
        full = folder_paths.get_full_path("controlnet", name)
        if full:
            return full
    except Exception:
        pass
    import os

    return name if os.path.isfile(name) else None


#: (path, mtime, size) -> bool  ("does this file carry control_embedder weights?")
_HEADER_CACHE: dict = {}


def is_control_lora_file(path: str) -> bool:
    """Cheap header probe: does the file contain ``control_embedder.*`` weights?

    ``safetensors.safe_open`` only reads the header, so forty LoRAs cost ~a second
    once, and the result is cached until the file changes. Non-safetensors files
    (rare for adapters) are accepted optimistically -- they cannot be probed
    cheaply and the runtime check still rejects them with a clear message.
    """
    if not path.lower().endswith(".safetensors"):
        return True
    try:
        stamp = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return False
    cached = _HEADER_CACHE.get(stamp)
    if cached is not None:
        return cached
    try:
        keys = control_embedder.read_lora_keys(path)
        result = any(control_embedder.EMBEDDER_MARK in k for k in keys)
    except Exception as e:
        print(f"{LOG} header probe failed for {os.path.basename(path)}: {e}")
        result = True  # do not hide a file because probing failed
    _HEADER_CACHE[stamp] = result
    return result


def list_control_lora_files():
    """``models/loras`` entries that are actually control-LoRAs.

    Returns ``None`` when the list cannot be produced (missing folder_paths,
    unreadable folder), so callers can fall back to every LoRA instead of showing
    an empty dropdown.
    """
    try:
        import folder_paths

        names = list(folder_paths.get_filename_list("loras"))
    except Exception:
        return None
    if not names:
        return []

    kept = []
    for name in names:
        try:
            full = folder_paths.get_full_path("loras", name) or name
        except Exception:
            full = name
        if is_control_lora_file(full):
            kept.append(name)
    return kept


def apply_lora_deltas(model, sd: dict, strength: float, mapping: dict | None = None):
    """Patch the model with the adapter's LoRA half.

    Returns ``(patched_model, applied, candidates)``. ``applied == 0`` is an
    error: silently sampling without the delta looks like "the control did
    nothing", which is exactly the failure mode we want to make loud.
    """
    import comfy.lora
    import comfy.lora_convert

    lora_sd = lora_part(sd)
    candidates = len(lora_sd)
    if not candidates:
        return model, 0, 0

    if mapping is not None:
        lora_sd, moved = remap_lora_state_dict(lora_sd, mapping)
        if moved:
            print(f"{LOG} control-LoRA: remapped {moved} block-indexed keys")

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    converted = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(converted, key_map)
    if not loaded:
        raise RuntimeError(
            f"{LOG} control-LoRA: none of the {candidates} LoRA keys matched this model. "
            "Pass the adapter through the Remap node with the right target_model, "
            "or set control_lora_lora_strength to 0 to use only the pose embedder."
        )

    patched = model.clone()
    applied = patched.add_patches(loaded, strength)
    count = len(applied) if applied is not None else len(loaded)
    print(f"{LOG} control-LoRA: LoRA half applied to {count}/{candidates} keys (strength={strength:g})")
    return patched, count, candidates


def describe_file(path: str) -> str:
    """Short human summary of a control-LoRA file (for logs / errors)."""
    try:
        keys = control_embedder.read_lora_keys(path)
        has_emb = any(control_embedder.EMBEDDER_MARK in k for k in keys)
        n_blocks = block_count_of_keys(keys)
        return (
            f"{len(keys)} keys, blocks={describe_count(n_blocks)}, "
            f"control_embedder={'yes' if has_emb else 'no'}"
        )
    except Exception as e:  # pragma: no cover - diagnostics only
        return f"<unreadable: {e}>"
