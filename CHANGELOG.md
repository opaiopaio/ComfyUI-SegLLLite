# Changelog

All notable changes to **ComfyUI-SegLLLite**. SemVer: new node = minor, bug fix = patch, node key change = major.

## [1.2.0] — 2026-11-14

### Changed (breaking for saved workflows)
- **`LLLite Apply (SEGS, Remap)` inputs regrouped**: LLLite block -> Control-LoRA block -> **remap block
  last**, because remapping describes the *parent DiT* and therefore governs both weight sets at once.
  ComfyUI stores widget values positionally, so after updating re-check these five values:
  `enable_remap`, `target_model`, `source_model`, `remap_mode`, `extend_strength`.

### Added
- The single remap block now covers the control-LoRA delta as well (Keys only / Extend).

### Hardened (review friendliness)
- No `os.environ` access, no `ctypes`, no `sys.path` mutation anywhere in the shipped package;
  language detection now reads an env var and the system locale only (`locale.getlocale()` handles
  the Windows `"Chinese (Simplified)_China"` form).
- `.pt`/`.ckpt` weights are loaded with `weights_only=True`, so a malicious adapter cannot execute code.

## [1.1.0] — 2026-11-14

### Added
- **Control-LoRA (pose) support inside both apply nodes.** New switch `enable_control_lora` plus
  `control_lora_name` / `control_lora_strength` (pose) / `control_lora_lora_strength` (LoRA half) /
  `control_lora_hint` (skeleton source) and a `vae` input. One SEG now gets LLLite redraw **and**
  control-LoRA pose control: LLLite owns the single `model_function_wrapper` slot while the
  control-LoRA appends a keyed `wrappers["diffusion_model"]` entry.
- **`LLLite Apply (SEGS, Remap)`** also remaps the control-LoRA block indices.
- **`Control-LoRA Apply (SEGS)`**: the same control-LoRA support as a standalone SEGS node (no LLLite),
  for "pose only" passes and for chaining after an LLLite node.
- **`Control-LoRA Apply (Anima)`**: `MODEL -> MODEL` version for plain txt2img / img2img. Applies the
  pose embedder (`strength`) and optionally the adapter LoRA half (`lora_strength`), with
  `enable_remap` / `target_model` / `source_model` so a 28-layer adapter works on 2.9B / 3.8B. Both
  strengths at 0 return the model untouched (no clone, no hook).
- `compat/control_embedder.py`: self-implemented control embedder (weight loading cached per file,
  ceil-correct 2x2 patch packing, token projection into `x_embedder` space, temporary forward hook).
- **Control-LoRA Extend**: the adapter's LoRA half can now be extended onto the 12 inserted blocks
  (`remap_mode = Extend to new layers`), so a 28-layer adapter drives all 40/52 layers instead of
  leaving a third of the model uncontrolled. `extend_strength` scales the copied output projections.- `compat/control_lora.py`: the adapter LoRA half (block-index remapping included), applied with a hard
  error when zero keys match instead of silently sampling without it.
- Bilingual strings for every new option.

### Changed
- The `control_lora_name` dropdown lists only files that actually contain `control_embedder.*`
  (header-only probe cached by mtime: ~5 ms for 40 LoRAs), falling back to the full LoRA list.
- New inputs are appended after every previously published input, so existing workflows keep their
  widget mapping.

### Fixed
- `crop_region` was read as `(x, y, w, h)`; Impact stores `(x1, y1, x2, y2)`, which shifted and
  stretched every `control_image` hint (and its preview).
- Every file is written without a UTF-8 BOM (the BOM broke the `pyproject.toml` TOML parse).
- The `x_embedder` output is asserted to be 5-D `(B, T, H, W, D)`; a clear error replaces a shape mishap.
- Validation order: a missing Anima-LLLite package is reported before "not an LLLite file".

## [1.0.0] — 2026-11-14

### Added
- `LLLiteApplySEGS` — attach a ControlNet-LLLite to every SEG of an Impact/LanPaint detailer pipeline, with per-segment hint images (`image` / `pose` / `control_image`), chainable via `preserve_wrapper`.
- `LLLiteClearSEGS` — drop LLLite wrappers from a SEGS chain (debug / reset).
- `LLLiteHintPreviewSEGS` — preview the hint image each SEG would receive.
- `Sam2DinoSegment` — SAM2 + GroundingDINO text-prompted segmentation that does not depend on APIs removed in transformers 5.
- Bilingual UI (system language, `OPNODES_LANG` override).

### Fixed
- Nodes did not appear in ComfyUI: the subpackage was called `nodes/`, and ComfyUI already
  has a top-level `nodes` module in `sys.modules`, so `from nodes.lllite_segs import ...`
  resolved to ComfyUI's own `nodes.py`. Renamed to `seg_nodes/` and switched every
  intra-package import to a **relative** import (also removed the `sys.path` injection).
- Added `tests/test_package_load.py`, which reproduces that environment (fake top-level
  `nodes` module) and asserts the package still registers its nodes.

- Fixed crash in the SAM2/DINO node: the PIL image handed to comfyui-sam2 must be
  **RGBA** (its `create_tensor_output` writes a 4-value colour and builds the mask from
  alpha), we passed RGB → `ValueError: shape mismatch: value array of shape (4,) could not
  be broadcast to indexing result of shape (N,3)`.

- Fixed `ImportError: attempted relative import with no known parent package` when loading
  Anima-LLLite: its `nodes.py` uses relative imports, so it must be imported **as a package**
  (reuse ComfyUI's already-imported module, else import by package name, else load
  `__init__.py` with `submodule_search_locations`). Loading `nodes.py` as a standalone file
  cannot work.

- Fixed `AttributeError: 'ModelPatcherDynamic' object has no attribute 'LanPaint_cfg_BIG'`:
  `ModelPatcher.clone()` (done by Anima-LLLite when patching) drops ad-hoc attributes that
  other nodes attach to the patcher. After patching we now restore any attribute the clone is
  missing, so LanPaint (and similar nodes) keep their state.

- **New:** `LLLite Apply (SEGS, Remap)` — rewrites LLLite block indices so a 28-layer LLLite can
  drive 40- (Anima 2.9B) / 52-layer (3.8B) models, with `Keys only` and `Extend` modes and manual
  `target_model` / `source_model` overrides.
- **Fixed:** a file without `lllite.*` metadata is now rejected with a clear error instead of
  silently loading zero weights (which looked like "the control had no effect").

### Notes
- MIT, zero GPL linkage (Impact Pack is duck-typed, never imported).
- No code is copied from the GPL projects.
