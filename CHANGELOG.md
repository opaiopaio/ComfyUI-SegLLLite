# Changelog

All notable changes to **ComfyUI-SegLLLite**. SemVer: new node = minor, bug fix = patch, node key change = major.

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

### Notes
- MIT, zero GPL linkage (Impact Pack is duck-typed, never imported).
- No code is copied from the GPL projects.
