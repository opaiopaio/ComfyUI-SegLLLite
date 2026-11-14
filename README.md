# ComfyUI-SegLLLite

> Part of the **OPNodes** node pack · MIT
> 🧩 **This package**: makes Impact/LanPaint **SEGS detailers compatible with ControlNet-LLLite** — every detected region is sampled with its own LLLite hint. Also ships text-prompted SAM2+GroundingDINO segmentation
> 🎯 **Sister package**: [ComfyUI-LoRAStackAutoTriggers](https://github.com/opaiopaio/ComfyUI-LoRAStackAutoTriggers) — LoRA stacking / auto trigger words / Anima cross-generation remap

**English** | [中文](README.zh-CN.md)

---

## What this is / what it is not

Impact Pack's detailer crops each detected region and samples it separately. ControlNet-LLLite
(used by Anima/DiT models) captures **one global image** and rescales it to the current latent —
so a 576x576 hand crop receives the whole picture squeezed into it. Hints and latents end up
geometrically mismatched; that is a data-flow problem, not a parameter problem.

This package adds the missing glue: a wrapper that plugs into Impact's `control_net_wrapper`
protocol — invoked **inside** the per-segment loop with the already-cropped image — and patches a
model with an LLLite whose hint *is* that crop.

It does **not** implement LLLite or SAM2 itself. No GPL code is linked: the Impact protocol is
duck-typed (method names only), never imported, which keeps this package MIT.

## Nodes

| Node | What it does | Needs |
|---|---|---|
| **LLLite Apply (SEGS)** | Attach a ControlNet-LLLite to every SEG; per-segment hint from `image` (the crop), `pose` (run a SEGS preprocessor on the crop), or `control_image` (crop from a global control image). Chainable. | torch (ships with ComfyUI) + Anima-LLLite |
| **LLLite Apply (SEGS, Remap)** | Same as above **plus cross-generation block remapping**: a 28-layer LLLite can drive a 40-layer (Anima 2.9B) or 52-layer (3.8B) model | torch + Anima-LLLite |
| **LLLite Clear (SEGS)** | Remove LLLite wrappers from a SEGS chain | torch |
| **LLLite Hints (SEGS)** | Preview the hint each SEG would receive | torch |
| **SAM2 + GroundingDINO Segment** | Text prompt → boxes (GroundingDINO) → masks (SAM2). Adds a runtime compatibility shim so it works on **transformers ≥ 5**. | `comfyui-sam2`, `transformers` |

A missing optional dependency disables only the affected nodes; ComfyUI always starts.

## Install

1. Copy this folder into `ComfyUI/custom_nodes/ComfyUI-SegLLLite`.
2. Restart ComfyUI.
3. Optional, per node: install `ComfyUI-Anima-LLLite` (LLLite nodes) and `comfyui-sam2` (segmentation node).

There is **no** `pip install` step: heavy packages are imported lazily.

## Quick start (per-region pose control)

```
LoadImage ─► [SAM2 + GroundingDINO Segment]  prompt "hand", threshold 0.3
          ─► MaskToSEGS                              (Impact Pack)
          ─► [LLLite Apply (SEGS)]                   hint_mode=pose,
             segs_preprocessor = DWPreprocessor Provider (SEGS)   ← Inspire
             lllite_name = your anima pose LLLite, strength 0.8
          ─► [LLLite Apply (SEGS)]                   hint_mode=image, tile LLLite  (chain)
          ─► DetailerForEachPipe / Detailer Lanpaint (SEGS/pipe)
          ─► SaveImage
```

Use **LLLite Hints (SEGS)** anywhere upstream of the detailer to verify each segment gets its own
hint — that is the whole point of the package.

## Anima cross-generation remap (LLLite)

A ControlNet-LLLite is structurally a LoRA family: per-block modules
(`lllite_dit_blocks_<N>_<atom>.down/.mid/.up` + a per-module `depth_embed`) plus one shared
`lllite_conditioning1.*` network. The block index is part of the weight key, so the same index
means a different block on a model with a different depth.

Anima ships as 28 → 40 → 52 block variants (depth expansions: new blocks spliced into the middle).
Loading a 28-block LLLite onto a 40-block model therefore either fails
(`depth_embed slices missing`) or, for files without `depth_embed`, silently lands on the wrong
blocks. **LLLite Apply (SEGS, Remap)** rewrites the block index in the weight keys before the load:

| Option | Meaning |
|---|---|
| `enable_remap` | Off = behave exactly like the plain node |
| `target_model` | `Auto (detect)` reads the layer count from the connected model; override if wrong |
| `source_model` | `Auto` reads it from the LLLite weight keys |
| `remap_mode` | `Keys only` moves the indices; `Extend` also copies source weights onto the newly inserted blocks |
| `extend_strength` | Extend mode only: scales the copied `.up.*` tensors |

Direction is one-way (28→40, 28→52, 40→52); older-into-newer only, and a mismatch in the other
direction raises instead of silently applying half of the control.

> A file without `lllite.*` metadata is rejected with a clear error. That matters: a plain LoRA
> (e.g. a "pose" LoRA) used to load zero matching keys and produce **no control at all**, with no
> message anywhere.

### Which files can be used here

Only **ControlNet-LLLite** files (`.safetensors` carrying `lllite.*` metadata, keys like
`lllite_dit_blocks_<N>_*`). Plain LoRA, traditional ControlNet, and **control-LoRA** files are
rejected with an explicit error — an Anima pose model built as a control-LoRA (`control_embedder.*`
keys) needs its own `AnimaControlApply` node and cannot be driven from here.

## Compatibility

| LLLiteSEGS | ComfyUI | Impact Pack | Anima-LLLite | transformers |
|---|---|---|---|---|
| v0.1.x | ≥ 0.4x | ≥ 8.x (optional) | ≥ 1.0 (optional) | any (≥5 fixed at runtime) |

A startup log line reports the language and which nodes registered, e.g.
`[SegLLLite] loaded nodes: LLLiteApplySEGS, …`.

## Language

Titles, categories, descriptions and tooltips follow the **system language** (zh/en). Force it with
`OPNODES_LANG=zh` (or `en`) before starting ComfyUI. Combo *values* are never translated — ComfyUI
stores them in workflow JSON.

## FAQ

| Symptom | Cause / fix |
|---|---|
| LLLite nodes missing | `ComfyUI-Anima-LLLite` not found in `custom_nodes`; the startup log says so |
| SAM2/DINO node missing | `comfyui-sam2` not installed, or its own imports failed (see log) |
| Segments look wrong / misaligned | Check **LLLite Hints (SEGS)**; `hint_mode=image` must show the segment crop, not the whole image |
| `hint_mode=pose` errors | connect a SEGS preprocessor (e.g. Inspire's DWPreprocessor Provider) |
| Very slow with many segments | LLLite weights are re-read per segment; keep segment count moderate |

## License

MIT. Impact Pack (GPL-3.0) is **not** imported or copied — only its method-name protocol is used.
Anima-LLLite (Apache-2.0) and comfyui-sam2 are optional runtime libraries, not bundled.
