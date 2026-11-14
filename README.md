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
| **Control-LoRA Apply (SEGS)** | Attach a control-LoRA (pose adapter) to every SEG: the adapter's LoRA half + the zero-init control embedder. No LLLite involved — chain it after an LLLite node to run both. | torch + a control-LoRA file |
| **Control-LoRA Apply (Anima)** | Same kernel for a whole **MODEL** (no SEGS): pose control on plain txt2img / img2img. Accepts an encoded `control_latent` or `image` + `vae`. `strength=0` returns the model untouched. | torch + a control-LoRA file |
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

## Pose control in the same node (control-LoRA)

Both apply nodes can additionally attach a **control-LoRA** (Anima pose adapter), so one segment gets
"redraw (LLLite) + pose control (control-LoRA)" at the same time. The two mechanisms live in different
places -- LLLite owns `model_options["model_function_wrapper"]` (single slot), the control-LoRA adds a
keyed `wrappers["diffusion_model"]` entry (multi slot) -- which is why they can coexist while two
LLLites cannot.

| Option | Default | Meaning |
|---|---|---|
| `enable_control_lora` | off | master switch |
| `control_lora_name` | -- | adapter file from `models/loras`; it must contain `control_embedder.*` |
| `control_lora_strength` | 1.2 | pose signal (the `control_embedder` half) |
| `control_lora_lora_strength` | 1.0 | the file's LoRA half; `0` = pose embedder only |
| `control_lora_hint` | `segs_preprocessor` | skeleton source: SEGS preprocessor / global control image / same as the LLLite hint |

`vae` (optional input) becomes required when the switch is on: the skeleton is VAE-encoded into the
control latent. The skeleton tokens are added to the DiT patch-embed output through a temporary
`x_embedder` forward hook (removed in a `finally`), and are resized with a warning when the token grid
does not line up (odd crop sizes).

On Anima 2.9B / 3.8B use **LLLite Apply (SEGS, Remap)**: it remaps the block indices of the LoRA half
as well. The plain node loads the LoRA half as-is and fails loudly when nothing matches.

> The pose adapter weights are **non-commercial** (CircleStone NC + NVIDIA Open Model). Generated
> images are unaffected; check the licence before commercial use.

## Pose control on a normal model (no SEGS)

`Control-LoRA Apply (Anima)` is the standalone `MODEL -> MODEL` version of the same kernel, for plain
txt2img / img2img:

```
Control-LoRA Apply (Anima)
  model                 ← LoraLoaderModelOnly(model, adapter, lora_strength)   (reads the LoRA half)
  control_lora_name     = the same adapter file                                (this node reads control_embedder.proj.*)
  control_latent        ← VAEEncode(skeleton)                                  (or use image + vae below)
  strength              = 1.0   (0 = model returned untouched, no clone, no hook)
  [optional] image + vae → encoded here, saving the VAEEncode node
```

Same file, two nodes -- exactly like upstream, but with no third-party package: only the weight file is
needed (its licence is non-commercial; generated images are unaffected).

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
