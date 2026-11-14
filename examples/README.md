# Example workflows

`quickstart_segs.json` — the minimal shape of this package:

```
MaskToSEGS (ComfyUI-Impact-Pack) → LLLite Apply (SEGS) → LLLite Hints (SEGS)
```

Required nodes outside this package:

| Node | Package | Why |
|---|---|---|
| `MaskToSEGS` | ComfyUI-Impact-Pack | turns a mask into SEGS (any SEGS producer works) |
| `DetailerForEachPipe` | ComfyUI-Impact-Pack or this repo's sister project | consumes the SEGS produced here |
| `Apply Anima ControlNet-LLLite (sd-scripts)` | ComfyUI-Anima-LLLite | performs the actual LLLite patch |

If a dependency is missing the node says so explicitly instead of failing silently
(`comfyui-sam2` gates the segmentation node, Anima-LLLite gates both apply nodes).
Connect your own `MASK` to `MaskToSEGS.mask` — it is intentionally left unconnected.
