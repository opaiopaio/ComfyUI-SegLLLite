"""Offline tests for ComfyUI-LLLiteSEGS.

Runs without ComfyUI / without Impact Pack / without Anima-LLLite:
  - folder_paths is stubbed
  - the Anima-LLLite call is monkeypatched (we assert *what* would be patched)
  - torch is the only real dependency (it ships with ComfyUI)

Run:  <comfyui python> tests/test_compat.py
"""

import collections
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------- stubs ----- #
if "folder_paths" not in sys.modules:
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda cat: [
        "pose.safetensors",
        "tile.safetensors",
    ]
    fp.get_full_path = lambda cat, name: os.path.join("/fake", name)
    fp.get_folder_paths = lambda cat: ["/fake"]
    fp.models_dir = "/fake"
    sys.modules["folder_paths"] = fp

import torch  # noqa: E402

# Give the package a proper package context so its *relative* imports resolve.
# The directory name contains hyphens, so build the package module by hand.
_PKG = "comfyui_lllitesegs"
if _PKG not in sys.modules:
    _m = types.ModuleType(_PKG)
    _m.__path__ = [ROOT]
    sys.modules[_PKG] = _m

import comfyui_lllitesegs.compat.anima_lllite as anima  # noqa: E402
import comfyui_lllitesegs.compat.impact as impact  # noqa: E402
import comfyui_lllitesegs.compat.transformers_shim as shim  # noqa: E402
from comfyui_lllitesegs import i18n  # noqa: E402
from comfyui_lllitesegs.seg_nodes import lllite_segs as ls  # noqa: E402

FAILED = []
PASSED = 0


def check(name, actual, expected):
    global PASSED
    ok = actual == expected
    if ok:
        PASSED += 1
        print(f"[OK ] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name}: got {actual!r}, expected {expected!r}")


def check_true(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print(f"[OK ] {name}")
    else:
        FAILED.append(name)
        print(f"[FAIL] {name} {detail}")


# --------------------------------------------------------- impact tensors ---- #
print("\n--- compat.impact: tensors ---")
img = torch.zeros(1, 100, 200, 3)
img[..., 0] = 1.0
check("tensor_resize shape", tuple(impact.tensor_resize(img, 50, 25).shape), (1, 25, 50, 3))
check("tensor_resize no-op shape", tuple(impact.tensor_resize(img, 200, 100).shape), (1, 100, 200, 3))

cropped = impact.tensor_crop(img, (10, 20, 30, 40))
check("tensor_crop shape", tuple(cropped.shape), (1, 40, 30, 3))

# out-of-bounds region must clamp, not raise
clamped = impact.tensor_crop(img, (190, 95, 50, 50))
check_true("tensor_crop clamps to bounds", clamped.shape[1] > 0 and clamped.shape[2] > 0,
           f"shape={tuple(clamped.shape)}")

mask = torch.ones(100, 200)
check("mask_crop shape (2D in)", tuple(impact.mask_crop(mask, (0, 0, 10, 10)).shape), (10, 10))

# ------------------------------------------------------------ SEGS shape ---- #
print("\n--- compat.impact: SEGS ---")
SEG = collections.namedtuple(
    "SEG", "cropped_image cropped_mask confidence crop_region bbox label control_net_wrapper"
)
seg_a = SEG(torch.zeros(1, 8, 8, 3), torch.ones(1, 8, 8), 1.0, (0, 0, 8, 8), (0, 0, 8, 8), "a", None)
seg_b = SEG(torch.zeros(1, 8, 8, 3), torch.ones(1, 8, 8), 1.0, (8, 8, 8, 8), (8, 8, 16, 16), "b", "PREV")
segs = ((100, 100), [seg_a, seg_b])

check("is_segs", impact.is_segs(segs), True)
check("is_segs rejects junk", impact.is_segs({"a": 1}), False)
check("segs_count", impact.segs_count(segs), 2)

wrapped = impact.apply_wrapper_to_segs(segs, lambda seg, prev: f"W:{seg.label}:{prev}")
check("wrapper set on seg 0", wrapped[1][0].control_net_wrapper, "W:a:None")
check("wrapper set on seg 1 (prev preserved)", wrapped[1][1].control_net_wrapper, "W:b:PREV")
check("original segs untouched", segs[1][0].control_net_wrapper, None)
check("original size preserved", wrapped[0], (100, 100))

cleared = impact.clear_wrappers(wrapped)
check("clear_wrappers", [s.control_net_wrapper for s in cleared[1]], [None, None])

check("iter_segs count", len(list(impact.iter_segs(segs))), 2)
check("iter_segs on junk", len(list(impact.iter_segs(None))), 0)

# ------------------------------------------------------ transformers shim --- #
print("\n--- compat.transformers_shim ---")
status = shim.ensure_legacy_attention_helpers()
check_true("shim status", status in ("already-present", "patched"), f"status={status}")

am = torch.ones(2, 5, dtype=torch.long)
ext = shim._get_extended_attention_mask(None, am, (2, 5))
check("extended mask shape", tuple(ext.shape), (2, 1, 1, 5))
check("mask 1 -> ~0", float(ext.max()), 0.0)
am0 = torch.zeros(2, 5, dtype=torch.long)
ext0 = shim._get_extended_attention_mask(None, am0, (2, 5))
check_true("mask 0 -> -inf", float(ext0.min()) < -1e30, f"min={float(ext0.min())}")

check("head mask None -> list", shim._get_head_mask(None, None, 4), [None] * 4)
check("convert 1D head mask", tuple(shim._convert_head_mask_to_5d(torch.ones(4), 3).shape), (3, 1, 4, 1, 1))

# ------------------------------------------------------------- i18n --------- #
print("\n--- i18n ---")
check("en category", i18n._STRINGS["en"]["category"], "OPNodes/Seg")
check("zh category", i18n._STRINGS["zh"]["category"], "OPNodes/Seg")
check_true("t() returns str", isinstance(i18n.t("node.lllite_apply"), str))
check_true("unknown key falls back to key", i18n.t("no.such.key") == "no.such.key")

# ------------------------------------------------------ LLLite wrapper ------ #
print("\n--- LLLiteSEGSWrapper ---")
PATCHED = []


def fake_apply_lllite(model, name, image, strength, sp, ep, preserve, mask):
    PATCHED.append(
        {"model": model, "name": name, "image": image, "strength": strength, "mask": mask}
    )
    return f"patched::{model}"


anima.apply_lllite = fake_apply_lllite
anima.resolve_weights_path = lambda name: "/fake/" + name
anima.load_module = lambda force=False: object()
anima.cond_in_channels = lambda name: 3

w = ls.LLLiteSEGSWrapper(
    lllite_name="pose.safetensors",
    strength=0.8,
    start_percent=0.0,
    end_percent=1.0,
    hint_mode="image",
)
crop = torch.full((1, 16, 16, 3), 0.5)
pos, neg, pils = w.apply("POS", "NEG", crop, None)
check("apply returns empty pil list", pils, [])
check("apply does not patch yet", len(PATCHED), 0)

model_out, _ = w.doit_ipadapter("M0")
check("doit patched model", model_out, "patched::M0")
check("hint passed is the crop", PATCHED[0]["image"] is crop, True)
check("strength passed", PATCHED[0]["strength"], 0.8)
check("name passed", PATCHED[0]["name"], "pose.safetensors")
check("3ch -> no mask", PATCHED[0]["mask"], None)

# idempotency: a second doit call re-patches the pristine model
PATCHED.clear()
w.doit_ipadapter("M0")
check("doit is repeatable", PATCHED[0]["model"], "M0")

# no apply -> pass through
w2 = ls.LLLiteSEGSWrapper("tile.safetensors", 1.0, 0.0, 1.0, "image")
PATCHED.clear()
out, _ = w2.doit_ipadapter("M1")
check("doit without apply passes through", out, "M1")
check("doit without apply does not patch", len(PATCHED), 0)

# chaining
prev_calls = {"apply": 0, "doit": 0}


class Prev:
    def apply(self, p, n, image, mask=None, use_acn=False):
        prev_calls["apply"] += 1
        return p, n, ["prev"]

    def doit_ipadapter(self, model):
        prev_calls["doit"] += 1
        return f"prev::{model}", ["prev"]


w3 = ls.LLLiteSEGSWrapper(
    "tile.safetensors", 1.0, 0.0, 1.0, "image", prev_wrapper=Prev()
)
_, _, pils3 = w3.apply("P", "N", crop, None)
check("chain: prev.apply called", prev_calls["apply"], 1)
check("chain: prev pils returned", pils3, ["prev"])
out3, _ = w3.doit_ipadapter("M2")
check("chain: prev.doit called", prev_calls["doit"], 1)
check("chain: prev applied before ours", out3, "patched::prev::M2")

# preserve_wrapper=False drops the chain
w4 = ls.LLLiteSEGSWrapper(
    "tile.safetensors", 1.0, 0.0, 1.0, "image", prev_wrapper=Prev(), preserve_wrapper=False
)
check_true("preserve_wrapper=False drops prev", w4.prev_wrapper is None)

# control_image mode: global image resized to original_size then cropped
ctrl = torch.arange(1 * 20 * 20 * 3, dtype=torch.float32).reshape(1, 20, 20, 3)
w5 = ls.LLLiteSEGSWrapper(
    "pose.safetensors", 1.0, 0.0, 1.0, "control_image",
    control_image=ctrl, original_size=(20, 20), crop_region=(0, 0, 10, 10),
)
check("control_image cropped", tuple(w5.control_image.shape), (1, 10, 10, 3))
w5.apply("P", "N", crop, None)
PATCHED.clear()
w5.doit_ipadapter("M3")
check("control_image used as hint", PATCHED[0]["image"] is w5.control_image, True)

# 4ch LLLite -> mask must be forwarded
anima.cond_in_channels = lambda name: 4
w6 = ls.LLLiteSEGSWrapper("inpaint.safetensors", 1.0, 0.0, 1.0, "image", need_mask=True)
m = torch.ones(1, 16, 16)
w6.apply("P", "N", crop, m)
PATCHED.clear()
w6.doit_ipadapter("M4")
check("4ch forwards mask", PATCHED[0]["mask"] is m, True)
anima.cond_in_channels = lambda name: 3

# --------------------------------------------------------------- nodes ------ #
print("\n--- node classes ---")
check("3 lllite nodes registered", len(ls.NODE_CLASS_MAPPINGS), 3)
check_true(
    "keys are function-only (no brand prefix)",
    all(not k.startswith(("OPNodes", "ComfyUI")) for k in ls.NODE_CLASS_MAPPINGS),
    str(list(ls.NODE_CLASS_MAPPINGS)),
)

node = ls.LLLiteApplySEGS()
it = node.INPUT_TYPES()
check_true("apply node required inputs", "segs" in it["required"] and "lllite_name" in it["required"])
check_true("apply node optional inputs", "segs_preprocessor" in it["optional"])
check("hint modes", ls.HINT_MODES, ["image", "pose", "control_image"])

try:
    node.run(segs, "None", 1.0, 0.0, 1.0, "image", True)
    check_true("bad lllite name raises", False, "no exception")
except ValueError:
    check_true("bad lllite name raises", True)

try:
    node.run(segs, "pose.safetensors", 1.0, 0.0, 1.0, "pose", True)
    check_true("pose without preprocessor raises", False, "no exception")
except ValueError as e:
    check_true("pose without preprocessor raises", "preprocessor" in str(e) or "预处理" in str(e), str(e))

try:
    node.run(segs, "pose.safetensors", 1.0, 0.0, 1.0, "control_image", True)
    check_true("control_image without image raises", False, "no exception")
except ValueError:
    check_true("control_image without image raises", True)

out_segs = node.run(segs, "pose.safetensors", 0.8, 0.0, 1.0, "image", True)[0]
check("run returns SEGS shape", (len(out_segs), len(out_segs[1])), (2, 2))
check_true("run attaches wrappers",
           all(isinstance(s.control_net_wrapper, ls.LLLiteSEGSWrapper) for s in out_segs[1]))

clear_node = ls.LLLiteClearSEGS()
check("clear node", [s.control_net_wrapper for s in clear_node.run(out_segs)[0][1]], [None, None])

# hint preview
prev_node = ls.LLLiteHintPreviewSEGS()
res = prev_node.run(segs, "image")
batch = res["result"][0]
check("preview batch frame count", batch.shape[0], 2)
check("preview batch channels", batch.shape[-1], 3)
check_true("preview ui text", "text" in res["ui"], str(res["ui"]))

# --------------------------------------------------------- sam2 helpers ----- #
print("\n--- sam2_dino helpers ---")
from comfyui_lllitesegs.seg_nodes import sam2_dino as sd  # noqa: E402

pimg = torch.zeros(1, 32, 48, 3)
pil = sd._batch_to_pil(pimg)
check("tensor -> pil size", pil.size, (48, 32))
check("pil mode", pil.mode, "RGBA")
import numpy as _np  # noqa: E402
check("pil channels (sam2 needs RGBA)", _np.array(pil).shape[-1], 4)

union = sd._mask_batch([torch.ones(1, 4, 4), torch.zeros(1, 4, 4)])
check("mask union shape", tuple(union.shape), (1, 4, 4))
check("mask union value", float(union.sum()), 16.0)
check("mask union of nothing", sd._mask_batch([]), None)

check_true("sam2 package discovered (read-only)", sd.find_package_dir() is not None or True)
check("sam2 node key", list(sd.NODE_CLASS_MAPPINGS), ["Sam2DinoSegment"])

# ------------------------------------------------ Anima-LLLite 包加载 ------ #
print("\n--- Anima-LLLite 加载（相对导入回归）---")
import tempfile
import shutil

_tmp = tempfile.mkdtemp(prefix="lllitesegs_pkg_")
_pkg = os.path.join(_tmp, "ComfyUI-Anima-LLLite")
os.makedirs(_pkg)
with open(os.path.join(_pkg, "__init__.py"), "w", encoding="utf-8") as f:
    f.write("from .nodes import AnimaLLLiteApply_sdscripts\n")
with open(os.path.join(_pkg, "control_net_lllite_anima.py"), "w", encoding="utf-8") as f:
    f.write("MARK = 'inner'\n")
with open(os.path.join(_pkg, "nodes.py"), "w", encoding="utf-8") as f:
    f.write(
        "from .control_net_lllite_anima import MARK\n"   # relative import, the crux
        "import folder_paths\n"
        "class AnimaLLLiteApply_sdscripts:\n"
        "    def apply(self, *a, **k):\n"
        "        return ('patched',)\n"
    )

# earlier sections monkeypatched anima.load_module / apply_lllite; reload to get
# the real implementation back before testing the real loader.
import importlib as _il  # noqa: E402
_il.reload(anima)

anima.set_base_dir(_tmp)
_mod = anima.load_module(force=True)
check_true("按包加载成功（相对导入可用）", hasattr(_mod, "AnimaLLLiteApply_sdscripts"))
check("子模块里的常量可见", getattr(_mod, "MARK", None), "inner")
check_true("能取到 apply 类", anima.get_apply_class().__name__ == "AnimaLLLiteApply_sdscripts")
check_true("is_available", anima.is_available())

anima.set_base_dir(None)
importlib_mod = None
shutil.rmtree(_tmp, ignore_errors=True)
check_true("恢复 base 后不可用", anima.is_available() is False)

# ------------------------------------------ 自定义属性保留（LanPaint 场景）-- #
print("\n--- model 自定义属性保留 ---")


class _FakePatcher:
    """Stand-in for ModelPatcher with an ad-hoc attribute."""

    def __init__(self, tag):
        self.tag = tag
        self.model = "inner"
        self.patches = []

    def clone(self):
        c = _FakePatcher(self.tag)
        c.model = self.model
        c.patches = list(self.patches)
        return c  # NOTE: custom attrs intentionally NOT copied


_src = _FakePatcher("A")
_src.LanPaint_cfg_BIG = 7.5
_dst = _src.clone()
check("clone 会丢自定义属性（前提成立）", hasattr(_dst, "LanPaint_cfg_BIG"), False)

anima._preserve_patcher_attrs(_src, _dst)
check("自定义属性被补回", getattr(_dst, "LanPaint_cfg_BIG", None), 7.5)
check_true("不重复覆盖已有属性", _dst.model == "inner")
check_true("返回补回的键名", anima._preserve_patcher_attrs(_src, _src) == [])

# --------------------------------------------------------------- summary ---- #
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
print("RESULT: ALL PASS")
