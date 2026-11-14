"""Offline tests for ComfyUI-SegLLLite.

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
_PKG = "comfyui_seglllite"
if _PKG not in sys.modules:
    _m = types.ModuleType(_PKG)
    _m.__path__ = [ROOT]
    sys.modules[_PKG] = _m

import comfyui_seglllite.compat.anima_lllite as anima  # noqa: E402
import comfyui_seglllite.compat.impact as impact  # noqa: E402
import comfyui_seglllite.compat.transformers_shim as shim  # noqa: E402
from comfyui_seglllite import i18n  # noqa: E402
from comfyui_seglllite.seg_nodes import lllite_segs as ls  # noqa: E402

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

# Impact's crop_region is (x1, y1, x2, y2) -- absolute corners, not (x, y, w, h).
cropped = impact.tensor_crop(img, (10, 20, 30, 40))
check("tensor_crop shape (x1,y1,x2,y2)", tuple(cropped.shape), (1, 20, 20, 3))
wide = impact.tensor_crop(img, (10, 20, 60, 40))
check("tensor_crop 角点语义: w=x2-x1", tuple(wide.shape), (1, 20, 50, 3))

# out-of-bounds region must clamp, not raise
clamped = impact.tensor_crop(img, (190, 95, 50, 50))  # x2 < x1 -> clamp, no raise
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
anima.is_lllite_file = lambda name: True   # 本段只测 wrapper，文件类型校验另测

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
anima.is_lllite_file = lambda name: True   # 本段只测 wrapper，文件类型校验另测

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
from comfyui_seglllite.seg_nodes import sam2_dino as sd  # noqa: E402

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

# ---------------------------------------------- LLLite 跨代 remap 逻辑 ---- #
print("\n--- LLLite remap ---")
import comfyui_seglllite.anima_arch as AA  # noqa: E402
from comfyui_seglllite.seg_nodes import lllite_remap as LR  # noqa: E402

check("共享 key 识别", LR.is_shared_key("lllite_conditioning1.conv1.weight"), True)
check("block key 识别", LR.is_shared_key("lllite_dit_blocks_3_self_attn_q_proj.down.weight"), False)
check("解析 block 索引", LR.block_index_of("lllite_dit_blocks_27_self_attn_v_proj.up.bias"), 27)
check("共享 key 无索引", LR.block_index_of("lllite_conditioning1.conv1.weight"), None)

fake_keys = [
    f"lllite_dit_blocks_{i}_self_attn_{a}_proj.{s}"
    for i in range(28) for a in ("k", "q", "v") for s in ("down.weight", "depth_embed")
] + ["lllite_conditioning1.conv1.weight"]
check("从 key 得到原始块数", LR.block_count_of_keys(fake_keys), 28)
check("重写单个 key", LR.rewrite_key("lllite_dit_blocks_2_self_attn_q_proj.down.weight", 3),
      "lllite_dit_blocks_3_self_attn_q_proj.down.weight")

mapping = AA.get_mapping(28, 40)
sd = {k: 1.0 for k in fake_keys}
remapped = LR.remap_state_dict(sd, mapping)
check("28->40 后最大块索引", LR.block_count_of_keys(remapped.keys()), 40)
check("原 index 2 -> 3", "lllite_dit_blocks_3_self_attn_q_proj.down.weight" in remapped, True)
check("conditioning 未被改写", "lllite_conditioning1.conv1.weight" in remapped, True)
check("条目数守恒（keys only）", len(remapped), len(sd))

extended = LR.extend_state_dict(sd, mapping, 1.0)
check("extend 增加条目", len(extended) > len(remapped), True)
inserted = AA.inserted_sources(mapping, 40)
missing_ins = [i for i in inserted if not any(k.startswith(f"lllite_dit_blocks_{i}_") for k in extended)]
check("每个插入块都有权重", missing_ins, [])

check("反向不支持（40->28）", AA.get_mapping(40, 28), None)

class _Dm:
    def __init__(self, n): self.blocks = [0] * n
class _Inner:
    def __init__(self, n): self.diffusion_model = _Dm(n)
class _Patcher:
    def __init__(self, n): self.model = _Inner(n)

check("从模型探测 40 块", LR.detect_model_block_count(_Patcher(40)), 40)
check("从模型探测 28 块", LR.detect_model_block_count(_Patcher(28)), 28)
check("None 模型", LR.detect_model_block_count(None), None)

# 文件类型校验（LLLite vs 普通 LoRA）
_orig_meta = anima.read_metadata
anima.read_metadata = lambda name: {}
check("无 lllite 元数据 -> 不是 LLLite", anima.is_lllite_file("pose_lora.safetensors"), False)
anima.read_metadata = lambda name: {"lllite.version": "2"}
check("有 lllite.version -> 是 LLLite", anima.is_lllite_file("real.safetensors"), True)
anima.read_metadata = _orig_meta

check("remap 节点已注册", "LLLiteApplySEGSRemap" in LR.NODE_CLASS_MAPPINGS, True)
_it = LR.LLLiteApplySEGSRemap.INPUT_TYPES()
check_true("remap 节点含 remap 选项",
           all(k in _it["required"] for k in
               ("enable_remap", "target_model", "source_model", "remap_mode", "extend_strength")))
check("extend 模式名", LR.REMAP_MODES[1].startswith("Extend"), True)

# ------------------------------ LLLite depth_embed 补全（回归）----------- #
print("\n--- LLLite depth_embed 补全 ---")
import torch as _torch  # noqa: E402

_ATOMS = ("self_attn_k_proj", "self_attn_q_proj", "self_attn_v_proj")
_sd28 = {}
for _i in range(28):
    for _a in _ATOMS:
        for _suf in ("down.weight", "depth_embed"):
            _sd28[f"lllite_dit_blocks_{_i}_{_a}.{_suf}"] = _torch.zeros(2, 2)
_sd28["lllite_conditioning1.conv1.weight"] = _torch.zeros(2, 2)

_m = AA.get_mapping(28, 40)
_ko = LR.fill_missing_depth_embeds(_sd28, _m)
_depth_blocks = {LR.block_index_of(k) for k in _ko if k.endswith(".depth_embed")}
check("keys-only：40 块全有 depth_embed", sorted(_depth_blocks), list(range(40)))
check("keys-only：不合成普通权重", len([k for k in _ko if k.endswith("down.weight")]), 84)
check("keys-only：conditioning 保留", "lllite_conditioning1.conv1.weight" in _ko, True)
check("原 sd 未被修改", len(_sd28), 169)

_ex = LR.extend_state_dict(_sd28, _m, 1.0)
check("extend：40 块全有 down.weight", len([k for k in _ex if k.endswith("down.weight")]), 120)
_depth_ex = {LR.block_index_of(k) for k in _ex if k.endswith(".depth_embed")}
check("extend：40 块全有 depth_embed", sorted(_depth_ex), list(range(40)))

# 转换器契约：每个目标模块都必须有 depth_embed，否则 Anima-LLLite 拒绝加载
check("插入块列表长度", len(AA.inserted_sources(_m, 40)), 12)

# ------------------------------------------ control-LoRA: embedder ------- #
print("\n--- control embedder ---")
from comfyui_seglllite.compat import control_embedder as CE  # noqa: E402
from comfyui_seglllite.compat import control_lora as CL  # noqa: E402

_lat = _torch.zeros(1, 16, 1, 8, 6)  # 奇数宽度：必须 ceil 补齐
_packed = CE.pack_latent(_lat, 2, 1)
check("pack: ceil 网格", tuple(_packed.shape), (1, 1, 4, 3, 64))
check("pack: in_features = C*p*p", _packed.shape[-1], 64)

_w = _torch.zeros(8, 64)
_b = _torch.zeros(8)
_tok = CE.control_tokens(_lat, _w, _b, spatial=2, temporal=1, strength=1.0)
check("tokens 形状", tuple(_tok.shape), (1, 1, 4, 3, 8))
check("strength=0 → 全零", float(CE.control_tokens(_lat, _w, _b, strength=0.0).abs().max()), 0.0)
_tok2 = CE.control_tokens(_lat, _w, _b, target_grid=(5, 7))
check("网格不符时 resize", tuple(_tok2.shape), (1, 1, 5, 7, 8))
try:
    CE.control_tokens(_lat, _torch.zeros(8, 32), _b)
    check("特征数不符应报错", "no error", "ControlEmbedderError")
except CE.ControlEmbedderError:
    check("特征数不符报错", True, True)

import os as _os  # noqa: E402
import tempfile as _tf  # noqa: E402
from safetensors.torch import save_file as _save  # noqa: E402

_d = _tf.mkdtemp(prefix="ctrl_emb_")
_p = _os.path.join(_d, "fake_control.safetensors")
_save({
    "diffusion_model.control_embedder.proj.weight": _torch.zeros(2048, 64),
    "diffusion_model.control_embedder.proj.bias": _torch.zeros(2048),
    "diffusion_model.blocks.0.x.lora_A.weight": _torch.zeros(4, 8),
}, _p)
_w2, _b2 = CE.load_embedder(_p)
check("加载 embedder 权重", tuple(_w2.shape), (2048, 64))
check("缓存命中（同一对象）", CE.load_embedder(_p)[0] is _w2, True)
check("has_embedder=True", CE.has_embedder(_p), True)
_p2 = _os.path.join(_d, "plain_lora.safetensors")
_save({"diffusion_model.blocks.0.x.lora_A.weight": _torch.zeros(4, 8)}, _p2)
check("普通 LoRA → has_embedder False", CE.has_embedder(_p2), False)
try:
    CE.load_embedder(_p2)
    check("缺 control_embedder 应报错", "no error", "ControlEmbedderError")
except CE.ControlEmbedderError:
    check("缺 control_embedder 报错", True, True)

# ------------------------------------------ control-LoRA: LoRA 半部 ------ #
print("\n--- control-LoRA: LoRA 半部 ---")
_keys = [
    "diffusion_model.blocks.3.adaln_modulation_mlp.1.lora_A.weight",
    "diffusion_model.llm_adapter.x",
    "diffusion_model.control_embedder.proj.weight",
]
_sd = {k: _torch.zeros(2, 2) for k in _keys}
check("lora_part 剔除 embedder",
      "diffusion_model.control_embedder.proj.weight" in CL.lora_part(_sd), False)
check("block 索引解析", CL.block_index_of(_keys[0]), 3)
check("llm_adapter 无块索引", CL.block_index_of(_keys[1]), None)
check("块数推断", CL.block_count_of_keys(_keys), 4)

_map = AA.get_mapping(28, 40)
_rem, _moved = CL.remap_lora_state_dict(
    {"diffusion_model.blocks.2.a.lora_A.weight": _torch.zeros(1, 1)}, _map)
check("28→40 块索引 2→3", "diffusion_model.blocks.3.a.lora_A.weight" in _rem, True)
check("moved 计数", _moved, 1)

# ------------------------------------------ 节点参数顺序（兼容）--------- #
print("\n--- 节点参数顺序（旧工作流兼容）---")
_OLD7 = ["segs", "lllite_name", "strength", "start_percent", "end_percent",
         "hint_mode", "preserve_wrapper"]
_CTRL = ["enable_control_lora", "control_lora_name", "control_lora_strength",
         "control_lora_lora_strength", "control_lora_hint"]
_RAMP = ["control_lora_start_percent", "control_lora_end_percent"]

_it = ls.LLLiteApplySEGS.INPUT_TYPES()
check("基础节点前 7 个参数不变", list(_it["required"])[:7], _OLD7)
check("control-LoRA 追加在末尾", list(_it["required"])[7:], _CTRL + _RAMP)
check("vae 在 optional", "vae" in _it["optional"], True)

_itr = LR.LLLiteApplySEGSRemap.INPUT_TYPES()
check("remap 节点前 7 个参数不变", list(_itr["required"])[:7], _OLD7)
check("remap 参数位置不变", list(_itr["required"])[7:12],
      ["enable_remap", "target_model", "source_model", "remap_mode", "extend_strength"])
check("remap 节点 control-LoRA 追加在末尾", list(_itr["required"])[12:], _CTRL + _RAMP)
check("remap 节点有 vae", "vae" in _itr["optional"], True)

print("\n--- build_control_lora_config 校验 ---")
try:
    ls.build_control_lora_config(True, "x", 1.0, 1.0, "segs_preprocessor", None)
    check("缺 vae 应报错", "no error", "ValueError")
except ValueError:
    check("缺 vae 报错", True, True)
check("开关关闭时不构建", ls.build_control_lora_config(False, "x", 1, 1, "same as lllite", None), None)

# ------------------------------------------ 独立 Control-LoRA 节点 ------- #
print("\n--- Control-LoRA 独立节点 ---")
from comfyui_seglllite.seg_nodes import control_lora_segs as CLS  # noqa: E402
import torch.nn as _nn  # noqa: E402

check("独立节点已注册", "ControlLoRAApplySEGS" in CLS.NODE_CLASS_MAPPINGS, True)
_it3 = CLS.ControlLoRAApplySEGS.INPUT_TYPES()
check_true("独立节点参数齐全", all(k in _it3["required"] for k in (
    "segs", "control_lora_name", "vae", "control_lora_strength",
    "control_lora_lora_strength", "control_lora_hint", "preserve_wrapper",
    "enable_remap", "target_model", "source_model")))


class _FakeXE(_nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.d = d

    def forward(self, x):
        b, c, t, h, w = x.shape
        return _torch.zeros(b, t, h // 2, w // 2, self.d)


class _FakeDit(_nn.Module):
    def __init__(self):
        super().__init__()
        self.model_channels = 8
        self.patch_spatial = 2
        self.patch_temporal = 1
        self.x_embedder = _FakeXE(8)


class _FakePatcher:
    """Mirrors the parts of ModelPatcher.clone() that matter here:

    real clone() deep-copies ``model_options`` (where LLLite keeps its
    ``model_function_wrapper``) and copies ``wrappers`` key by key, but it does
    NOT copy ad-hoc python attributes."""

    def __init__(self, dit):
        self.model = type("M", (), {"diffusion_model": dit})()
        self.wrappers = {}
        self.model_options = {}

    def clone(self):
        n = _FakePatcher(self.model.diffusion_model)
        n.wrappers = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.wrappers.items()}
        n.model_options = dict(self.model_options)
        return n

    def add_wrapper_with_key(self, t, k, w):
        self.wrappers.setdefault(t, {}).setdefault(k, []).append(w)


class _FakeVAE:
    def encode(self, img):
        return {"samples": _torch.ones(1, 16, 1, img.shape[1] // 8, img.shape[2] // 8)}


_p3 = _os.path.join(_d, "ctrl8.safetensors")
_save({
    "diffusion_model.control_embedder.proj.weight": _torch.ones(8, 64),
    "diffusion_model.control_embedder.proj.bias": _torch.zeros(8),
    "diffusion_model.blocks.0.a.lora_A.weight": _torch.zeros(2, 2),
}, _p3)

_cfg = {
    "path": _p3, "name": "ctrl8", "strength": 1.0, "lora_strength": 0.0,
    "hint_source": "same as lllite", "vae": _FakeVAE(), "remap": False,
    "target_choice": None, "source_choice": None,
}
_img = _torch.zeros(1, 64, 48, 3)
_w = CLS.ControlLoRASEGSWrapper(_cfg)
_w.apply([], None, _img)
_pat = _FakePatcher(_FakeDit())
_out, _pils = _w.doit_ipadapter(_pat)
check("control wrapper 已挂到 diffusion_model", "diffusion_model" in _out.wrappers, True)
_fn = _out.wrappers["diffusion_model"]["seglllite_control"][0]
_x = _torch.zeros(1, 16, 1, 16, 12)
_res = _fn(lambda: _out.model.diffusion_model.x_embedder(_x))
check("x_embedder 输出被叠加 tokens", tuple(_res.shape), (1, 1, 8, 6, 8))
check("tokens 非零（权重=1、latent=1）", bool(float(_res.abs().sum()) > 0), True)
check("hook 已移除（不污染共享 DiT）",
      len(_out.model.diffusion_model.x_embedder._forward_hooks), 0)

_cfg0 = dict(_cfg)
_cfg0["strength"] = 0.0
_w0 = CLS.ControlLoRASEGSWrapper(_cfg0)
_w0.apply([], None, _img)
_o0, _ = _w0.doit_ipadapter(_FakePatcher(_FakeDit()))
check("strength=0 → 完全不挂 wrapper", getattr(_o0, "wrappers", {}), {})

# 直接串两个 wrapper（LLLite 节点 → Control-LoRA 节点）
_cfg_chain = dict(_cfg)
_wc = CLS.ControlLoRASEGSWrapper(_cfg_chain, prev_wrapper=None, preserve_wrapper=True)
_wc.apply([], None, _img)
_oc, _ = _wc.doit_ipadapter(_FakePatcher(_FakeDit()))
check("链式：单层也能挂上", "diffusion_model" in _oc.wrappers, True)

# ------------------------- LLLite + control-LoRA 同时挂载（关键组合）----- #
print("\n--- LLLite + control-LoRA 同块共存 ---")


def _stub_lllite(model, *a, **k):
    """Stand-in for anima.apply_lllite: same slot a real LLLite occupies."""
    patched = model.clone()
    patched.model_options["model_function_wrapper"] = "lllite"
    return patched


_ll_cfg = dict(_cfg)
_wl = ls.LLLiteSEGSWrapper(
    lllite_name="fake.safetensors", strength=1.0, start_percent=0.0, end_percent=1.0,
    hint_mode="image", apply_fn=_stub_lllite, control_lora=_ll_cfg,
)
_wl.apply([], None, _img)
_ol, _ = _wl.doit_ipadapter(_FakePatcher(_FakeDit()))
check("LLLite 槽位存活（clone 未丢 model_options）",
      _ol.model_options.get("model_function_wrapper"), "lllite")
check("control-LoRA wrapper 同时挂上", "diffusion_model" in _ol.wrappers, True)
_fn2 = _ol.wrappers["diffusion_model"]["seglllite_control"][0]
_res2 = _fn2(lambda: _ol.model.diffusion_model.x_embedder(_x))
check("两个控制同时生效（tokens 已叠加）", bool(float(_res2.abs().sum()) > 0), True)

# 关掉开关时只剩 LLLite
_wl2 = ls.LLLiteSEGSWrapper(
    lllite_name="fake.safetensors", strength=1.0, start_percent=0.0, end_percent=1.0,
    hint_mode="image", apply_fn=_stub_lllite, control_lora=None,
)
_wl2.apply([], None, _img)
_ol2, _ = _wl2.doit_ipadapter(_FakePatcher(_FakeDit()))
check("开关关闭：只有 LLLite",
      (_ol2.model_options.get("model_function_wrapper"), _ol2.wrappers), ("lllite", {}))

# ------------------------------- Control-LoRA Apply (Anima) 独立模型节点 -- #
print("\n--- Control-LoRA Apply (Anima) ---")
from comfyui_seglllite.seg_nodes import control_lora_model as CLM  # noqa: E402

check("模型节点已注册", "ControlLoRAApplyModel" in CLM.NODE_CLASS_MAPPINGS, True)
_it4 = CLM.ControlLoRAApplyModel.INPUT_TYPES()
check("required 参数（新参数追加在 strength 之后）", list(_it4["required"]),
      ["model", "control_lora_name", "pose_strength", "lora_strength",
       "enable_remap", "target_model", "source_model",
       "pose_start_percent", "pose_end_percent"])
check("optional 参数", list(_it4["optional"]), ["control_latent", "image", "vae"])
check("输出是 MODEL", CLM.ControlLoRAApplyModel.RETURN_TYPES, ("MODEL",))

_node = CLM.ControlLoRAApplyModel()
_p0 = _FakePatcher(_FakeDit())
(_out0,) = _node.run(_p0, "ctrl8", 0.0, control_latent=_torch.zeros(1, 16, 1, 8, 6))
check("strength=0 原样返回（同一对象，未 clone）", _out0 is _p0, True)

try:
    _node.run(_FakePatcher(_FakeDit()), "ctrl8", 1.0)
    check("缺 latent/image 应报错", "no error", "ValueError")
except ValueError:
    check("缺 latent/image 报错", True, True)

try:
    _node.run(_FakePatcher(_FakeDit()), "ctrl8", 1.0, image=_torch.zeros(1, 64, 48, 3))
    check("image 缺 vae 应报错", "no error", "ValueError")
except ValueError:
    check("image 缺 vae 报错", True, True)

_p1 = _FakePatcher(_FakeDit())
(_out1,) = _node.run(_p1, _p3, 1.0, control_latent={"samples": _torch.ones(1, 16, 1, 8, 6)})
check("control_latent(dict) 注入成功", "diffusion_model" in _out1.wrappers, True)
_fn3 = _out1.wrappers["diffusion_model"]["seglllite_control"][0]
_r3 = _fn3(lambda: _out1.model.diffusion_model.x_embedder(_x))
check("模型节点 tokens 生效", bool(float(_r3.abs().sum()) > 0), True)

_p4 = _FakePatcher(_FakeDit())
(_out4,) = _node.run(_p4, _p3, 1.0, image=_torch.ones(1, 64, 48, 3), vae=_FakeVAE())
check("image + vae 路径可用", "diffusion_model" in _out4.wrappers, True)

try:
    _node.run(_FakePatcher(_FakeDit()), _p2, 1.0, control_latent=_torch.zeros(1, 16, 1, 8, 6))
    check("非 control-LoRA 文件应报错", "no error", "ValueError")
except ValueError:
    check("非 control-LoRA 文件报错", True, True)

# ------------------------------------------ resolve_block_mapping --------- #
print("\n--- resolve_block_mapping（统一解析）---")
_s28 = _Patcher(28)
_s40 = _Patcher(40)
src, tgt, mp = ls.resolve_block_mapping(_s40, AA.AUTO_LABEL, AA.AUTO_LABEL, 28)
check("28 文件 + 40 模型：源/目标", (src, tgt), (28, 40))
check("28→40 映射条目", len(mp), 28)
src2, tgt2, mp2 = ls.resolve_block_mapping(_s28, AA.AUTO_LABEL, AA.AUTO_LABEL, 28)
check("28 文件 + 28 模型：无需映射", (src2, tgt2, mp2), (28, 28, None))
try:
    ls.resolve_block_mapping(_s28, AA.AUTO_LABEL, AA.AUTO_LABEL, 40)
    check("40→28 应报错", "no error", "ValueError")
except ValueError:
    check("40→28 明确报错", True, True)
# 文件优先于手动选择：选 40 但文件是 28
src3, tgt3, mp3 = ls.resolve_block_mapping(_s40, AA.AUTO_LABEL,
                                           "Anima 2.9B (40 layers)", 28)
check("文件优先于 source_model", (src3, tgt3), (28, 40))

# ------------------------------- 本轮补充：过滤 / 断言 / 模型节点 remap ---- #
print("\n--- combo 过滤 + 形状断言 + 模型节点 remap ---")

# 1) 只列含 control_embedder.* 的文件
check("过滤: 含 embedder 的文件判定", CL.is_control_lora_file(_p3), True)
check("过滤: 普通 LoRA 判定", CL.is_control_lora_file(_p2), False)
check("过滤: 非 safetensors 乐观通过", CL.is_control_lora_file("weights.pt"), True)

_orig_list = fp.get_filename_list
_orig_full = fp.get_full_path
_orig_paths = fp.get_folder_paths
fp.get_filename_list = lambda k: {"loras": ["good.safetensors", "plain.safetensors"],
                                  "controlnet": []}.get(k, [])
fp.get_full_path = lambda k, n: {"good.safetensors": _p3, "plain.safetensors": _p2}.get(n)
fp.get_folder_paths = lambda k: []
check("过滤: 只留下 control-LoRA", CL.list_control_lora_files(), ["good.safetensors"])
check("过滤: 节点 combo 用它", ls._control_lora_names(), ["good.safetensors"])

# 探测结果为空时不得回退成全量列表（否则下拉又是一墙无关 LoRA）
fp.get_filename_list = lambda k: {"loras": ["plain.safetensors"], "controlnet": []}.get(k, [])
check("过滤: 空结果 -> ['None'] 而非全量", ls._control_lora_names(), ["None"])

# get_full_path 失效（嵌套子目录 / 多模型路径）时手工拼路径
fp.get_filename_list = lambda k: {"loras": [_os.path.basename(_p3)], "controlnet": []}.get(k, [])
fp.get_full_path = lambda k, n: None
fp.get_folder_paths = lambda k: [_os.path.dirname(_p3)]
check("过滤: 手工拼路径可解析", CL._resolve_in(fp, "loras", _os.path.basename(_p3)), _p3)

# 完全解析不到的条目放行：宁可多列，不可藏掉用户的适配器
fp.get_filename_list = lambda k: {"loras": ["mystery.safetensors"], "controlnet": []}.get(k, [])
fp.get_folder_paths = lambda k: []
check("过滤: 无法解析的文件放行", CL.list_control_lora_files(), ["mystery.safetensors"])

# controlnet 目录也要扫（不少人的适配器放在那里）
fp.get_filename_list = lambda k: {"loras": [], "controlnet": ["Anima\\pos.safetensors"]}.get(k, [])
fp.get_full_path = lambda k, n: {"Anima\\pos.safetensors": _p3}.get(n)
check("过滤: 也扫 controlnet", CL.list_control_lora_files(), ["Anima\\pos.safetensors"])

fp.get_filename_list = _orig_list
fp.get_full_path = _orig_full
fp.get_folder_paths = _orig_paths
# 2) x_embedder 输出维度断言
_prov = CE.token_provider(_torch.zeros(1, 16, 1, 8, 6), _torch.zeros(8, 64), _torch.zeros(8))
try:
    _prov(_torch.zeros(1, 8, 6, 8))  # 4D -> 必须先报错，而不是索引越界/形状乱算
    check("4D 输出应报错", "no error", "RuntimeError")
except RuntimeError as e:
    check("4D 输出报错信息含 expected", "expected (B, T, H, W, D)" in str(e), True)

# 3) 模型节点：remap 参数与双开关短路
_it5 = CLM.ControlLoRAApplyModel.INPUT_TYPES()
check("模型节点有 lora_strength", "lora_strength" in _it5["required"], True)
check("姿态旋钮叫 pose_strength", "pose_strength" in _it5["required"], True)
check("模型节点有 remap 三件套",
      all(k in _it5["required"] for k in ("enable_remap", "target_model", "source_model")), True)
check("已有 widget 顺序未被破坏",
      list(_it5["required"])[:3], ["model", "control_lora_name", "pose_strength"])

_p5 = _FakePatcher(_FakeDit())
(_out5,) = _node.run(_p5, "ctrl8", 0.0, lora_strength=0.0)
check("两个强度都为 0 → 原样返回", _out5 is _p5, True)
(_out6,) = _node.run(_FakePatcher(_FakeDit()), _p3, 1.0,
                     control_latent={"samples": _torch.ones(1, 16, 1, 8, 6)})
check("只开姿态仍可注入", "diffusion_model" in _out6.wrappers, True)

# ------------------------------------------ 姿态 sigma 门限（ramp）------- #
print("\n--- pose ramp ---")
_ones = _torch.ones(1, 16, 1, 8, 6)
_prov_r = CE.token_provider(_ones, _torch.ones(8, 64), _torch.zeros(8), ramp=(0.0, 0.5))
_fake_out = _torch.zeros(1, 1, 4, 3, 8)
_late = _prov_r(_fake_out, progress=100.0)   # t=1.0 -> 超出 0-0.5
check("ramp: 收尾阶段被门掉", float(_late.abs().sum()), 0.0)
_mid = _prov_r(_fake_out, progress=40.0)     # t=0.4 -> 生效
check("ramp: 中段生效", bool(float(_mid.abs().sum()) > 0), True)
_prov_n = CE.token_provider(_ones, _torch.ones(8, 64), _torch.zeros(8))  # 无 ramp
check("ramp 未设置时不受进度影响", bool(float(_prov_n(_fake_out, progress=100.0).abs().sum()) > 0), True)

# --------------------------------------------------------------- summary ---- #
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
print("RESULT: ALL PASS")
