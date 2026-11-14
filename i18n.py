"""ComfyUI-LLLiteSEGS — minimal bilingual strings.

Language selection (first hit wins):
  1. env ``OPNODES_LANG`` (``zh`` / ``en``)
  2. Windows UI language (via ``GetUserDefaultUILanguage``)
  3. ``LC_ALL`` / ``LC_MESSAGES`` / ``LANG``
  4. ``locale.getdefaultlocale()``
  5. ``en``

Only UI text is translated. Combo *values* are never translated: ComfyUI stores
them inside workflow JSON, so translating them would break workflows across
languages.
"""

import os

_LOG = "[LLLiteSEGS]"


def _detect_lang() -> str:
    forced = (os.environ.get("OPNODES_LANG") or "").strip().lower()
    if forced.startswith("zh"):
        return "zh"
    if forced.startswith("en"):
        return "en"

    try:  # Windows: most reliable signal
        import ctypes

        lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        if (lang_id & 0x3FF) == 0x04:  # LANG_CHINESE
            return "zh"
        return "en"
    except Exception:
        pass

    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        val = (os.environ.get(var) or "").strip().lower()
        if val.startswith("zh"):
            return "zh"
        if val.startswith("en"):
            return "en"

    try:
        import locale

        loc = (locale.getlocale()[0] or "").lower()
        if loc.startswith("zh"):
            return "zh"
    except Exception:
        pass

    return "en"


LANG = _detect_lang()

_STRINGS = {
    "en": {
        "category": "OPNodes/Seg",
        "node.lllite_apply": "LLLite Apply (SEGS)",
        "node.lllite_clear": "LLLite Clear (SEGS)",
        "node.lllite_preview": "LLLite Hints (SEGS)",
        "node.sam2dino": "SAM2 + GroundingDINO Segment",
        "node.sam2dino_loader": "GroundingDINO Loader",
        "desc.lllite_apply": (
            "Attach a ControlNet-LLLite to every SEG so each detected region gets its own "
            "hint image (crop / pose / control image) before sampling."
        ),
        "desc.lllite_clear": "Remove LLLite wrappers from a SEGS chain (debug / reset).",
        "desc.lllite_preview": "Preview the hint image each SEG would receive.",
        "desc.sam2dino": (
            "Text-prompted segmentation: GroundingDINO boxes -> SAM2 masks. Works with "
            "transformers>=5 (restores the helpers it removed) and needs comfyui-sam2 installed."
        ),
        "tip.hint_mode": "crop = the segment's own crop; pose = run the SEGS preprocessor on it; control_image = crop from a global control image.",
        "tip.seg_processor": "Only used when hint_mode=pose. e.g. DWPreprocessor Provider (SEGS) from Inspire.",
        "hint.no_segs": "no SEGS",
        "hint.seg_count": "{n} SEG",
        "err.no_anima_lllite": "ComfyUI-Anima-LLLite not found in custom_nodes.",
        "err.bad_lllite": "LLLite file not found: {name}",
        "err.pose_needs_preprocessor": "hint_mode=pose requires an segs_preprocessor input.",
        "err.control_needs_image": "hint_mode=control_image requires a control_image input.",
    },
    "zh": {
        "category": "OPNodes/Seg",
        "node.lllite_apply": "应用 LLLite（SEGS 分区）",
        "node.lllite_clear": "清除 LLLite（SEGS）",
        "node.lllite_preview": "LLLite 提示图预览（SEGS）",
        "node.sam2dino": "SAM2 + GroundingDINO 分区",
        "node.sam2dino_loader": "GroundingDINO 加载器",
        "desc.lllite_apply": (
            "给每个 SEG 挂一个 ControlNet-LLLite，让每个检测区域在采样前拥有自己的提示图"
            "（本块裁剪 / 姿态 / 控制图），解决全局图被压缩到小块导致的几何错位。"
        ),
        "desc.lllite_clear": "清除 SEGS 链上的 LLLite wrapper（调试 / 重置）。",
        "desc.lllite_preview": "预览每个 SEG 实际会拿到的提示图。",
        "desc.sam2dino": (
            "文本提示分区：GroundingDINO 出框 → SAM2 出掩码。兼容 transformers>=5"
            "（自动补回被移除的接口），需要已安装 comfyui-sam2。"
        ),
        "tip.hint_mode": "crop=用本块裁剪图；pose=对本块跑 SEGS 预处理器；control_image=从全局控制图按区域裁块。",
        "tip.seg_processor": "仅 hint_mode=pose 时使用，例如 Inspire 的 DWPreprocessor Provider (SEGS)。",
        "hint.no_segs": "无 SEGS",
        "hint.seg_count": "{n} 块",
        "err.no_anima_lllite": "未在 custom_nodes 中找到 ComfyUI-Anima-LLLite。",
        "err.bad_lllite": "找不到 LLLite 文件：{name}",
        "err.pose_needs_preprocessor": "hint_mode=pose 需要连接 segs_preprocessor。",
        "err.control_needs_image": "hint_mode=control_image 需要连接 control_image。",
    },
}


def t(key: str, **fmt) -> str:
    """Translate a UI string; unknown keys fall back to the key itself."""
    table = _STRINGS.get(LANG) or _STRINGS["en"]
    val = table.get(key)
    if val is None:
        val = _STRINGS["en"].get(key, key)
    if fmt:
        try:
            return val.format(**fmt)
        except Exception:
            return val
    return val


def language_name() -> str:
    return {"zh": "中文", "en": "English"}.get(LANG, LANG)


__all__ = ["LANG", "t", "language_name", "_LOG"]
