"""ComfyUI-SegLLLite — minimal bilingual strings.

Language selection (first hit wins):
  1. ``lang.txt`` next to this file, containing ``zh`` or ``en``
  2. the system locale (``locale.getlocale()``)
  3. ``en``

  Deliberately reads **no** environment variables and uses **no** FFI: the override
  is a plain file so that both the behaviour and the code stay trivially reviewable.

Only UI text is translated. Combo *values* are never translated: ComfyUI stores
them inside workflow JSON, so translating them would break workflows across
languages.
"""

import os

_LOG = "[SegLLLite]"


def _lang_from_tag(value) -> "str | None":
    """Map a locale tag / env value to ``zh`` / ``en`` (None when unknown).

    Windows reports ``"Chinese (Simplified)_China"`` rather than ``zh_CN``, so
    matching only on the ``zh`` prefix would miss it.
    """
    v = (value or "").strip().lower()
    if not v:
        return None
    if v.startswith("zh") or "chinese" in v:
        return "zh"
    if v.startswith("en") or "english" in v:
        return "en"
    return None


def _lang_override():
    """Optional ``lang.txt`` next to this module (contents ``zh`` or ``en``)."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lang.txt")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return _lang_from_tag(fh.read())
    except Exception:
        pass
    return None


def _detect_lang() -> str:
    forced = _lang_override()
    if forced:
        return forced

    try:
        import locale

        hit = _lang_from_tag(locale.getlocale()[0])
        if hit:
            return hit
    except Exception:
        pass

    return "en"


LANG = _detect_lang()

_STRINGS = {
    "en": {
        "category": "OPNodes/Seg",
        "category_control": "OPNodes/Control",
        "node.lllite_apply": "LLLite Apply (SEGS)",
        "node.lllite_clear": "LLLite Clear (SEGS)",
        "node.lllite_preview": "LLLite Hints (SEGS)",
        "node.lllite_apply_remap": "LLLite Apply (SEGS, Remap)",
        "node.control_lora_apply": "Control-LoRA Apply (SEGS)",
        "node.control_lora_apply_model": "Control-LoRA Apply (Anima)",
        "node.sam2dino": "SAM2 + GroundingDINO Segment",
        "desc.lllite_apply": (
            "Attach a ControlNet-LLLite to every SEG so each detected region gets its own "
            "hint image (crop / pose / control image) before sampling."
        ),
        "desc.lllite_clear": "Remove LLLite wrappers from a SEGS chain (debug / reset).",
        "desc.control_lora_apply_model": (
            "Attach a Control-LoRA (pose adapter) to a whole model: works on plain txt2img / "
            "img2img, no third-party node required. Applies the pose embedder (strength) and/or the "
            "adapter's LoRA half (lora_strength, with cross-generation block remapping). Accepts an "
            "encoded control_latent, or an image + vae. Both strengths 0 = model untouched."
        ),
        "desc.control_lora_apply_model": (
            "给整个模型挂一个 Control-LoRA（姿态适配器），普通文生图/图生图也能用，无需第三方节点。"
            "可分别控制姿态 embedder（strength）与适配器的 LoRA 半部（lora_strength，支持跨代 block 重映射）。"
            "可接已编码的 control_latent，或接 image + vae 自动编码。两个强度都为 0 时模型原样返回。"
        ),
        "desc.control_lora_apply": (
            "Attach a Control-LoRA (pose adapter) to every segment: the adapter's LoRA half plus the "
            "zero-init control embedder, so an SEGS detailer redraws the region while the skeleton "
            "holds the pose. No LLLite involved -- chain it with an LLLite node to run both."
        ),
        "desc.control_lora_apply_model": (
            "Attach a Control-LoRA (pose adapter) to a whole model: works on plain txt2img / "
            "img2img, no third-party node required. Applies the pose embedder (strength) and/or the "
            "adapter's LoRA half (lora_strength, with cross-generation block remapping). Accepts an "
            "encoded control_latent, or an image + vae. Both strengths 0 = model untouched."
        ),
        "desc.control_lora_apply_model": (
            "给整个模型挂一个 Control-LoRA（姿态适配器），普通文生图/图生图也能用，无需第三方节点。"
            "可分别控制姿态 embedder（strength）与适配器的 LoRA 半部（lora_strength，支持跨代 block 重映射）。"
            "可接已编码的 control_latent，或接 image + vae 自动编码。两个强度都为 0 时模型原样返回。"
        ),
        "desc.control_lora_apply": (
            "给每个分区挂一个 Control-LoRA（姿态适配器）：文件里的 LoRA 部分 + 零初始化 control "
            "embedder，让 Detailer 重绘的同时由骨架锁住姿态。不含 LLLite；要和 LLLite 同时用就在后面"
            "再串一个 LLLite 节点。"
        ),
        "desc.lllite_apply_remap": (
            "Like LLLite Apply (SEGS), but rewrites the block indices inside the LLLite "
            "weights so a 28-layer LLLite can drive a 40-layer (Anima 2.9B) or 52-layer "
            "(3.8B) model. Older -> newer only."
        ),
        "desc.lllite_preview": "Preview the hint image each SEG would receive.",
        "desc.sam2dino": (
            "Text-prompted segmentation: GroundingDINO boxes -> SAM2 masks. Works with "
            "transformers>=5 (restores the helpers it removed) and needs comfyui-sam2 installed."
        ),
        "tip.hint_mode": "crop = the segment's own crop; pose = run the SEGS preprocessor on it; control_image = crop from a global control image.",
        "tip.lllite_target": "Auto reads the block count from the connected model; override it if detection is wrong.",
        "tip.lllite_source": "Generation the LLLite was trained on. Auto reads it from the weight keys.",
        "tip.lllite_extend": "Extend mode only: scales the copied .up.* tensors for the newly inserted blocks.",
        "tip.seg_processor": "Only used when hint_mode=pose. e.g. DWPreprocessor Provider (SEGS) from Inspire.",
        "tip.enable_control_lora": "Also attach a Control-LoRA (pose adapter) to every segment, so redraw and pose control run together.",
        "tip.control_lora_name": "Control-LoRA file from models/loras; it must contain control_embedder weights.",
        "tip.control_lora_strength": "Strength of the pose signal added at the patch-embed output.",
        "tip.control_lora_lora_strength": "Strength of the file's LoRA half; 0 = pose embedder only.",
        "tip.control_lora_hint": "Where the pose skeleton comes from: SEGS preprocessor / global control image / the LLLite hint itself.",
        "tip.ctrl_vae": "Required by the control-LoRA backend to encode the skeleton into a control latent.",
        "tip.ctrl_latent": "Already encoded control latent (e.g. VAEEncode of a skeleton). Takes precedence over image.",
        "tip.ctrl_pose_strength": "Pose signal strength (the control_embedder half). 0 disables pose control.",
        "tip.pose_ramp": "Where in the sampling run the pose applies (0 = start, 1 = end). Setting e.g. 0.0-0.6 keeps the pose out of the final detail steps, which otherwise adds grain / muddy colour.",
        "tip.ctrl_pose_strength": "姿态信号强度（control_embedder 那一半）。0 = 关闭姿态控制。",
        "tip.pose_ramp": "姿态在采样进程的哪一段生效（0 = 起始，1 = 结束）。设成 0.0-0.6 可让收尾阶段不再被姿态扰动，否则容易出现噪点、颜色发脏。",
        "tip.ctrl_model_lora_strength": "Also apply the adapter's LoRA half here (0 = leave it to LoraLoaderModelOnly). On 2.9B/3.8B it is remapped automatically.",
        "tip.ctrl_image": "Raw skeleton/control image; encoded with the vae input (saves a VAEEncode node).",
        "hint.no_segs": "no SEGS",
        "hint.seg_count": "{n} SEG",
        "err.no_anima_lllite": "ComfyUI-Anima-LLLite not found in custom_nodes.",
        "err.bad_lllite": "LLLite file not found: {name}",
        "err.not_lllite": (
            "{name} is not a ControlNet-LLLite file (no lllite.* metadata). "
            "A plain LoRA / traditional ControlNet cannot be used here."
        ),
        "err.remap_direction": "cannot remap {src} -> {tgt}: only older -> newer is supported",
        "err.pose_needs_preprocessor": "hint_mode=pose requires an segs_preprocessor input.",
        "err.control_needs_image": "hint_mode=control_image requires a control_image input.",
        "err.ctrl_needs_vae": "enable_control_lora requires a vae input (the skeleton must be encoded).",
        "err.ctrl_bad_file": "Control-LoRA file not found: {name}",
        "err.ctrl_no_embedder": "{name} carries no control_embedder weights; it is not a control-LoRA.",
        "err.ctrl_needs_preprocessor": "control_lora_hint=segs_preprocessor requires an segs_preprocessor input.",
        "err.ctrl_control_image": "control_lora_hint=control_image requires a control_image input.",
        "err.ctrl_need_source": "needs control_latent, or image + vae.",
        "err.ctrl_image_needs_vae": "image needs a vae input to be encoded.",
        "err.ctrl_need_source": "需要 control_latent，或者 image + vae。",
        "err.ctrl_image_needs_vae": "接 image 时必须同时接 vae 才能编码。",
        "err.ctrl_lora_no_match": "control-LoRA: none of {n} LoRA keys matched this model (use the Remap node, or set control_lora_lora_strength=0).",
        "err.ctrl_dim_mismatch": "control embedder outputs {w} channels but this model has {m}.",
        "err.unknown_source": "cannot determine the generation of the weights file; set source_model manually",
    },
    "zh": {
        "category": "OPNodes/Seg",
        "category_control": "OPNodes/Control",
        "node.lllite_apply": "应用 LLLite（SEGS 分区）",
        "node.lllite_clear": "清除 LLLite（SEGS）",
        "node.lllite_preview": "LLLite 提示图预览（SEGS）",
        "node.lllite_apply_remap": "应用 LLLite（SEGS 分区 · 跨代重映射）",
        "node.control_lora_apply": "应用 Control-LoRA（SEGS 分区 · 姿态）",
        "node.control_lora_apply_model": "应用 Control-LoRA（Anima 姿态）",
        "node.sam2dino": "SAM2 + GroundingDINO 分区",
        "desc.lllite_apply": (
            "给每个 SEG 挂一个 ControlNet-LLLite，让每个检测区域在采样前拥有自己的提示图"
            "（本块裁剪 / 姿态 / 控制图），解决全局图被压缩到小块导致的几何错位。"
        ),
        "desc.lllite_clear": "清除 SEGS 链上的 LLLite wrapper（调试 / 重置）。",
        "desc.control_lora_apply_model": (
            "Attach a Control-LoRA (pose adapter) to a whole model: works on plain txt2img / "
            "img2img, no third-party node required. Applies the pose embedder (strength) and/or the "
            "adapter's LoRA half (lora_strength, with cross-generation block remapping). Accepts an "
            "encoded control_latent, or an image + vae. Both strengths 0 = model untouched."
        ),
        "desc.control_lora_apply_model": (
            "给整个模型挂一个 Control-LoRA（姿态适配器），普通文生图/图生图也能用，无需第三方节点。"
            "可分别控制姿态 embedder（strength）与适配器的 LoRA 半部（lora_strength，支持跨代 block 重映射）。"
            "可接已编码的 control_latent，或接 image + vae 自动编码。两个强度都为 0 时模型原样返回。"
        ),
        "desc.control_lora_apply": (
            "Attach a Control-LoRA (pose adapter) to every segment: the adapter's LoRA half plus the "
            "zero-init control embedder, so an SEGS detailer redraws the region while the skeleton "
            "holds the pose. No LLLite involved -- chain it with an LLLite node to run both."
        ),
        "desc.control_lora_apply_model": (
            "Attach a Control-LoRA (pose adapter) to a whole model: works on plain txt2img / "
            "img2img, no third-party node required. Applies the pose embedder (strength) and/or the "
            "adapter's LoRA half (lora_strength, with cross-generation block remapping). Accepts an "
            "encoded control_latent, or an image + vae. Both strengths 0 = model untouched."
        ),
        "desc.control_lora_apply_model": (
            "给整个模型挂一个 Control-LoRA（姿态适配器），普通文生图/图生图也能用，无需第三方节点。"
            "可分别控制姿态 embedder（strength）与适配器的 LoRA 半部（lora_strength，支持跨代 block 重映射）。"
            "可接已编码的 control_latent，或接 image + vae 自动编码。两个强度都为 0 时模型原样返回。"
        ),
        "desc.control_lora_apply": (
            "给每个分区挂一个 Control-LoRA（姿态适配器）：文件里的 LoRA 部分 + 零初始化 control "
            "embedder，让 Detailer 重绘的同时由骨架锁住姿态。不含 LLLite；要和 LLLite 同时用就在后面"
            "再串一个 LLLite 节点。"
        ),
        "desc.lllite_apply_remap": (
            "与「应用 LLLite（SEGS 分区）」相同，但会改写 LLLite 权重里的 block 索引，"
            "让 28 层 LLLite 也能驱动 40 层（Anima 2.9B）或 52 层（3.8B）模型。仅支持老 → 新。"
        ),
        "desc.lllite_preview": "预览每个 SEG 实际会拿到的提示图。",
        "desc.sam2dino": (
            "文本提示分区：GroundingDINO 出框 → SAM2 出掩码。兼容 transformers>=5"
            "（自动补回被移除的接口），需要已安装 comfyui-sam2。"
        ),
        "tip.hint_mode": "crop=用本块裁剪图；pose=对本块跑 SEGS 预处理器；control_image=从全局控制图按区域裁块。",
        "tip.lllite_target": "Auto = 从所接模型读层数；探测不准时手动指定。",
        "tip.lllite_source": "这颗 LLLite 训练时用的代。Auto = 从权重 key 里读。",
        "tip.lllite_extend": "仅 Extend 模式：对新插入块复制的 .up.* 权重做缩放。",
        "tip.seg_processor": "仅 hint_mode=pose 时使用，例如 Inspire 的 DWPreprocessor Provider (SEGS)。",
        "tip.enable_control_lora": "额外给每个分区挂一个 Control-LoRA（姿态适配器），重绘与姿态控制同时生效。",
        "tip.control_lora_name": "models/loras 里的 Control-LoRA 文件，必须含 control_embedder 权重。",
        "tip.control_lora_strength": "姿态信号强度（加到 patch-embed 输出上）。",
        "tip.control_lora_lora_strength": "文件里 LoRA 那部分的强度；0 = 只用姿态 embedder。",
        "tip.control_lora_hint": "姿态骨架来源：SEGS 预处理器 / 全局控制图 / 与 LLLite 同一个 hint。",
        "tip.ctrl_vae": "control-LoRA 后端需要它把骨架图编码成 control latent。",
        "tip.ctrl_latent": "已编码的 control latent（例如骨架图过 VAEEncode）。优先于 image。",
        "tip.ctrl_pose_strength": "Pose signal strength (the control_embedder half). 0 disables pose control.",
        "tip.ctrl_pose_strength": "姿态信号强度（control_embedder 那一半）。0 = 关闭姿态控制。",
        "tip.ctrl_model_lora_strength": "顺便在这里应用适配器的 LoRA 半部（0 = 交给 LoraLoaderModelOnly）。2.9B/3.8B 上会自动重映射。",
        "tip.ctrl_image": "原始骨架/控制图；用 vae 输入自动编码（省掉一个 VAEEncode）。",
        "hint.no_segs": "无 SEGS",
        "hint.seg_count": "{n} 块",
        "err.no_anima_lllite": "未在 custom_nodes 中找到 ComfyUI-Anima-LLLite。",
        "err.bad_lllite": "找不到 LLLite 文件：{name}",
        "err.not_lllite": "「{name}」不是 ControlNet-LLLite 文件（没有 lllite.* 元数据）。普通 LoRA、传统 ControlNet、以及 control-LoRA（例如带 control_embedder 的 Anima 姿态模型）都不能在这里用——AnimaControl 需要它自己的节点。",
        "err.remap_direction": "无法从 {src} 重映射到 {tgt}：只支持老 → 新",
        "err.pose_needs_preprocessor": "hint_mode=pose 需要连接 segs_preprocessor。",
        "err.control_needs_image": "hint_mode=control_image 需要连接 control_image。",
        "err.ctrl_needs_vae": "enable_control_lora 需要连接 vae（骨架图必须编码成 latent）。",
        "err.ctrl_bad_file": "找不到 Control-LoRA 文件：{name}",
        "err.ctrl_no_embedder": "「{name}」没有 control_embedder 权重，不是 Control-LoRA。",
        "err.ctrl_needs_preprocessor": "control_lora_hint=segs_preprocessor 需要连接 segs_preprocessor。",
        "err.ctrl_control_image": "control_lora_hint=control_image 需要连接 control_image。",
        "err.ctrl_need_source": "needs control_latent, or image + vae.",
        "err.ctrl_image_needs_vae": "image needs a vae input to be encoded.",
        "err.ctrl_need_source": "需要 control_latent，或者 image + vae。",
        "err.ctrl_image_needs_vae": "接 image 时必须同时接 vae 才能编码。",
        "err.ctrl_lora_no_match": "control-LoRA：{n} 个 LoRA 权重没有一个匹配当前模型（改用 Remap 节点并选对 target_model，或把 control_lora_lora_strength 设为 0）。",
        "err.ctrl_dim_mismatch": "control embedder 输出 {w} 通道，但当前模型是 {m}。",
        "err.unknown_source": "无法判断权重文件的代，请手动设置 source_model",
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
