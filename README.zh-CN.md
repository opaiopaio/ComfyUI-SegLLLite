# ComfyUI-SegLLLite

> **OPNodes** 节点包成员 · MIT
> 🧩 **本包**：让 Impact/LanPaint 的 **SEGS Detailer 兼容 ControlNet-LLLite** —— 每个检测区域用自己的 LLLite 提示图。另含文本提示的 SAM2+GroundingDINO 分区
> 🎯 **姊妹包**：[ComfyUI-LoRAStackAutoTriggers](https://github.com/opaiopaio/ComfyUI-LoRAStackAutoTriggers) —— LoRA 堆叠 / 自动触发词 / Anima 跨代映射

[English](README.md) | **中文**

---

## 这是什么 / 不是什么

Impact Pack 的 Detailer 会把每个检测区域**裁剪出来单独采样**；而 ControlNet-LLLite（Anima/DiT
用）是**捕获整张图**再重采样到当前 latent 尺寸。于是裁出的 576×576 手部小块里被塞进了整张图 →
提示图与 latent **几何错位**。这是数据流问题，不是参数能调的。

本包补上缺失的那层胶水：一个符合 Impact `control_net_wrapper` 协议的 wrapper。该协议的两个方法
**在逐块循环内部**被调用、且拿到的就是**已裁剪**的图，所以我们只需存下这块图、用它去 patch 模型，
提示图与 latent 就天然 1:1 对齐。

本包**不实现** LLLite 或 SAM2 本身；也**不链接任何 GPL 代码**——只 duck-type Impact 的方法名，
从不 import，因此保持 MIT。

## 节点

| 节点 | 作用 | 依赖 |
|---|---|---|
| **应用 LLLite（SEGS 分区）** | 给每个 SEG 挂 LLLite；提示图来源可选 `image`（本块裁剪）/ `pose`（对本块跑 SEGS 预处理器）/ `control_image`（从全局控制图裁块）；支持链式叠加 | torch（ComfyUI 自带）+ Anima-LLLite |
| **应用 LLLite（SEGS 分区 · 跨代重映射）** | 同上，**外加跨代 block 重映射**：28 层 LLLite 也能驱动 40 层（Anima 2.9B）/ 52 层（3.8B）模型 | torch + Anima-LLLite |
| **清除 LLLite（SEGS）** | 清空 SEGS 链上的 wrapper | torch |
| **LLLite 提示图预览（SEGS）** | 预览每块实际会拿到的提示图 | torch |
| **SAM2 + GroundingDINO 分区** | 文本提示 → 框（GroundingDINO）→ 掩码（SAM2）；内置兼容层，**transformers ≥ 5 也能跑** | `comfyui-sam2`、`transformers` |

缺哪个可选依赖，就只禁用相关节点；**ComfyUI 永远能正常启动**。

## 安装

1. 把本文件夹复制到 `ComfyUI/custom_nodes/ComfyUI-SegLLLite`
2. 重启 ComfyUI
3. 可选：LLLite 节点需要 `ComfyUI-Anima-LLLite`；分区节点需要 `comfyui-sam2`

**没有** `pip install` 步骤：重依赖全部惰性 import。

## 快速上手（分区姿态控制）

```
LoadImage ─► [SAM2 + GroundingDINO 分区]  prompt "hand"，threshold 0.3
          ─► MaskToSEGS                              （Impact Pack）
          ─► [应用 LLLite（SEGS 分区）]              hint_mode=pose，
             segs_preprocessor = DWPreprocessor Provider (SEGS)   ← Inspire
             lllite_name = 你的 anima 姿态 LLLite，strength 0.8
          ─► [应用 LLLite（SEGS 分区）]              hint_mode=image，分块细节 LLLite（链式第二级）
          ─► DetailerForEachPipe / Detailer Lanpaint (SEGS/pipe)
          ─► SaveImage
```

在 Detailer 之前接一个 **LLLite 提示图预览（SEGS）**，就能确认每块拿到的是自己的提示图——这正是本包的全部意义。

## Anima 跨代重映射（LLLite）

ControlNet-LLLite 在结构上就是一个 LoRA 家族：每块有 `lllite_dit_blocks_<N>_<atom>.down/.mid/.up`
+ 每模块的 `depth_embed`，再加一个全局共享的 `lllite_conditioning1.*`。**block 索引写在 key 里**，
所以同一个索引在不同层数的模型上代表不同块。

Anima 有 28 → 40 → 52 三种深度（插入式加深，新块插在中间）。把 28 层 LLLite 装到 40 层模型上：
当前实现要么直接报 `depth_embed slices missing`，要么（文件里没有 `depth_embed` 时）**静默落到错误
的块上**。**应用 LLLite（SEGS 分区 · 跨代重映射）** 会在加载前改写 key 里的 block 索引：

| 选项 | 含义 |
|---|---|
| `enable_remap` | 关掉 = 与普通节点完全一致 |
| `target_model` | `Auto (detect)` 从所接模型读层数；探测不准时手动指定 |
| `source_model` | `Auto` 从 LLLite 权重 key 推断 |
| `remap_mode` | `Keys only` 只搬索引；`Extend` 额外把源块权重复制到新插入的块 |
| `extend_strength` | 仅 Extend：对复制的 `.up.*` 权重做缩放 |

只支持单向（28→40、28→52、40→52）：老 → 新；反方向直接报错，不会只装一半。

> 没有 `lllite.*` 元数据的文件会被明确拒绝。这点很重要：普通 LoRA（比如"姿势 LoRA"）当初被当
> LLLite 加载时，**一个权重都没匹配上、等于完全没控制**，而且没有任何提示。

### 什么文件能用

**只支持 ControlNet-LLLite**（带 `lllite.*` 元数据、key 形如 `lllite_dit_blocks_<N>_*` 的 safetensors）。
普通 LoRA、传统 ControlNet、以及 **control-LoRA** 都会被明确拒绝——例如用 `control_embedder.*` 实现的
Anima 姿态模型，它需要官方 `AnimaControlApply` 节点，不能从这里驱动。

## 兼容矩阵

| LLLiteSEGS | ComfyUI | Impact Pack | Anima-LLLite | transformers |
|---|---|---|---|---|
| v0.1.x | ≥ 0.4x | ≥ 8.x（可选） | ≥ 1.0（可选） | 任意（≥5 已在运行时修好） |

启动日志会打印语言与已注册节点，例如
`[SegLLLite] loaded nodes: LLLiteApplySEGS, …`

## 语言

节点标题、分类、说明与 tooltip 跟随**系统语言**（中/英）。启动 ComfyUI 前设 `OPNODES_LANG=zh`（或 `en`）可强制指定。
下拉菜单的**取值不翻译**——ComfyUI 会把它存进工作流。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 看不到 LLLite 节点 | `custom_nodes` 里没有 `ComfyUI-Anima-LLLite`；启动日志会写明 |
| 看不到 SAM2/DINO 节点 | 没装 `comfyui-sam2`，或它自身 import 失败（看日志） |
| 分区效果错位 | 用 **LLLite 提示图预览（SEGS）** 检查：`hint_mode=image` 必须显示该块裁剪图，而不是整图 |
| `hint_mode=pose` 报错 | 需要接 SEGS 预处理器（如 Inspire 的 DWPreprocessor Provider） |
| 分区多时很慢 | LLLite 权重按块重复读取；建议控制分区数量 |

## 许可

MIT。Impact Pack（GPL-3.0）**未被 import、未被复制**，只使用其方法名协议。
Anima-LLLite（Apache-2.0）与 comfyui-sam2 属于可选的运行时库，不随本包分发。
