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
| **应用 Control-LoRA（SEGS 分区 · 姿态）** | 给每个分区挂姿态适配器：文件里的 LoRA 部分 + 零初始化 control embedder；不含 LLLite，接在 LLLite 节点后面即可两者同时生效 | torch + Control-LoRA 文件 |
| **应用 Control-LoRA（Anima 姿态）** | 同一内核对整个 **MODEL** 生效（不涉 SEGS）：普通文生图/图生图加姿态控制；可接已编码的 `control_latent`，也可接 `image` + `vae`；`strength=0` 时模型原样返回 | torch + Control-LoRA 文件 |
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

## 同一节点里做姿态控制（control-LoRA）

两个应用节点都可以额外挂一个 **control-LoRA**（Anima 姿态适配器），让同一个分区**同时**做「重绘（LLLite）+ 姿态控制」。两种机制挂在不同层，所以能共存（两个 LLLite 则不行）：

- LLLite → `model_options["model_function_wrapper"]`（单槽）
- control-LoRA → `wrappers["diffusion_model"]`（多槽，DiT forward 内部消费）

| 参数 | 默认 | 说明 |
|---|---|---|
| `enable_control_lora` | 关 | 总开关 |
| `control_lora_name` | — | `models/loras` 里的适配器文件，必须含 `control_embedder.*` |
| `control_lora_strength` | 1.2 | 姿态信号强度（`control_embedder` 那一半） |
| `control_lora_lora_strength` | 1.0 | 文件里 LoRA 那一半；`0` = 只用姿态 embedder |
| `control_lora_hint` | `segs_preprocessor` | 骨架来源：SEGS 预处理器 / 全局控制图 / 与 LLLite 同一个 hint |

开关打开后 `vae` 必接：骨架图要被编码成 control latent。注入通过临时的 `x_embedder` forward hook 完成（`finally` 移除），token 网格对不上时会自动 resize 并警告（奇数裁剪尺寸）。

Anima 2.9B / 3.8B 请用 **应用 LLLite（SEGS 分区 · 跨代重映射）**：它会把 LoRA 那一半的 block 索引一起改写；基础节点只原样加载，匹配不上会明确报错。

> 姿态适配器权重**非商用**（CircleStone NC + NVIDIA Open Model），生成图不受限。

## 普通模型上的姿态控制（不涉 SEGS）

`应用 Control-LoRA（Anima 姿态）` 是同一内核的 `MODEL → MODEL` 版本，用于普通文生图 / 图生图：

```
应用 Control-LoRA（Anima 姿态）
  model            ← 任意 checkpoint（不再需要 LoraLoaderModelOnly）
  control_lora_name = 适配器文件        （下拉只列真正的 control-LoRA）
  strength          = 1.0   姿态 embedder；0 = 关闭
  lora_strength     = 0.0   顺便应用适配器的 LoRA 半部；0 = 交给 LoraLoaderModelOnly
  enable_remap      = true  代不同时对那半部做 block 重映射（28 -> 40 / 52）
  control_latent    ← VAEEncode(骨架图)   （或改用下面的 image + vae）
  [可选] image + vae → 在这里编码，省掉 VAEEncode 节点
```

一个文件、一个节点：适配器的两半都能在这里应用，包括 LoRA 半部的跨代重映射。把 `lora_strength`
设为 0 就回到官方那种「LoRA 由 LoraLoaderModelOnly 加载」的用法。只需要权重文件（非商用，生成图不受限）。

`control_lora_name` 下拉**只列真正含 `control_embedder.*` 的文件**（只读 header，按 mtime 缓存；
40 个 LoRA 约 5 ms）。探测失败时回退到全部 LoRA，由节点给出明确的「不是 control-LoRA」错误。

### 与 ComfyUI-Anima-LLLite 的交互方式

为了在加载前改写 block 索引，`应用 LLLite（SEGS 分区 · 跨代重映射）` 会在**单次 apply 期间**临时替换
Anima-LLLite 模块对象上的 `load_lllite_weights`，并在 `finally` 里还原。不改磁盘上的任何文件，该调用之外
其它节点看不到这次替换，包内其它功能按原样使用对方的实现。

### 什么文件能用

**只支持 ControlNet-LLLite**（带 `lllite.*` 元数据、key 形如 `lllite_dit_blocks_<N>_*` 的 safetensors）。
普通 LoRA、传统 ControlNet、以及 **control-LoRA** 都会被明确拒绝——例如用 `control_embedder.*` 实现的
Anima 姿态模型，它需要官方 `AnimaControlApply` 节点，不能从这里驱动。

## 示例

`examples/quickstart_segs.json` 是最小可跑图，`examples/README.md` 列出它依赖的外部节点
（Impact 的 `MaskToSEGS` + ComfyUI-Anima-LLLite）。

## 兼容矩阵

| LLLiteSEGS | ComfyUI | Impact Pack | Anima-LLLite | transformers |
|---|---|---|---|---|
| v0.1.x | ≥ 0.4x | ≥ 8.x（可选） | ≥ 1.0（可选） | 任意（≥5 已在运行时修好） |

启动日志会打印语言与已注册节点，例如
`[SegLLLite] loaded nodes: LLLiteApplySEGS, …`

## 语言

节点标题、分类、说明与 tooltip 跟随**系统语言**（中/英）。想强制指定，就在包目录里建一个 `lang.txt`
（内容写 `zh` 或 `en`），重启 ComfyUI 生效。**本包不读取任何环境变量**。
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
