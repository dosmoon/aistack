---
title: NVIDIA Parakeet TDT 在消费级 GPU 上跑长音频
slug: parakeet-on-consumer-gpu
date: 2026-05-07
tags: [asr, parakeet, nemo, nvidia, 8gb-vram, long-audio]
---

# NVIDIA Parakeet TDT 在消费级 GPU 上跑长音频

> **TL;DR** 在 8 GB 消费卡（RTX 4060 Laptop 等）上跑 Parakeet TDT 处理 50 分钟以上的长音频，需要三组配置同时到位：local attention、subsampling chunking、**不**碰 decoding strategy 的 `preserve_alignments`。这个组合在 NVIDIA 任何一份单独的文档里都不完整，今天通过实测把缺失的连带效应补上。

## 谁该读这篇

- 在 8–12 GB 消费 GPU 上自部署 Parakeet TDT 0.6B v2 / v3
- 长音频（30 分钟以上）出现 OOM、慢得离谱、或 segment 时间戳返回空数组
- 想理解为什么 aistack 的 `aistack/asr/parakeet.py` 里要这么调几个开关

## 上下文

NVIDIA Parakeet TDT 是当前消费级 ASR 的最佳选择之一——50 分钟英文政论演说在 RTX 4060 Laptop（8 GB VRAM）上 cache 命中跑 62 秒，RTF ≈ 0.021，约 **80× 实时**。但默认配置跑不动；需要三层独立的旋钮配合，且每层都有陷阱。

下面给出的是**经过 50 分钟真音频实测**（卢比奥 5/5 伊朗问题记者会，47.8 MB mp3）验证的可工作组合。

## 可工作的配置组合

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

# (1) 注意力阶段：避免 attention 矩阵 O(N²) 爆炸
model.change_attention_model("rel_pos_local_attn", [256, 256])

# (2) Subsampling 阶段：避免 downsampling 一次性吞下整段音频
model.change_subsampling_conv_chunking_factor(1)   # 1 = 自动选 chunk 大小

# 注意：DO NOT 调用 change_decoding_strategy 把 preserve_alignments 开成 True
# 见后面"陷阱 #3"

model.eval()

# 转写
results = model.transcribe([wav_path], timestamps=True, num_workers=0, batch_size=1)
```

`num_workers=0` 是 Windows 下避免 NeMo 内部 manifest.json 与 DataLoader 子进程的 file-lock 竞争——这是 NeMo 自己的 `examples/asr/transcribe_speech.py` 的默认值，不是我们发明的。

## 三个旋钮分别解决什么

### 旋钮 #1：`change_attention_model("rel_pos_local_attn", [256, 256])`

**默认问题**：Parakeet TDT 默认使用全自注意力（rel_pos），算量是 **O(N²)**，N = audio token 数。在 8 GB 卡上 2-3 分钟音频就 OOM。NVIDIA 的官方上限：A100 80GB + 全注意力 = 24 分钟音频。换算到 8 GB 卡 ≈ 几分钟。

**怎么解**：换成 Longformer 风格的局部注意力——每个位置只看左右 256 帧（在 80 ms/帧的速率下 ≈ ±20 秒上下文），算量降到 **O(N × 256)**，对音频长度变成线性。

**代价**：约 1–3% WER（模型失去全局上下文的能力，跨句指代/共指消解略差）。8 GB 卡值得这个交换。

**官方依据**：
- Parakeet HF model card：`change_attention_model("rel_pos_local_attn", att_context_size=...)` 是官方推荐路径
- FastConformer 论文 + NVIDIA blog：local attention 的设计目标就是支持小卡长音频

### 旋钮 #2：`change_subsampling_conv_chunking_factor(1)`

**默认问题**：FastConformer encoder 第一层是 subsampling（4× downsampling，把原始声学帧打成可被 attention 处理的 token）。这一步**默认会把整段音频的中间激活一次性塞进 VRAM**——50 分钟音频在这一层就能炸 8 GB，注意力都还没轮上跑。NVIDIA 自己 FastConformer 研究博客原话："the downsampling module at the earliest stage can take more memory than the actual forward pass since it directly operates on the audio sequence which may not fit in memory for very long audio files"。

**怎么解**：调用 `change_subsampling_conv_chunking_factor(1)` 让 subsampling 改成分块处理，每次只算一段 chunk 的激活，处理完释放再处理下一块。`1` 表示自动选 chunk 大小（推荐值）。

**显式代价**：几乎没有——只影响显存分配模式，不改变计算结果，WER 不受影响。

**隐式收益（这里是**官方文档没写**的关键发现）**：在长音频 + local attention 的组合下，开 chunking **顺带修复了 NeMo 长音频路径上 segment timestamps 输出空数组的 bug**——这是 NeMo open issue 类型问题（见 [#14714](https://github.com/NVIDIA-NeMo/NeMo/issues/14714) 周边讨论）。

**实测对比**（同一份 50 分钟卢比奥音频，aistack `/v1/audio/transcriptions`）：

| 配置 | NeMo 原生返回的 segment 数 | 备注 |
|---|---|---|
| 仅 local attention（无 chunking） | **1**（整段 0–2986s 一段） | 长音频 segment 时间戳路径触发 bug |
| local attention + chunking | **788**（句子级，标点边界正确） | NeMo punctuation-aware 切分恢复正常 |

显然旋钮 #2 同时解决了两个问题，**但 NVIDIA 任何一份文档都没说**它有这第二个作用。这是本笔记最有价值的发现之一。

**官方依据**（说了什么）：
- [NeMo ASR API Reference](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/api.html)：`subsampling_conv_chunking_factor` 是可选参数，可设为 2 的幂 / 1（自动） / -1（禁用），仅支持 depthwise separable conv 的 subsampling 模型
- [Parakeet HF model card #15](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/15)：8 GB 卡推荐配置就是这两个一起用

**官方没说的**：开了它会顺带修 segment timestamps 路径。这条要靠实测才能发现。

### 旋钮 #3：`change_decoding_strategy(preserve_alignments=True)` ── **不要碰**

研究 NeMo 文档时会看到一段建议，大意是"想要 NeMo 自己输出 segment timestamps，应该显式调用 `change_decoding_strategy` 配置 `preserve_alignments=True / compute_timestamps=True / segment_seperators=['.','?','!']`"。听起来很合理。**这是陷阱**。

**实测代价**（同一台 RTX 4060 Laptop + 同一段 50 分钟 mp3）：

| 配置 | VRAM | Windows Shared System Memory | 总占用 | 50min cache-hit 推理时长 |
|---|---|---|---|---|
| 不调 `change_decoding_strategy` | 8 GB | 10 GB | 18 GB | **62 s** |
| 调了，`preserve_alignments=True` | 8 GB | **30 GB** | **38 GB** | **≥120 s（客户端 timeout）** |

多出来的 20 GB 工作集**从 VRAM 溢出到 Windows 共享系统内存（PCIe 路径）**。PCIe 带宽 ≈ GDDR6 的 1/30，每次 GPU kernel 触碰这部分内存都要走总线，半个工作集变成 IO bound——纯计算时间没变，但 IO 开销翻倍，最终 2x 减速。

**机制（实测后从源码反推）**：

1. NeMo 源码注释明示：`preserve_alignments is not implemented for Frame-Looping + CUDA graphs` —— RNNT/TDT 解码器从 CUDA graph 快速路径**回退**到 Python 循环路径，每个 decoder step 不再复用工作区张量
2. `preserve_alignments=True` 让 hypothesis 保留每帧的对齐 logits 张量，长度等于声学帧数 T。50 分钟 ≈ 75000 帧 × 词表大小 V+1 ≈ 1000 × 4 字节 = 数百 MB 单独 alignment 数据
3. (1) 与 (2) 复合，加上禁用 CUDA graph 后中间张量不复用的累积，最终工作集膨胀到 +20 GB

**和 NeMo issue #14714 的关系**：那个 OPEN 的 issue 报的是 `preserve_alignments=True` + parakeet-tdt-0.6b-v3 + timestamps 路径下"Boolean value of Tensor"运行时错误。我们没遇到崩溃但遇到了同根因的性能塌方——属同一 bug 类。

**正确做法**：不调用 `change_decoding_strategy`。`transcribe(timestamps=True)` 就足够拿到 word-level 时间戳；segment timestamps 已经被旋钮 #2 修好。如果哪天 NeMo 长音频还是吐空 segment（应不会），后处理一层 word→sentence 切分（拿 word 时间戳按句末标点 + 静音间隙合并）作为兜底比开 `preserve_alignments` 安全得多。

## 综合性能基线

机型：RTX 4060 Laptop（8 GB VRAM）+ i9 13 代 + 64 GB DDR5 + Windows 共享 GPU 内存上限 31 GB，torch 2.7.1+cu126，cuDNN 9.7.1，NeMo 2.7+。

完整端到端基线（含 ffmpeg 转码、模型推理、word/segment 时间戳计算、verbose_json 序列化、HTTP 响应回传）：

| 音频长度 | 冷启动 | Cache 命中（良好状态） | RTF（最佳） |
|---|---|---|---|
| ~80 秒 | ~25 秒 | ~3 秒 | ~0.04 |
| 4 分钟 | ~37 秒 | ~13 秒 | 0.05 |
| 12 分钟 | ~30 秒 | **5.7 秒** | **0.008** |
| 17 分钟 | ~12 秒 | ~10 秒 | 0.010 |
| 25 分钟 | ~52 秒 | 13-20 秒 | 0.009 |
| 50 分钟 | ~120 秒 | 60-80 秒 | 0.021 |
| 99 分钟 | — | ~490 秒 | 0.082（共享内存触顶）|

短音频 RTF 偏高（4 min 0.05）是 fixed overhead（ffmpeg 转码 + JSON 序列化 + HTTP 传输）摊不下去，不是 GPU 不行。中长音频（12-25 min）才看到 GPU 真实速度，**RTF 0.008 ≈ 125× 实时**是本机配置下的稳态最优。

## 请求间内存动力学

> 本节是 aistack 团队 2026-05-07 实测发现的工程现象，整理成机制说明。NVIDIA / PyTorch / NeMo 任何官方文档都没把这套放在一起讨论。

### 现象

**同一段音频在不同前置请求历史下，wall time 漂移 2-4×。最坏情况比冷启动还慢**。

实测对照（同一段 25 min 音频，aistack `/v1/audio/transcriptions`）：

| Warm-up 路径 | wall #1 | wall #2 | reserved peak |
|---|---|---|---|
| 冷 → 50min × 2 → 25min | 20 s | 20 s | 24.7 GB |
| 冷 → 12min × 2 → 25min | 15 s | **13 s** | 14.5 GB |
| 冷 → 25min × 3（无前置） | 52 s | 27 s | 13.4 GB |

实测对照（50 min 音频在不同上下文下）：

| 场景 | wall |
|---|---|
| 50 min cache 命中 baseline | 69 s |
| 25 min × 1 后跑 50 min | **175 s**（比冷启动还慢 56 s）|
| aistack 重启冷启动 50 min | 119 s（含 ~25 s 模型加载）|

### 机制

来自 PyTorch 官方文档 + ZDevito 的 caching allocator 详解 + NeMo / Parakeet TDT 架构分析：

**1. PyTorch 维护一个 GPU 内存池（caching allocator）**

上次请求的张量释放后，**block 留在池里复用**，不还给 CUDA driver。设计初衷是避免 `cudaFree`（同步设备调用，会打断 CUDA-CPU pipeline）。

**2. 池子的形状由上次请求的工作量决定**

- 上次跑 50min Rubio 留下 24 GB 池子，里面是"50min-shape"的 free block
- 下次 25min 来要新 shape：cuDNN 重新选 conv 算法、申请新 size workspace
- 必须 split 老 block 或申请新 block → bookkeeping + 同步开销

**3. 不同 shape 兼容程度差很多**

- 12min → 25min：cuDNN 算法基本同源，pool 直接扩展，wall 最低（13s）
- 50min → 25min：cuDNN 重选算法，pool 碎片化，wall 中等（20s）
- 冷 → 25min：完全重建 pool + 模型加载，wall 最高（52s 含 25s 加载）

**4. Parakeet TDT 因架构特点更敏感**

- TDT decoder 每 timestep 跑两个网络（prediction + joint），duration 决定跳几帧 → Python 控制流频繁同步 GPU
- FastConformer 8x subsampling + local attention 的 cuDNN workspace 较大且与输入长度强相关
- 比简单 CTC 模型 stream event 密度高，pool 形状变化空间也大

### 实战影响

**正向**：连续跑同形状（同长度同模型同语种）音频时，cache 复用红利明显，最快可达 RTF 0.008（12min Trump 重复跑实测）。

**负向**：混合长度时 wall 不可预测漂移；最坏可超过冷启动，因为"被污染的状态" + "pool 重排开销" > "模型加载开销"。

### 不能做什么

我们尝试过 `torch.cuda.empty_cache()` 自动清池来"修复碎片化"——失败：

- 第一次请求后调 empty_cache，第二次请求 hang 死在 worker thread 内部 CUDA 调用
- GPU 锁永远不释放，整个 ASR 端点不可用，必须 kill 进程
- 机制：`empty_cache()` 内部对每个 free block 调 `cudaFree`；`cudaFree` 是同步调用（NVIDIA 文档定义），等所有 stream 完成
- NeMo / cuDNN 内部 stream 上若有未完成 event，同步可能与 cuDNN descriptor 生命周期形成死锁
- 不是 Windows 特有，是 PyTorch 维护者明确反对的 empty_cache 反模式（参 [zdevito 详解](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)）

### 能做什么

1. **批量同长度同语种工作量**——性能最优、最稳定
2. **混合工作量先批一类再切一类**——避免高频 shape 切换
3. **切换工作量类别前 kill aistack 重启**（25-30 秒模型加载成本，比"被污染状态"快得多）
4. **接受 wall 漂移作为本地 ASR 的工程现实**——写文档让客户端知道，不要承诺严格 SLA

aistack 当前**不会自动检测 + 修复**这个现象——经实测后认为没有可靠的启发式（变量太多、规则不存在），且自动修复路径已被证伪。

## 文档为什么"天书"

NVIDIA 关于这套配置的信息分布在 7 层：

| 层级 | 信息源 | 说了什么 | 缺什么 |
|---|---|---|---|
| 1 | NeMo User Guide | "long audio inference is supported" | 怎么配 |
| 2 | NeMo API Reference | 参数语义、合法值范围 | 副作用、组合规则 |
| 3 | FastConformer 研究博客 | 设计动机（subsampling 内存问题） | 操作细节 |
| 4 | NeMo 源码 docstring | "preserve_alignments not implemented for CUDA graphs" 等关键警告 | 不量化 |
| 5 | Parakeet HF model card | A100 + 全注意力上限、推荐 API | 消费卡场景 |
| 6 | HF model card discussion | 社区实测的 8 GB 配方 | 非官方，不保证 |
| 7 | GitHub issue tracker | open bug 现象（如 #14714） | 没修 |

**没有任何一层独自完整**。例如：
- 看 Layer 1 "默认 1" 会以为不用调用，实际必须显式调
- 看 Layer 4 知道 preserve_alignments 关 CUDA graph，但不知道会撑爆 30 GB
- 看 Layer 6 知道哪两个开关要开，但不知道为什么

我们这篇笔记的目标就是**横切 7 层把缺失的连带效应补上**——尤其是 Layer 5（旋钮 #2 修 segment timestamps）和 Layer 7（preserve_alignments 真实代价是 20 GB）。

## Open questions

- **chunking → segment-timestamp 修复机制**：经实测确认这两件事相关，但 NeMo 源码里具体哪段逻辑因 chunking 启用而走通的，我们还没逐行确认。猜测是 chunking 让 subsampling 输出在边界处插入了某种 marker，让下游 segment 检测器能正常切分；但只是猜测。
- **att_context_size 的最优值**：当前用 `[256, 256]`（NVIDIA model card 推荐），但我们没测过 `[128, 128]`（HF discussion #15 用的）会不会显存更省、WER 更接近全注意力。在 4060 Laptop 上 256 已经够用，没去调。
- **比 50min 更长的音频边界**：本笔记基线测到 50 分钟。NVIDIA 官方说法是"local attention + chunking 配 8 GB 可达 11 小时"，但我们没实际跑过那个上限。
- **为什么"默认 1"还要显式调用**：NeMo API Reference 说 `subsampling_conv_chunking_factor` 默认值是 1（auto），但我们实测**不显式调用就触发 segment-timestamp bug**，调了就修好。猜测 `change_attention_model` 在内部重置了 subsampling 状态，强制需要重新 set。值得有人翻 NeMo 源码确认。

## 配套代码

aistack 中这套配置的实现见 [`aistack/asr/parakeet.py`](https://github.com/dosmoon/aistack/blob/main/aistack/asr/parakeet.py)：

- `_get_model()` 调用 `_maybe_switch_to_local_attention()` 与 `_maybe_enable_subsampling_chunking()`
- `_configure_timestamp_decoding()` 函数体保留作未来 opt-in 路径，但 `_get_model` **不**调用它（含详细注释说明 20 GB 实测代价）
- `_segments_from_words()` 作为防御兜底，在 NeMo 回归吐空 segment 时接管

## 引用

- [NeMo ASR Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [NeMo ASR API Reference (24.07)](https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/asr/api.html)
- [Parakeet TDT 0.6B v3 — HuggingFace model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Parakeet TDT 0.6B v2 — HF Discussion #15（8 GB 长音频可工作配方）](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/15)
- [NVIDIA Research — Fast Conformer with Linearly Scalable Attention](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/)
- [NeMo Issue #14714 (OPEN)](https://github.com/NVIDIA-NeMo/NeMo/issues/14714) — `preserve_alignments=True` 在 parakeet-tdt-0.6b-v3 timestamps 路径下的失败
- [NeMo PR #10950](https://github.com/NVIDIA-NeMo/NeMo/pull/10950) — Timestamps to transcribe（segment_seperators 设计来源）

## 致谢

aistack 团队成员的实测数据（特别是 8 GB VRAM + 30 GB shared RAM 这组对照），让 `preserve_alignments=True` 的真实代价从"docstring 暗示的不兼容"变成了"量化到 GB 的内存溢出"。如果只读源码注释，我们会把代价低估两个数量级。

---

*这是 dosmoon aistack 项目研究笔记之一。其他笔记见 [README.md](README.md)。*
