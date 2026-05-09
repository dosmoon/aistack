---
title: 消费级 GPU 本地 ASR 性能基线
date: 2026-05-07
updated: 2026-05-08
tags: [asr, parakeet, benchmark, consumer-gpu, self-hosting]
sidebar:
  order: 1
---

# 消费级 GPU 本地 ASR 性能基线

> **TL;DR** RTX 4060 Laptop（8 GB VRAM，消费级笔记本独显）跑 NVIDIA Parakeet TDT 0.6B v3：稳态最佳 RTF 0.008（约 125× 实时），实际单次请求的 wall time 因 GPU 内存状态而漂移 2-4×（取决于前一次请求形态）。**最坏情况下，wall time 超过冷启动 30%+**。这篇笔记给出可独立复现的硬件、软件、负载、性能数据 + 请求间内存动力学的实测对照，读者自行决定本地 ASR 是否适合自己的场景。

> **2026-05-08 增补**：本笔记表格里的 wall time（特别是 50/99 min 上的 60-490 秒和 2-4× 漂移）是 aistack **单次 NeMo 调用** 的数据，反映"未经包装的 Parakeet 在 8 GB 卡的真实表现"。aistack 当前生产路径已切到**应用层 12 分钟切片 + LCS 缝合**，跨任意时长 RTF 稳定在 0.007-0.009、reserved VRAM 锁定 7.8 GB、wall 不再漂移（97 min 音频 44 秒）。详细机制和实测数据见 [parakeet-on-consumer-gpu.md](parakeet-on-consumer-gpu.md) 的"应用层切片"章节。本笔记的数字保留作为"裸 NeMo"对照基线。

## 这篇是干什么的

aistack 的产品立场是：**对于本地 GPU 性能足够的 AI 任务，没必要走云端 API**。但"足够"是个具体的工程问题，不是口号。这篇笔记把"足够"拆成可被验证的数字——硬件长什么样、跑什么样的负载、出什么样的性能——读者拿这些数字去和自己的具体场景（每月音频时长、对延迟的容忍度、数据合规要求、等等）对比，自己判断。

笔记**不做横向对比**：不报任何商业 ASR 服务的价格、不画 cost-per-minute 对比图、不说 aistack "比 X 便宜"。这样的对比一个月就过期，对每个使用场景的解释也都不一样。读者拿到我们的数字之后，自己拉自己关心的服务的当前定价做这道乘法题更靠谱。

## 测试硬件

| 项 | 规格 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop（8 GB VRAM，消费级笔记本独显，SKU 入门款） |
| CPU | Intel Core i9 13 代 |
| 系统 RAM | 64 GB DDR5 |
| Windows 共享 GPU 内存上限 | 31 GB（手动配，非默认） |
| 操作系统 | Windows 11 |
| 驱动 | NVIDIA Studio Driver 5xx 系列 |
| 整机功率（推理峰值） | ~70 W（实测，CPU + GPU + 显存合计；非 GPU-only） |

中端游戏 / 创作笔记本配置——独显是 8 GB 入门款（不是 4090 / 4080 高显存型号），CPU 与 RAM 是创作类工作站常见的 i9 + 64 GB 组合。

**Windows 共享 GPU 内存** 是 WDDM 驱动给 GPU 提供的"溢出区"——当 VRAM 占满时，GPU 通过 PCIe 访问主机 RAM 的一部分作为延伸显存。本机配置允许最多 31 GB（默认是系统 RAM 的一半，本机手动放大）。这是后面性能数据的重要背景。

## 软件栈

| 组件 | 版本 |
|---|---|
| Python | 3.12.13 |
| torch | 2.7.1+cu126 |
| torchaudio | 2.7.1+cu126 |
| 系统 cuDNN（torch 自带） | 9.7.1 |
| CUDA runtime | 12.6 |
| NeMo toolkit | 2.7+ (`[asr,cu12]` extras) |
| ASR 模型 | `nvidia/parakeet-tdt-0.6b-v3`（HuggingFace 公开权重） |

模型权重一次性下载约 1.2 GB，本地缓存在 NEMO_CACHE_DIR，复用不重新下载。零 API 费用，零账户注册。

关键运行配置（详见 [parakeet-on-consumer-gpu.md](parakeet-on-consumer-gpu.md)）：

```python
model.change_attention_model("rel_pos_local_attn", [256, 256])
model.change_subsampling_conv_chunking_factor(1)
```

WER 代价相对全注意力 ~1-3%（NVIDIA 在 model card 中给出）。

## 测试负载

| 项 | 描述 |
|---|---|
| 音频 | 50 分钟英文政论演说（卢比奥外交记者会，47.8 MB mp3） |
| 语种 | 英语，含外交专有名词、地名、数字 |
| 噪声 | 干净棚内拾音，少量房间回声 |
| 扬声器数 | 单人主讲 + 偶尔记者提问 |

选这一段是因为它**代表难度中等的真实场景**：长音频 + 大量专有名词 + 自然语速变化。不是干净朗读样本（Common Voice 那种），也不是极端嘈杂场景（开放街头）。

## 性能数据

aistack `/v1/audio/transcriptions` 全链路端到端实测（含 ffmpeg 转码、模型推理、word/segment 时间戳计算、verbose_json 序列化、HTTP 响应回传）。

> RTF（real-time factor）= wall time ÷ 音频时长。RTF < 1 比真实音频走得快；RTF 0.01 ≈ 100× 实时。

### 分长度的稳态参考点

各长度的"良好状态下" cache 命中数据（具体含义见下面的内存动力学节）：

| 音频长度 | wall (s) | RTF | 加速 | 备注 |
|---|---|---|---|---|
| 4.4 min | ~13 s | 0.05 | 20× | fixed overhead 主导 |
| 12 min | ~6-8 s | 0.008-0.011 | 90-125× | **GPU 真实巡航段** |
| 17 min | ~10 s | 0.010 | 100× | 仍在巡航段 |
| 25 min | 13-20 s | 0.009-0.013 | 75-110× | 巡航段上沿 |
| 50 min | 60-80 s | 0.020-0.027 | 35-50× | 工作集开始溢到 PCIe shared |
| 99 min | ~490 s | 0.082 | 12× | 共享 GPU 内存上限触顶，page file 介入 |

短音频 RTF 偏高（4.4 min 0.05）是因为 fixed overhead（ffmpeg 转码、JSON 序列化、HTTP 传输）在短音频上摊不下去，不是 GPU 算力不行。中长音频（12-25 min）才看到 GPU 真实速度。

### 冷启动的固定代价

模型从磁盘加载到 VRAM 大约 **25-30 秒**——这是首次请求要付的"开门票"。aistack 默认 5 分钟空闲后释放（可调 `AISTACK_MODEL_KEEP_ALIVE_SEC`）。如果 5 分钟内有第二个请求，模型不重载，节省这 25-30 秒。

### 长音频的悬崖

99 min 音频耗 490 秒（RTF 0.082）——比 50 min 慢得**远不成比例**。机制是工作集触到 Windows 共享 GPU 内存上限（本机 31 GB）后，溢出部分被 Windows 分页到 SSD page file，每次 GPU kernel 访问那部分要走 SSD→PCIe→GDDR6 三跳，相对 GDDR6 速度退化 ~30 倍。

90-100 min 大约是本机配置的"舒适音频长度"上沿。再长能跑（Parakeet 不会崩），但 RTF 退化到 0.08-0.15+。要质变得换 24+ GB VRAM 显卡。

## 请求间的内存动力学（重要）

**同一段音频，wall time 可在不同前置请求历史下漂移 2-4×**。这是工程上不可忽视的现象，不是测量噪声。

### 实测对照：同一段 25 min 音频，在三种 warm-up 路径下

| Warm-up 路径 | 25min #1 wall | 25min #2 wall | reserved peak |
|---|---|---|---|
| 冷启动 → 50min × 2 → 25min | 20 s | 20 s | 24.7 GB |
| 冷启动 → 12min × 2 → 25min | 15 s | **13 s** | 14.5 GB |
| 冷启动 → 25min × 3（无前置） | 52 s（cold）| 27 s（warm） | 13.4 GB |

**4× wall 差距**——同一台机器、同一份代码、同一段音频，仅因前一次请求形态不同。

### 实测对照：同一段 50 min 音频，被 25 min 工作量"污染"后

| 场景 | wall | 备注 |
|---|---|---|
| 50 min cache 命中（warm baseline）| 69 s | 同形状反复跑的稳态 |
| 25 min × 1 后跑 50 min | **175 s** | **比冷启动还慢 56 秒** |
| aistack 重启冷启动 50 min | 119 s | 含 25-30 s 模型加载 |

**最坏情况下，"warm 状态被污染"反而比 kill 进程从零开始慢**。

### 机制

来自 PyTorch 官方文档 + 维护者 ZDevito 的 caching allocator 详解：

- PyTorch 维护一个 GPU 内存池（caching allocator），上次请求的张量释放后，**block 留在池里复用**，不还给 CUDA driver
- 复用的前提是**形状对得上**——cuDNN 为不同输入 shape 选不同 conv 算法、申请不同 size 的 workspace
- 上次跑 50min 留下的池子全是"50min-shape"的 block，下次 25min 来要新 shape，pool 必须 split / 重排
- 走过 split 路径的请求 wall 偏高；shape 同源（如 12min → 25min）则池子直接复用，wall 最低

详细机制与七层文档梳理见 [parakeet-on-consumer-gpu.md § 请求间内存动力学](parakeet-on-consumer-gpu.md#请求间内存动力学)。

### 推荐使用模式（按性能可预测性排序）

1. **批量同长度同语种工作量**（如全部转 5-15 min 视频字幕、或全部转 30-60 min 播客）→ 性能最优、最稳定
2. **混合工作量先批一类再切一类**（先转完所有短视频，再转所有长音频）→ 良好
3. **随机长度任意切换** → 接受 2-4× wall 漂移；或者在切换工作量类别前 kill aistack 重启（25-30 秒成本，比"被污染"快得多）

aistack 当前**不会自动检测内存状态触发清理**——这是实测后的设计决定，不是缺失功能。我们尝试过 `torch.cuda.empty_cache()` 自动清池，结果导致第二次请求 hang 死（机制：cudaFree 是同步设备调用，与 NeMo 内部 stream event 死锁），已撤销。判断"现在该不该重启"的启发式规则太多变量耦合，不存在可靠实现。

## 资源占用

| 资源 | 50 min 音频时占用 | 备注 |
|---|---|---|
| VRAM | 8 GB（全卡满载） | Parakeet 设计上预分配 workspace，VRAM 永远满载与音频长度无关 |
| Windows 共享 GPU 内存 | 14 GB | 此项**按音频长度近似线性**：50min 14 GB，120min 22 GB，99min 触顶 31 GB |
| 系统 RAM | ~20 GB | 含 Python 进程 + 音频缓冲 + Windows 共享 GPU 内存重叠映射 |
| 整机功率 | ~70 W 推理期间 / ~25 W 模型空闲驻留 |  |

**Parakeet 显存饥饿但伸缩稳健**：不会因为工作集超过 VRAM 而崩——溢出部分自动落到 Windows 共享内存（PCIe 路径）。代价是 PCIe 比 GDDR6 慢 ~30 倍，所以工作集越接近上限越慢。这是 Parakeet TDT 的真实工程特性：用显存换计算并行度，整段音频一次性铺开喂 GPU 算（FastConformer 全局工作集），所以 8 GB VRAM 永远满。详见 [parakeet-on-consumer-gpu.md](parakeet-on-consumer-gpu.md)。

## 这数据意味着什么

读者拿这些数字回到自己的场景算一下：

**问 1：每月需要转写多少小时音频？**
按"良好状态"30-50× 实时估算（不是最佳的 100×，而是能稳定预期的中位数），1 小时音频耗 1-2 分钟。100 小时音频在 1 张消费卡 1 张白班里跑完。

**问 2：你愿意接受冷启动 100-120 秒吗？**
- 接受 → 服务可以按需启动 / 按需驻留模型
- 不接受 → aistack 进程常驻 + cache TTL 调长（到几小时），后续请求免冷启动开销

**问 3：电费怎么算？**
推理峰值 70 W，转 100 小时音频 ≈ 2 小时 × 70 W = 140 Wh = **0.14 度电**。按民用电价你自己算。

**问 4：硬件投入怎么摊？**
6000-8000 元的游戏本可以跑这个负载。如果是台式机配独立 8 GB 卡（4060 / 二手 3060 12 GB / 老 2080 Ti 等），更便宜。这部分摊销取决于硬件还有多少残值年限。

**问 5：网络 / 数据合规怎么处理？**
本地推理音频和文字都不出本机。没有 vendor 数据保留政策、没有跨国传输、没有日志合规附加成本。

把这五个问题的答案拼起来，对比你正在使用或考虑使用的任意云服务的总账单（含基础费率 + 任何附加 feature 的钱 + 未利用的最低消费 + 数据合规 / 隐私评审的人力成本），就是你自己的判断依据。**aistack 不替你做这道题**。

## 什么场景下本地不合适

诚实的限制：

1. **极低用量**（每月 10 小时以内）—— 摊不掉硬件成本，云端按量付费可能更经济
2. **峰值并发要求高** —— 单台 8 GB 卡同时只跑一个 ASR 任务（aistack 的设计选择，避免 OOM）。需要并行处理几十个流的场景，本地多机部署或云端弹性更合适
3. **要求 wall time 严格可预测** —— 本节"内存动力学"展示的是 wall time 在不可预测前置历史下漂移 2-4×。如果你的场景需要"5 分钟音频必须 X 秒内返回"的严格 SLA，本地是错的工具，应该上云端按需弹性
4. **零运维要求** —— 本地需要 GPU 驱动、Python 环境、模型权重的维护。不愿意碰这些的，云端 API 就是一行 HTTP 调用
5. **没有 GPU** —— Parakeet/Whisper 在 CPU 上能跑但慢得多（RTF 接近 1，近实时但谈不上加速）。aistack 也支持 CPU 模式但不建议作为主路径
6. **极端低延迟流式**（< 500ms 端到端）—— 当前 Parakeet 不支持原生流式，aistack 走的是降级路径。要真正实时字幕（电话客服 / 直播）应选 streaming-trained 模型 + 专门优化
7. **超过 ~90 分钟单段音频且 wall 敏感** —— 本机 90-100 min 是悬崖，再长虽不崩但 RTF 退化到 0.08-0.15+。要稳定吃长音频应当切片预处理或上 24+ GB 显存卡

如果你的场景命中以上任意一条，本地未必是更好的选择。

## 什么场景下本地是赢的

- **批量离线转写**（视频字幕生成、播客存档、采访整理、法律取证）—— 本地常驻一个 aistack 进程，把待转写文件丢进去就行
- **私域数据**（企业内部会议、医疗记录、未公开访谈）—— 数据完全不出本机
- **价格敏感型大用量**（每月几十到几百小时）—— 硬件成本一次摊，电费基本可以忽略
- **自主可控**（不想被云服务的 deprecation policy 绑架，比如 OpenAI 关停 Whisper-1 时自己的 pipeline 不受影响）

## 数据复现路径

1. 装 aistack：`uv pip install -e .[asr-parakeet]`（按 `pyproject.toml` 的 extras）
2. 启服务：`scripts/dev.bat`
3. 调 API：

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
  -F "file=@your-50min-audio.mp3" \
  -F "model=parakeet" \
  -F "language=en" \
  -F "response_format=verbose_json"
```

4. 看 `/admin` 或 `/admin/api/metrics` 拿 RTF / latency / 资源占用。

完整配置依据见 [parakeet-on-consumer-gpu.md](parakeet-on-consumer-gpu.md)，包括为什么要那两个 `change_*` 调用、为什么不要碰 `preserve_alignments`、以及实测下来的 7 层文档梳理。

## Open questions

- **更长音频的边界**（11 小时 / 8 GB 卡）—— NVIDIA 的官方说法是 local attention + chunking 配 8 GB 可达 11 小时，我们没实测过那个上限。50 分钟以内已确认稳定。
- **更便宜显卡的下沿**（4 GB / 6 GB VRAM）—— 当前基线是 8 GB。Parakeet 0.6B 模型本身约 2.4 GB，加 KV cache 和工作区，4 GB 应该不够，6 GB 边缘。需要实测确认。
- **CPU-only 的可行性**—— 不建议作为主路径但会跑通。具体 RTF 数字未测。

如果你测了上面任意一项，欢迎 PR 把数据加进来——这个目录就是为这种发现存在的。

## 致谢

本笔记的所有数字都来自 aistack 项目的真实开发过程。50 分钟基线测试由项目维护者（@OldApeTalk）在 RTX 4060 Laptop 上跑出，包括开关组合的对照实验（`preserve_alignments` 那 +20 GB 共享内存的代价就是这么发现的）。没有这些实测数据，本笔记最多就是一篇文档抄录。

---

*这是 dosmoon aistack 项目研究笔记之一。其他笔记见 [README.md](README.md)。*
