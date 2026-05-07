---
title: 消费级 GPU 本地 ASR 性能基线
slug: consumer-gpu-asr-baseline
date: 2026-05-07
tags: [asr, parakeet, benchmark, consumer-gpu, self-hosting]
---

# 消费级 GPU 本地 ASR 性能基线

> **TL;DR** RTX 4060 Laptop（8 GB VRAM，消费级笔记本独显）跑 NVIDIA Parakeet TDT 0.6B v3：50 分钟英文新闻演说 → 缓存命中 62 秒完成（RTF ≈ 0.021，约 80× 实时）；冷启动含模型加载 ~120 秒。整机推理峰值功率约 70 W。这篇笔记给出可独立复现的硬件、软件、负载、性能数据，让读者自行决定本地 ASR 是否适合自己的场景。

## 这篇是干什么的

aistack 的产品立场是：**对于本地 GPU 性能足够的 AI 任务，没必要走云端 API**。但"足够"是个具体的工程问题，不是口号。这篇笔记把"足够"拆成可被验证的数字——硬件长什么样、跑什么样的负载、出什么样的性能——读者拿这些数字去和自己的具体场景（每月音频时长、对延迟的容忍度、数据合规要求、等等）对比，自己判断。

笔记**不做横向对比**：不报任何商业 ASR 服务的价格、不画 cost-per-minute 对比图、不说 aistack "比 X 便宜"。这样的对比一个月就过期，对每个使用场景的解释也都不一样。读者拿到我们的数字之后，自己拉自己关心的服务的当前定价做这道乘法题更靠谱。

## 测试硬件

| 项 | 规格 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop（8 GB VRAM，消费级笔记本独显，SKU 入门款） |
| CPU | Intel Core i7-13620H |
| 系统 RAM | 32 GB DDR5 |
| 操作系统 | Windows 11 |
| 驱动 | NVIDIA Studio Driver 5xx 系列 |
| 整机功率（推理峰值） | ~70 W（实测，CPU + GPU + 显存合计；非 GPU-only） |

这是一台**普通 6000-8000 RMB 价位的游戏笔记本**——2024-2025 年消费级主流配置的下沿。不是工作站、不是数据中心卡、不是 24 GB 大显存型号。

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

aistack `/v1/audio/transcriptions` 全链路端到端测得（含 ffmpeg 转码、模型推理、word/segment 时间戳计算、verbose_json 序列化、HTTP 响应回传）：

| 指标 | 冷启动（含模型加载） | Cache 命中 |
|---|---|---|
| 50 分钟音频 端到端时长 | 100-120 秒 | **62 秒** |
| RTF（real-time factor） | ~0.040 | **0.021** |
| 加速倍数（vs 实时） | 25× | **48×** |
| 每秒能转写多少秒音频 | 25 | **48** |

> RTF（real-time factor）= 推理时长 ÷ 音频时长。RTF < 1 意味着比真实音频走得快，<0.05 已经是"几乎瞬时返回"的体感。

短音频对照基线（17 分钟英文新闻片段，5/6 单独测得）：

| 音频长度 | Cache 命中时长 | RTF |
|---|---|---|
| 17 分钟 | ~10 秒 | ~0.01 |

短音频 RTF 更低（≈0.01）是因为模型加载等固定开销在短音频上没被分摊。模型常驻 + 持续负载下，**RTF 稳态在 0.01-0.025 区间**。

## 资源占用

50 分钟音频推理过程中（任务管理器 + GPU-Z 实测）：

| 资源 | 占用 |
|---|---|
| VRAM | 8 GB（全卡满载，模型 + 工作区 + 激活） |
| Windows 共享 GPU 内存 | 10 GB（PCIe 路径，正常工作集） |
| 系统 RAM | 18 GB（含 Python 进程 + 音频缓冲 + 操作系统） |
| 整机功率 | ~70 W 推理期间，~25 W 模型空闲驻留 |

**模型常驻（idle keep-alive）**：默认 5 分钟内有第二个请求就复用已加载模型，第二次请求 latency 从 100s+ 跳到 62s。aistack 内置 `_model_cache` 管理这个驻留窗口，可调（`AISTACK_MODEL_KEEP_ALIVE_SEC`）。

## 这数据意味着什么

读者拿这些数字回到自己的场景算一下：

**问 1：每月需要转写多少小时音频？**
按 48× 实时计算，1 小时音频耗 75 秒、能 1 张消费卡转 100 小时音频在不到 2 小时 wall-clock 内做完。

**问 2：你愿意接受冷启动 100 秒吗？**
- 接受 → 服务可以按需启动 / 按需驻留模型
- 不接受 → 让 aistack 进程常驻 + cache TTL 调长（到几小时），首次外的所有请求都是 62s/50min

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
2. **峰值并发要求高** —— 单台 8 GB 卡同时只跑一个 ASR 任务（aistack 的设计选择，避免 OOM）。需要并行处理几十个流的场景，本地多机部署或者云端弹性更合适
3. **零运维要求** —— 本地需要 GPU 驱动、Python 环境、模型权重的维护。不愿意碰这些的，云端 API 就是一行 HTTP 调用
4. **没有 GPU** —— Parakeet/Whisper 在 CPU 上能跑但慢得多（RTF 接近 1，近实时但谈不上加速）。aistack 也支持 CPU 模式但不建议作为主路径
5. **极端低延迟流式**（< 500ms 端到端）—— 当前 Parakeet 不支持原生流式，aistack 走的是降级路径。要真正实时字幕（电话客服 / 直播）应选 streaming-trained 模型 + 专门优化

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
