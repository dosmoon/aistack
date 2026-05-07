---
status: WIP — data collection in progress, do not publish
title: Parakeet 在消费 GPU 上的内存动力学（实验数据档）
date: 2026-05-07
---

# Parakeet 内存动力学实验数据档（WIP）

> **状态**：数据采集中。本文件**不发布**——仅保存 rid → 场景 → 关键数字的对应关系，供后续分析与文档撰写复用。
>
> 完整原始数据在 `logs/access-2026-05-07.jsonl`（每条带 `request_id`、`audio_sec`、`latency_ms`、`extra.rtf`、`extra.vram_peak_mb`、`extra.vram_reserved_peak_mb`）。本文件是这些原始数据的**实验语义索引**。

## 硬件 / 软件基线

- **机器**：i9 13 代 / 64 GB DDR5 / RTX 4060 Laptop 8 GB VRAM / Win11
- **Windows shared GPU memory 上限**：31 GB（手动配，非默认）
- **软件**：Python 3.12.13 / torch 2.7.1+cu126 / NeMo 2.7+ / Parakeet TDT 0.6B v3
- **关键运行配置**：`change_attention_model("rel_pos_local_attn", [256,256])` + `change_subsampling_conv_chunking_factor(1)`

## 实验组 1 — 单一长音频反复跑（Rubio 50min，同一 mp3）

audio_sec=2986.37, 47.8 MB

| Time | rid | wall (s) | RTF | alloc_peak (MB) | reserved_peak (MB) | 备注 |
|---|---|---|---|---|---|---|
| 16:08:37 | 680e246cb309bf8b | 100.4 | 0.034 | — | — | 当时无 vram 字段；冷启动 |
| 16:10:37 | 7001128374d1ebba | 61.9  | 0.021 | — | — | cache hit |
| 16:17:42 | 38eacb5c3512e2d6 | 23.5  | 0.286 | — | — | 不同音频（80s）插入 |
| 19:22:44 | 5701f04b0a60c88b | 117.0 | 0.038 | — | — | 重启后冷启动 |
| 19:29:09 | b1bf3c79902822d6 | 143.5 | 0.048 | — | — | 二次但慢 |
| 19:35:39 | 328c32f4c55f870c | 202.1 | 0.068 | — | — | 三次更慢 |

后段 19:22-19:35 三次递增的 117→143→202 是 `_configure_timestamp_decoding(preserve_alignments=True)` bug 期 — 已 revert（commit `a0eb0c7`），数据本身不属于"正常"baseline，仅作历史记录。

## 实验组 2 — 99min 极限测（Trump 0326）

audio_sec=5946.99 (99.1 min), 142.7 MB mp3

| Time | rid | wall (s) | RTF | alloc_peak (MB) | reserved_peak (MB) | 备注 |
|---|---|---|---|---|---|---|
| 20:02:13 | 85133db9fb417400 | 489.8 | 0.082 | — | — | shared RAM 触顶 31 GB（人眼观察）；page file 启动 |

## 实验组 3 — 4-25min 短音频曲线（vram 字段尚未启用）

| Time | rid | audio (min) | wall (s) | RTF | 备注 |
|---|---|---|---|---|---|
| 20:16:49 | 454fe95a40465f42 | 4.40 | 37.4 | 0.142 | cold start |
| 20:17:13 | 99809acab05ad42e | 4.40 | 12.9 | 0.049 | warm |
| 20:18:49 | 9eacc07a0e1f3af1 | 12.10 | 25.4 | 0.035 | warm |
| 20:21:16 | e8df721f95ac3bd0 | 25.03 | 57.5 | 0.038 | warm |

## 实验组 4 — 12min Trump 阅兵反复跑（vram 字段已启用）

audio_sec=726.27 (12.1 min), 同一 mp3，dc0b857 之后

| Time | rid | wall (s) | RTF | alloc_peak (MB) | reserved_peak (MB) | 备注 |
|---|---|---|---|---|---|---|
| 20:41:33 | 3418092a2f6b2dfc | 8.05 | 0.011 | 7110 | 13412 | warm baseline |
| 20:42:07 | 05f8e89c36b6a853 | 8.10 | 0.011 | 7110 | 13412 | warm |
| 20:56:07 | e8023ddf7c9afd48 | 8.19 | 0.011 | 7110 | 13412 | warm |
| 20:56:56 | 7110be8d83163d44 | 7.94 | 0.011 | 7110 | 13412 | warm |
| 20:57:07 | 9470ef087986630a | 7.61 | 0.011 | 7110 | 13412 | warm |
| 21:07:22 | 546c9e5b1d9fd723 | **5.77** | **0.008** | 7159 | **19446** | **跑过 50min 之后变快** |
| 21:07:33 | 3f738383d0c80232 | 5.83 | 0.008 | 7159 | 19446 | warm |
| 21:07:44 | 5a6d43e3d877a70d | 5.77 | 0.008 | 7159 | 19446 | warm |

**关键发现**：50min 跑过把 reserved pool 涨到 19 GB 之后，12min Trump 反过来从 8s → 5.77s。pool 形状对路 + 充裕余量 = 真实 GPU 巡航速度（RTF 0.008）。

## 实验组 5 — 50min Rubio 在不同前置工作量下的 wall 漂移

| Time | rid | wall (s) | RTF | alloc_peak (MB) | reserved_peak (MB) | 前置场景 |
|---|---|---|---|---|---|---|
| 20:48:52 | a447d112eb476bda | 80.3 | 0.027 | 16884 | — | warm baseline |
| 21:00:09 | 7aa4498d0970d17a | **175.3** | **0.059** | 16808 | — | **跑过 25min WH 之后** |
| 21:01:33 | 465d618c3b46937e | 69.3 | 0.023 | 16884 | — | 再跑回 50min 立刻恢复 |
| 21:04:06 | d97821691eb3c04d | 119.6 | 0.040 | 16731 | — | 重启后冷启动 |
| 21:05:29 | 6138477275055680 | 76.6 | 0.026 | 16859 | — | warm |
| 21:07:01 | cbde50dac0e730b6 | 72.8 | 0.024 | 16859 | — | warm |

**关键发现**：175s 那次（跑过 25min 之后）甚至比冷启动 119s 还慢——25min 留下的小尺寸 buffer 不能给 50min 直接复用，被迫重排。

## 实验组 6 — 25min WH 在三种 warm-up 路径下的对比

audio_sec=1501.63 (25 min), 同一 mp3

| Time | rid | wall (s) | RTF | alloc_peak (MB) | reserved_peak (MB) | 前置场景 |
|---|---|---|---|---|---|---|
| 21:25:34 | e8811aa62dc77479 | 20.0 | 0.013 | 12609 | **24676** | 冷→50min×2→25min #1 |
| 21:26:02 | c105da46ed56eb1a | 20.3 | 0.013 | 12609 | 24676 | 同上 #2 |
| 21:27:59 | d0dbc4c5edf8f389 | 15.5 | 0.010 | 12024 | 17242 | 冷→12min×2→25min #1 |
| 21:28:21 | cbc3d859a3f06c84 | **13.4** | **0.009** | 12076 | 14522 | **冷→12min×2→25min #2** |
| 21:28:37 | a7505523d6d8693c | 13.5 | 0.009 | 12076 | 14522 | 同上 #3 |
| 21:29:59 | 1225481c0096ec90 | 52.3 | 0.035 | 11973 | 15542 | 纯冷启动 #1 |
| 21:30:30 | 7a1ae4fee35c07b0 | 27.5 | 0.018 | 12051 | 13412 | 冷启动 warm #2 |
| 21:31:00 | 41e9404e132a6518 | 26.4 | 0.018 | 12051 | 13412 | 冷启动 warm #3 |

**关键发现**：
- 同一段 25min 音频，wall 可在 13s ~ 52s 之间漂移（4× 差距）
- 用稍小的同模型工作量（12min）预热 → **最优**
- 用大很多的工作量（50min）预热 → 中等（pool 大但碎片化）
- 纯冷启动 → 最差（含模型加载 25-30s）
- alloc_peak 始终在 12 GB ± 5%，**与 pool 状态无关**——这才是 25min 的真实工作集

## empty_cache 实验结果与机制修订（important）

### 实测结果

加 `AISTACK_PARAKEET_CLEAR_CACHE_BETWEEN=1` 后第二个请求挂死：

```
21:46:02 rid=3a319d9b6318960e   ← cold start，正常完成 52s
[第二次请求 rid=9b52556b...]   ← 完全没进 JSONL（observability finally 没跑）
[第三次请求]                    ← 同样 hang
```

请求挂在 `asyncio.to_thread(module.transcribe, ...)` 内部，CUDA 调用永远不返回。GPU 锁永远不释放。整个 ASR 端点不可用直到 kill 进程。

### 机制（修订版，基于 PyTorch 官方文档而非凭空推测）

**之前我把现象归因到"Windows WDDM decommit/recommit race"——错误**。该机制是凭空编的，没有任何官方文档、PyTorch issue、社区帖支持。修订后的机制基于 PyTorch 维护者公开文档：

来源：[zdevito.github.io/2022/08/04/cuda-caching-allocator.html](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) + PyTorch 官方 `empty_cache` 文档

**实际机制**：

1. `torch.cuda.empty_cache()` 内部对每个 free block 调 `cudaFree`
2. **`cudaFree` 是同步 CUDA API 调用**（NVIDIA 官方文档定义）—— 隐式 `cudaDeviceSynchronize()`，等所有挂起 GPU 操作完成
3. PyTorch caching allocator 的存在目的就是**避免** `cudaFree`，因为同步障碍打断 CPU-GPU pipeline
4. `empty_cache()` 强行触发 cudaFree，**与 NeMo / cuDNN 内部 stream 上的未完成 event 形成同步竞争**
5. NeMo TDT 内部因架构特点（每 timestep 双网络协同 + frame skipping）stream event 密度高
6. 同步 + descriptor 生命周期 + 内部 stream 依赖 → 可能死锁

**和 Windows / Linux 无关，是 empty_cache 在活跃 CUDA 工作期间调用的通用反模式**。Windows 工作集大、stream 操作多，触发概率高；Linux 不是免疫，只是窗口小。

PyTorch 维护者 [明确反对常规调用 empty_cache](https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232)：
> "If you need it repeatedly, your program is in an 'unhealthy state'. ... Performance cost is significant."

**已撤销**：`AISTACK_PARAKEET_CLEAR_CACHE_BETWEEN` env flag + `dev-clearcache.bat` launcher 都已 revert（commit `782b70d`）。`_reset_gpu_peak()` 退回到只 reset peak stats。

## 关键洞察（基于 6 组实验综合）

### 洞察 1：前后任务共享的状态没有唯一规则

跨调用持久存在的状态包括：模型权重、cuDNN workspace cache、PyTorch caching allocator pool（带形状/碎片信息）、cuDNN benchmark 算法选择缓存、CUDA streams + 未完成 events、pinned memory 缓冲、`cfg.decoding` 字段。

**每一项独立受上次工作量影响，相互之间又有交叉作用**。不存在一条简单规则告诉你"前面跑过 X 之后下次性能会是 Y"。

每次试图把这个多变量耦合简化成单条规则（"reserved 越大越快"、"同形状最优"、"Windows WDDM 锅"）都被新数据打脸。结构性原因：规则不存在。

### 洞察 2：最恶劣情况超过冷启动

数据：
- Cold start 50min（含模型加载 ~25-30s）：**119 s**
- 25min × 1 → 50min Rubio（碎片化最严重）：**175 s**

**175 s > 119 s** ── "warm 状态被污染" 比 "kill 进程从零开始" 慢 56 秒。

这反直觉但很重要：**在某些工作负载切换场景下，主动重启 aistack 比沿用 warm cache 更快**。原本"warm cache 永远更优"的简化模型被这条数据彻底否定。

### 洞察 3：alloc_peak 是真实工作量、reserved_peak 是池子状态

```
12 min Trump 反复跑：alloc_peak 永远 7110 MB，reserved 13412 (warm baseline)
50 min Rubio 跑过后再跑 12 min：alloc_peak 仍 7159，reserved 跳到 19446
```

**alloc_peak 反映音频内在 GPU 工作量**（非常稳定，跨场景 5% 浮动）。**reserved_peak 是 pool 的当前最高水位**（跨场景跨度大）。
benchmark 用 alloc_peak 作为指标比 wall time 信噪比高一个数量级。

## 设计原则（基于上述洞察）

**原则 A：放弃"自动修复"**

任何"嗅探到状态不好，自动 empty_cache / 重置 / warm-up"的逻辑都不可靠（empty_cache 已证伪）。状态空间太大，没有可定义边界的启发式。

**原则 B：暴露"reset"作为产品能力**

既然用户真的可能撞到"重启反而更快"的场景，aistack 应该提供：

```
admin 端点 /admin/api/reset-asr-state
  - 释放当前 ASR 模型（_model_cache.evict_category）
  - GC + 重新加载（下一次请求触发，~25-30s 模型加载）
  - 不杀 aistack 本身，不影响 LLM/TTS 路径
```

技术上很简单（`_model_cache.evict_category("asr-main")` 已实现）。**未实现，作为 follow-up**。

**原则 C：诚实文档化**

把 wall 漂移、最坏情况、推荐使用模式写进对外文档（已完成，见 `consumer-gpu-asr-baseline.md` 的"请求间内存动力学"节 + `parakeet-on-consumer-gpu.md` 的同名节）。

## 后续工作（不阻塞当前发布）

- 实现 `/admin/api/reset-asr-state` 端点（~半小时工作）
- 测 `torch.backends.cudnn.benchmark = False` 是否能让 pool 形状更可预测（代价：单次推理略慢）
- 用"短视频拼接"做严谨基线（剔除文字密度变量，跑 5/10/15/30/60/90 min 完整曲线）
- 用 `py-spy` 或 `gdb attach` 抓一次 hang 进程的 stack trace，确认 cudaFree 死锁的具体堆栈位置（不是必须但能 close 机制猜测的最后一公里）

## 数据落盘归档

JSONL 在 `logs/access-2026-05-07.jsonl`，截至 21:31 包含全部上述记录。
git ignore 中已包含 `logs/`，**实测数据本身不入 repo**——本文件保留 rid 索引就够了，需要详查时按 rid 在本机 JSONL grep。
