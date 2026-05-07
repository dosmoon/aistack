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

## 待做的实验（pending empty_cache 开关）

加 `AISTACK_PARAKEET_CLEAR_CACHE_BETWEEN=1` 之后重跑，验证假设：

1. **50min → 25min**（pool 碎片化场景）：清缓存后 wall 从 20s 降到 ~15s？（接近 12min 预热的最优）
2. **50min → 50min**（同形状最优场景）：清缓存后 wall 从 69s 升到 ~120s？（失去复用红利）
3. **冷 → 25min**（无残留场景）：清缓存影响极小（pool 本来就空）

如果 1 改善 + 2 退化 + 3 不变 → 假设成立。否则机制还要重新理解。

## 数据落盘归档

JSONL 在 `logs/access-2026-05-07.jsonl`，截至 21:31 包含全部上述记录。
git ignore 中已包含 `logs/`，**实测数据本身不入 repo**——本文件保留 rid 索引就够了，需要详查时按 rid 在本机 JSONL grep。
