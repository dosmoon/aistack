---
title: 研究笔记
description: 围绕 aistack 涉及的技术与决策做出来的稳定结论 —— 实测、选型、能力边界、立场判断。
sidebar:
  order: 5
---

# 研究笔记

围绕 aistack 涉及的技术与决策做出来的稳定结论。形态包括但不限于:

- **实测笔记** —— 自家硬件跑出来的数字、配置、可重现的现象
- **选型调研** —— 业界全景汇总 + 选型推荐
- **能力边界** —— 澄清某个工具的真实能力 / 限制
- **立场判断** —— 项目自身方向、不做什么、为什么不做

共同点是**以后还会被引用**:跨项目演进周期保持有效。

## 笔记列表

| 笔记 | 主题 |
|---|---|
| [消费级 GPU 本地 ASR 性能基线](consumer-gpu-asr-baseline/) | 50 分钟英文音频 / RTX 4060 8GB / 62s 端到端 / RTF 0.021。读者拿这数据自行决定本地 ASR 是否适合自己场景。 |
| [NVIDIA Parakeet TDT 在消费级 GPU 上跑长音频](parakeet-on-consumer-gpu/) | 8 GB 卡上跑 Parakeet 长音频的可工作配置:哪几个旋钮要开、哪几个不能碰、为什么 NVIDIA 官方文档没把这事说全。 |
| [Whisper 翻译能力的真实边界](whisper-translation-capability/) | Whisper `task=translate` 是 X→English only,不能做 EN→ZH 或任何 non-English 目标翻译。需要 EN→ZH 字幕时三条可行路径的对比。 |
| [中文 ASR 引擎选型基线研究](chinese-asr-engine-survey/) | FireRedASR-AED / Paraformer / SenseVoice / Whisper-large-v3 / Fun-ASR / FireRedASR2S 全景对比。AISHELL/WenetSpeech 公开 CER + 设计意图 + 评估方法陷阱 + 8GB 卡集成成本。**desk research,实测前必读**。 |
| [Whisper 生态系统全景](whisper-ecosystem-tools/) | 6 大类 25+ 项目分类整理。推理引擎 / 蒸馏小模型 / 功能增强层 / 流式化 / Whisper-style 重训 / 跨域反演。aistack 该集成什么、不该集成什么、未来产品形态用得上什么的判断。 |

## 风格约定

- 每篇开头一段 TL;DR:3 行内回答"读这篇我能拿走什么"
- 量化要带单位与机型("8 GB VRAM @ RTX 4060 Laptop",不只是"内存大了一倍")
- 引用上游文档与社区帖必须给链接
- 文末有 **Open questions** 一节列出未确认的事——研究笔记不是教科书,承认未知比假装全知更可信
- 中文版本译自英文原稿。如发现表述偏差以英文版为准。
