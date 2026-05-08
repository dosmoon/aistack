# Research Notes

> 实证研究笔记。发布到 dosmoon pages 的源材料目录。

## 这里收什么

只收**经过我们自己实测验证、并且在上游官方文档中没有完整说明**的研究成果。三条准入门槛：

1. **三条腿都站住** —— 官方文档 / 社区经验 / 源码-运行时实测，三者交叉印证
2. **量化优于定性** —— "慢了 2x"不算结论，"30 GB 共享 RAM、PCIe 带宽 ≈ GDDR6 的 1/30、所以 2x 减速"才算
3. **可重现** —— 别人能拿这篇笔记 + 我们写到的代码与 env 复现出同样的现象

不收的内容：

- 纯文档抄录或翻译（去看上游文档就行）
- 单次试错日记（那是 `docs/progress/` 的事）
- API 契约（那是 `docs/public/api/` 的事）
- 产品设计或架构决策（那是 `docs/design/` 的事）

## 为什么单建一个目录

aistack 这一年多调过的"上游没写的实情"在累积——尤其是消费级 GPU 上跑大模型的边界配置（VRAM/PCIe/Windows 共享内存这几个维度）。这些发现：

- **不属于** API 契约——它们是后端实现细节，但对任何想自己部署、调优、复用的人都是关键资料
- **不属于** progress chronicle——一篇日志记的是"那一天发生了什么"，而研究笔记记的是"事情发生过之后我们提炼出的稳定结论"
- **不属于** AI Coding 失败日志——那本是 AI 工作方式的元日志，记"AI 没做对什么"；研究笔记记"我们最终搞清楚了什么"

把它们沉淀成一类独立的 first-class 资产，未来：
- 别人来读 aistack 源码，问"为什么这里要调这个 API"，答案在 research-note
- 我们升级依赖（NeMo 2.7 → 2.8）想知道哪些 workaround 还成立，跑回归对照基线就在这
- 把 dosmoon 这边踩过的坑公开发布，对中文圈消费级 GPU 部署社区是真有用的资料

## 目录

| 笔记 | 主题 | 状态 |
|---|---|---|
| [consumer-gpu-asr-baseline.md](consumer-gpu-asr-baseline.md) | 消费级 GPU 跑 ASR 的实测基线：50 分钟英文音频 / RTX 4060 8GB / 62s 端到端 / RTF 0.021。读者拿这数据自行决定本地 ASR 是否适合自己场景。 | 2026-05-07 |
| [parakeet-on-consumer-gpu.md](parakeet-on-consumer-gpu.md) | NVIDIA Parakeet TDT 在 8 GB 卡上跑长音频的**可工作配置**：哪几个旋钮要开、哪几个不能碰、为什么 NVIDIA 官方文档没把这事说全。 | 2026-05-07 |
| [aistack-positioning-and-product-paths.md](aistack-positioning-and-product-paths.md) | 为什么 aistack 必然是研发形态、不可能演化成终端用户产品；以及未来若做产品形态可行的几条路径（whisper.cpp 离线优先 / Docker 全家桶 / sherpa-onnx 多引擎）。**强项是定位判断，不是技术决策**——具体技术细节留到新仓库再展开。 | 2026-05-08 |
| [whisper-translation-capability.md](whisper-translation-capability.md) | Whisper `task=translate` 的真实边界：**X→English only**，不能做 EN→ZH 或任何 non-English 目标翻译，paper §2.1 数据集决定的硬限制。需要 EN→ZH 字幕时三条可行路径（Whisper+LLM 级联 / SeamlessM4T / Qwen2-Audio）的对比与待实测项。 | 2026-05-08 |
| [chinese-asr-engine-survey.md](chinese-asr-engine-survey.md) | 中文 ASR 引擎选型基线研究（2026 视角）：FireRedASR-AED / Paraformer-large / SenseVoice / Whisper-large-v3 / Fun-ASR / FireRedASR2S 全景对比。AISHELL-1/2 + WenetSpeech 公开 CER 数据 + 各引擎设计意图 + CER 评估方法陷阱 + dosmoon 中文 bench 数据集选型策略 + 长音频处理 + 8GB 卡集成成本评估 + 实测计划。**desk research, 实测前必读**。 | 2026-05-08 |
| [whisper-ecosystem-tools.md](whisper-ecosystem-tools.md) | Whisper 生态系统全景：6 大类 25+ 项目分类整理。推理引擎（faster-whisper / whisper.cpp / insanely-fast-whisper / WhisperKit）/ 蒸馏小模型（distil-large-v3 是被低估的最优升级）/ 功能增强层（WhisperX / CrisperWhisper / stable-ts）/ 流式化（whisper_streaming / SimulStreaming / WhisperLive）/ Whisper-style 重训（OWSM / Belle-whisper）/ 跨域反演。aistack 该集成什么、不该集成什么、未来产品形态用得上什么的判断。**填补 Whisper 生态信息盲点的入口**。 | 2026-05-08 |

## 风格约定

- 中文为主，关键技术术语保留英文
- 每篇开头一段 TL;DR：3 行内回答"读这篇我能拿走什么"
- 量化要带单位与机型（"8 GB VRAM + 30 GB shared RAM @ RTX 4060 Laptop"，不只是"内存大了一倍"）
- 引用上游文档与社区帖必须给链接，不要"听说"
- 文末有 **Open questions** 一节列出未确认的事——研究笔记不是教科书，承认未知比假装全知更可信
