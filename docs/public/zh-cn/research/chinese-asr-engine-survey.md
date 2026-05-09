---
title: 中文 ASR 引擎选型基线研究
date: 2026-05-08
tags: [asr, chinese, mandarin, benchmark, sensevoice, paraformer, fireredasr]
sidebar:
  order: 4
---

# 中文 ASR 引擎选型基线研究（2026 视角）

> **TL;DR**
>
> 1. aistack 当前默认把 `lang=zh*` 路由到 SenseVoiceSmall，**但这个选择从未被实测验证过** —— 我们既没跑过中文 bench，bench/audio 里也全是英文音频
> 2. 公开 benchmark 数据（AISHELL-1/2 + WenetSpeech net/meeting）的 2024-2025 排序：FireRedASR2S-LLM (2.89%) > FireRedASR-AED (3.18%) ≈ FireRedASR-LLM (3.05%) ≫ SenseVoice-L (4.47%) ≈ Paraformer-large (4.56%)。**SenseVoice 的中文 CER 在主流引擎里垫底**
> 3. 但 SenseVoice 不是被错选 —— 它的设计点是"轻量+多语种+情感/事件标签"，不是"中文质量天花板"。"中文质量"该选谁是另一个问题
> 4. **8 GB VRAM 消费级卡上 FireRedASR-AED (1.1B) 是当前性价比最高的中文 ASR 候选**，而且 Apache-2.0 + 集成成本可接受
> 5. **必须先做的基础工作**：搭中文 bench（真实音频 + ground truth + CER 评估脚本）。没有这一步任何"换引擎"的决定都是空谈

---

## 一、问题陈述：为什么要重做这件事

aistack 的 `auto` 路由（`aistack/api/asr.py:_select_for_auto`）对 `lang=zh*` 已经路由到 SenseVoice，但这个选择只基于"它支持中文 + FunASR runtime 已经装着"，**没有任何质量数据支撑**。具体地：

- `bench/audio/` 全是英文音频（`perf-12min.mp3` ~ `perf-97min.mp3`），那些 `*_zh.srt` 是英文音频的中文翻译，不是中文 ASR 的 ground truth
- `bench/asr_eval.py` 只跑 LibriSpeech，是英文 WER bench，没有中文路径
- backlog 里"Mandarin ASR 数据集 (Common Voice zh-CN or AISHELL dev)"长期 ready 但没动

研究形态的 aistack **本来就应该量化这件事**。这篇笔记是动手前的 desk research，把"既有的前人经验"先吸取干净，避免实测时走弯路。

## 二、当前真实可选的中文 ASR 引擎（2024-2026）

按"对 8 GB VRAM 消费级硬件 + aistack 的研究形态友好度"排序：

| 引擎 | 来源 | 参数 | 公开 avg CER¹ | 8GB | License | 设计定位 |
|---|---|---|---|---|---|---|
| **SenseVoiceSmall** | Alibaba / FunASR | ~234 M | 不在主流 benchmark² | ✅ 极轻松 | MIT (FunASR) | 轻量多语种 + 情感/事件标签 |
| **SenseVoice-Large** | Alibaba / FunASR | ~1.6 B | 4.47% | ✅ 轻松 | MIT | 多语种通用大版本 |
| **Paraformer-large** | Alibaba / FunASR | ~220 M | 4.56% | ✅ 极轻松 | MIT | 阿里中文专精 SOTA 老大哥 |
| **Paraformer-large-vad-punc** | Alibaba / FunASR | ~220 M + VAD/标点 | 同上 + 自带标点 | ✅ 极轻松 | MIT | Paraformer + 标点恢复 + VAD，开箱即用 |
| **FireRedASR-AED** | 小红书 / 2025-01 | **1.1 B** | **3.18%** | ✅ 较紧但能跑³ | Apache-2.0 | **当前性价比之王**，CER 比 Paraformer-large 低 30%，参数才 5x |
| **FireRedASR-LLM** | 小红书 / 2025-01 | 8.3 B | 3.05% | ❌ | Apache-2.0 | 质量天花板，但 8 GB 卡跑不动 |
| **FireRedASR2S-AED** | 小红书 / 2025 | 未公布 | 未公布 | ✅ 推测 | Apache-2.0 | v2 升级版，加 VAD/LID/Punc |
| **FireRedASR2S-LLM** | 小红书 / 2025 | 未公布 | **2.89%** | ❌ | Apache-2.0 | 当前公开 SOTA，但太大 |
| **Fun-ASR-Nano** | Alibaba / Tongyi 2025-12 | 未公布⁴ | 未公布 | 推测 ✅ | 商业 SaaS 为主⁵ | 多方言（7 大方言+26 口音）、低幻觉（78.5%→10.7%） |
| **Whisper-large-v3** | OpenAI | ~1.5 B | 在中文上明显落后专精模型⁶ | ✅ | MIT | 多语种通用基线 |
| **Belle-whisper-large-v3-zh** | BELLE / 2024 | ~1.5 B | 比 Whisper-v3 提升 24-65% | ✅ | Whisper 衍生 | 中文微调版，社区 fine-tune |
| **Qwen2-Audio** | Alibaba | 7 B+ | — | ⚠ 紧 | 商业许可 | 多模态 LLM，强在"听懂"不在纯转写效率 |

¹ avg CER = AISHELL-1 + AISHELL-2-ios + WenetSpeech-net + WenetSpeech-meeting 四集平均。来源 [FireRedASR paper](https://arxiv.org/abs/2501.14350)
² SenseVoiceSmall 在 FunASR 论文里没列在主流四集对比表，只在 SenseVoice 自家 paper 里（其他设置）报中文 CER
³ FireRedASR-AED 1.1B 在 fp16 下约 ~2.2 GB VRAM；加激活和 KV 后 4-5 GB 范围，8 GB 卡有空间但不如 Paraformer 宽裕
⁴ Fun-ASR 是 Alibaba 2025 年 9 月推出的新一代语音识别大模型；Nano 版 2025-12 发布，强调实时性和多方言
⁵ Fun-ASR 主要走 Tongyi DashScope 商业 API；GitHub 仓库已开源但训练数据规模决定它仍是产品向，不是研究向
⁶ Whisper-large-v3 在中文上的 CER 没有官方 AISHELL/WenetSpeech 数字，但社区一致结论是"明显落后专精模型"，Belle 等微调版的"24-65% 相对改进"反向佐证

## 三、各引擎的设计意图（避免误解卖点）

### SenseVoice 系列（小+大）
- **真正的卖点**：234M 小模型同时做中/英/日/韩 + 情感识别 + 声学事件检测，**用一个模型解决"语音理解"而不是只做转写**
- **不是天花板的原因**：阿里自家做"纯中文 SOTA"用的是 Paraformer 与 Fun-ASR，SenseVoice 走的是另一条产品路径
- **适合什么**：需要语种切换 + 情感标注 + 实时字幕的场景（直播、会议）
- **不适合什么**：高质量中文转写——它的中文 CER 在主流对比里垫底

### Paraformer-large
- **设计点**：非自回归（NAR）端到端，**没有 Whisper 的 30s padding 限制**，单步并行解码所以快
- **质量定位**：长期被认为开源中文 ASR SOTA 的强基线，FireRedASR 出现前是首选
- **变体**：`speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`（裸）+ `speech_paraformer-large-vad-punc_asr_*`（带 VAD 和标点）。**带 VAD-Punc 的版本更适合直接生产**
- **集成成本**：FunASR 已经在 aistack 里装着（SenseVoice 用的同一个 runtime），加 Paraformer 是最小增量

### FireRedASR (v1 + v2S)
- **设计点**：小红书自家工业级模型，AED 和 LLM 双轨。AED 是"实用版"，LLM 是"质量天花板"
- **公开 SOTA**：FireRedASR-LLM 在 paper 里把 Paraformer-large 的 5.80% 压到 3.48%（38.6% 相对改进）。FireRedASR2S-LLM 进一步到 2.89%
- **AED 的实操价值**：1.1B 参数，公开 CER 3.18%，**比 Paraformer-large 低近 30%，仅是 5x 参数代价**。这是"一上来就该接进来"的引擎
- **风险**：① 不是 FunASR runtime，需要单独安装依赖（自带的 inference 包）② **40-60s 输入硬限制**，长音频必须应用层分段 ③ 社区比 FunASR 小一档，"Issues 多久回复"不确定
- **代码许可**：Apache-2.0，干净

### Fun-ASR (Tongyi 2025)
- **设计点**：阿里 Tongyi Lab 2025 年新一代 ASR 大模型，强调多方言、低幻觉、实时
- **关键数字**：在高噪场景把 hallucination 率从 78.5% 砍到 10.7%（这是 Whisper 时代以来最值得注意的改进方向之一）
- **风险**：主要走 Alibaba DashScope 商业 API；开源版本（GitHub `FunAudioLLM/Fun-ASR`）能用，但训练数据规模、weights 完整性、长期维护承诺都不如 Paraformer 那条传统线确定
- **何时考虑**：等 Nano 版本（端到端实时 ASR）的开源完整度更确认后再评估

### Whisper-large-v3 (基线参考)
- **意义**：作为多语种通用模型，在中文上的表现是"参照系"，不是产品选项
- **fine-tune 路径**：BELLE-2 出的 `Belle-whisper-large-v3-zh` 把中文 CER 改善 24-65%，比裸 Whisper 强很多但仍不及 FireRedASR
- **何时用**：① 需要同时支持中英日韩多语种且模型存储紧张时 ② 跨语种通用 baseline 测试时

### 不在候选范围
- **FireRedASR-LLM (8.3B)** / **FireRedASR2S-LLM** / **Qwen2-Audio (7B+)** —— 8 GB 卡跑不动或紧到不实用；研究形态记录其存在但不实测
- **Conformer/Branchformer 各种学术变体** —— 没有公开生产级权重，研究形态可以引用论文，不投入实测

## 四、公开 benchmark 数据（前人结论汇总）

四个标准测试集的 CER（数字越低越好），来源 [FireRedASR paper Table 1](https://arxiv.org/abs/2501.14350)、[FunASR paper](https://arxiv.org/abs/2305.11013)、阿里 ModelScope 模型卡：

| 模型 | AISHELL-1 | AISHELL-2-ios | WenetSpeech-net | WenetSpeech-meeting | avg |
|---|---|---|---|---|---|
| FireRedASR-LLM (8.3B) | 0.76 | 2.15 | 4.60 | 4.67 | **3.05** |
| FireRedASR-AED (1.1B) | **0.55** | 2.52 | 4.88 | 4.76 | 3.18 |
| FireRedASR2S-LLM (?) | — | — | — | — | **2.89** (Mandarin avg) |
| SenseVoice-L (~1.6B) | — | — | — | — | 4.47 |
| Paraformer-large (~220M) | 1.95 | 2.85 | — | 6.97 | 4.56 (Fire 论文同设置)¹ |
| Whisper-large-v3 (~1.5B) | — | — | — | — | 远高于专精模型² |

¹ FireRedASR 论文中的 Paraformer-large 5.80% 与 FunASR 自报 1.95% 数字差异说明：不同论文采用不同的 Paraformer 版本（vad-punc 与否）、不同的归一化策略、不同的 WenetSpeech 子集分组，**单数比较要慎重看 setup**
² Whisper 的中文 CER 在公开对比中通常被报为"明显落后"，没有标准数字。Belle-whisper-large-v3-zh 的 24-65% 相对改进暗示原版 Whisper 中文 CER 在 10%+ 量级（具体待我们实测）

**关键观察**：
- AISHELL-1（朗读、安静、专业播音）的 CER 已经被卷到 < 1%，**作为 dosmoon 实际用例（新闻/播客/评论）的代表性极差** —— 这不是真实应用场景的难度
- WenetSpeech 才是真实场景代理：net 域是网络音频（B站、播客、视频博主），meeting 域是会议录音，**CER 普遍在 4-7% 区间**，这才是实操难度
- 模型间的真正差距体现在 WenetSpeech 上，AISHELL-1 上几乎都 < 1%（区分度低）

## 五、CER 评估方法的陷阱

中文 ASR 评测必须用 **CER（字错误率）** 不是 WER（词错误率）。原因和注意事项：

### 为什么不用 WER
- **中文没有自然词边界**（"我去北京" 是 4 字 / 2 词 / 3 词都有人切）
- 不同分词工具（jieba / pkuseg / hanlp）切出的词数和边界都不同，作为 metric 单位不稳定
- 字符级度量是"无歧义"的最小颗粒，不依赖分词

### 必须做的归一化
评测前必须把 ASR 输出和 ground truth 归一化到一致表示，否则数字不可比：

| 维度 | 问题 | 处理 |
|---|---|---|
| 数字 | "5000" vs "五千" | 统一为汉字数字（或都转阿拉伯）|
| 标点 | 模型有的输出有的没有；"。" vs "."| 测纯字时把标点全删；测带标点版本时统一标点形式 |
| 全角/半角 | "ＡＢＣ" vs "ABC" | 全部转半角 |
| 英文大小写 | "OK" vs "ok" | 统一小写或大写 |
| 空格 | 中英混排的空格位置 | 删除所有空格再算 CER（或用一致的 tokenization） |
| 繁简 | "雲端" vs "云端" | 统一转简体（或保留原样要看 ground truth）|
| 同音异形不处理 | "在" vs "再"是真错，不该归一化 | 不处理 |

### 评测工具
- **`jiwer`**（Python 库）—— 最常用，支持自定义 transformation。计算 CER 直接 `jiwer.cer(reference, hypothesis)`
- **ESPnet `sclite` wrapper** —— 学术圈常用，跟论文报数字最一致
- **`zhconv`** —— 繁简转换
- 推荐 aistack 用 jiwer + zhconv + 自写的数字归一化函数（参考 FunASR `funasr/utils/postprocess_utils.py` 里的现成实现）

### 已知坑
- 论文里报的 CER 数字**不一定都用同一套归一化**，跨论文比要看 setup
- AISHELL-1 几乎所有模型都 < 1%，**用它单独排序模型基本没用**，必须看 WenetSpeech
- 如果某个引擎的输出带标点而 ground truth 不带，**直接算 CER 会被标点错配灌水**

## 六、数据集选型策略

### 标准公开测试集
| 集 | 时长 | 内容 | 难度 | dosmoon 场景代表性 |
|---|---|---|---|---|
| AISHELL-1 test | 5h | 专业播音员朗读 | 极低 | 极差 |
| AISHELL-2 test-ios | 5h | iOS 录音、家庭环境 | 中 | 中 |
| WenetSpeech test-net | 24h | B站/播客/视频博主 | 中-高 | **高** ✓ |
| WenetSpeech test-meeting | 15h | 会议录音 | 高 | 中（dosmoon 不主做会议）|
| SpeechIO test sets | 多 | 真实多场景（YouTube/TV/播客）| 中-高 | **高** ✓ |
| Common Voice zh-CN | 多 | 众包朗读 | 低 | 低 |

### 推荐的 dosmoon 中文 bench 组合

考虑到 dosmoon 的实际内容形态（新闻、政治、经济、播客评论），建议三层组合：

**L1: 标准对照基线**（几分钟，跑得快）
- AISHELL-1 dev 抽 5-10 段（不超过 30s）—— 跟论文数字对得上的"sanity check"
- WenetSpeech test-net 抽 5-10 段 —— 真实音频代表

**L2: 长音频（reflects real use）**
- 自采或公开来源的 1 段 17-25 min 中文新闻/访谈片段，人工或半自动出 ground truth
- 这一层是 dosmoon 真用例的代理；跨引擎在长音频上的稳定性差异会在这里暴露

**L3: 困难样本（catch the failure modes）**
- 带背景音乐的播客片段
- 多人对话片段
- 含专有名词、外语词、数字密集的政经评论
- 这一层不是为打分，是为"看哪个引擎在难点上崩"

**首版只做 L1+L2**，L3 留作后续研究。

### 不推荐的做法
- ❌ 只用 AISHELL-1 评测 —— 区分度低，结论无意义
- ❌ 用合成或 TTS 音频做 ground truth —— 跟真实场景脱节
- ❌ 跑超过 6 小时的全量测试集 —— 实测开销大，研究形态不需要论文级 reproducibility

## 七、长音频处理策略

dosmoon 的实际内容（17-50 min 新闻、播客）远超所有引擎的单次输入限制：

| 引擎 | 单次输入限制 | 推荐长音频策略 |
|---|---|---|
| Whisper / faster-whisper | 30s 内部窗口（自动分段）| 引擎自己分段，但要 VAD filter |
| Paraformer | 无硬限制（内存允许就行）| 直接喂，或前置 VAD 切 30s 段 |
| SenseVoice | 单次 ~30s 推荐 | 前置 VAD（FunASR pipeline 自带 fsmn-vad）|
| FireRedASR-AED | **60s 硬限制** | 应用层 VAD 分段 + 后处理拼接 |
| FireRedASR-LLM | **40s 硬限制** | 同上，更紧 |
| FireRedASR2S | 同 v1 | 同上 |

aistack 已经在 Parakeet 路径上实现过应用层 12 分钟切片 + word-LCS 缝合（见 `parakeet-on-consumer-gpu.md` 增补部分）。**这套机制可以复用到 FireRedASR-AED 上**，但分段粒度要从 12min 改到 ~50s（贴近 60s 上限留余量），缝合策略可能要从 word-LCS 改为字符-LCS（中文没有词边界）。

## 八、运行时实操考量（aistack 8 GB 卡视角）

### 集成成本对比
| 引擎 | 已装依赖 | 增量依赖 | 模型大小 | 工作量 |
|---|---|---|---|---|
| Paraformer-large(-vad-punc) | ✅ FunASR + funasr 已装 | 0 | ~880 MB 下载 | **~2 小时**（仿照 sensevoice.py 写一个 paraformer.py）|
| SenseVoiceSmall | ✅ 已装 + 已集成 | 0 | 已下载 | 0（已在生产路径上）|
| FireRedASR-AED | ❌ | FireRedASR 自家 inference 包 + PyTorch | ~4.4 GB 下载 | **~半天**（新建 aistack/asr/fireredasr.py，写 chunked 模式适配 60s 限制）|
| FireRedASR2S-AED | ❌ | 同上但更新 | 未知 | 半天-1 天，等社区稳定一点再做 |
| Whisper-large-v3 | ✅ faster-whisper 已装 | 0 | ~3 GB 下载 | 0（API 层 `model=large-v3` 即可）|
| Fun-ASR (开源版) | 部分 | 待评估 | 待评估 | 等 weights/inference 稳定后再做 |

### VRAM 共存预算（关键约束）
8 GB 卡上同时挂多模型时：
- Paraformer + SenseVoice + Whisper-small 共存 ≈ 总占用 < 4 GB，宽松
- FireRedASR-AED 单独驻留 ~5 GB，**与 Ollama 大模型不能共存**，必须靠 `_model_cache` 的 `evict_category` 机制热切换
- aistack 现有的 hot-swap 机制（asr-main 互斥驻留）已经能处理这个，FireRedASR 新增时延续相同 `category="asr-main"` 即可

### License 一览
全部 permissive，无传染：

| 引擎 | License | 商用 | aistack/产品形态接进来 |
|---|---|---|---|
| Paraformer / SenseVoice (FunASR) | MIT | ✅ | ✅ |
| FireRedASR / FireRedASR2S | Apache-2.0 | ✅ | ✅ |
| Whisper / faster-whisper / CTranslate2 | MIT | ✅ | ✅ |
| Belle-whisper-large-v3-zh | 受 Whisper 上游 MIT 影响 | ✅ | ✅ |
| Fun-ASR | 待确认（GitHub 仓库的开源条款）| 待确认 | 暂缓 |
| Qwen2-Audio | 商业许可（Alibaba 自有）| ⚠ 需读条款 | 暂不评估 |

## 九、aistack 的中文 ASR 实测计划（动手时分阶段）

> 本文是 desk research，本节只列计划不动手。真正实施时按 backlog 排期。

### Phase 0（基础设施，必须先做）
- 在 `bench/audio/` 加 L1+L2 中文音频（AISHELL 抽样 + WenetSpeech 抽样 + 1 段长音频）
- 写 `bench/zh_eval.py`，jiwer + zhconv + 数字归一化，输出 CER
- 把现有 `bench/asr_eval.py` 的英文 WER 评估留着，并行不冲突

### Phase 1（最小增量价值）
- 加 `aistack/asr/paraformer.py`（仿 sensevoice.py 结构，FunASR runtime 复用）
- 在 `_select_for_auto` 增加 zh 的 Paraformer 优先分支
- 跑 SenseVoice vs Paraformer-large 在 L1+L2 上的 CER 对比，写日报

### Phase 2（接 FireRedASR-AED）
- 加 `aistack/asr/fireredasr.py`，集成 FireRedASR 自家 inference 包
- 写 60s 分段 + 拼接逻辑，参考 Parakeet chunked 模式
- 对比 Paraformer / SenseVoice / FireRedASR-AED 在 L1+L2 上

### Phase 3（决定 zh 路由的最终选择）
- 看 Phase 2 数据，决定 `_select_for_auto` 对 zh 默认路由到谁
- 把决策依据写入日报和 aistack 产品路径设计文档（内部）

### 永远不在 aistack 里做的
- FireRedASR-LLM (8.3B+) 集成 —— 太大，不适合 aistack 的研究硬件配置
- Qwen2-Audio 集成 —— 商业许可 + 部署复杂度，跨过 aistack 边界
- Fun-ASR DashScope API 集成 —— 那是云服务，不是本地引擎，跟 aistack 定位冲突

---

## Open questions（待我们实测验证）

1. **SenseVoiceSmall 的真实中文 CER**：公开 benchmark 表里只列 SenseVoice-L 的 4.47%，Small 版没有标准数字。我们需要自己量一份
2. **FireRedASR-AED 在 dosmoon 长音频上的 60s 分段稳定性**：分段策略不同会不会让 CER 漂移？word-LCS 在中文（无词边界）上能不能复用，还是必须切到字符级？
3. **WenetSpeech-net 子集对 dosmoon 真实场景的代表性**：B 站/播客内容跟 dosmoon 关心的国际新闻评论有多接近？必要时自采小样本
4. **Whisper-large-v3 + 中文 fine-tune（Belle 系）的实际改进幅度**：报的 24-65% 相对改进，在长音频上能保持多少？
5. **`speech_paraformer-large-vad-punc_*` 自带的 VAD 与 aistack 现有的 fsmn-vad 是否冲突**：FunASR 的 pipeline 配置层会不会有路径打架
6. **多个 FunASR 模型同时驻留**（SenseVoice + Paraformer 共存）的真实 VRAM 占用：模型卡上的"234 M"和"220 M"加起来不等于 ~500 MB，运行时 buffers 会怎样

## 参考文献

### 论文
- [FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration](https://arxiv.org/abs/2501.14350) (2025-01)
- [FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System](https://arxiv.org/html/2603.10420v1) (2025)
- [Fun-ASR / FunAudio-ASR Technical Report](https://arxiv.org/abs/2509.12508) (2025)
- [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://www.isca-archive.org/interspeech_2023/gao23g_interspeech.pdf) (Interspeech 2023)
- [Advocating Character Error Rate for Multilingual ASR Evaluation](https://aclanthology.org/2025.findings-naacl.277.pdf) (NAACL 2025)
- [What is lost in Normalization? Exploring Pitfalls in Multilingual ASR Model Evaluations](https://arxiv.org/html/2409.02449v3)
- [Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation](https://arxiv.org/abs/2510.06961)

### 仓库与模型卡
- [FireRedTeam/FireRedASR (GitHub)](https://github.com/FireRedTeam/FireRedASR)
- [FireRedTeam/FireRedASR2S (GitHub)](https://github.com/FireRedTeam/FireRedASR2S)
- [FireRedTeam/FireRedASR-AED-L (HuggingFace model card)](https://huggingface.co/FireRedTeam/FireRedASR-AED-L)
- [modelscope/FunASR (GitHub)](https://github.com/modelscope/FunASR)
- [FunAudioLLM/Fun-ASR (GitHub)](https://github.com/FunAudioLLM/Fun-ASR)
- [BELLE-2/Belle-whisper-large-v3-zh (HuggingFace)](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh)
- [SpeechColab/Leaderboard (GitHub)](https://github.com/SpeechColab/Leaderboard)

### 评测工具
- [jiwer (Python WER/CER library)](https://github.com/jitsi/jiwer)
- [zhconv (繁简转换)](https://github.com/gumblex/zhconv)
- [Open ASR Leaderboard (HuggingFace Space)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
