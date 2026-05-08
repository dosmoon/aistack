# Whisper 翻译能力的真实边界

> TL;DR
>
> Whisper 的 `task=translate` 是 **X → English only**，训练数据决定，不是配置问题。
> 它**不能**把英文音频翻成中文，也不能做任何 "non-English target" 翻译。
> 想要 "EN→ZH 字幕" 必须级联：Whisper 转写 + LLM 翻译；或者换模型（SeamlessM4T / Qwen2-Audio）。

---

## 一、Whisper 翻译方向是单向的

Whisper 的 transcribe API 接受一个 `task` 字段，OpenAI 文档列出两个值：

| task | 行为 | 输出语言 |
|---|---|---|
| `transcribe` | 语音转写 | 源语言原文 |
| `translate` | 语音翻译 | **固定为英文** |

这个限制在 [Whisper paper](https://arxiv.org/abs/2212.04356) §2.1 ("Data Processing") 写明：
> "We then construct a dataset by combining audio with transcripts in the same language, as well as audio with English translations of the speech in the audio."

训练数据只有两种 pair：① 同语言音频+文本 ② 任意语言音频 + 英文翻译。**没有 X→Y（non-English target）的样本**。因此模型根本没学过把语音翻成英文以外的任何语言。

实测表现：

- 给 Whisper 喂英文音频 + `task=translate` → 输出仍是英文（等于 transcribe，没翻）
- 给 Whisper 喂中文音频 + `task=translate` + initial_prompt="output in Spanish" → 仍输出英文，prompt 无效
- 社区 issues 反复确认：[openai/whisper#649](https://github.com/openai/whisper/issues/649), [#2046](https://github.com/openai/whisper/issues/2046)

**这是模型能力的硬边界**，不是 API 限制，不能通过 prompt engineering 绕过。

## 二、X → English 方向的实际质量

这是 Whisper 真正会做的方向。社区与官方评测的共识：

| 源语言 | 质量定位 | 备注 |
|---|---|---|
| 法 / 德 / 西 / 葡 / 意 → 英 | 字幕级可用 | BLEU 在 large-v3 上约 25-32（CoVoST-2），不及现代 frontier LLM 直接做翻译 |
| 中 / 日 / 韩 → 英 | 中等 | 准确但意译倾向重，长句结构英化 |
| 阿拉伯 / 印地 / 越南 等 → 英 | 偏弱 | hallucination 风险升高 |
| 低资源（如斯瓦希里、孟加拉） → 英 | 不可靠 | 整段编造常见 |

已知毛病（与 transcribe 共享 + 翻译特有）：

- **意译胜过直译**——专业字幕需要的"贴原文"不是 Whisper 的强项
- **长片段截短** —— 30 秒窗口边界处会丢内容
- **数字/专有名词丢失** —— "$3.5 million in 2023" 可能变成 "millions of dollars"
- **VAD 失效段落 hallucinate** —— 静音段编出整句翻译

## 三、想做 EN→ZH 字幕的三条可行路径

| 方案 | 形态 | 可控性 | 离线友好 | 质量上限 | 部署成本 |
|---|---|---|---|---|---|
| **A. Whisper(EN→EN) + LLM(EN→ZH)** | 级联 | 高（两段独立调）| ✅ 全本地 | 高（取决于 LLM） | 低（aistack 已具备）|
| **B. SeamlessM4T-v2** (Meta) | 一次性 speech→text translation | 中 | ✅ | 中-高（直接 EN→ZH） | 中（模型 2-9 GB，PyTorch 部署）|
| **C. Qwen2-Audio Instruct** | 一次性，prompt 驱动 | 高（自然语言指令）| ✅ | 高 | 中-高（vLLM 部署，~14 GB）|

**方案 A 是 aistack 上最自然的路径**：
- Whisper 走 `transcribe`（不是 `translate`），出英文原文
- 把英文文本喂给 Ollama 上的中等以上 LLM（qwen2.5:14b、qwen2.5:32b、deepseek-v2 等）做翻译
- 这一套 aistack 当前的 ASR + LLM-proxy 已经具备，只差封装

**方案 B 的吸引力**：少一次 LLM 调用，端到端延迟可能更低。但 SeamlessM4T 的模型大、vocabulary 控制不如 LLM 灵活、专有名词处理与 prompt 注入不便。**长片段转写的稳定性也未经我们实测**。

**方案 C 的吸引力**：质量上限最高（多模态 LLM 真正"听懂"再翻），但部署最重，跟 aistack 的研究形态契合度也是最高的（dosmoon 真要研究"音频理解+翻译一体化"的天花板，C 是该走的路）。

## 四、对 aistack 与未来产品的影响

- **aistack 短期价值点**：增加一个"transcribe-and-translate" 的 convenience endpoint（方案 A 级联），不引入新依赖，对下游消费者（VideoCraft 等）立刻可用
- **aistack 研究价值点**：方案 A vs B vs C 的实测对比（同一段英文播客，三条路出中文字幕，比较 BLEU / 人工偏好评分 / 端到端延迟 / 显存峰值）—— 这正是 aistack 该做的事
- **离线产品形态的判断**：如果未来产品定位是"离线英中字幕工具"，**方案 A（whisper.cpp + 一个本地小 LLM）是唯一干净的路径**。SeamlessM4T 与 Qwen2-Audio 的体积/复杂度都超出"装上即用"形态能承受的范围

---

## Open questions（待实测）

1. **方案 A 的实操质量**：用 Whisper-large-v3 + qwen2.5:14b 跑一段 17 分钟英文演讲，跟 frontier LLM（Claude / GPT-4 级别）直接做语音→中文文本的输出做盲对比，能拉开多大差距？本地全离线方案离闭源天花板的距离是产品决策关键数据。**不与 DeepL 等专业 NMT 工具对比**——既有经验已确认现代 LLM 翻译质量整体优于 DeepL，没有研究价值
2. **方案 B 的长音频稳定性**：SeamlessM4T 官方 demo 都是 < 30 秒片段，长音频要不要分段？分段策略与 Whisper 一致还是另立？
3. **方案 C 的 prompt 控制力**：能不能通过 prompt 让 Qwen2-Audio 输出"贴原文"或"贴本地化"两种风格？这是产品差异化的可能旋钮
4. **Whisper 的"伪 EN→ZH"**：是否存在社区 fine-tuned 的 Whisper 变体能直接做 EN→ZH？（已知有少量尝试如 whisper-large-v2-cn-translate，但训练数据规模与质量不明）
