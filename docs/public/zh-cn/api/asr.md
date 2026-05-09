---
title: POST /v1/audio/transcriptions
description: 语音转文字端点。兼容 OpenAI Whisper API。支持 faster-whisper、NVIDIA Parakeet、阿里 SenseVoice，以及按语种自动路由。
sidebar:
  order: 3
---

# `POST /v1/audio/transcriptions`

转写一个音频文件。兼容 OpenAI Whisper API —— 针对 OpenAI Whisper 端点
写的客户端，把 base URL 指到 aistack 即可，无需改任何代码。

完整的请求与响应 schema（每个表单字段、每种响应变体、每个错误码）在
自动生成的 [ASR reference](./reference/asr/) 里。本页讲的是 *为什么* ——
后端选择动因、segment 粒度的设计取舍、各后端的具体行为、性能区间、
以及并发契约。

## 选择后端

`model` 字段接受 faster-whisper 各尺寸（`whisper-tiny` ...
`whisper-large-v3-turbo`、`whisper-distil-large-v3`）、Parakeet 的
HuggingFace id、SenseVoice 的 HuggingFace id，或者 `auto` 路由别名。
被接受的完整形式列表在
[reference](./reference/asr/) 里 —— 本节解释你应当 *用哪个*：

- **英语长音频、对吞吐敏感** → `parakeet`
  （非常快，不支持翻译，不支持流式）。
- **中文 / 粤语 / 日语 / 韩语** → `sensevoice`
  （为 CJK 而设计；`auto` 在 `language` 是 CJK 时会路由到这里）。
- **任何多语种、或英语短音频，需要流式或翻译** → `whisper-{size}`。
  按你场景需要的速度/质量权衡来挑尺寸。
- **不想选** → `auto`（或者省略 `model`）。aistack 按 `language`
  挑选已安装的最佳后端。

各后端覆盖速览：

| 后端                                  | 语言                                | 流式  | 翻译（`task=translate`） |
|--------------------------------------|------------------------------------|-----------|------------------------------|
| faster-whisper（Whisper 各尺寸）       | Whisper 支持的全部 99 种            | 支持      | 支持（X → English only）     |
| Parakeet TDT v3                      | 英语 + 24 种欧洲语言                 | 不支持¹    | **不支持**                    |
| SenseVoice Small                     | zh / yue / en / ja / ko             | 支持      | **不支持**                    |

¹ 网关在 Parakeet 上接受 `stream=true`，但会以单事件 SSE 降级 + 一个
`warning` 事件返回；它不会悄悄地把一个非流式模型切片，付掉 WER 的代价。

## Segment 粒度

`verbose_json` 中的 `segments` 数组（以及 `stream=true` 下逐段的 SSE
delta）可以按两种方式分组：

- `"sentence"`（默认）—— 每段是一个完整句子；句长不设上限，但有
  30 秒的安全上限。是 LLM 逐行翻译、语义检索、agent 推理、与另一份
  转写做对齐的正确输入。
- `"subtitle"` —— 每段 ≤ 70 字符、1–7 秒，遵循 stable-ts /
  字幕本地化行业标准。**句子太长时会在句中切。**

### 为什么默认 `"sentence"`

`"subtitle"` 是给某一种特定调用方（直接写 SRT 的）准备的形状，但把字
幕尺寸的 cue 喂进按行翻译的 LLM 会产生坏的翻译：半句失去时态和指代，
LLM 不得不去猜缺失的上下文，目标语种输出变得不连贯。

行业惯例（OpenAI Whisper verbose_json、WhisperX、stable-ts、FunASR）
是 **`segments` = 句级**，SRT 的 cue 尺寸是独立的下游步骤。aistack
遵循同样的惯例。

### 何时用 `"subtitle"`

- **只** 生成 SRT/VTT 且不做任何后续转换的流水线。省去自己实现 cue
  尺寸的工作。
- 已经有不错的下游 cue 切分、只想让 aistack 给出 ASR-only 路径终态
  格式的流水线。

### 何时坚持 `"sentence"`

- **任何 LLM 翻译流水线。** 即使最终输出是 SRT，也先在句级粒度上翻
  译，再对翻译后的文本做 cue 切分 —— 翻译质量压倒一切。
- 语义检索、对齐、摘要、agent 在转写上的推理。
- 任何会自己做句子检测、否则得把字幕 cue 重新拼回去的调用方。

对于既要翻译句子又要出 SRT 的流水线，推荐流程是：
`segment_granularity=sentence` 翻译，再在翻译后的句级 segment 上跑下
游 cue 切分（自己写的，或 stable-ts）。

`segment_granularity` 仅影响 Parakeet。faster-whisper 和 SenseVoice
原生就产出由 VAD 驱动的句级 segment。

## 各后端值得知道的怪癖

### faster-whisper（Whisper 各尺寸）

- 默认 `device="auto"`：`torch.cuda.is_available()` 时选 CUDA，否则
  选 CPU（int8 量化）。
- 尺寸越小越快但越不准；按你场景需要的速度/质量权衡来挑。
- 支持 `translate=true`，把非英语音频转为 **英语**（X → en only ——
  Whisper 训练数据里没有任何其他目标语种）。

### Parakeet（NVIDIA NeMo）

- 在 25 种欧洲语言加英语之间自动检测。`language` 提示会被记到响应里
  但不约束模型 —— 由模型决定。
- `translate=true` **不支持**；返回 400 `malformed`。
- 欧洲语种 ASR 的最佳选择；中文 / 粤语 / 日语 / 韩语不要往这里路由。

### SenseVoice（阿里 FunASR）

- 中文和粤语最强；同时覆盖日语、韩语、英语（英语输出会省略词间空格 ——
  CJK 中心 tokenizer 的怪癖；如有需要请后处理）。
- `language` 接受 `"auto"`、`"zh"`、`"en"`、`"yue"`、`"ja"`、`"ko"`。
- `translate=true` **不支持**；返回 400 `malformed`。
- VAD + 主模型作为一对加载；首次使用时两个会先后加载。

## 冷启动

对某个后端的第一次请求要付模型加载成本。在缓存空闲窗口内
（默认 5 分钟，可通过 `AISTACK_MODEL_KEEP_ALIVE_SEC` 配置）的后续请求，
模型是热的，只付推理延迟。

RTX 4060 Laptop（8 GB VRAM）上的近似首次调用延迟：

| 后端                    | 冷启动（加载 + 首次推理） | 热（200 秒音频） |
|------------------------|---------------------------:|------------------:|
| faster-whisper-small   |                       ~14s |             ~10 s |
| Parakeet TDT 0.6B      |                       ~25s |              ~2 s |
| SenseVoice Small       |                       ~20s |              ~6 s |

CPU 模式延迟会是数倍以上。完整基准和"在紧凑 VRAM 上仅 GPU 部署"的
理由见仓库里的 `docs/research-note/parakeet-on-consumer-gpu.md`
与 `consumer-gpu-asr-baseline.md`。

## 并发

aistack 串行化 ASR 推理：本 worker 上同时最多一个转写任务（同一个全
局 GPU 槽位与 TTS 和 LLM 共享）。并发请求会立刻得到 **HTTP 503** 加
`Retry-After: 5`（不排队）。槽位忙的响应结构形状以及调用方的重试模
式见 [errors](./errors/)。调用方应当尊重 retry 提示去退避，而不是猛
打。

## 稳定性

OpenAI 规范字段（`file`、`model`、`language`、`response_format`、
`translate`）以及响应形状（`text`、`verbose_json`）在 `/v1` 内稳定。

被接受的 `model` 值集合 **会变** —— 后端被加入或移除时；把它当作配
置，不是契约。用 `GET /v1/models` 在每个部署上发现当前集合。
