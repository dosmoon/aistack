---
title: 集成指南
description: 如何从客户端集成 aistack —— 能力发现、请求、错误、流式，以及完整的调用方接入路径。
sidebar:
  order: 1
---

# aistack 集成指南

> **本文是 aistack 向调用方公开的契约。** 它是"aistack 做什么、怎么用"
> 唯一的权威来源，适用于一切客户端 —— CLI 工具、像 VideoCraft 这样的
> GUI 应用、agent 框架、未来的 dosmoon 产品。aistack 不会迁就任何特定调
> 用方的需求；调用方按本文档化的内容来对接。
>
> 如果你是第一次集成 aistack，从这里开始。每个端点的设计页（在
> [API](./api/) 下面 —— `asr` / `tts` / `llm` / `models` / `errors` /
> `observability`）讲的是每项能力背后的 *为什么*；自动生成的
> [Reference](./api/reference/) 覆盖字段级别的 *是什么*。本指南把它们
> 串成一条连贯的接入路径。

---

## 1. aistack 是什么

aistack 是一个 **本地 AI 能力网关**。一台兼容 OpenAI API 的服务器
（默认 `127.0.0.1:11500`），对外暴露三种能力 —— 语音转文字（ASR）、
文字转语音（TTS）、对话补全（LLM）—— 并独自承担本地 GPU 调度，调用
方不必关心。

| 你（调用方） | aistack |
|---|---|
| 发送 OpenAI 形态的 HTTP 请求。 | 决定哪个后端来服务该请求。 |
| 读取 OpenAI 形态的响应。 | 协调本地 GPU 在所有后端之间分配。 |
| 不知道也不关心是哪个后端跑了你的请求。 | 在显存吃紧时热切换模型。 |
| 不需要在你的应用里携带 ML 库。 | 在本地承载 ML 库或代理它们。 |

这个契约刻意收得很窄：稳定的 HTTP 形状、稳定的错误响应结构、通过
`/v1/models` 做能力发现、以及 503 + `Retry-After` 作为通用的"我现在
忙不过来"信号。

---

## 2. 端点速览

| Method | Path | 用途 | Reference |
|---|---|---|---|
| GET | `/health` | 存活探针。 | 本页 §10。 |
| GET | `/v1/models` | 能力清单 —— 网关现在能提供哪些能力。 | [`models`](./api/models/) |
| POST | `/v1/audio/transcriptions` | 语音转文字。 | [`asr`](./api/asr/) |
| POST | `/v1/audio/speech` | 文字转语音（以及 `/v1/audio/*` 下其他相关 TTS 端点）。 | [`tts`](./api/tts/) |
| POST | `/v1/chat/completions` | 对话补全（代理到本地 Ollama）。 | [`llm`](./api/llm/) |

所有端点默认不做鉴权；aistack 设计上是绑定 `127.0.0.1` 或私有局域网。
如果要暴露到更广的网络，前面套一个带鉴权的反向代理。

---

## 3. 第一步 —— 发现可用能力

任何对接都从调 `GET /v1/models` 开始。这是网关向你做自我介绍的方式；
没有别的端点会告诉你哪些能力当前可用。

```bash
curl http://127.0.0.1:11500/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "auto",
      "object": "model",
      "owned_by": "aistack",
      "capabilities": ["asr"],
      "is_routing_alias": true
    },
    {
      "id": "whisper-small",
      "object": "model",
      "owned_by": "openai",
      "capabilities": ["asr"],
      "languages": ["en", "zh", "ja", "ko", "es", "fr", "..."]
    },
    {
      "id": "iic/SenseVoiceSmall",
      "object": "model",
      "owned_by": "alibaba",
      "capabilities": ["asr"],
      "languages": ["zh", "yue", "en", "ja", "ko"]
    },
    {
      "id": "qwen3-tts-12hz-0.6b-customvoice",
      "object": "model",
      "owned_by": "qwen",
      "capabilities": ["tts"]
    },
    {
      "id": "qwen3:4b",
      "object": "model",
      "owned_by": "ollama",
      "capabilities": ["llm"]
    }
  ]
}
```

### 字段语义

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 原样作为 `model` 传给能力端点。 |
| `object` | string | 永远是 `"model"`（OpenAI 规范要求）。 |
| `owned_by` | string | 模型作者归属，自由文本。仅供展示，不要在代码里据此分支。 |
| `capabilities` | array of string | aistack 扩展。是 `["asr","tts","llm"]` 的子集。模型选择器按这个过滤。 |
| `languages` | array of string（仅 ASR） | aistack 扩展。该模型可转写的 ISO 639-1 语言码。TTS / LLM 条目和路由别名上没有此字段。 |
| `is_routing_alias` | boolean | aistack 扩展。`true` 表示这是 aistack 内部解析的虚拟 id，不是真实模型。当前只有 ASR 的 `id="auto"`。 |

### 何时刷新

- 启动时拉一次，按能力缓存。
- 用户每次打开模型选择器时（可能刚装了新后端，Ollama 可能刚启动）。
- **不要** 在每次推理调用前都拉 —— 这个接口便宜但不免费，并且在
  秒-分钟尺度上是稳定的。

### 某个条目缺失意味着什么

清单反映的是 **此时此刻** 的状态，不是静态目录：

| 后端 | 仅在以下条件下出现 |
|---|---|
| ASR provider | venv 里能 import 对应的 Python 库 |
| TTS（Qwen3-TTS） | Docker 容器自身的 `/health` 有响应 |
| LLM（Ollama） | aistack 能访问到 Ollama 的 `/api/tags` |

如果 TTS 容器挂了，`/v1/audio/speech` 会返回 503；如果 Ollama 挂了，
`/v1/chat/completions` 会返回 503。把 `/v1/models` 当作发现层，它会
告诉你 *不要* 一开始就去派发到一个缺失的能力上。

---

## 4. 第二步 —— 转写音频（ASR）

`POST /v1/audio/transcriptions` 镜像 OpenAI 的 Whisper API。接受
`multipart/form-data`。返回语种、时长、完整文本、按段落的时间戳，以及
后端支持时按词的时间戳。

### 选模型

三个真实后端在对应库已安装时暴露出来，加上 `auto` 路由别名。

| 选项 | 行为 |
|---|---|
| `model=auto` | aistack 根据 `language` 表单字段决定：CJK → SenseVoice，欧洲语种 → Parakeet，其他 → faster-whisper-small。当首选后端没装时优雅降级。 |
| `model=whisper-small`（或任意 whisper 尺寸） | faster-whisper / CTranslate2。默认通用选项。 |
| `model=parakeet` | NVIDIA Parakeet TDT 0.6B v3。英语/欧洲语种准确率最强。词级时间戳由模型本身给出。 |
| `model=sensevoice` | 阿里 SenseVoice Small。中日韩最佳；同时也处理英语/日语/韩语。 |

大多数调用方应该把 `auto` 作为选择器的默认项；高级用户可以钉死某个具
体后端。

### 示例：curl

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
  -F "file=@speech.mp3" \
  -F "model=auto" \
  -F "language=en" \
  -F "response_format=verbose_json"
```

### 示例：Python（httpx）

```python
import httpx

with open("speech.mp3", "rb") as f:
    r = httpx.post(
        "http://127.0.0.1:11500/v1/audio/transcriptions",
        files={"file": f},
        data={
            "model": "auto",
            "language": "en",          # 可选提示；驱动 auto 路由
            "response_format": "verbose_json",
            "translate": "false",       # 设为 true 启用 Whisper 专属的"翻译到英语"模式
        },
        timeout=120.0,
    )
r.raise_for_status()
result = r.json()
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")
```

### 响应结构（`response_format=verbose_json`）

```json
{
  "language": "en",
  "duration": 17.18,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.81, "end": 7.14, "text": "..."}
  ],
  "words": [
    {"start": 0.81, "end": 0.99, "word": "The"}
  ]
}
```

`words[]` 在每个支持词级时间戳的后端上都会被填充；客户端应该把它的缺
席视为"该后端/配置下不可用"，而不是错误。

### 用 `stream=true` 流式转写

对长音频（或任何客户端希望边出结果边消费的场景），传 `stream=true`
作为表单字段。响应是 `text/event-stream` 而不是 JSON；事件遵循 OpenAI
的转写流式形状，附带一个扩展事件。

示例：

```bash
curl -N -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
  -F "file=@long_lecture.mp3" \
  -F "model=whisper-small" \
  -F "language=en" \
  -F "stream=true"
```

线缆格式（每帧一个 `data: { ... }\n\n` 事件）：

```
data: {"type": "transcript.text.delta", "delta": "Hello world.",
       "segment": {"start": 0.0, "end": 1.7,
                   "words": [{"start":0.0,"end":0.4,"word":"Hello"}, ...]}}

data: {"type": "transcript.text.delta", "delta": "This is the second segment.",
       "segment": {"start": 1.7, "end": 4.2, "words": [...]}}

... (随着模型产出更多 segment 继续推送 delta) ...

data: {"type": "transcript.text.done", "language": "en", "duration": 1020.0}
```

#### 事件类型

| `type` | 含义 |
|---|---|
| `transcript.text.delta` | 增量段。`delta` 是该段文本；`segment` 携带秒级 start/end 和按词时间戳。 |
| `transcript.text.done` | 转写结束。携带检测到的语种和总时长。 |
| `warning`（aistack 扩展） | 在所选模型不支持真正的流式时，**先于** 任何 delta 发出。携带 `code`、`model`、`message`。warning 之后转写仍然以单个 delta 的形式到达。 |
| `error`（aistack 扩展） | 流中失败。body 匹配标准错误响应结构（`{kind, provider, message}`）。后续不再有事件。 |

#### 支持流式 vs 不支持

模型通过 `/v1/models` 中的 `supports_streaming` 声明流式行为。当前契
约下：

- `whisper-small`（以及任意 whisper 尺寸） —— **原生流式**：每解码一段
  发一个 delta，无 warning。
- `iic/SenseVoiceSmall` —— **原生流式**：每个 VAD chunk 发一个 delta，
  无 warning。
- `nvidia/parakeet-tdt-0.6b-v3` —— **不支持流式**：客户端先收到一个
  `warning` 事件，然后是携带完整文本的单个 delta，最后是
  `transcript.text.done`。在选择器里挑 Parakeet 是允许的，但敏感
  的客户端应该按 `supports_streaming` 过滤，把它从只支持流式的工作
  流中隐藏。
- `auto` —— 当语种提示路由到支持流式的后端时流式，路由到不支持的后端
  时降级。别名条目的 `supports_streaming` 是候选池的 AND，所以只要装
  了 Parakeet 就是 `false`。

降级路径会以单个事件的形式投递完整转写，而不是让请求失败 —— 这样始
终发 `stream=true` 的 OpenAI 形态客户端也能拿到可用响应。`warning`
事件是可发现的信号；敏感的客户端应该据此分支。

#### 取消

和 LLM 流（§6）同一套：关闭 HTTP 连接。网关轮询
`request.is_disconnected()` 并把取消令牌传递给 worker 线程，worker
在段与段之间检查。流式后端上长音频转写在断开 ~1 秒内中止；Parakeet
（降级路径）只在粗粒度边界上响应取消。

#### 与 `response_format` 的相互作用

当 `stream=true` 时，`response_format` 被忽略 —— 响应永远是上述形状
的 SSE。需要在 plain text / json / verbose_json 之间选择的调用方应
当让 `stream=false`。

---

## 5. 第三步 —— 生成语音（TTS）

`POST /v1/audio/speech` 是到本地运行的 Qwen3-TTS 容器的透明代理。接
受标准 OpenAI 字段（`model`、`input`、`voice`、`response_format`）；
上游同时暴露用于声音克隆与声音设计的扩展字段，这些字段原样透传。

### 示例：最小语音合成

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3-tts-12hz-0.6b-customvoice",
    "input": "Hello, this is a test of the local TTS gateway.",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output out.wav
```

### 示例：声音克隆（扩展字段透传）

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3-tts-12hz-0.6b-customvoice",
    "input": "Cloned-voice output.",
    "task_type": "voice_clone",
    "ref_audio": "/path/to/reference.wav",
    "ref_text": "Reference transcript matching the audio."
  }' \
  --output cloned.wav
```

代理是透明的 —— 你传出的每个字段、收到的每个字节都直接来自
Qwen3-TTS。完整字段表参见上游文档；上游演化时 aistack 会持续转发。

---

## 6. 第四步 —— 对话补全（LLM）

`POST /v1/chat/completions` 镜像 OpenAI 的 chat API。背后 aistack 代
理到本地运行的 Ollama 守护进程，并在网关层加两件事：

1. **转发前驱逐 `asr-main` 缓存。** 如果 aistack 最近在服务 ASR 后端，
   它的模型会在 LLM 请求转发前从缓存中丢弃，腾出 VRAM 给 Ollama 加载
   模型。
2. **客户端省略时 `keep_alive` 默认 `"30s"`。** 30 秒内连续的 LLM 调
   用复用已加载的模型；闲置的 Ollama 会把 VRAM 还给下一个需要的人。
   长跑的 agent 会话请显式覆盖。

### 示例：非流式

```bash
curl -X POST http://127.0.0.1:11500/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [
      {"role": "user", "content": "Translate to Chinese: hello world"}
    ]
  }'
```

### 示例：流式 + 取消

```python
import httpx

req = {
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Write a short poem."}],
    "stream": True,
}

with httpx.stream("POST",
                   "http://127.0.0.1:11500/v1/chat/completions",
                   json=req,
                   timeout=600.0) as r:
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        chunk = json.loads(line[6:])
        delta = chunk["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)
        if user_pressed_cancel():
            break   # 关闭流会向上游传递 TCP RST；
                    # aistack 停止从 Ollama 拉取并释放槽位。
```

不需要单独的取消端点。**关闭 HTTP 响应就是取消信号。** aistack 的流
循环轮询客户端断连，并把中止传递给上游模型。

---

## 7. 单任务 GPU 槽位

这是最重要的一条网关行为，必须理解，因为它决定了客户端该如何组织并
发。

**任意时刻 GPU 上至多跑一个推理任务**，跨所有三种能力。对 ASR + LLM
（或 LLM + TTS 等）的并发调用会得到：

- 第一个请求拿到槽位并执行。
- 第二个请求立刻返回 HTTP **503** + 一个 `Retry-After: 5` 头，错误响
  应结构的 kind 为 `network`。
- 客户端决定是退避重试还是抛出错误。

这是有意为之。在 8 GB 消费级显卡上，跨能力的并发推理会让 worker OOM；
串行化是唯一让网关稳定的方式。在更大的硬件上，同一策略意味着可预测的
资源核算，而吞吐没有真正的损失。

**给客户端的指引**：

- 把 503 + `Retry-After` 当作 **传输层反压信号**，不是致命错误。睡
  指定的秒数然后重试。
- 不要对 aistack 做并行流水线，期望并发 —— 设计上做串行派发，遇竞争
  时重试。
- 对穿插 ASR / LLM / TTS 的 agent 循环，按顺序跑各步。aistack 的热
  切换策略会保证 VRAM 在步骤之间换模型时仍然够用。

一个简单的 Python 重试辅助函数：

```python
import time, httpx

def call_with_retry(method, url, max_attempts=5, **kw):
    for attempt in range(max_attempts):
        r = httpx.request(method, url, **kw)
        if r.status_code != 503:
            return r
        retry = float(r.headers.get("Retry-After", "5"))
        time.sleep(retry)
    r.raise_for_status()
```

---

## 8. 错误响应结构

所有非 2xx 响应都带这个形状，包括 §7 的槽位忙 503：

```json
{
  "error": {
    "kind": "network | malformed | overflow | cancelled | unknown",
    "provider": "aistack | Faster-Whisper | Parakeet | SenseVoice | ...",
    "message": "human-readable details safe to surface to users"
  }
}
```

按 `error.kind` 分支。五种 kind 的含义：

| Kind | 触发场景 | HTTP 状态 | 客户端响应 |
|---|---|---|---|
| `network` | 上游后端不可达、模型下载失败、传输错误 | 503 | 提示"服务未启动，请先启动"；延迟后重试 |
| `malformed` | 输入有问题 —— 不支持的音频格式、字段缺失、未知 model id | 400 | 把错误显示给用户；不要重试 |
| `overflow` | 输入对所选模型/VRAM 过大 | 413 | 建议改用更小的模型或更短的输入 |
| `cancelled` | 客户端在请求过程中断开 | 499 | 通常不需要 UI；用户已经知道自己取消了 |
| `unknown` | 其他没归到上面任何类的 | 500 | 记日志并显示 message；不诊断不要重试 |

完整说明：[`errors`](./api/errors/)。

---

## 9. 各能力的细节注意事项

### ASR

- `language` 表单字段是提示，不是约束。faster-whisper 在
  `language` 缺省时自动检测；SenseVoice 在音频实际语种与提示冲突时
  忽略提示。
- `response_format`：`json`（最小 `{text}`）、`verbose_json`（带
  segments + words 的完整结构）、`text`（纯文本 body）。
- `translate=true` 仅在 Whisper 系后端上有效。Parakeet 和 SenseVoice
  被要求翻译时会报 `malformed`。
- 长音频（>5 分钟）情况下，Parakeet 在 8 GB 硬件上会自动启用 local-
  attention encoder 模式 —— 准确率与全 attention 几乎无差，但内存上限
  从 ~3 分钟变成事实上无界。

### TTS

- 代理是透明的；如果 Qwen3-TTS 上游加了新字段，aistack 不改代码就能
  转发。
- TTS 容器在启动时占住固定的 VRAM 预留量（在
  `docker/tts_qwen3/docker-compose.yml` 里通过
  `gpu_memory_utilization` 配置）。在 8 GB 卡上，如果觉得网关吃紧可以
  调到 0.5。

### LLM

- 流式用 Server-Sent Events，OpenAI 标准形状。
- `keep_alive` 接受 Ollama 同样的字符串值（`"30s"`、`"5m"`、`"0"`
  等）。aistack 仅在该字段缺失时注入 `"30s"`。
- 仅云端的 LLM（DeepSeek、Claude、Gemini）明确 *不在* aistack 范围内。
  这些 API 从客户端直连即可，代理过来没有收益。

---

## 10. 健康检查

```bash
curl http://127.0.0.1:11500/health
```

```json
{"status": "ok", "version": "0.0.1"}
```

Connection refused 表示 aistack 没在跑。200 表示 worker 活着，但不
保证任何具体后端可用 —— 配合 `/v1/models` 才能知道当前哪些能力可用。

---

## 11. 版本与稳定性

### `/v1` 内的契约

下面这些是稳定的，且在 `/v1` 内只会以增量方式变化：

- §2 列出的端点集合。
- `/v1/models` 条目里的字段名与类型。
- 错误响应结构形状（`error.kind`、`error.provider`、`error.message`）。
- §8 中的五种 error kind 和它们的 HTTP 状态映射。
- 槽位竞争时的 503 + `Retry-After` 语义。
- 三种能力端点的 OpenAI 形态请求/响应 body。

### **不是** 契约的部分

- 某个具体的 model id 是否出现在 `/v1/models`。这取决于装了哪些后端，
  与环境相关。
- 具体的时间特性（延迟、吞吐）。
- 内部头、日志行、admin UI 形状。
- 某个具体后端是在进程内承载还是被代理。

### 破坏性变更

对上面契约清单中任何东西的破坏性变更需要升到 `/v2`。`/v1` 至少会保
留一个发布周期继续服务，让调用方可以不耦合部署地迁移。

---

## 12. 接下来看哪里

| 你想…… | 读 |
|---|---|
| 详细理解清单响应 | [`models`](./api/models/) |
| 构建 ASR 客户端 | [`asr`](./api/asr/) |
| 构建 TTS 客户端 | [`tts`](./api/tts/) |
| 构建 LLM 客户端 | [`llm`](./api/llm/) |
| 精确按错误分支 | [`errors`](./api/errors/) |
| 把你的 trace ID 串过 aistack（可选 `X-Request-ID`） | [`observability`](./api/observability/#optional-send-x-request-id-for-cross-system-correlation) |
| 读 aistack 关于你的流量记录的指标 / access log / 载荷捕获 | [`observability`](./api/observability/) |
| 理解 aistack 的内部架构（集成不需要） | aistack 仓库的 `docs/design/architecture.md`（仅内部） |

如果某个行为与本指南承诺的不一致，**以本指南为准** —— 请到
`github.com/dosmoon/aistack` 上提 issue。
