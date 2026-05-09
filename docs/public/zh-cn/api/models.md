---
title: GET /v1/models
description: 列出网关当前能服务的模型，含能力、语言、流式支持、以及路由别名。
sidebar:
  order: 2
---

# `GET /v1/models`

列出 aistack 当前可服务的每一个模型 —— 跨所有后端、所有能力 —— 加上
aistack 提供的路由别名。调用方用它来填充模型选择器、按语言过滤 ASR
模型、并在发请求前判断某个能力是否可用。

完整请求与响应 schema 在自动生成的
[inventory & health 的 reference](./reference/models/) 里。本页讲设
计动因：aistack 是怎么扩展 OpenAI `/v1/models` 形状的、动态可达性意味
着什么、以及何时该调这个端点。

## 为什么扩展 OpenAI 的形状

OpenAI 的 `/v1/models` 返回的是只带 `id` / `object` / `owned_by` 的
朴素条目。这够 *渲染* 一个选择器，但不够 *构建* 一个 —— 调用方分不
清哪些条目是 ASR 哪些是 LLM、转写模型支持什么语言、流式能不能用。
aistack 给每个条目加了四个扩展字段把这个缺口补上：

- `capabilities` —— `["asr"]` / `["tts"]` / `["llm"]`。让选择器按任
  务过滤，而不是从 id 字符串里猜。
- `languages` —— ASR 后端可转写的 ISO 639-1 语言码。TTS / LLM 条目
  上没有这个字段。
- `supports_streaming` —— 这个模型上 `stream=true` 是否能产出真正的
  增量 SSE 响应。对那些只能以质量代价"假流式"的后端（例如 Parakeet）
  为 false。
- `is_routing_alias` —— 标记虚拟的 `id="auto"` 条目，aistack 在请求
  时把它解析成一个真实后端。

OpenAI-only 客户端忽略未知字段也仍能拿到可用的选择器。理解 aistack
扩展的客户端用它们来精确过滤和分组。

## `auto` 路由别名

当至少装了一个 ASR 后端时，响应里会包含一个合成条目，`id="auto"`、
`is_routing_alias=true`。给 `POST /v1/audio/transcriptions` 传
`model=auto` 让网关来挑：

- CJK / 声调语言提示 → SenseVoice（已安装时）
- Parakeet 覆盖的欧洲语言提示 → Parakeet
- 其他，或没有语言提示 → faster-whisper-small

首选后端没装时这个别名优雅降级，所以调用方可以放心用 `model=auto`
而不用按部署做后端探测。

别名上 **没有** `languages` 字段 —— 它会解析到最适合该请求 `language`
提示的、已安装的真实后端，所以它的语言覆盖是已安装后端的并集。它的
`supports_streaming` 是候选池的 AND：当且仅当每个已安装的真实后端都
支持流式时为 true。

## 可达性语义

列表是 **动态** 的。一个模型只在它的后端当下真的能服务请求时才出现：

| 后端                                              | 仅在以下条件下出现                                   |
|--------------------------------------------------|--------------------------------------------------------|
| ASR（faster-whisper、Parakeet、SenseVoice）       | venv 里能 import 对应的 Python 库                       |
| TTS（Qwen3-TTS）                                  | Docker 容器自身的 `/health` 有响应                       |
| LLM（Ollama）                                     | aistack 能访问到 `localhost:11434`（守护进程在跑）        |

如果你启动了 aistack 但没启动 TTS Docker 容器，TTS 条目会从
`/v1/models` 中省略，`POST /v1/audio/speech` 会返回 `503 network`。
调用方应该把模型列表当作 **能力清单**，而不是静态目录 —— 部署拓扑变
化时（装了一个后端、TTS 容器启动了、Ollama 重启了）刷新它。

## 何时调用

- **客户端启动时** —— 缓存列表、填充 UI 选择器。
- **用户打开"挑模型"对话框时** —— 刷新一次，以防用户刚装了新后端
  或刚启动了 Ollama。
- **不要每次推理调用前都调** —— 没有理由在每次转写前重新拉。这个端
  点便宜（import 探测加一次 HTTP HEAD 检查），但不免费。

## 稳定性

`id`、`object`、`owned_by` 属于 OpenAI 规范契约，在 `/v1` 内稳定。
`capabilities`、`languages`、`supports_streaming`、`is_routing_alias`
是 aistack 扩展；可以加新的值（增量），但已有值的含义永不改变。某个
具体 model id 是否出现取决于已安装的后端，不是契约保证。
