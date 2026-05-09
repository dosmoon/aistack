---
title: HTTP API
description: aistack 的 HTTP 契约 —— ASR、TTS、LLM 代理、模型清单、错误、可观测性。
sidebar:
  order: 3
---

# aistack HTTP API

本节 **就是 aistack 向调用方公开的契约**。任何客户端 —— CLI 工具、像
VideoCraft 这样的 GUI 应用、agent 框架、未来的 dosmoon 产品 —— 都按
照这里文档化的内容来对接。aistack 不会迁就任何特定调用方；调用方按
此契约对接。

内部的实现选择（用哪个推理引擎、GPU 在哪里、调度怎么做）**不**属于
契约 —— 它们可以在不升 API 版本的情况下变。

## 本节的两类文档

- **叙述页**（本页、[`asr`](./asr/)、[`tts`](./tts/)、[`llm`](./llm/)、
  [`models`](./models/)、[`errors`](./errors/)、
  [`observability`](./observability/)）解释契约 *为什么* 长成这样 ——
  设计选择、集成路径、权衡取舍。顶层的 [集成指南](../integration/)
  把它们串成一条连贯的调用方接入路径。
- **[自动生成的 reference](./reference/)** 列出每个端点接受和返回
  *什么* —— 请求/响应字段表、枚举值、错误码、JSON schema。Reference
  在每次构建时从在跑的 FastAPI 应用的 OpenAPI spec 生成，所以不会与
  运行中的代码漂移。

如果你是第一次集成 aistack，先看顶层的 [集成指南](../integration/)。
当你需要某个具体字段的类型或者一份完整的错误码列表时，跳到
[Reference](./reference/)。

## 版本策略

端点都带 `/v1/` 前缀。`/v1/*` 内部的契约遵循增量向后兼容：

- 新增端点 —— 允许
- 给请求新增可选字段 —— 允许
- 给响应新增字段 —— 允许
- 收紧字段校验 —— 不鼓励；仅在当前行为本身有 bug 时才做
- 移除或重命名任何字段 —— 需要 `/v2`
- 改变响应形状 —— 需要 `/v2`

`/v2` 引入时，`/v1` 至少会保留一个发布周期继续服务，让调用方可以不
耦合部署地迁移。

## Base URL

```
http://127.0.0.1:11500
```

绑定地址在服务端配置，共享部署（局域网、云 GPU 租用）下可能不同。
调用方应当把 base URL 当作配置，而不是常量。

## 鉴权

aistack 默认 **不做鉴权**。它的设计场景是 localhost 或私有局域网部
署。如果你把 aistack 暴露在公网上，前面套一层反向代理或 VPN ——
协议本身没有逐请求的 auth。

## 协议风格

在 OpenAI spec 适用的地方兼容 OpenAI API：

- `POST /v1/audio/transcriptions` 镜像 OpenAI 的 Whisper API。
- `POST /v1/audio/speech` 镜像 OpenAI 的 TTS API。
- `POST /v1/chat/completions` 镜像 OpenAI 的 chat completion API。
- `GET /v1/models` 镜像 OpenAI 的 model list 形状，每个条目额外增加
  aistack 扩展字段（`capabilities`、`languages`、`is_routing_alias`、
  `supports_streaming`）。

aistack 在 OpenAI spec 之外加的能力（例如 SenseVoice 的 `task_type`
参数、vLLM-Omni 的声音克隆字段），都以额外的可选字段暴露。会忽略未
知字段的标准 OpenAI 客户端无需修改即可工作。

## 流式

三个能力端点在底层模型支持时都支持流式。发现机制是每个 `/v1/models`
条目上的 `supports_streaming` 字段；线缆格式是 OpenAI 在该能力上使
用的标准 SSE 形状（ASR 用 `transcript.text.delta` /
`transcript.text.done`，LLM 用 OpenAI chat-completion 增量，TTS 用
vLLM-Omni 流式端点的原始音频块）。在所选模型不原生支持流式的极少数
场景下，ASR 会额外发出一个 aistack `warning` 事件 —— 见
[集成指南](../integration/) §4 "用 `stream=true` 流式转写"。

## 错误

所有非 2xx 响应使用共同的响应结构：

```json
{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "aistack service is not reachable. ..."
  }
}
```

调用方按 `error.kind`（机器可读）分支，并展示 `error.message`（人类
可读，可安全显示）。[错误页](./errors/) 解释设计和调用方模式；
[reference](./reference/) 列出每个 kind 值及其 HTTP 状态映射。

## 在线 API 浏览器

aistack 运行时，FastAPI 自动生成的 Swagger UI 在：

```
http://127.0.0.1:11500/docs
```

它始终反映当前运行版本的真实 schema。[Reference 章节](./reference/)
是同一份 OpenAPI spec 的构建时渲染，二者应当严格一致。
