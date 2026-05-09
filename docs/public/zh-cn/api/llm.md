---
title: POST /v1/chat/completions
description: 对话补全端点。兼容 OpenAI。反向代理到本地 Ollama 守护进程，叠加 GPU 调度和合理的 keep_alive 默认值。
sidebar:
  order: 5
---

# `POST /v1/chat/completions`

兼容 OpenAI 的对话补全端点。aistack 把请求反向代理到本地 Ollama 守护
进程（默认 `http://127.0.0.1:11434`），并在直连之上叠加两件事。

完整的请求与响应 schema 以 OpenAI 的权威
[Chat Completion API reference](https://platform.openai.com/docs/api-reference/chat)
为准。aistack 原样代理这个 schema —— Ollama 自己的 OpenAI 兼容层就镜
像了它，aistack 不变换 body。
[自动生成的 LLM reference](./reference/llm/) 列出 aistack 自己包在代
理外面的那层（响应状态码、错误形状）长什么样。

本页讲的是 *为什么* —— 那两件叠加的事、GPU 调度、keep_alive 策略、
错误归因，以及并发在网关上的工作方式。

## aistack 在直连 Ollama 之上加了什么

1. **GPU 调度。** 转发之前，aistack 先清掉自己的 ASR `asr-main` 缓存
   条目 —— 进程内可能驻留在 VRAM 中的 Whisper / Parakeet / SenseVoice
   模型会被丢弃，并执行 `gc.collect()` 与 `torch.cuda.empty_cache()`，
   让 LLM 模型放得下。处理器随后在整个上游调用期间持有全局网关 GPU
   槽位。

2. **合理的 `keep_alive` 默认值。** 当请求省略 `keep_alive` 时，
   aistack 注入 `"30s"`。30 秒内连续的 LLM 调用复用已加载的模型；闲
   置释放 VRAM 还给 ASR。传一个显式值（`"5m"`、`"1h"`、`"-1"` 表示
   永不卸载、`"0"` 表示立刻卸载）来覆盖。

针对 OpenAI `/v1/chat/completions` 写的客户端不修改即可工作 —— aistack
不修改 `messages`、`tools`、`response_format` 或任何其他 OpenAI 字段。
只在 `keep_alive` 缺失时填入。

## 资源调度详解

这个端点收到请求时：

1. **获取全局 GPU 槽位。** 跨 ASR / TTS / LLM 的并发请求会得到 HTTP
   503 加 `Retry-After`。槽位代表的是"GPU 正在做推理"，跟谁拥有
   kernel 无关。
2. **驱逐 ASR。** `_model_cache.evict_category("asr-main")` 丢弃当前
   驻留的所有 ASR 主模型。
3. **注入 `keep_alive`。** 请求 body 缺少时，设为 `"30s"`。
4. **转发到 Ollama。** 当 `stream=true` 时按 chunk 流式转发响应，让
   首 token 的延迟尽快到达客户端。
5. **释放槽位。** 在正常完成、流结束、客户端断开或上游错误时。

这意味着：**LLM 调用之后立刻发起的新 ASR 调用可能要付冷启动加载延
迟**（asr-main 模型为腾出位置已被驱逐）。这个权衡是有意为之 —— 另
一种选择是在紧凑 VRAM 上 OOM。对连续多次调用 LLM 的工作流（例如批
量翻译），30 秒的 keep_alive 默认值能让 Ollama 跨多次调用保持热。
对紧密交错 ASR 和 LLM 的工作流（例如实时多模态 agent），8 GB 硬件上
两边都要付冷加载税；调低 Qwen3-TTS Docker 的 `gpu_memory_utilization`
是回收 VRAM 的主要手段。

## 流式

每个由 Ollama 服务的 LLM 在 `/v1/models` 中都宣称
`supports_streaming: true`。当 `stream=true` 时，aistack 原样转发
Ollama 的 SSE 块 —— 首 token 延迟是 Ollama 的 prefill 时间加上
localhost 上亚毫秒级的代理开销。

客户端断开会传播：aistack 在断开时关闭上游连接，让 Ollama 的 runner
中止生成，而不是在死掉的 socket 上跑到结束。

## 错误归因

aistack 区分起源于 **网关边界** 的错误和起源于 **Ollama 自身** 的错
误。

**网关源错误** 使用标准的 `{error: {kind, provider, message}}` 响应
结构（见 [errors](./errors/)）：

```json
{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "Ollama is not reachable at http://127.0.0.1:11434. Start it with: ollama serve"
  }
}
```

**后端源错误**（Ollama 因未知模型、消息格式错误等拒绝请求）会被
**原样** 透传。客户端直接看到 Ollama 的错误格式。这是有意为之：现存
的 OpenAI 兼容客户端对 Ollama / OpenAI 的错误形态有自己的预期，重新
包装会破坏这个预期。

所以：当 `error.kind` 存在时按它分支（这是 aistack 响应结构）；任何
不带 `kind` 字段的非 2xx 响应回退到 OpenAI / Ollama 错误解析。

## 健康与清单

- `GET /v1/models` 把 Ollama 已安装的模型聚合进来，`capabilities=
  ["llm"]`、`owned_by="ollama"`。Ollama 不可达时 LLM 条目被静默省略
  （没有条目，也没有错误）—— `/v1/models` 的其余部分继续服务。
- Ollama 不可达时直接 `POST /v1/chat/completions` 会返回
  `503 network`，message 是上面那条可执行的提示。

## 配置

服务端 env 变量：

```
AISTACK_OLLAMA_URL    默认 http://127.0.0.1:11434
```

当 Ollama 跑在非默认端口或局域网另一台主机上时，在 `scripts/dev.bat`
或你的启动器里设置。

## 稳定性

`/v1` 内 OpenAI 兼容的请求和响应形状跟随 OpenAI 公布的规范。流式格式
跟随 OpenAI 的 chunk schema。aistack 这一侧的行为（驱逐、`keep_alive`
注入）记录在上文，并在 `/v1` 内稳定；如果有意义地改变（比如一个不同
的默认 keep_alive），会作为带发布说明的 `/v1` 增量改动，而不是 `/v2`
破坏性变更。
