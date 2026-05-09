---
title: POST /v1/audio/speech
description: 文字转语音端点。兼容 OpenAI TTS API。代理到本地 Docker 容器中运行的 Qwen3-TTS 模型。
sidebar:
  order: 4
---

# `POST /v1/audio/speech`

从文本生成语音音频。在字段层兼容 OpenAI TTS API —— 针对 OpenAI
`/v1/audio/speech` 写的客户端可以无修改工作。aistack 代理到运行在
`aistack-qwen3-tts` Docker 容器内的 Qwen3-TTS-12Hz-0.6B-CustomVoice
模型。

aistack 在这一层 **不加任何业务逻辑**；它是一个透明反向代理。请求 body
和响应 body 原样穿过（除去 hop-by-hop 头）。

完整请求与响应 schema 在自动生成的 [TTS reference](./reference/tts/)
里。OpenAI 兼容的字段语义（`input`、`voice`、`response_format`、
`model`）以 OpenAI 的
[Audio API 文档](https://platform.openai.com/docs/api-reference/audio)
为权威。本页讲的是 *为什么* —— 透明代理的立场、输出格式怪癖、透传端
点表面、冷启动行为，以及 aistack 为什么 **不** 在 TTS 上仲裁并发。

## 为什么是透明代理

aistack 不变换 TTS 请求/响应有三个原因：

1. **OpenAI 的契约就是契约。** 针对 OpenAI TTS 端点写的客户端在
   aistack 上不修改即可工作。中间加业务逻辑就要持续跟 OpenAI 演化的
   规范保持同步 —— 一片漂移面，没有价值。
2. **上游拥有模型。** Qwen3-TTS 的扩展字段（声音克隆、声音设计、批
   量合成）是上游的领地。aistack 透传它们让调用方拿到完整能力集。
3. **音频是字节，不是 JSON。** 在网关做转码（WAV → MP3、采样率转换）
   会让每个请求都烧 CPU、加延迟。需要其他格式的客户端用 ffmpeg 在
   客户端转。

## 输出格式

输出是 vLLM-Omni 给什么就是什么 —— 截至本文撰写时，无论请求里的
`response_format` 提示如何，输出都是 **24 kHz 单声道 16-bit PCM WAV**。
vLLM-Omni 服务器未来可能加上 MP3/Opus/FLAC 编码；aistack 会原样透传，
无需改代码。

如果今天就需要某个特定格式，在客户端转码：

```bash
ffmpeg -i out.wav -codec:a libmp3lame out.mp3
```

## 透传端点表面

完整的 Qwen3-TTS 扩展表面通过 aistack 在 `/v1/audio/*` 下可达：

- `POST /v1/audio/speech` —— 合成（自定义声音 / 克隆声音 / 设计声音）
- `POST /v1/audio/speech/stream` —— 流式合成（边生成边切块）
- `POST /v1/audio/speech/batch` —— 批量合成
- `GET /v1/audio/voices` —— 列出可用的声音
- `POST /v1/audio/voices` —— 注册一个新声音
- `DELETE /v1/audio/voices/{name}` —— 删除一个已注册的声音

aistack 把这些全部原样代理。它们的 schema 由
[vLLM-Omni 的 Qwen3-TTS 文档](https://github.com/QwenLM/Qwen3-TTS)
拥有，aistack 不重复记录。aistack 未来版本可能加增值行为（遥测、跨多
后端部署的 voice list 聚合）而不改变 over-the-wire 格式。

TTS 在 `/v1/models` 中的条目宣称 `supports_streaming: true`，是因为
有 `/v1/audio/speech/stream` 透传路径；这条流式的线缆格式是
vLLM-Omni 在那里给出什么（分块的音频字节），**不是** ASR 用的
`transcript.text.delta` SSE 形状。

## 冷启动

对刚启动的 Docker 容器的第一次请求会触发 vLLM-Omni 内部的
`torch.compile` + CUDA Graph 捕获 —— 这要 **~60 到 ~150 秒**，依请
求和机器状态而定。代理超时设到 600 秒以吸收最坏情况；客户端应当展示
"预热中"而不是把长延迟当成挂死。

预热之后，RTX 4060 Laptop 上的稳态延迟通常 RTF 0.7–1.1（短句几百毫
秒）。

## 并发

aistack **会** 把单个全局 GPU 槽位应用到 TTS 请求上：Qwen3-TTS 容器
和进程内 ASR 与 LLM 代理共享物理 GPU，而槽位代表的是"GPU 正在做推
理"，跟谁拥有 kernel 无关。跨能力的并发请求会得到 HTTP 503 加
`Retry-After`。

vLLM-Omni 在 Docker 容器内部处理自己的请求排队。aistack 不在网关层
实现队列 —— 你想要队列就在客户端围着 busy 信号建一个。

## 容器宕机后的恢复

最常见的失败模式是"Docker 容器没在跑"：

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "Qwen3-TTS container is not reachable. Start it with: docker compose -f docker/tts_qwen3/docker-compose.yml up -d"
  }
}
```

错误 message 里附带恢复命令。在冷启动中途崩溃的容器可能卡到一种状
态，请求会在代理的 600 秒读超时上超时 —— 这种情况下重启容器。

## 稳定性

OpenAI 兼容的字段层（`input`、`voice`、`response_format`、`model`）
在 `/v1` 内稳定。Qwen3-TTS 扩展字段（`task_type`、`language`、
`ref_audio`、……）是上游的契约，权威文档在
[vLLM-Omni Qwen3-TTS 仓库](https://github.com/QwenLM/Qwen3-TTS)。

透传端点路径（`/v1/audio/speech/stream`、`/voices`、……）跟随
vLLM-Omni 的契约；如果未来某个 TTS 后端暴露不同的表面，aistack 会
在单独的迁移说明里记录映射，而不是悄悄改写契约。
