---
title: aistack
description: 本地优先的 ASR / TTS / LLM-proxy 网关，提供兼容 OpenAI 的 API。研究形态的开源项目，面向想要研究本地 AI 能力的开发者与研究者。
sidebar:
  order: 0
---

# aistack

一个自托管、兼容 OpenAI API 的网关，把开源 ASR（faster-whisper、NVIDIA Parakeet、阿里 SenseVoice）、TTS（Qwen3-TTS via vLLM）以及一个本地 LLM 代理（转发到 Ollama）封装在同一个 HTTP 端点之后。

## 定位

aistack **是研究形态的，不是产品化的**。它服务于想要对比后端、衡量质量/延迟权衡、把本地 ASR/TTS 集成进自己工具链的开发者与研究者 —— 而不是期待一键安装、下载即用的终端用户。重点放在可观测性、多后端对比、以及不带感情色彩的基准测试上。

如果你想要开箱即用的本地 AI 体验，LLM 直接用 [Ollama](https://ollama.com)，ASR/TTS 等下游产品化工具出现。如果你想 *研究* 本地 ASR/TTS，aistack 是为你而做的。

## 本站包含什么

| 章节 | 内容 |
|---|---|
| [集成指南](integration/) | 从 Hello World 到上线：能力发现、请求、错误、流式。新接触 aistack 的话先读这页。 |
| [HTTP API](api/) | 每个端点的设计动因（*为什么*），加上自动生成的 [Reference](api/reference/)（*是什么*）。 |

更多章节（配置、部署说明、研究结论）会陆续发布。完整代码仓库在 [github.com/dosmoon/aistack](https://github.com/dosmoon/aistack)。

## 快速上手

```bash
git clone https://github.com/dosmoon/aistack
cd aistack
pip install -e .[asr-fasterwhisper]
python -m uvicorn aistack.main:app --port 11500
curl http://127.0.0.1:11500/health
```

完整的安装方式与 extras 布局（按 ASR 后端按需安装、TTS Docker 容器、Ollama 同机并行）在仓库的 `README.md` 里有详细说明。

## API 速览

```
GET  /health                        存活检查
GET  /v1/models                     能力清单
POST /v1/audio/transcriptions       语音转文字  (Whisper / Parakeet / SenseVoice)
POST /v1/audio/speech               文字转语音  (Qwen3-TTS)
POST /v1/chat/completions           对话补全    (代理到 Ollama)
```

默认 base URL：`http://127.0.0.1:11500`。每个端点的详细说明请看 [HTTP API 章节](api/)。
