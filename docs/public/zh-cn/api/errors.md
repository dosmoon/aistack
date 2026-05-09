---
title: 错误响应结构
description: aistack 每个端点共用的 JSON 错误响应结构。按 error.kind 分支，展示 error.message。
sidebar:
  order: 6
---

# 错误响应结构

aistack 所有非 2xx 响应都使用同一个 JSON 响应结构。调用方按
`error.kind`（机器可读）分支，并展示 `error.message`（人类可读，可
安全显示）。无论由哪个端点产生，这个结构都一样。

线缆格式 schema（每个字段、每个 kind 值、HTTP 状态映射）在自动生成
的 [错误响应结构 reference](./reference/asr/#schema-errorenvelope) 里。
本页讲设计动因和调用方侧的模式 —— *为什么* 和 *怎么用*，不是 *是什么*。

## 为什么所有端点用同一个响应结构

aistack 对每个非 2xx 响应都用同一个 `{error: {kind, provider,
message}}` 形状有三个理由：

1. **每个调用方一处处理。** 一个同时集成 ASR、TTS、LLM 的客户端只
   需写一次错误路径，而不是三次。
2. **`kind` 稳定；`message` 不稳定。** 代码按 `kind` 分支（一个小而
   稳定的枚举）；人类读 `message`（自由文本，可能在不同版本之间被
   重新措辞）。对 `message` 做字符串匹配是 bug；按 `kind` 分支才是
   契约。
3. **`provider` 归因失败。** 出问题时你想知道"是网关路由、是上游
   Ollama、还是 Whisper 后端"——`provider` 不用挖日志就能回答。
   `"aistack"` 表示是网关本身；一个像 `"Parakeet"` 这样的后端名表
   示是该子系统拒绝了请求。

## 值得知道的状态码语义

kind → status 的映射在
[reference 表](./reference/asr/#schema-errorenvelope)。两点不那么显
然的：

- **`499`（`cancelled`）** 遵循 nginx 的"client closed request"约
  定 —— 它不是标准 HTTP 码，但被广泛理解。调用方已经走了，这个状态
  仅供信息用；它会出现在你的 access log 里，但永远不会出现在一个还
  活着的响应里。
- **`503` 在两种真实情况之间被复用**："Ollama 挂了 / 模型加载失
  败"和"GPU 槽位忙在服务另一个推理，稍后重试"。靠 `Retry-After`
  头来区分 —— 仅在槽位忙的路径上出现。见下文反压一节。

## 错误示例

### 未知 model id（`malformed`，400）

```bash
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
     -F file=@audio.mp3 -F model=bogus-model
```

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": {
    "kind": "malformed",
    "provider": "aistack",
    "message": "Unknown model: 'bogus-model'. Use whisper-{size}, parakeet, or sensevoice."
  }
}
```

### 后端未安装（`network`，503）

当用户请求 Parakeet 但 venv 里没有 `nemo_toolkit`：

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": {
    "kind": "network",
    "provider": "Parakeet",
    "message": "NeMo toolkit not installed. Run: pip install nemo_toolkit[asr]"
  }
}
```

### TTS 上游容器宕机（`network`，503）

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

### GPU 槽位忙（`network` + `Retry-After`，503）

单任务 GPU 锁的反压 —— 服务器健康，只是已经在服务另一个推理：

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 5
Content-Type: application/json

{
  "error": {
    "kind": "network",
    "provider": "aistack",
    "message": "aistack GPU slot is busy (held by asr); rejected llm. Retry after a few seconds."
  }
}
```

槽位忙这条路径用的响应结构形状跟其他 `503` 一样，因为它 *本来就是* 一
个传输层反压信号。靠 `Retry-After` 头与"Ollama 挂了"区分 —— 仅在槽
位忙路径上出现。其他 `network` 错误（模型下载失败、上游守护进程不可
达）也是 `503`，否则会无法区分。

## 调用方侧的处理模式

一份能处理全部五种 kind 加上槽位忙重试场景的参考 Python 实现：

```python
import httpx

class AistackError(Exception):
    def __init__(self, kind, provider, message, status):
        self.kind, self.provider, self.message, self.status = kind, provider, message, status
        super().__init__(f"[{kind}/{provider}] {message}")

class BusyError(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after

def call_aistack(method, url, **kw):
    r = httpx.request(method, url, **kw)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 503 and r.headers.get("Retry-After"):
        raise BusyError(retry_after=int(r.headers["Retry-After"]))
    try:
        env = r.json().get("error", {})
        kind = env.get("kind", "unknown")
        provider = env.get("provider", "aistack")
        message = env.get("message", r.text)
    except (ValueError, AttributeError):
        kind, provider, message = "unknown", "aistack", r.text
    raise AistackError(kind, provider, message, status=r.status_code)
```

## 稳定性

`kind` 值的集合是契约的一部分：

- `/v1` 内允许新增 kind。
- 重命名或移除已有 kind 需要 `/v2`。

`message` 文本 **不** 稳定 —— 措辞可能在不同版本之间变化。代码必须
按 `kind` 分支，而不是对 `message` 做字符串匹配。
