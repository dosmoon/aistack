---
title: 可观测性
description: 内置的性能与可用性层 —— 滚动 p50/p95/p99、JSONL access log、按请求的载荷捕获、X-Request-ID 关联。
sidebar:
  order: 7
---

# 可观测性 —— 性能与可用性分析

aistack 自带一层可观测性，用于 **性能分析**（延迟分布、吞吐、RTF）
和 **可用性分析**（错误率、槽位忙拒绝、断连）。三个独立开关，除载荷
捕获外默认全开。

线缆格式 schema（指标 JSON、access log JSONL 字段、载荷捕获布局）由
代码拥有，每次构建时渲染进 [admin reference](./reference/admin/)。本
页讲 *为什么* —— 设计动因、何时用各开关、可选的 `X-Request-ID` 跨系
统关联故事，以及分析配方。

## 三个独立开关

为什么是三个开关而不是一个总开关：三条流的代价画像 **差异极大**，捆
在一起会强迫"全开或全关"的选择，没有任何真实工作流契合这种粒度。

- `metrics` —— 进程内每能力的滚动直方图 + 计数器。代价：每请求 < 10
  µs。**永远想要。**
- `access_log` —— 每请求一行 JSONL，追加到 `AISTACK_OBS_LOG_DIR` 下
  按天滚动的文件。代价：入队 + 后台 flush。**几乎永远想要。**
- `payload` —— 请求 body 和响应 body 落盘以备回放/诊断。代价：磁盘
  IO；尺寸 + 年龄有上限。**稳态下几乎永远不要**；仅在某个具体 bug
  需要这些字节时开启。

启动时通过 env 变量切换（见仓库 README 中的配置），或从 `/admin`
仪表盘上实时切换。实时切换仅会话有效 —— 重启回到 env 驱动的默认值。
开关 **对调用方的线缆效果为零** —— 所有可观测性特性都在服务端，且完
全向后兼容。已有客户端（CLI 脚本、VideoCraft、OpenAI 形态的 SDK）无
需修改即可工作，并立刻开始出现在指标 / access log 中。

## 可选：发 `X-Request-ID` 用于跨系统关联

如果调用方有自己的作业 id（一个 VideoCraft 流水线步骤、一个 agent
任务 id、一个面向用户的请求 id），把它作为 `X-Request-ID` 头传过来。
aistack 会：

1. 用它作为 access log JSONL 中的 `request_id` 字段。
2. 在载荷捕获开启时用它作为目录名
   （`<PAYLOAD_DIR>/<date>/<X-Request-ID>/`）。
3. 把它附加到内存中的指标采样上（在 `/admin/api/metrics` 的 `recent`
   下可见）。
4. 在响应头里回传。

```python
import httpx

trace_id = f"videocraft-{pipeline_id}-step-{step_idx}"
resp = httpx.post(
    "http://127.0.0.1:11500/v1/audio/transcriptions",
    headers={"X-Request-ID": trace_id},
    files={"file": open("clip.mp3", "rb")},
    data={"model": "auto", "language": "en"},
)
# 即使你没传，也会回传（aistack 生成 16-hex）：
returned_id = resp.headers["X-Request-ID"]
```

发自己 id 时的 **格式约束**：

- ASCII 字母、数字、`-_:.` —— 其他字符会被拒绝，aistack 改用自己生
  成的 16-hex id。
- 最长 128 字符 —— 超长值也会被拒绝。
- 唯一性由调用方负责（我们不去重）。

**不传也完全可以。** aistack 会生成一个 16-hex id（例如
`825e53134e178cd1`），并在响应头里返回。只想日志里记下"aistack 分
配的 id"的客户端只需读响应头 —— 请求侧不用动。

### 何时关联值得做

跳过：

- 一个进程在跟 aistack 说话，且不需要跨系统关联。
- 请求靠时间戳 + 端点本身就能区分。

加上：

- 长流水线（transcribe → LLM → TTS）需要端到端调试。
- 慢的 / 失败的请求需要从 VideoCraft 的日志追到 aistack 的 access
  log + 载荷目录，而不必猜时间戳。
- 多个客户端共享一个 aistack 实例，你需要归因流量。

## 状态分类（为什么 `503-busy` 自成一类）

指标计数器把每个能力的流量切成五类：

- `2xx` —— 成功。
- `4xx` —— 客户端错误（校验、未知模型）。
- `5xx` —— 服务端错误（provider 崩溃、上游宕机）。
- `503-busy` —— GPU 槽位拒绝。**Load shedding 不是错误。** 健康的
  网关在负载下返回 `503-busy` *正是因为系统在做它该做的事*。把它算
  作 `5xx` 会让健康的网关在流量高峰时看起来像坏了。
- `client-disconnect` —— 客户端在请求过程中关闭了连接。也不是错误
  —— 请求被调用方放弃了。

指标快照中的 `error_rate` 是 `(4xx + 5xx) / total` —— 故意排除 load
shedding 和断连。

## 载荷脱敏 —— 什么安全、什么不安全

`payload` 开启时，每个请求和响应 body 都会持久到磁盘。两件事要知道：

- **头会被脱敏**，在每请求的 `meta.json` 中：`Authorization`、
  `Cookie`、`X-Api-Key`、`Proxy-Authorization` 被替换成 `***`。
- **body 不脱敏。** 载荷捕获是 opt-in 的，仅用于受信环境的诊断。如
  果你的请求 body 含 PII，生产上不要开启。

磁盘清扫器（启动时和每 30 分钟运行一次）会丢弃：

1. 比 `AISTACK_OBS_PAYLOAD_MAX_DAYS`（默认 7）老的捕获目录。
2. 然后只要总量 > `AISTACK_OBS_PAYLOAD_MAX_GB`（默认 5），就丢最老
   的幸存者。

顺序很重要 —— 年龄裁剪先于尺寸裁剪运行，所以做磁盘预算时，1 天前的
大请求优先于 6 天前的小请求被保留。

## 长期分析配方

```bash
# 最近 24 小时的慢 ASR 请求
jq 'select(.category=="asr" and .latency_ms > 1000)' logs/access-2026-05-08.jsonl

# 每个 LLM 模型的错误率
jq -r 'select(.category=="llm") | [.model, .status] | @tsv' logs/*.jsonl \
    | sort | uniq -c

# 一天内 SenseVoice 的平均 RTF
jq -r 'select(.extra.provider=="sensevoice") | .extra.rtf' logs/access-2026-05-08.jsonl \
    | awk '{s+=$1;n++} END{print s/n}'

# 回放一个被捕获的请求
curl -X POST http://127.0.0.1:11500/v1/audio/transcriptions \
    -F file=@captures/2026-05-08/<rid>/req.bin \
    -F model=auto
```

## 性能开销

热路径上，所有开关取默认值时（`metrics` + `access_log` 开，`payload`
关）实测：

- metrics：~5 µs/请求（一次 dict 更新 + 一次 deque append）。
- access_log：~10 µs/请求（dict 取出 + queue.put）。
- request-id 中间件：~3 µs/请求。

总计 < 20 µs/请求 —— 跟动辄数百毫秒的 ASR / LLM / TTS 工作相比可以
忽略。

`payload` 开启时，开销主要由请求音频的磁盘 IO 主导（通常几百 KB 到
几 MB）。诊断场景可接受，不推荐在持续高吞吐下使用。

## 为什么没有 Prometheus exporter

aistack 出厂提供一个 JSON `/admin/api/metrics` 端点，不是 OpenMetrics。
两个原因：(1) aistack 是单机网关，所以 scraper + 保留策略是架构其余
部分并未要求的开销；(2) JSON 形状已经是脚本和仪表盘需要的样子。如果
某天另一头真的接上了 Grafana，加一个 Prometheus exporter 是 `/v1`
的增量改动。
