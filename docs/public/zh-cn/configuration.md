---
title: 配置
description: aistack 各配置开关的默认值为什么这样定，以及你应当在何时偏离默认值。配套于自动生成的环境变量参考。
sidebar:
  order: 2
---

# aistack 配置

aistack 从环境变量读取配置。没有配置文件，也不支持 `.env` ——
`aistack/config.py` 直接读 `os.environ`。admin UI 可能在运行时切换
少数几个会话级开关，但不会持久化：重启回到 env 默认值。

本页讲的是 *为什么* —— 每个默认值背后的设计动因、产出这些数字的扫
描结果，以及何时该偏离。每个变量逐字段的清单（名称、类型、默认值、
一行说明）见自动生成的
[Configuration Reference](./reference/configuration/)，每次构建时都
会从 `aistack/config.py` 重新渲染。

## 在哪里设这些变量

启动脚本是设 env 变量的规范位置。当前项目自带的唯一启动器是
`scripts/dev.bat`（Windows）。它已经把最常被调的几个变量以
`set KEY=VALUE` 注释模板的形式写好 —— 取消需要的行的注释，或者新增
一行：

```bat
REM 在 scripts/dev.bat 里，`uvicorn` 那行之前：
set HF_HOME=D:\AI_Models\hf
set AISTACK_OBS_PAYLOAD=on
```

`dev.bat` 里 `if "%KEY%"==""` 的守卫意味着：你 shell 里已经导出的变
量优先于脚本里的默认值 —— 一次性覆盖很方便
（`set HF_HOME=E:\... && scripts\dev.bat`）。

其他平台或生产部署，需要你自己提供启动器：在 `python -m uvicorn
aistack.main:app` 之前 export 变量的 shell 脚本、systemd unit 的
`Environment=` 行、`docker run -e` 标志，等等。aistack 当前没出厂这
些，但下面列出的变量名在哪里设都一样生效。

## 约定

- 布尔开关接受 `1 | 0` 或 `on | off`。
- 时间值以 `_SEC`（秒）或 `_MIN`（分钟）结尾；尺寸值以 `_MB` 或 `_GB`
  结尾。
- 变量在进程启动时读一次。服务起来后再改 env 需要重启 —— 三个可观测
  性开关是例外（见下文）。

## 模型存放位置（第三方 SDK）

告诉每个 SDK 去哪里找预下载的权重。不设这些的话，Hugging Face /
ModelScope / NeMo 会在首次运行时往用户目录里拉好几 GB。

- `HF_HOME` —— faster-whisper / Parakeet / Qwen3-TTS 的权重缓存。
- `MODELSCOPE_CACHE` —— SenseVoice + FunASR VAD 缓存。
- `NEMO_CACHE_DIR` —— Parakeet `.nemo` 归档缓存。

`scripts/dev.bat` 把这三个都指到 `D:\AI_Models\<vendor>` —— 一棵被
所有后端共享的目录树。这些是上游 SDK 的约定，不是 aistack 自己定义的
env 变量，所以不会出现在自动生成的参考表里。

## 模型生命周期

aistack 把每个加载的模型在最后一次请求后驻留一段宽限期，然后清出释
放 VRAM。两个相关旋钮是 `AISTACK_MODEL_KEEP_ALIVE_SEC`（默认 300 秒）
和 `AISTACK_MODEL_SCAN_INTERVAL_SEC`（默认 60 秒）。

更高的 keep-alive 适合"在同一个模型上做交互式爆发"（加载成本只付一
次）；更低的 keep-alive 适合"在紧凑的 VRAM 上轮换多个模型"。

## Parakeet ASR

两组旋钮：**attention 模式**（内存策略）和 **chunking**（长音频如何切
分给推理）。

### attention 模式 —— 为什么是 `local` + `256,256`

- `local` attention 的内存随音频长度 O(N) 线性增长；`full` 是 O(N²)，
  在 8 GB 卡上超过 ~2-3 分钟就 OOM。12+ GB 的卡可以设 `full` 换取一
  点 WER 收益，但对 aistack 瞄准的消费级硬件画像来说，任何超过几分
  钟的音频，`local` 是唯一可行的模式。
- `256,256` 窗口是在 25 分钟音频上做 128/256/512 扫描的甜点：128 牺
  牲 1.2 个百分点的召回换 3 秒墙钟时间；512 反方向也只能买到 ≤ 0.3
  个百分点。256 也是 Parakeet HF 模型卡推荐的，所以训练-推理分布对
  得上。

### chunking —— 为什么 `window=720`、`overlap=120`

aistack 把任何长于 `WINDOW_SEC` 的音频切成多个窗口，相邻 chunk 之间
共享 `OVERLAP_SEC`，独立跑每个窗口，再在重叠区按 word-LCS 拼接结果。
这让每一遍都待在短输入的 VRAM 区间里，避免 cuDNN workspace + caching
allocator 的相互作用在长音频上把使用量推过 8 GB。

默认值来自在 25 分钟和 50 分钟真实录音上的扫描：

- `overlap=60` 在 25 分钟音频上意外产出一个 14 分钟的尾段（tail
  merge）。
- `overlap=120` 重新分布成三段均衡的 chunk，召回也最高：25 分钟上
  98.1%，50 分钟上 95.5%。
- `overlap=180` 在 50 分钟音频上回退，把最后一段吹到 13.8 分钟，把
  reserved VRAM 推到 13 GB。

`AISTACK_PARAKEET_CHUNK_DISABLE=1` 在 24+ GB 卡上是合适的旋钮 ——
那时整段音频走全程不再 OOM，你宁愿付线性的 cuDNN workspace 成本，
也不愿每次调用付 15-30 秒的 LCS 拼接开销。

## 后端上游

aistack 把 LLM 和 TTS 代理给本地服务器。仅在上游运行在不同端口或主
机时才覆盖：

- `AISTACK_OLLAMA_URL` —— `/v1/chat/completions` 走的 Ollama base URL。
- `AISTACK_QWEN3_TTS_UPSTREAM` —— `/v1/audio/*` 走的 vLLM-Omni base
  URL。

默认值匹配 Ollama 和 Qwen3-TTS 标准 Docker 绑定，所以同机上一次干净
的 `ollama serve` + `docker compose up -d` 不需要任何覆盖。

## 可观测性

指标 / access-log / payload 捕获三层各自独立的开关。**每个都可以通过
admin UI 在运行时切换；运行时切换不会跨重启持久化** —— env 变量是耐
久的旋钮。

- `AISTACK_OBS_METRICS=on`（默认）—— 滚动直方图。每请求 ~5 µs；
  **永远想要**。
- `AISTACK_OBS_ACCESS_LOG=on`（默认）—— 按天滚动的 JSONL。每请求
  ~10 µs；**几乎永远想要**。
- `AISTACK_OBS_PAYLOAD=off`（默认）—— 把请求和响应字节落盘。受磁盘
  IO 制约；**稳态下几乎永远不要**。仅在某个具体 bug 需要这些字节时
  开启。

为什么是三个开关而不是一个总开关：三条流的代价画像差异极大，捆在一
起会强迫"全开或全关"的选择，没有任何真实工作流契合这种粒度。叙述层
面记录在
[可观测性叙述](./api/observability/)；线缆格式在
[admin reference](./api/reference/admin/)（自动从代码生成）。

互相影响的磁盘预算旋钮：

- `AISTACK_OBS_PAYLOAD_MAX_GB=5` —— 总磁盘预算。超出时清扫器先删最
  老的。
- `AISTACK_OBS_PAYLOAD_MAX_DAYS=7` —— 年龄预算；更老的目录树先被删，
  在尺寸裁剪运行之前。顺序很重要：在做预算时，1 天前的大请求优先于
  6 天前的小请求被保留。
- `AISTACK_OBS_PAYLOAD_RESP_MAX_MB=50` —— 单个响应上限。超过上限的
  body 只保存元数据（请求 body 始终保存）。

## 完整默认值快照

可直接放进 `scripts/dev.bat` 的起点。下面每行都匹配代码里已经烘焙好
的默认值，所以删掉一行回退到一样的行为。改了的留下，剩下的删掉。

```bat
REM 模型存放位置（指向你的共享缓存）
set HF_HOME=D:\AI_Models\hf
set MODELSCOPE_CACHE=D:\AI_Models\modelscope
set NEMO_CACHE_DIR=D:\AI_Models\nemo

REM 模型生命周期
set AISTACK_MODEL_KEEP_ALIVE_SEC=300
set AISTACK_MODEL_SCAN_INTERVAL_SEC=60

REM Parakeet —— attention
set AISTACK_PARAKEET_ATTENTION_MODE=local
set AISTACK_PARAKEET_ATT_CONTEXT_SIZE=256,256

REM Parakeet —— chunking
set AISTACK_PARAKEET_CHUNK_DISABLE=0
set AISTACK_PARAKEET_CHUNK_WINDOW_SEC=720
set AISTACK_PARAKEET_CHUNK_OVERLAP_SEC=120
set AISTACK_PARAKEET_CHUNK_MIN_LAST_SEC=300

REM 上游
set AISTACK_OLLAMA_URL=http://127.0.0.1:11434
set AISTACK_QWEN3_TTS_UPSTREAM=http://127.0.0.1:17860

REM 可观测性
set AISTACK_OBS_METRICS=on
set AISTACK_OBS_ACCESS_LOG=on
set AISTACK_OBS_PAYLOAD=off
set AISTACK_OBS_LOG_DIR=.\logs
set AISTACK_OBS_PAYLOAD_MAX_GB=5
set AISTACK_OBS_PAYLOAD_MAX_DAYS=7
set AISTACK_OBS_PAYLOAD_RESP_MAX_MB=50
set AISTACK_OBS_METRICS_WINDOW_MIN=60
```

POSIX shell 用户把 `set KEY=VALUE` 换成 `export KEY=VALUE`，并把路径
分隔符翻一下。
