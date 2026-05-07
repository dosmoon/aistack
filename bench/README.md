# aistack 评测工具

不是测试，是评测。测试看正确性 pass/fail；评测看性能数字 + 质量数字 + 自身功能正确性。

## 当前一个工具：ASR 评测

```bash
# 启动 aistack
dev.bat

# 跑 LibriSpeech dev-clean（首次自动下载 337 MB 到 ~/.cache/aistack-bench/）
python -m bench.asr_eval --model whisper-small
python -m bench.asr_eval --model parakeet
python -m bench.asr_eval --model auto

# 全量（5.4 小时音频，2703 条）
python -m bench.asr_eval --model whisper-small --limit 0

# 输出 JSON 归档
python -m bench.asr_eval --model whisper-small --json results-whisper.json
```

## 报告里的三件事

| 维度 | 指标 | 含义 |
|---|---|---|
| **正确性**（aistack 自身） | HTTP 错误数 / 解析失败数 | 评测顺带发现 bug |
| **质量** | WER（Word Error Rate） | 转写得对不对 |
| **性能** | wall time / RTF | 跑得快不快 |

## 数据集

LibriSpeech dev-clean（CC-BY-4.0）—— ASR 行业事实标准基准。朗读类英文公共领域有声书，5.4 小时，2703 条，每条带精确 GT。

来源：<https://www.openslr.org/12/>

## 后续

- Mandarin 数据集（Common Voice zh-CN 或 AISHELL dev）
- 多长度评测（不同时长档分别报数）
- 历史对照（保存历次结果，看模型 / 配置变化的影响）
