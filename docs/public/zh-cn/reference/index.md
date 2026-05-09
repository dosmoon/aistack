---
title: 参考总览
description: 自动生成的参考资料 —— 每次构建从代码生成的 *是什么*。这些数字背后的 *为什么*，请看对应的叙述页。
sidebar:
  order: 4
---

# Reference

本节在每次构建时 **从 aistack 源代码生成**。这里的页面把每个参数、字
段、类型和默认值 **按运行中的代码所定义** 的样子原样列出 —— 没有会
漂移的手写表格。

要看设计动因（为什么默认值是这样、有哪些权衡），从每一页 reference
回链到对应的手写叙述。

## 章节

- [Configuration](./configuration/) —— 每个 `AISTACK_*` 环境变量，从
  `aistack/config.py` 生成。叙述配套：[配置指南](../configuration/)。
- HTTP API 端点 reference 在 [`/api/reference/`](../api/reference/) 下。
  在我们确认布局对"调用方读 API"和"运维读配置"两类读者都好用之
  后，它大概率会在以后的一次重组中搬到这里。
