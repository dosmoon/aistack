# aistack 必然是研发形态 · 产品形态的可行路径

> TL;DR
>
> aistack 的依赖链与价值取向决定了它不可能演化成"装上即用"的终端用户产品。
> 想做产品形态需要单独立项、另开仓库。本文记录这个判断的依据，以及未来产品项目的几条可行路径，
> 防止两种形态在同一个仓库里互相妥协。

---

## 一、为什么 aistack 不会变成产品形态

四条结构性原因，每一条单独都不致命，叠加起来就锁死了形态：

### 1. 依赖链异构，无法统一打包

aistack 当前需要并存：

- **NeMo Toolkit**（Parakeet）—— PyTorch + 一整套 NVIDIA 私有依赖
- **FunASR**（SenseVoice）—— PyTorch + ModelScope 模型分发
- **faster-whisper** —— CTranslate2，相对独立
- **vLLM-Omni**（Qwen3-TTS）—— Docker 容器，必须独立进程
- **Ollama**（LLM）—— 外部二进制

这五个东西**没有共同的运行时**。Ollama 把 llama.cpp 一把封住能做到单二进制，aistack 不行 —— 砍掉任何一个后端都会丢失"对比研究"这个核心价值。

### 2. CUDA / cuDNN 强绑，不能"插哪都能跑"

`pyproject.toml` 已经记录的细节：torch 必须 ≥ 2.7（cuDNN 9.7），nemo_toolkit 要 [cu12] extra，
torchaudio 要锁版本，pyarrow / pandas / datasets 都得固定具体小版本。

这是研发形态可以接受的代价（写进文档 + dev.bat 自动化），但对终端用户每一条都是劝退点。
要消除这些约束就得换技术栈，等同于另起一个项目。

### 3. 价值在"测量与对比"，不在"装上即用"

aistack 已经投入到比产品形态更深的可观测性：

- 滚动窗口 p50/p95/p99 per-capability
- per-request payload capture（重放级别）
- bench harness（多后端 WER/RTF 对比）
- 跨后端语言路由（auto routing alias）

这些**对终端用户毫无价值**——他们只想"按一下出字幕"。但对研发用户、对未来产品决策、
对像 VideoCraft 这样的下游集成方，这些恰恰是核心资产。

把 aistack 优化成产品会先砍掉这一层，等于自我阉割最有价值的部分。

### 4. 研发用户与终端用户对"复杂性可见度"的需求是相反的

| 维度 | 研发用户要的 | 终端用户要的 |
|---|---|---|
| 后端选择 | 看得到、能切、能对比 | 自动选，看不到 |
| 错误信息 | 完整 traceback | 友好的错误提示 |
| GPU 内存 | 实时监控 | 不存在 |
| 模型缓存 | 能手动驱逐 | 自动管理 |
| API 兼容 | OpenAI + aistack 自有扩展（is_routing_alias、supports_streaming）| 只 OpenAI 子集 |

试图同一个产品同时服务两边会同时让两边不满意 —— 这是软件设计常识，不需要再实证。

---

## 二、产品形态的可行路径

如果未来要做"装上即用"的终端用户产品，**必须另开仓库**，而且可行的路径不多。
本节只列方向不展开 —— 真做时在新仓库的 design 文档里详细论证。

### 路径 A：whisper.cpp 为核心 + 离线优先（推荐）

- **技术核心**：whisper.cpp（C++ + GGML，自带 CUDA/Metal/Vulkan/CPU），单二进制
- **差异化**：**离线可用**——installer 自带模型权重，air-gap 环境可用，零遥测，可在飞机/SCIF/工厂车间使用
- **TTS**：v1 不做；v2 加 Piper（同样 native + 小体积）
- **License**：上游 whisper.cpp / Whisper 权重 / Piper 全部 MIT，无传染、无商用限制
- **成本估算**：MVP 3-4 周，产品化（系统服务 + 升级器 + UI） 4-6 周

### 路径 B：Docker 全家桶 + 一键 compose

- **技术核心**：把 aistack + Ollama + Qwen3-TTS 打成一个 docker-compose，写好启动脚本
- **差异化**：能力最全（保留 TTS 高质量、保留多 ASR 后端）
- **代价**：用户必须装 Docker Desktop —— Windows 用户劝退率高，跟"产品形态"目标冲突
- **真实定位**：与其说是产品形态，不如说是"研发形态的友好版"
- **成本估算**：1-2 周，但需求侧不清晰

### 路径 C：sherpa-onnx + ONNX 模型

- **技术核心**：sherpa-onnx（C++ + ONNX Runtime），可加载 Whisper / Paraformer / SenseVoice 等多家 ONNX 导出
- **差异化**：单二进制 + 多模型选择（中文质量比纯 Whisper 强）
- **代价**：模型 ONNX 导出有工程门槛；社区比 whisper.cpp 小一档
- **何时考虑**：路径 A 跑通后，作为引擎扩展加进来。**不适合作为第一条路**

### 不可行/不推荐

- **PyInstaller 把 aistack 整体打包** —— 包体 5-8 GB、冷启动数十秒、调试困难、CUDA 版本绑死。劳而无功
- **重写所有后端为 native** —— NeMo/FunASR 的等价 native 实现要么不存在要么质量打折，工程量以年计
- **Electron + Python 后端** —— Python 后端的所有问题原样保留，再加上 Electron 自身的体积代价

---

## 三、aistack 与未来产品项目的关系

aistack **不是**未来产品的代码基座，**是**未来产品的**研究先驱**：

| aistack 产出 | 给未来产品的输入 |
|---|---|
| 多后端 WER/RTF 实测数据 | "我们应该选哪个 ASR 引擎做产品" 有数据答案 |
| 消费级 GPU 边界（8 GB VRAM 怎么跑大模型） | 产品的最低硬件要求有依据 |
| OpenAI-API 兼容 + 必要扩展字段 | 产品的对外 HTTP 形状已经验证可用 |
| 可观测性指标体系 | 产品后续需要排错时知道该看什么 |

dosmoon 真做产品时，新仓库可以参考 aistack 的 API 形状、复用观测指标的定义，
但**不应该共享代码** —— 两边语言、运行时、目标用户完全不同。

---

## Open questions

- **TTS 的离线产品形态究竟用 Piper 还是放弃 TTS** —— 跟产品定位精确度有关，等产品立项时定
- **离线产品要不要捆绑 LLM**（一个嵌入式 Ollama-like 的 stripped-down llama.cpp）—— 视用户需求强度
- **whisper.cpp 的中文质量是否够用** —— 如果 dosmoon 产品最终面向中文场景，这一条得跑实测，结论可能逼回路径 C
- **air-gap 环境的模型分发** —— USB stick？签名校验？上游模型版本怎么追？这是产品工程的一个完整子课题
