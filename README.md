# CosyVoice Minimal（模块化、可嵌入、最小依赖）

CosyVoice Minimal 的目标：将 CosyVoice 的核心推理能力模块化，尽量将依赖收敛到本项目中，实现“可在任意环境快速集成”的最小可用组件（CLI/WebUI/SDK）。

本项目内置必要的前端、声学与声码器推理逻辑，默认关闭 JIT/TRT 等可选优化，优先保证开箱即用与可移植性；需要时可在相同 API 上开启加速。

## 特性一览

- 模块化 API：`CosyVoice2` 提供统一推理接口（SFT/Zero-shot/跨语种/指令）。
- 可嵌入：作为 SDK 直接导入使用，或通过 CLI、Gradio WebUI 快速验证。
- 最小依赖：依赖列表精简并随仓库提供，避免拉取完整上游工程。
- 离线友好：默认设置禁用部分在线拉取（见 `webui.py` 顶部环境变量）。
- 跨环境：CPU/GPU 皆可运行；JIT/TRT/VLLM 等可选能力按需开启。

## 版本更新 v0.5

- 新增对 macOS 的官方支持（Apple Silicon 与 Intel 均可）。
- 提供 `requirements-mac.txt`，便于在 macOS 上一键安装最小依赖。
- 默认配置与 API 接口保持不变，按需开启 `fp16/load_trt/load_vllm`（macOS 上建议使用 CPU 或 MPS）。

## 目录结构

```
cosyvoice_minimal/
├── cosyvoice/                   # 核心模块（前端/模型/声码器/工具）
│   ├── cli/
│   │   ├── cosyvoice.py         # CosyVoice / CosyVoice2 高层封装与推理接口
│   │   ├── model.py             # 模型装载与推理实现
│   │   └── frontend.py          # 文本/提示/说话人前端处理
│   ├── utils/                   # 文件/前端工具等
│   ├── flow/, hifigan/, llm/, transformer/, tokenizer/, dataset/
├── run.py                       # CLI 示例脚本（最小推理用法）
├── webui.py                     # WebUI（Gradio）
├── requirements.txt             # 依赖清单（最小化）
├── install.sh                   # 一键安装（CPU 版示例）
├── audios/                      # 示例参考音频
└── README.md
```

## 环境与安装

建议使用 Conda 并启用 Python 3.12 的环境（遵循本地约定“312”环境）。

```bash
conda create -n 312 python=3.12 -y
conda activate 312
pip install -r requirements.txt
# 或使用脚本（CPU 版 PyTorch 源）：
bash install.sh
```

提示：如需 GPU 版本 PyTorch/ONNX Runtime，请根据你的 CUDA 版本改用相应官方索引安装。

### macOS（Apple Silicon/Intel）安装

在 macOS 上推荐同样使用 3.12 的 Conda 环境，并安装 `requirements-mac.txt`：

```bash
conda create -n 312 python=3.12 -y
conda activate 312
pip install -r requirements-mac.txt
```

说明：

- Apple Silicon（M1/M2/M3）上，PyTorch 默认支持 MPS（Metal）。目前本项目默认以 CPU/MPS 运行，无需 CUDA。
- 如果你希望使用 MPS，可保持默认设置；若遇到显存相关报错，可适当缩短文本或分段推理。
- 其余用法与 Linux 一致，WebUI/CLI/SDK API 无差异。

## 模型准备

默认模型目录（可自定义）：

```
/home/caden/models/CosyVoice2-0_5B/
```

该目录应包含（CosyVoice2）至少：

- `cosyvoice2.yaml`
- `llm.pt`, `flow.pt`, `hift.pt`
- `campplus.onnx`, `speech_tokenizer_v2.onnx`
- `spk2info.pt`
- `CosyVoice-BlankEN/`（前端所需 Qwen 资源路径会在加载时以该目录为根）

你也可以放置其他尺寸或带 `-Instruct` 后缀的 CosyVoice2 模型目录，API 保持一致。

## 快速开始

### 方式一：命令行示例（最小用法）

```bash
python run.py
```

默认行为（见 `run.py`）：

- 使用 `CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)` 初始化（保证通用性）。
- 预置执行 SFT（预训练音色）推理并保存到 `sft_spk_speech_0.wav`。
- 可自行取消注释 Zero-shot/跨语种/指令示例代码段进行测试。

支持通过环境变量覆盖部分路径：

- `COSYVOICE_PROMPT_AUDIO`：参考音频路径（默认 `./audios/我当然知道了.wav`）。

### 方式二：WebUI（Gradio）

```bash
python webui.py --model_dir /home/caden/models/CosyVoice2-0_5B --port 8000
```

功能：

- 四种模式：`预训练音色`、`3s极速复刻`（Zero-shot）、`跨语种复刻`、`自然语言控制`（需 Instruct 模型）。
- 支持上传或录音作为 prompt 音频，支持语速调节与一键保存 zero-shot 复刻音色。
- 默认设置了若干离线/兼容性环境变量（如禁用 HF datasets 在线、torchaudio dispatcher 等）。

## 作为模块嵌入（SDK 用法）

最小代码骨架如下（同步 `cosyvoice/cli/cosyvoice.py` 的接口）：

```python
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

model_dir = "/path/to/CosyVoice2"
cv = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

# 1) 预训练音色（SFT）
for m in cv.inference_sft("你好，世界。", spk_id="woman", stream=False):
    audio = m["tts_speech"].numpy()

# 2) Zero-shot 复刻
prompt_text = "我当然知道了"
prompt = load_wav("./audios/我当然知道了.wav", 16000)
for m in cv.inference_zero_shot("待合成文本", prompt_text, prompt, stream=False):
    audio = m["tts_speech"].numpy()

# 3) 跨语种复刻
for m in cv.inference_cross_lingual("こんにちは", prompt, stream=False):
    audio = m["tts_speech"].numpy()

# 4) 自然语言控制（CosyVoice2-Instruct 模型）
for m in cv.inference_instruct2("请用激昂的情绪朗读。", "更有激情一些", prompt, stream=False):
    audio = m["tts_speech"].numpy()
```

可选增强：

- `fp16=True`：半精度（需 GPU）。
- `load_trt=True`：启用 TensorRT 解码（需准备 `.plan/onnx` 资源）。
- `load_vllm=True`（仅 CosyVoice2）：外接 vLLM（见代码）。

## 依赖最小化与环境说明

- `requirements.txt` 提供最小依赖集合；`install.sh` 给出 CPU 参考安装流程。
- `webui.py` 顶部设置了：
  - `HF_DATASETS_OFFLINE=1`（尽量离线）；
  - `TORCHAUDIO_USE_BACKEND_DISPATCHER=0`（兼容性）；
  - `CURL_CA_BUNDLE=''`（某些网络环境下的证书问题绕开）。
- 默认关闭 JIT/TRT/FP16，保证在无 GPU/驱动环境也能跑通；需要时在初始化参数中开启。

## 常见问题（FAQ）

- 没有输出/推理报错：确认模型目录完整（见“模型准备”），日志中如提示缺文件请补齐。
- CUDA OOM：尝试 `fp16=True`、缩短输入文本、避免过多并发；或先用 CPU 验证流程。
- 提示文本与音频不一致导致效果差：确保 Zero-shot 的 `prompt_text` 与参考音频语义一致且语种匹配。
- 采样率问题：WebUI 下 prompt 音频需 ≥16kHz；CLI/SDK 请用 `load_wav(..., 16000)`。
- 速度调节：WebUI 内置 `speed_change`；SDK 可在各 `inference_*` 中使用 `speed` 参数。

## 许可证

本项目基于 Apache License 2.0，与上游 CosyVoice 项目保持一致。
