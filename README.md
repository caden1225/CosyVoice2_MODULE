# CosyVoice Minimal Inference Package

这是CosyVoice的精简推理包，只包含必要的推理功能，无需下载完整的CosyVoice项目。

## 文件结构

```
cosyvoice_minimal/
├── cosyvoice/                 # 核心推理模块
│   ├── cli/                  # 命令行接口
│   ├── utils/                 # 工具函数
│   ├── llm/                   # 语言模型
│   ├── flow/                  # 流模型
│   ├── hifigan/               # 声码器
│   ├── transformer/           # Transformer组件
│   ├── tokenizer/             # 分词器
│   └── dataset/               # 数据处理
├── run.py                     # 推理脚本
├── requirements.txt           # 依赖包
└── README.md                  # 说明文档
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型

确保CosyVoice2模型已下载到本地，默认路径为：
```
/home/caden/models/CosyVoice2-0_5B/
```

模型目录应包含以下文件：
- `cosyvoice2.yaml` - 配置文件
- `llm.pt` - 语言模型权重
- `flow.pt` - 流模型权重
- `hift.pt` - 声码器权重
- `campplus.onnx` - 说话人编码器
- `speech_tokenizer_v2.onnx` - 语音分词器
- `spk2info.pt` - 说话人信息
- `CosyVoice-BlankEN/` - Qwen模型目录

### 3. 准备参考音频

将参考音频文件放在指定位置：
```
/home/caden/workspace/data/audio_samples/打开车门.wav
```

### 4. 运行推理

```bash
python run.py
```

## 功能说明

脚本支持三种推理模式：

1. **Zero-shot推理**: 使用参考音频和提示文本进行语音合成
2. **细粒度控制**: 支持在文本中添加特殊标记（如[laughter]）
3. **指令推理**: 根据指令改变语音风格（如方言）

## 自定义配置

如需修改配置，请编辑`run.py`文件中的以下变量：

```python
# 模型路径
model_dir = '/path/to/your/model'

# 参考音频路径
prompt_audio_path = '/path/to/your/audio.wav'

# 合成文本
text_to_synthesize = '你要合成的文本'

# 提示文本（用于zero-shot）
prompt_text = '参考音频对应的文本'

# 指令（用于指令推理）
instruction = '用某种方言说这句话'
```

## 注意事项

1. 首次运行时会自动下载wetext模型，需要网络连接
2. 确保有足够的GPU内存（建议8GB以上）
3. 音频文件格式支持wav、mp3等常见格式
4. 合成文本长度建议不超过200个字符

## 故障排除

### 常见问题

1. **CUDA内存不足**: 尝试设置`fp16=True`或减少文本长度
2. **模型文件缺失**: 检查模型目录是否完整
3. **音频加载失败**: 检查音频文件路径和格式
4. **网络连接问题**: wetext模型下载需要网络连接

### 性能优化

- 使用GPU加速：确保安装了`onnxruntime-gpu`
- 启用半精度：设置`fp16=True`
- 批量处理：可以修改脚本支持批量文本处理

## 许可证

本包基于Apache License 2.0许可证，与原始CosyVoice项目保持一致。
