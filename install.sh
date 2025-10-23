#!/bin/bash
# CosyVoice Minimal Package Installation Script

echo "正在安装CosyVoice精简推理包的依赖..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python版本: $python_version"

# 安装PyTorch (CPU版本，如果需要GPU版本请修改)
echo "安装PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
echo "安装其他依赖..."
pip install numpy onnxruntime hyperpyyaml omegaconf tqdm modelscope
pip install wetext inflect whisper kaldifst

echo "依赖安装完成！"
echo ""
echo "使用方法："
echo "1. 确保模型文件在正确位置: /home/caden/models/CosyVoice2-0_5B/"
echo "2. 确保参考音频文件存在: /home/caden/workspace/data/audio_samples/打开车门.wav"
echo "3. 运行推理: python run.py"
