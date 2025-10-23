#!/usr/bin/env python3
"""
CosyVoice Minimal Package Test Script
测试精简包是否能正常工作
"""

import sys
import os

def test_imports():
    """测试所有必要的模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        print("✓ CosyVoice2 导入成功")
    except Exception as e:
        print(f"✗ CosyVoice2 导入失败: {e}")
        return False
    
    try:
        from cosyvoice.utils.file_utils import load_wav
        print("✓ load_wav 导入成功")
    except Exception as e:
        print(f"✗ load_wav 导入失败: {e}")
        return False
    
    try:
        import torchaudio
        print("✓ torchaudio 导入成功")
    except Exception as e:
        print(f"✗ torchaudio 导入失败: {e}")
        return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    
    model_dir = '/home/caden/models/CosyVoice2-0_5B'
    
    if not os.path.exists(model_dir):
        print(f"✗ 模型目录不存在: {model_dir}")
        return False
    
    print(f"✓ 模型目录存在: {model_dir}")
    
    # 检查必要的模型文件
    required_files = [
        'cosyvoice2.yaml',
        'llm.pt',
        'flow.pt', 
        'hift.pt',
        'campplus.onnx',
        'speech_tokenizer_v2.onnx'
    ]
    
    # 可选文件
    optional_files = [
        'spk2info.pt'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 缺失")
            missing_files.append(file)
    
    # 检查可选文件
    for file in optional_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} 存在")
        else:
            print(f"⚠ {file} 不存在（可选）")
    
    if missing_files:
        print(f"缺失必要文件: {missing_files}")
        return False
    
    return True

def test_audio_loading():
    """测试音频文件加载"""
    print("\n=== 测试音频文件加载 ===")
    
    audio_path = '/home/caden/workspace/data/audio_samples/打开车门.wav'
    
    if not os.path.exists(audio_path):
        print(f"✗ 音频文件不存在: {audio_path}")
        return False
    
    print(f"✓ 音频文件存在: {audio_path}")
    
    try:
        from cosyvoice.utils.file_utils import load_wav
        audio = load_wav(audio_path, 16000)
        print(f"✓ 音频加载成功，形状: {audio.shape}")
        return True
    except Exception as e:
        print(f"✗ 音频加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("CosyVoice 精简包测试")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 模块导入测试失败")
        return
    
    # 测试模型
    if not test_model_loading():
        print("\n❌ 模型加载测试失败")
        return
    
    # 测试音频
    if not test_audio_loading():
        print("\n❌ 音频加载测试失败")
        return
    
    print("\n✅ 所有测试通过！精简包可以正常使用")
    print("\n使用方法:")
    print("python run.py")

if __name__ == "__main__":
    main()
