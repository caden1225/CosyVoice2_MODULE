#!/usr/bin/env python3
"""
CosyVoice Minimal Inference Script
精简版CosyVoice推理脚本，只包含必要的推理功能
"""

import sys
import os
import torchaudio
import soundfile as sf

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def main():
    # 模型路径 - 从环境变量获取或使用默认值
    model_dir = os.getenv('COSYVOICE_MODEL_DIR', '/home/caden/models/CosyVoice2-0_5B')

    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"错误：模型目录不存在: {model_dir}")
        print("请设置环境变量 COSYVOICE_MODEL_DIR 或确保默认模型路径存在")
        return

    # Wetext模型路径 - 从环境变量获取或使用默认值
    wetext_model_dir = os.getenv('COSYVOICE_WETEXT_DIR', '/home/caden/models/Cosyvoice_Wetext')

    # 初始化CosyVoice2模型
    print("正在初始化CosyVoice2模型...")
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False, wetext_model_dir=wetext_model_dir)
    print("模型初始化完成！")

    # 加载参考音频 - 从环境变量获取或使用默认值
    prompt_audio_path = os.getenv('COSYVOICE_PROMPT_AUDIO', '/home/caden/workspace/data/audio_samples/打开车窗.wav')
    if not os.path.exists(prompt_audio_path):
        print(f"警告：参考音频文件不存在: {prompt_audio_path}")
        print("请设置环境变量 COSYVOICE_PROMPT_AUDIO 或确保默认音频文件存在")
        return
    for output in cosyvoice.inference_sft(
        tts_text="这是一段很长的文本，从前有个山，山里有个庙，庙里有个和尚，和尚在念经，念经的声音很好听。他喜欢念，从前有个山...",
        spk_id='caden_zh',
        stream=False,
        speed=1.0
    ):
        tts_speech = output['tts_speech']
        audio_data = tts_speech.squeeze().cpu().numpy()
        sf.write(f'test_speed.wav', audio_data, cosyvoice.sample_rate)

    # prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    # print(f"已加载参考音频: {prompt_audio_path}")
    
    # # 示例1: Zero-shot推理
    # print("\n=== Zero-shot推理 ===")
    # text_to_synthesize = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    # prompt_text = '希望你以后能够做的比我还好呦。'
    
    # print(f"合成文本: {text_to_synthesize}")
    # print(f"提示文本: {prompt_text}")
    
    # for i, result in enumerate(cosyvoice.inference_zero_shot(text_to_synthesize, prompt_text, prompt_speech_16k, stream=False)):
    #     output_file = f'zero_shot_{i}.wav'
    #     # 使用 soundfile 保存音频文件
    #     import soundfile as sf
    #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    #     print(f"已保存: {output_file}")
    
    # # 示例2: 细粒度控制推理
    # print("\n=== 细粒度控制推理 ===")
    # text_with_control = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
    
    # print(f"合成文本: {text_with_control}")
    
    # for i, result in enumerate(cosyvoice.inference_cross_lingual(text_with_control, prompt_speech_16k, stream=False)):
    #     output_file = f'fine_grained_control_{i}.wav'
    #     # 使用 soundfile 保存音频文件
    #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    #     print(f"已保存: {output_file}")
    
    # # 示例3: 指令推理
    # print("\n=== 指令推理 ===")
    # instruction = '用英文说这句话'
    # text_to_synthesize = '今天天气真好，我们去公园散步吧。'
    
    # print(f"合成文本: {text_to_synthesize}")
    # print(f"指令: {instruction}")
    
    # for i, result in enumerate(cosyvoice.inference_instruct2(text_to_synthesize, instruction, prompt_speech_16k, stream=False)):
    #     output_file = f'instruct_{i}.wav'
    #     # 使用 soundfile 保存音频文件
    #     import soundfile as sf
    #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    #     print(f"已保存: {output_file}")
    
    # print("\n所有推理完成！")

if __name__ == "__main__":
    main()
