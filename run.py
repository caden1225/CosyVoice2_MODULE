#!/usr/bin/env python3
import os
import soundfile as sf

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def main():
    # 模型路径 - 从环境变量获取或使用默认值
    model_path = '/home/caden/models/CosyVoice2-0_5B'
    # 初始化CosyVoice2模型
    print("正在初始化CosyVoice2模型...")
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
    print("模型初始化完成！")

    # cosyvoice.add_zero_shot_spk(
    #     prompt_text="오늘 날씨가 어때요",
    #     prompt_speech_16k=load_wav("/home/caden/workspace/cosyvoice_minimal/audios/오늘 날씨가 어때요.wav", 16000),
    #     zero_shot_spk_id="kr"
    # )

    # 加载参考音频 - 从环境变量获取或使用默认值
    prompt_audio_path = os.getenv('COSYVOICE_PROMPT_AUDIO', '/home/caden/workspace/cosyvoice_minimal/audios/我当然知道了.wav')
    prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    print(f"已加载参考音频: {prompt_audio_path}")
    
    # 示例1: Zero-shot推理
    print("\n=== Zero-shot推理 ===")
    text_to_synthesize = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    prompt_text = '我当然知道了'
    
    print(f"合成文本: {text_to_synthesize}")
    print(f"提示文本: {prompt_text}")
    # # 示例1：Zero-shot推理 DONE
    # print(f"接收参数：text_to_synthesize, prompt_text, prompt_speech_16k") 
    # for i, result in enumerate(cosyvoice.inference_zero_shot(text_to_synthesize, prompt_text, prompt_speech_16k, stream=False)):
    #     output_file = f'zero_shot_prompt_speech_{i}.wav'
    #     # 使用 soundfile 保存音频文件
    #     import soundfile as sf
    #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    #     print(f"已保存: {output_file}")
    
    # 示例2: 细粒度控制推理
    print("\n=== 细粒度控制推理 ===")
    print(f"接收参数：text_with_control, prompt_speech_16k") 
    text_with_control = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
    print(f"合成文本: {text_with_control}")
    
    for i, result in enumerate(cosyvoice.inference_cross_lingual(text_with_control, prompt_speech_16k, stream=False)):
        output_file = f'fine_grained_control_{i}.wav'
        # 使用 soundfile 保存音频文件
        sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
        print(f"已保存: {output_file}")
    
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

    # # 示例4: 预置音色推理 DONE
    # print("\n=== 预置音色推理 ===")
    # print(f"接收参数：text_to_synthesize, spk_id:woman") 
    # for i, result in enumerate(cosyvoice.inference_sft(text_to_synthesize, spk_id="woman", stream=False)):
    #     output_file = f'sft_spk_speech_{i}.wav'
    #     # 使用 soundfile 保存音频文件
    #     import soundfile as sf
    #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    #     print(f"已保存: {output_file}")




    # print("\n所有推理完成！")

if __name__ == "__main__":
    main()
