#!/usr/bin/env python
#coding=utf-8
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

print(ROOT_DIR)

import argparse


# import multiprocessing


def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


import shutil


import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import ffmpeg


import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--port',
                    type=int,
                    default=8000)
parser.add_argument('--model_dir',
                    type=str,
                    default='/home/caden/models/CosyVoice2-0_5B',
                    help='local path or modelscope repo id')
args = parser.parse_args()
cosyvoice = CosyVoice2(args.model_dir)
sft_spk = cosyvoice.list_available_spks()
prompt_sr, target_sr = 16000, 22050
default_data = np.zeros(target_sr)


def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


reference_wavs = ["请选择参考音频或者自己上传"]
for name in os.listdir(f"{ROOT_DIR}/audios"):
    reference_wavs.append(name)

spk_new = ["无"]

for name in os.listdir(f"{ROOT_DIR}/voices"):
    # print(name.replace(".pt",""))
    spk_new.append(name.replace(".pt",""))


def refresh_choices():

    spk_new = ["无"]

    for name in os.listdir(f"{ROOT_DIR}/voices"):
        # print(name.replace(".pt",""))
        spk_new.append(name.replace(".pt",""))
    
    return {"choices":spk_new, "__type__": "update"}


def change_choices():

    reference_wavs = ["选择参考音频,或者自己上传"]

    for name in os.listdir(f"{ROOT_DIR}/audios"):
        reference_wavs.append(name)
    
    return {"choices":reference_wavs, "__type__": "update"}


def change_wav(audio_path):

    text = audio_path.replace(".wav","").replace(".mp3","").replace(".WAV","")

    return f"{ROOT_DIR}/audios/{audio_path}",text


def save_name(name):

    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False

    shutil.copyfile(f"{ROOT_DIR}/output.pt",f"{ROOT_DIR}/voices/{name}.pt")
    gr.Info("音色保存成功,存放位置为voices目录")

    

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2.点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3.点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，若同时提供，优先选择prompt音频文件\n2.点击生成音频按钮',
                 '自然语言控制': '1. 提供prompt音频（16kHz）\n2. 输入instruct文本\n3. 点击生成音频按钮'}
def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, speed_factor):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # instruct 模式（CosyVoice2: inference_instruct2）需要 instruct 文本 与 prompt 音频
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.instruct is False:
            gr.Warning('当前模型不支持自然语言控制模式，请使用包含 -Instruct 的 CosyVoice2 模型: {}'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('自然语言控制模式需要输入 instruct 文本')
            return (target_sr, default_data)
    # if cross_lingual mode, please make sure that model is speech_tts/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用speech_tts/CosyVoice-300M模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            return (target_sr, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
    # zero_shot 模式仅使用 prompt_wav 与 prompt_text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, sft_dropdown)
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info('get instruct inference request (CosyVoice2 instruct2)')
        if prompt_wav is None:
            gr.Warning('自然语言控制模式需要提供prompt音频（>=16kHz）')
            return (target_sr, default_data)
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)

    
    # 推理函数返回生成器，这里取最后一个输出作为最终音频
    last_model_output = None
    try:
        for m in output:
            last_model_output = m
    except Exception as e:
        logging.error(f"推理失败: {e}")
        return (target_sr, default_data)

    if last_model_output is None or 'tts_speech' not in last_model_output:
        logging.warning('未获得有效的合成结果')
        return (target_sr, default_data)

    if speed_factor != 1.0:
        try:
            numpy_array = last_model_output['tts_speech'].numpy()
            audio = (numpy_array * 32768).astype(np.int16)
            audio_data = speed_change(audio, speed=speed_factor, sr=int(target_sr))
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
            audio_data = last_model_output['tts_speech'].numpy().flatten()
    else:
        audio_data = last_model_output['tts_speech'].numpy().flatten()


    return (target_sr, audio_data)


def main():
    with gr.Blocks() as demo:

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        speed_factor = gr.Slider(minimum=0.25,maximum=4,step=0.05,label="语速调节",value=1.0,interactive=True)

        # 已移除无需的切分/混合相关控件

        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=None, scale=0.25)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            wavs_dropdown = gr.Dropdown(label="参考音频列表",choices=reference_wavs,value="请选择参考音频或者自己上传",interactive=True)
            refresh_button = gr.Button("刷新参考音频")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[wavs_dropdown])
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value='')

        save_button = gr.Button("保存刚刚推理的zero-shot音色")

        save_button.click(save_name, inputs=[new_name])

        wavs_dropdown.change(change_wav,[wavs_dropdown],[prompt_wav_upload,prompt_text])

        generate_button = gr.Button("生成音频")
        # generate_button_stream = gr.Button("流式生成")

        # audio_output = gr.Audio(label="合成音频")
        audio_output = gr.Audio(label="合成音频",value=None,
            streaming=True,
            autoplay=False,  # disable auto play for Windows, due to https://developer.chrome.com/blog/autoplay#webaudio
            interactive=False,
            show_label=True,show_download_button=True)
        
        # result2 = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)")

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, speed_factor],
                              outputs=[audio_output])

        # generate_button_stream.click(generate_audio_stream,
        #                       inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed,speed_factor,new_dropdown],
        #                       outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_port=args.port,inbrowser=True)



if __name__ == '__main__':
    
    main()


    
