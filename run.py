#!/usr/bin/env python3
import os
import soundfile as sf

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import numpy as np
from pathlib import Path
import json
import time as _time

def main():
    set_all_random_seed(0)
    # 模型路径 - 从环境变量获取或使用默认值
    model_path = '/home/caden/models/CosyVoice2-0_5B'
    # model_path = '/Users/caden/models/CosyVoice2-0.5B'
    print("正在初始化CosyVoice2模型...")
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
    print("模型初始化完成！")

    # ========== 调试/打点：保存并打印每阶段与模块的输入输出 ==========
    # 日志根目录：优先用环境变量 COSY_LOG_ROOT；否则按 transformers 版本命名
    try:
        import transformers as _tf
        _tf_ver = getattr(_tf, '__version__', 'unknown')
    except Exception:
        _tf_ver = 'unknown'
    log_root_env = os.getenv('COSY_LOG_ROOT')
    log_root = Path(log_root_env) if log_root_env else Path(f'./debug_logs_{_tf_ver}')
    log_root.mkdir(parents=True, exist_ok=True)

    def _to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(x, np.ndarray):
            return x
        # 标量或列表
        try:
            return np.array(x)
        except Exception:
            return np.array([str(type(x))])

    def _save_item(base_dir: Path, name: str, obj):
        base_dir.mkdir(parents=True, exist_ok=True)
        npy_path = base_dir / f'{name}.npy'
        meta_path = base_dir / f'{name}.meta.json'
        arr = _to_numpy(obj)
        try:
            np.save(npy_path, arr, allow_pickle=False)
        except Exception:
            # 回退为可pickle
            np.save(npy_path, arr, allow_pickle=True)
        try:
            shape = list(arr.shape)
        except Exception:
            shape = []
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else 'unknown',
                'shape': shape,
                'saved_at': _time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        print(f"[saved] {name}: shape={shape}, path={npy_path}")

    # --- Hook frontend._extract_speech_feat ---
    fe = cosyvoice.frontend
    if hasattr(fe, '_extract_speech_feat'):
        _orig_extract_speech_feat = fe._extract_speech_feat
        def _wrapped_extract_speech_feat(speech):
            stage_dir = log_root / 'frontend_extract_speech_feat'
            _save_item(stage_dir, 'input_speech', speech)
            out = _orig_extract_speech_feat(speech)
            speech_feat, speech_feat_len = out
            _save_item(stage_dir, 'output_speech_feat', speech_feat)
            _save_item(stage_dir, 'output_speech_feat_len', speech_feat_len)
            print(f"frontend._extract_speech_feat: feat_shape={_to_numpy(speech_feat).shape}, len={int(_to_numpy(speech_feat_len)[0])}")
            return out
        fe._extract_speech_feat = _wrapped_extract_speech_feat

    # --- Hook frontend._extract_spk_embedding ---
    if hasattr(fe, '_extract_spk_embedding'):
        _orig_extract_spk_embedding = fe._extract_spk_embedding
        def _wrapped_extract_spk_embedding(speech):
            stage_dir = log_root / 'frontend_extract_spk_embedding'
            _save_item(stage_dir, 'input_speech_16k', speech)
            emb = _orig_extract_spk_embedding(speech)
            _save_item(stage_dir, 'output_embedding', emb)
            print(f"frontend._extract_spk_embedding: emb_shape={_to_numpy(emb).shape}")
            return emb
        fe._extract_spk_embedding = _wrapped_extract_spk_embedding

    # --- Hook frontend.frontend_zero_shot 以捕获 model_input ---
    if hasattr(fe, 'frontend_zero_shot'):
        _orig_frontend_zero_shot = fe.frontend_zero_shot
        def _wrapped_frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
            model_input = _orig_frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id)
            stage_dir = log_root / 'model_input_zero_shot'
            # 保存关键项
            for k, v in model_input.items():
                try:
                    _save_item(stage_dir, k, v)
                except Exception as e:
                    print(f"[warn] save model_input[{k}] failed: {e}")
            # 打印概览
            overview = {k: list(_to_numpy(v).shape) for k, v in model_input.items()}
            print("model_input(keys->shape):", overview)
            return model_input
        fe.frontend_zero_shot = _wrapped_frontend_zero_shot

    # --- Hook CosyVoiceModel.llm_job 以记录 LLM 输入输出 ---
    from cosyvoice.cli import model as _cosy_model_mod
    if hasattr(_cosy_model_mod.CosyVoiceModel, 'llm_job'):
        _orig_llm_job = _cosy_model_mod.CosyVoiceModel.llm_job
        def _wrapped_llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
            stage_dir = log_root / 'llm'
            # 保存输入
            _save_item(stage_dir, f'{uuid}_text', text)
            _save_item(stage_dir, f'{uuid}_prompt_text', prompt_text)
            _save_item(stage_dir, f'{uuid}_llm_prompt_speech_token', llm_prompt_speech_token)
            _save_item(stage_dir, f'{uuid}_llm_embedding', llm_embedding)
            print("LLM 输入 shapes:",
                  { 'text': list(_to_numpy(text).shape),
                    'prompt_text': list(_to_numpy(prompt_text).shape),
                    'llm_prompt_speech_token': list(_to_numpy(llm_prompt_speech_token).shape),
                    'llm_embedding': list(_to_numpy(llm_embedding).shape) })
            # 调用原始逻辑
            _orig_llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid)
            # 保存输出（token 序列）
            try:
                tokens = self.tts_speech_token_dict.get(uuid, [])
                _save_item(stage_dir, f'{uuid}_llm_output_tokens', np.array(tokens, dtype=np.int32))
                print(f"LLM 输出 token 数量: {len(tokens)}")
            except Exception as e:
                print(f"[warn] save llm outputs failed: {e}")
        _cosy_model_mod.CosyVoiceModel.llm_job = _wrapped_llm_job

    # --- 进一步细化：为 llm 实例及其子模块增加打点（模块间 I/O）---
    try:
        llm_inst = getattr(cosyvoice.model, 'llm', None)
        import torch
        if llm_inst is not None:
            # 1) 包装 llm.inference（逐步保存输出）
            if hasattr(llm_inst, 'inference'):
                _orig_llm_infer = llm_inst.inference
                def _wrapped_llm_infer(*args, **kwargs):
                    stage_dir = log_root / 'llm_inference'
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    # 保存输入概要
                    try:
                        for idx, a in enumerate(args):
                            _save_item(stage_dir, f'input_arg{idx}', a)
                        for k, v in kwargs.items():
                            _save_item(stage_dir, f'input_kw_{k}', v)
                    except Exception as e:
                        print(f"[warn] save llm.inference inputs failed: {e}")
                    # 追踪输出序列
                    step = 0
                    for out in _orig_llm_infer(*args, **kwargs):
                        try:
                            _save_item(stage_dir, f'output_step_{step}', out)
                        except Exception:
                            pass
                        print(f"llm.inference -> step {step} saved")
                        step += 1
                        yield out
                llm_inst.inference = _wrapped_llm_infer
            # 2) 包装 llm.inference_bistream（如存在）
            if hasattr(llm_inst, 'inference_bistream'):
                _orig_llm_bi = llm_inst.inference_bistream
                def _wrapped_llm_bi(*args, **kwargs):
                    stage_dir = log_root / 'llm_inference_bistream'
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        for idx, a in enumerate(args):
                            _save_item(stage_dir, f'input_arg{idx}', a)
                        for k, v in kwargs.items():
                            _save_item(stage_dir, f'input_kw_{k}', v)
                    except Exception as e:
                        print(f"[warn] save llm.inference_bistream inputs failed: {e}")
                    step = 0
                    for out in _orig_llm_bi(*args, **kwargs):
                        try:
                            _save_item(stage_dir, f'output_step_{step}', out)
                        except Exception:
                            pass
                        print(f"llm.inference_bistream -> step {step} saved")
                        step += 1
                        yield out
                llm_inst.inference_bistream = _wrapped_llm_bi
            # 2.5) 包装 llm.sampling_ids 以捕获每步采样前的分布与选择
            try:
                if hasattr(llm_inst, 'sampling_ids'):
                    _orig_sampling_ids = llm_inst.sampling_ids
                    _call_idx = { 'step': 0 }
                    def _wrapped_sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos=True):
                        step = _call_idx['step']
                        stage_dir = log_root / 'llm_sampling'
                        stage_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            import numpy as _np
                            # 保存原始logits(对数概率)的topk摘要与全量统计
                            ws = weighted_scores.detach().float().cpu().numpy()
                            # 统计
                            _save_item(stage_dir, f'step_{step}_logp_stats', _np.array([
                                ws.min() if ws.size else 0.0,
                                ws.max() if ws.size else 0.0,
                                ws.mean() if ws.size else 0.0,
                                ws.std() if ws.size else 0.0
                            ], dtype=_np.float32))
                            # topk
                            try:
                                import torch as _t
                                topk = 20
                                topv, topi = _t.topk(_t.from_numpy(ws), k=min(topk, ws.shape[-1]))
                                _save_item(stage_dir, f'step_{step}_topk_scores', topv.numpy())
                                _save_item(stage_dir, f'step_{step}_topk_indices', topi.numpy())
                            except Exception:
                                pass
                            # 记录入参
                            _save_item(stage_dir, f'step_{step}_sampling_param', _np.array([int(sampling)], dtype=_np.int32))
                            _save_item(stage_dir, f'step_{step}_ignore_eos', _np.array([1 if ignore_eos else 0], dtype=_np.int32))
                            # 当前已解码token长度
                            try:
                                _save_item(stage_dir, f'step_{step}_decoded_len', _np.array([len(decoded_tokens)], dtype=_np.int32))
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"[warn] save sampling_ids info failed at step {step}: {e}")
                        out = _orig_sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos)
                        try:
                            # 保存选择结果
                            _save_item(stage_dir, f'step_{step}_chosen', _to_numpy(out))
                        except Exception:
                            pass
                        _call_idx['step'] += 1
                        return out
                    llm_inst.sampling_ids = _wrapped_sampling_ids
                    print('[hook] wrapped llm.sampling_ids for logits/topk capture')
            except Exception as e:
                print(f"[warn] wrap sampling_ids failed: {e}")
            # 3) 为 text_encoder 与 llm 主干注册 forward hooks（捕获张量 I/O）
            def _register_forward_hooks(module, name_prefix: str):
                calls = {'count': 0}
                def pre_hook(mod, inputs):
                    idx = calls['count']
                    try:
                        for j, inp in enumerate(inputs):
                            _save_item(log_root / 'llm_modules' / name_prefix, f'pre_input_{idx}_{j}', inp)
                    except Exception:
                        pass
                def fwd_hook(mod, inputs, output):
                    idx = calls['count']
                    try:
                        _save_item(log_root / 'llm_modules' / name_prefix, f'post_output_{idx}', output)
                    except Exception:
                        pass
                    calls['count'] += 1
                try:
                    module.register_forward_pre_hook(pre_hook)
                    module.register_forward_hook(fwd_hook)
                    print(f"[hook] registered forward hooks for {name_prefix}")
                except Exception as e:
                    print(f"[warn] register hooks for {name_prefix} failed: {e}")

            if hasattr(llm_inst, 'text_encoder') and isinstance(llm_inst.text_encoder, torch.nn.Module):
                _register_forward_hooks(llm_inst.text_encoder, 'text_encoder')
            if hasattr(llm_inst, 'llm') and isinstance(llm_inst.llm, torch.nn.Module):
                _register_forward_hooks(llm_inst.llm, 'llm_core')
    except Exception as e:
        print(f"[warn] extra llm hooks setup failed: {e}")

    # cosyvoice.add_zero_shot_spk(
    #     prompt_text="오늘 날씨가 어때요",
    #     prompt_speech_16k=load_wav("/home/caden/workspace/cosyvoice_minimal/audios/오늘 날씨가 어때요.wav", 16000),
    #     zero_shot_spk_id="kr"
    # )

    # prompt_audio_path = '/Users/caden/workspace/audios/我当然知道了.wav'
    prompt_audio_path = './audios/我当然知道了.wav'
    prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    print(f"已加载参考音频: {prompt_audio_path}")
    
    # 示例1：Zero-shot推理 DONE
    print("\n=== Zero-shot推理 ===")
    text_to_synthesize = '收到好友从远方寄来的生日礼物'
    prompt_text = '我当然知道了'
    
    print(f"合成文本: {text_to_synthesize}")
    print(f"提示文本: {prompt_text}")
    import time
    start = time.perf_counter()
    print(f"接收参数：text_to_synthesize, prompt_text, prompt_speech_16k") 
    for i, result in enumerate(cosyvoice.inference_zero_shot(text_to_synthesize, prompt_text, prompt_speech_16k, stream=False)):
        output_file = f'orin_zero_shot_prompt_speech_{i}.wav'
        # 使用 soundfile 保存音频文件
        print(f"time cost: {time.perf_counter() - start}")
        sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
        print(f"已保存: {output_file}")
    
    # # 示例2: 细粒度控制推理
    # print("\n=== 细粒度控制推理 ===")
    # print(f"接收参数：text_with_control, prompt_speech_16k") 
    # text_with_control = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
    # print(f"合成文本: {text_with_control}")
    
    # for i, result in enumerate(cosyvoice.inference_cross_lingual(text_with_control, prompt_speech_16k, stream=False)):
    #     output_file = f'fine_grained_control_{i}.wav'
    #     # 使用 soundfile 保存音频文件
    #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    #     print(f"已保存: {output_file}")
    
    # # # 示例3: 指令推理
    # # print("\n=== 指令推理 ===")
    # # instruction = '用英文说这句话'
    # # text_to_synthesize = '今天天气真好，我们去公园散步吧。'
    
    # # print(f"合成文本: {text_to_synthesize}")
    # # print(f"指令: {instruction}")
    
    # # for i, result in enumerate(cosyvoice.inference_instruct2(text_to_synthesize, instruction, prompt_speech_16k, stream=False)):
    # #     output_file = f'instruct_{i}.wav'
    # #     # 使用 soundfile 保存音频文件
    # #     import soundfile as sf
    # #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    # #     print(f"已保存: {output_file}")

    # # # 示例4: 预置音色推理 DONE
    # # print("\n=== 预置音色推理 ===")
    # # print(f"接收参数：text_to_synthesize, spk_id:woman") 
    # # for i, result in enumerate(cosyvoice.inference_sft(text_to_synthesize, spk_id="woman", stream=False)):
    # #     output_file = f'sft_spk_speech_{i}.wav'
    # #     # 使用 soundfile 保存音频文件
    # #     import soundfile as sf
    # #     sf.write(output_file, result['tts_speech'].squeeze().cpu().numpy(), cosyvoice.sample_rate)
    # #     print(f"已保存: {output_file}")




    # print("\n所有推理完成！")

if __name__ == "__main__":
    main()
