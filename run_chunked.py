#!/usr/bin/env python3
import os
import argparse
from typing import List
import soundfile as sf

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed


def synthesize_chunks(
    cosyvoice: CosyVoice2,
    spk_id: str,
    output_dir: str,
    stream: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    for chunk_idx, text_chunk in enumerate(["这","是一个","切块","语音","合成的","测试"]):
        text_chunk = (text_chunk or "").strip()
        if not text_chunk:
            continue

        print(f"[Chunk {chunk_idx}] 文本: {text_chunk} | spk_id={spk_id}")
        for seg_idx, result in enumerate(
            cosyvoice.inference_sft(text_chunk, spk_id=spk_id, stream=stream)
        ):
            wav = result["tts_speech"].squeeze().cpu().numpy()
            sr = cosyvoice.sample_rate
            out_path = os.path.join(output_dir, f"chunk_{chunk_idx:03d}_{seg_idx:02d}.wav")
            sf.write(out_path, wav, sr)
            print(f"  -> 保存: {out_path} (sr={sr})")


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice2 按文本分块合成脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("COSYVOICE2_MODEL", "/home/caden/models/CosyVoice2-0_5B"),
        help="模型目录",
    )
    parser.add_argument(
        "--spk_id",
        type=str,
        default="caden_zh",
        help="预置音色ID，例如 woman / man 等",
    )
    parser.add_argument(
        "--chunks_file",
        type=str,
        help="从文件读取分块文本，按行作为一个chunk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./chunk_outputs",
        help="输出目录",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="关闭模型内部流式产出，仅整段返回",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_random_seed(args.seed)

    print("正在初始化CosyVoice2模型...")
    cosyvoice = CosyVoice2(
        args.model_path,
        load_jit=False,
        load_trt=False,
        fp16=False,
    )
    print("模型初始化完成！")


    synthesize_chunks(
        cosyvoice=cosyvoice,
        spk_id=args.spk_id,
        output_dir=args.output_dir,
        stream=(not args.no_stream),
    )

    print("全部分块合成完成！")


if __name__ == "__main__":
    main()


