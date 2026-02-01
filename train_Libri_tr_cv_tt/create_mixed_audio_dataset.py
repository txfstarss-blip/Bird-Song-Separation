import os
import random
import json5
import argparse
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
import soundfile as sf
from tqdm import tqdm  # 引入 tqdm 用于进度条
import json


def load_config(config_file="config.json5"):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json5.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到配置文件: {config_file}")
    except json5.Json5DecoderException as e:
        raise ValueError(f"配置文件格式错误: {e}")


def parse_args():
    """解析命令行参数，允许覆盖配置文件中的 n_sources"""
    parser = argparse.ArgumentParser(description="生成混合音频数据集")
    parser.add_argument('--n_sources', type=int, nargs='+', default=None,
                        help='需要混合的音源数量列表，例如: --n_sources 2 3')
    return parser.parse_args()


def generate_mixed_csv(config, n_sources_list):
    """生成多源混合音频的 CSV 文件"""
    input_root = config["input_root"]
    output_dir = config["output_dir"]
    total_files = config["total_files"]
    split_ratio = config["split_ratio"]

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"输入音频根目录不存在: {input_root}")

    subdirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if not subdirs:
        raise ValueError(f"输入目录 {input_root} 中没有子文件夹")
    if len(subdirs) < max(n_sources_list):
        raise ValueError(f"子文件夹数量 ({len(subdirs)}) 小于最大音源数 ({max(n_sources_list)})")

    split_sizes = {split: int(total_files * ratio) for split, ratio in split_ratio.items()}
    split_sizes["tt"] = total_files - split_sizes["tr"] - split_sizes["cv"]

    for n_sources in n_sources_list:
        output_subdir = os.path.join(output_dir, f"{n_sources}speakers_csv")
        os.makedirs(output_subdir, exist_ok=True)

        all_data = []
        for _ in range(total_files):
            selected_dirs = random.sample(subdirs, n_sources)
            audio_files = []
            for d in selected_dirs:
                wavs = [f for f in os.listdir(os.path.join(input_root, d)) if f.endswith('.wav')]
                if not wavs:
                    raise FileNotFoundError(f"文件夹 {d} 中没有 .wav 文件")
                audio_files.append(os.path.join(d, random.choice(wavs)))

            snrs = []
            if n_sources == 2:
                snr = round(random.uniform(0, 5), 5)
                snrs = [snr, -snr]
            elif n_sources == 3:
                # 为 3 人混合采样三个非零 SNR，避免静音说话者
                snrs = []
                while len(snrs) < 3:
                    v = random.uniform(-5, 5)
                    if abs(v) < 0.5:
                        continue
                    snrs.append(round(v, 5))
            elif n_sources == 4:
                snr1 = round(random.uniform(0, 5), 5)
                snr2 = round(random.uniform(0, 5), 5)
                snrs = [snr1, -snr1, snr2, -snr2]
            elif n_sources == 5:
                # 为 5 人混合采样五个非零 SNR，避免静音说话者
                snrs = []
                while len(snrs) < 5:
                    v = random.uniform(-5, 5)
                    if abs(v) < 0.5:
                        continue
                    snrs.append(round(v, 5))

            row = [item for pair in zip(audio_files, snrs) for item in pair]
            all_data.append(row)

        columns = []
        for i in range(1, n_sources + 1):
            columns.extend([f's{i}', f'snr{i}'])
        df = pd.DataFrame(all_data, columns=columns)

        for split, size in split_sizes.items():
            split_df = df.sample(n=size)
            df = df.drop(split_df.index)
            split_df.to_csv(os.path.join(output_subdir, f'mix_{n_sources}_spk_{split}.csv'), index=False)
            print(f"生成 {n_sources} 源 {split} CSV 文件，包含 {size} 条记录")


def mix_audio_from_csv(config, n_sources_list):
    """根据 CSV 文件生成混合音频，并显示实时处理信息和进度条"""
    input_root = config["input_root"]
    output_dir = config["output_dir"]
    audio_output_dir = config["audio_output_dir"]
    min_length_sec = config["min_length_sec"]
    sample_rate = config["sample_rate"]

    for n_sources in n_sources_list:
        csv_subdir = os.path.join(output_dir, f"{n_sources}speakers_csv")
        audio_subdir = os.path.join(audio_output_dir, f"{n_sources}speakers_wav{sample_rate // 1000}k")

        if not os.path.exists(csv_subdir):
            print(f"警告: {csv_subdir} 不存在，跳过 {n_sources} 源音频生成")
            continue

        for split in ['tr', 'cv', 'tt']:
            csv_file = os.path.join(csv_subdir, f'mix_{n_sources}_spk_{split}.csv')
            if not os.path.exists(csv_file):
                print(f"警告: {csv_file} 不存在，跳过 {split} 数据集")
                continue

            split_dir = os.path.join(audio_subdir, 'min', split)
            os.makedirs(os.path.join(split_dir, 'mix'), exist_ok=True)
            for i in range(1, n_sources + 1):
                os.makedirs(os.path.join(split_dir, f's{i}'), exist_ok=True)

            df = pd.read_csv(csv_file)
            print(f"处理 {n_sources} 源 {split} 数据集，{len(df)} 个文件")

            expected_columns = [f's{i}' for i in range(1, n_sources + 1)] + [f'snr{i}' for i in range(1, n_sources + 1)]
            if not all(col in df.columns for col in expected_columns):
                print(f"错误: CSV 文件 {csv_file} 列名不完整")
                print(f"期望列名: {expected_columns}")
                print(f"实际列名: {list(df.columns)}")
                print(f"前几行数据:\n{df.head()}")
                raise ValueError(f"CSV 文件缺少必要的列，期望 {n_sources} 源数据")

            # JSON 索引列表，用于 Wsj0mixVariable 加载
            mix_json_list = []
            src_json_lists = [[] for _ in range(n_sources)]

            # 使用 tqdm 显示进度条
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"混合 {n_sources} 源 {split}"):
                audio_data = []
                for i in range(1, n_sources + 1):
                    path = os.path.join(input_root, row[f's{i}'])
                    try:
                        fs, data = wavfile.read(path)
                    except Exception as e:
                        print(f"错误: 无法读取 {path} - {e}")
                        continue
                    if fs != sample_rate:
                        data = resample(data, int(len(data) * sample_rate / fs))
                    audio_data.append(data)

                if len(audio_data) != n_sources:
                    print(f"警告: 跳过行 {idx}，音频文件读取不完整")
                    continue

                min_length = int(min_length_sec * sample_rate)
                lengths = [len(d) for d in audio_data]
                target_length = min(min(lengths), min_length) if min(lengths) > min_length else min_length
                audio_data = [d[:target_length] for d in audio_data]

                mixed = np.zeros(target_length, dtype=np.float32)
                for i, data in enumerate(audio_data, 1):
                    snr = float(row[f'snr{i}'])
                    weight = 10 ** (snr / 20)
                    scaled_data = weight * data
                    mixed += scaled_data
                    audio_data[i - 1] = scaled_data

                max_amp = np.max(np.abs([mixed] + audio_data))
                if max_amp > 0:
                    scale = 0.9 / max_amp
                    mixed *= scale
                    audio_data = [d * scale for d in audio_data]

                mix_name_parts = [os.path.splitext(os.path.basename(row[f's{i}']))[0] + f'_{row[f"snr{i}"]}'
                                  for i in range(1, n_sources + 1)]
                mix_name = '_'.join(mix_name_parts)

                try:
                    mix_path = os.path.join(split_dir, 'mix', f'{mix_name}.wav')
                    mix_path_abs = os.path.abspath(mix_path)
                    sf.write(mix_path_abs, mixed, sample_rate)
                    for i, data in enumerate(audio_data, 1):
                        src_path = os.path.join(split_dir, f's{i}', f'{mix_name}.wav')
                        src_path_abs = os.path.abspath(src_path)
                        sf.write(src_path_abs, data, sample_rate)
                        src_json_lists[i - 1].append([src_path_abs, target_length])
                    mix_json_list.append([mix_path_abs, target_length])
                    # 打印实时处理信息
                    print(f"完成混合: {mix_name}.wav")
                except Exception as e:
                    print(f"错误: 无法保存音频 {mix_name} - {e}")

            # 写出 Wsj0mixVariable 期望的 JSON 索引文件
            with open(os.path.join(split_dir, 'mix.json'), 'w', encoding='utf-8') as f:
                json.dump(mix_json_list, f, indent=0)
            for i in range(1, n_sources + 1):
                with open(os.path.join(split_dir, f's{i}.json'), 'w', encoding='utf-8') as f:
                    json.dump(src_json_lists[i - 1], f, indent=0)


def main():
    """主函数：加载配置、解析命令行参数并执行任务"""
    args = parse_args()
    config = load_config()

    n_sources_list = args.n_sources if args.n_sources is not None else config["n_sources_list"]

    required_keys = ["input_root", "output_dir", "audio_output_dir", "total_files",
                     "min_length_sec", "sample_rate", "split_ratio"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"配置文件缺少参数: {key}")

    if not n_sources_list or not all(isinstance(n, int) and n > 1 for n in n_sources_list):
        raise ValueError("n_sources_list 必须是非空整数列表，且每个值大于 1")

    print("开始生成 CSV 文件...")
    generate_mixed_csv(config, n_sources_list)

    print("\n开始生成混合音频...")
    mix_audio_from_csv(config, n_sources_list)

    print("\n所有任务完成！")


if __name__ == "__main__":
    main()