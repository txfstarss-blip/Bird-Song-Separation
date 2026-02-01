import os
import random
import csv
import logging
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


class AudioDatasetProcessor:
    def __init__(self, input_path=None, output_path=None, csv_path=None, log_file=None):
        """初始化数据集处理器"""
        self.input_path = input_path
        self.output_path = output_path
        self.csv_path = csv_path
        self.target_sr = 8000  # 目标采样率
        self.target_duration = 4  # 目标时长（秒）
        self.max_files_per_dir = 20  # 每个目录保留的文件数
        self.num_mixtures = 10000  # 生成的混合音频对数量

        # 设置日志
        self.logger = logging.getLogger('AudioDatasetProcessor')
        self.logger.setLevel(logging.INFO)
        # 避免重复添加处理器
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # 创建输出目录
        try:
            os.makedirs(self.output_path, exist_ok=True)
            self.logger.info(f"输出目录已创建或已存在: {self.output_path}")
        except Exception as e:
            self.logger.error(f"创建输出目录失败: {str(e)}")
            raise

    def process_audio_file(self, input_file, output_file):
        """处理单个音频文件：转换格式、裁剪、检查长度"""
        try:
            # 读取音频
            audio, sr = librosa.load(input_file, sr=None, mono=True)
            self.logger.debug(f"读取音频文件: {input_file}, 原始采样率: {sr}")

            # 转换为目标采样率
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

            # 计算目标样本数
            target_samples = int(self.target_duration * self.target_sr)

            # 检查音频长度
            if len(audio) < target_samples:
                self.logger.warning(f"音频 {input_file} 长度不足 {self.target_duration} 秒，已跳过")
                return False

            # 裁剪到 4 秒
            audio = audio[:target_samples]

            # 保存到 WAV 文件
            sf.write(output_file, audio, self.target_sr, subtype='PCM_16')
            self.logger.info(f"成功处理音频: {input_file} -> {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"处理音频 {input_file} 时出错: {str(e)}")
            return False

    def process_dataset(self):
        """处理数据集：转换、裁剪并保留指定数量的音频"""
        self.logger.info("开始处理数据集...")
        try:
            for dir_name in os.listdir(self.input_path):
                dir_path = os.path.join(self.input_path, dir_name)
                if not os.path.isdir(dir_path):
                    self.logger.debug(f"跳过非目录项: {dir_path}")
                    continue

                # 获取二级目录
                subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
                if not subdirs:
                    self.logger.warning(f"目录 {dir_name} 下没有子目录")
                    continue

                # 随机选择一个二级目录
                selected_subdir = random.choice(subdirs)
                subdir_path = os.path.join(dir_path, selected_subdir)
                self.logger.info(f"处理目录: {dir_name}, 选择子目录: {selected_subdir}")

                # 获取所有音频文件
                audio_files = [f for f in os.listdir(subdir_path) if f.endswith(('.wav', '.mp3', '.flac'))]
                if not audio_files:
                    self.logger.warning(f"目录 {subdir_path} 下没有音频文件")
                    continue

                # 创建对应的输出目录
                output_dir = os.path.join(self.output_path, dir_name)
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    self.logger.debug(f"创建输出子目录: {output_dir}")
                except Exception as e:
                    self.logger.error(f"创建输出子目录 {output_dir} 失败: {str(e)}")
                    continue

                # 随机选择最多 max_files_per_dir 个文件
                selected_files = random.sample(audio_files, min(len(audio_files), self.max_files_per_dir))

                # 处理每个文件
                valid_files = 0
                for audio_file in selected_files:
                    input_file = os.path.join(subdir_path, audio_file)
                    output_file = os.path.join(output_dir, f"{valid_files:03d}_{os.path.splitext(audio_file)[0]}.wav")

                    if self.process_audio_file(input_file, output_file):
                        valid_files += 1
                        if valid_files >= self.max_files_per_dir:
                            break

                self.logger.info(f"处理完成: {dir_name}, 保留 {valid_files} 个音频文件")
            self.logger.info("数据集预处理完成！")

        except Exception as e:
            self.logger.error(f"处理数据集时发生错误: {str(e)}")
            raise

    def generate_mixing_csv(self):
        """生成混合音频的 CSV 文件"""
        self.logger.info("开始生成混合 CSV 文件...")
        try:
            # 获取所有一级目录
            dirs = [d for d in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, d))]
            if len(dirs) < 2:
                self.logger.error("目录数量不足 2，无法生成混合对")
                return

            # 获取所有音频文件路径
            audio_files = {}
            for dir_name in dirs:
                dir_path = os.path.join(self.output_path, dir_name)
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]
                if files:
                    audio_files[dir_name] = files
                else:
                    self.logger.warning(f"目录 {dir_name} 下没有 WAV 文件")

            if len(audio_files) < 2:
                self.logger.error("可用音频目录不足 2，无法生成混合对")
                return

            # 生成 CSV
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['audio1_path', 'snr1', 'audio2_path', 'snr2'])

                for _ in range(self.num_mixtures):
                    # 随机选择两个不同目录
                    dir1, dir2 = random.sample(list(audio_files.keys()), 2)

                    # 从每个目录中随机选择一个音频文件
                    audio1 = random.choice(audio_files[dir1])
                    audio2 = random.choice(audio_files[dir2])

                    # 生成随机信噪比 (-5 到 5)
                    snr = round(random.uniform(-5, 5), 5)

                    # 写入 CSV，转换为相对路径
                    # rel_audio1 = audio1.replace(self.output_path, ".\\dataset")
                    # rel_audio2 = audio2.replace(self.output_path, ".\\dataset")
                    writer.writerow([audio1, -snr, audio2, snr])
                    self.logger.debug(f"生成混合对: {audio1} ({-snr}), {audio2} ({snr})")

            self.logger.info(f"CSV 文件已生成: {self.csv_path}")

        except Exception as e:
            self.logger.error(f"生成 CSV 文件时发生错误: {str(e)}")
            raise

    def run(self):
        """运行完整的处理流程"""
        self.logger.info("开始完整处理流程...")
        try:
            self.process_dataset()
            self.generate_mixing_csv()
            self.logger.info("所有任务完成！")
        except Exception as e:
            self.logger.error(f"运行完整流程时发生错误: {str(e)}")
            raise


if __name__ == "__main__":
    # 创建处理器实例
    processor = AudioDatasetProcessor(
        input_path=r"D:\train-LibriSpeech",
        output_path=r"D:\train_Libri_tr_cv_tt\train-LibriSpeech_cutstr",
        csv_path=r"D:\train_Libri_tr_cv_tt\train-LibriSpeech_mixing.csv",
        log_file=r"audio_processing.log"
    )

    # 运行完整流程


    # processor.run()
    processor.generate_mixing_csv()