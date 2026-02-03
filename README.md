# MultiDecoderDPRNN with EnhancedSelector
一款针对**鸟鸣分离**场景优化的多源分离模型，基于DPRNN（Dual-Path RNN）架构改进，支持动态识别重合鸟鸣的数量并自适应选择对应解码器，可处理多个鸟鸣重合的分离任务（不局限于4个），适配复杂的鸟类声景分离需求。

## 项目简介
该模型专为鸟鸣分离任务设计，在经典MultiDecoderDPRNN基础上进行针对性改进，核心解决多鸟鸣重合场景下的源数量识别与精准分离问题：
- 引入**SENet (Squeeze-and-Excitation)** 通道注意力机制，对鸟鸣时频特征进行通道重校准，强化有效鸟鸣特征提取
- 设计**EnhancedSelector**增强选择器，通过并行卷积分支（标准卷积+空洞卷积）捕捉鸟鸣的时间上下文特征，精准预测重合鸟鸣数量
- 支持**动态解码器选择**，可根据预测的鸟鸣数量自动匹配对应解码器，打破固定源数量的限制，适配不局限于4个的多鸟鸣重合场景
- 基于Asteroid语音/音频分离框架开发，适配鸟鸣音频的特征特性，具备良好的可扩展性和泛化性

## 核心特性
- 🐦 专为鸟鸣分离优化，适配鸟类声景的音频特征与分离需求
- 🎯 动态识别重合鸟鸣数量，自适应选择解码器，支持多源鸟鸣分离（不局限于4个）
- 🧠 集成注意力机制与多分支卷积，提升复杂声景下的特征提取与分离精度
- 🔧 基于Asteroid框架开发，兼容其生态系统，支持灵活的参数配置与功能扩展
- 📋 提供完整的训练、分离、评估端到端流程，可直接适配自定义鸟鸣数据集

## 环境依赖
```bash
# 基础核心依赖
pip install torch>=1.9.0 numpy>=1.21.0

# 音频分离核心框架
pip install asteroid==0.5.4

# 音频处理工具（必备，用于鸟鸣音频读写、预处理）
pip install soundfile librosa
```

## 模型架构
### 核心模块
1. **SENet**: 通道注意力机制，对鸟鸣时频特征的各通道进行重校准，抑制背景噪声，强化鸟鸣特征
2. **EnhancedSelector**: 改进型源数量选择器，通过双分支卷积捕捉鸟鸣的时间上下文，精准预测重合鸟鸣的数量
3. **DPRNN_MultiStage**: 多阶段双路径RNN，针对鸟鸣的时频特征设计，高效提取多源鸟鸣的时空关联特征
4. **Decoder_Select**: 动态解码器选择模块，根据EnhancedSelector的预测结果，自动匹配对应数量的解码器完成分离
5. **SingleDecoder**: 单源数量解码器，针对特定鸟鸣数量设计，完成该数量下的鸟鸣源分离

## 使用说明

### 1. 训练模型 (train.py)
基于自定义鸟鸣数据集训练模型，支持配置训练参数、数据集路径、模型超参等，适配不同的鸟鸣分离场景。
```bash
# 基础训练命令
python train.py \
  --config configs/birdsep_dprnn.yaml \
  --exp_dir exp/birdsep_multidecoder \
  --sample_rate 16000
```
**关键参数说明**：
- `--config`: 模型训练配置文件路径（包含模型、优化器、数据集、训练策略等所有参数）
- `--exp_dir`: 实验结果保存目录，自动存储模型权重、训练日志、配置文件、验证结果等
- `--sample_rate`: 鸟鸣音频采样率（需与数据集一致，建议16000/8000Hz）

### 2. 鸟鸣分离 (separate.py)
使用训练好的预训练模型，对单/批量混合鸟鸣音频进行分离，输出各独立鸟鸣源音频文件。
```bash
# 批量混合鸟鸣分离命令
python separate.py \
  --model_path exp/birdsep_multidecoder/checkpoints/best_model.ckpt \
  --input_dir data/mixed_bird_audio \
  --output_dir data/separated_bird_audio \
  --sample_rate 16000
```
**关键参数说明**：
- `--model_path`: 训练好的模型权重文件路径（ckpt格式）
- `--input_dir`: 混合鸟鸣音频文件目录（仅支持.wav格式，需与训练采样率一致）
- `--output_dir`: 分离后的独立鸟鸣音频保存目录，按鸟鸣源自动分轨/命名
- `--sample_rate`: 输入混合鸟鸣音频的采样率，需与模型训练时一致

### 3. 模型评估 (eval.py)
在标注好的鸟鸣分离测试集上评估模型性能，支持多种音频分离经典指标，量化模型的分离效果。
```bash
# 模型性能评估命令
python eval.py \
  --model_path exp/birdsep_multidecoder/checkpoints/best_model.ckpt \
  --test_dir data/birdsep_test_set \
  --metrics sisdr stoi sdr \
  --sample_rate 16000
```
**关键参数说明**：
- `--test_dir`: 鸟鸣分离测试集目录（需包含混合鸟鸣音频和对应的干净独立鸟鸣音频标注）
- `--metrics`: 评估指标，支持SISDR、SDR、STOI等音频分离常用指标，多指标用空格分隔
- `--sample_rate`: 测试集音频采样率，需与模型训练时一致

### 4. 加载预训练模型推理
直接加载训练好的最佳模型，进行单条/批量鸟鸣音频的分离推理，适配自定义的业务代码集成。
```python
import json
import torch
from model import load_best_model  # 导入模型文件中的加载函数

# 加载训练配置文件
with open("exp/birdsep_multidecoder/config.json", "r") as f:
    train_conf = json.load(f)

# 加载训练好的最佳模型
model = load_best_model(
    train_conf=train_conf,
    exp_dir="exp/birdsep_multidecoder",
    sample_rate=16000
)

# 模型推理（eval模式关闭梯度计算，提升速度）
model.eval()
with torch.no_grad():
    # 输入形状：[batch_size, audio_time_length]，示例为1条3秒16k采样的混合鸟鸣
    mixed_bird_audio = torch.rand(1, 48000)
    separated_audio, selector_output = model(mixed_bird_audio)
    print(f"分离后鸟鸣音频形状: {separated_audio.shape}")  # [batch, 源数量, 时间长度]
    print(f"预测重合鸟鸣数量: {selector_output.argmax(-1).item()}")
```

## 代码结构
项目采用清晰的模块化结构，核心模型与业务脚本分离，便于维护、修改和扩展，适配不同的鸟鸣分离数据集与场景需求。
```
├── configs/                  # 配置文件目录
│   └── birdsep_dprnn.yaml    # 鸟鸣分离模型训练配置文件（含所有超参、数据集配置）
├── model.py                  # 核心模型代码（SENet、EnhancedSelector、DPRNN等所有模块）
├── train.py                  # 模型训练脚本（端到端训练，支持断点续训、验证、日志记录）
├── separate.py               # 鸟鸣分离推理脚本（支持单/批量音频，输出分离后的音频文件）
├── eval.py                   # 模型性能评估脚本（支持多指标量化，输出评估报告）
└── README.md                 # 项目说明与使用文档
```

## 总结
1. **核心定位**：专为**鸟鸣分离**设计的多源音频分离模型，解决多鸟鸣重合场景下的源数量识别与精准分离问题，支持不局限于4个的多鸟鸣分离；
2. **核心优势**：动态识别重合鸟鸣数量、自适应选择解码器、集成注意力机制强化鸟鸣特征提取，适配复杂的鸟类声景；
3. **运行脚本**：
   - `train.py`: 基于自定义鸟鸣数据集训练模型，需搭配yaml配置文件；
   - `separate.py`: 使用预训练模型对混合鸟鸣音频进行分离，输出独立鸟鸣源；
   - `eval.py`: 在标注测试集上量化评估模型分离性能，生成多指标评估报告；
4. **环境要求**：核心依赖为PyTorch和Asteroid音频分离框架，搭配SoundFile、Librosa完成音频处理，建议使用Python3.8+版本运行。