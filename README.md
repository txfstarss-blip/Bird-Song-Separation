# MDSEDPRNN
A multi-source separation model optimized for **bird song separation** scenarios, improved based on the DPRNN (Dual-Path RNN) architecture. It supports dynamic identification of the number of overlapping bird songs and adaptive selection of corresponding decoders, capable of handling separation tasks with multiple overlapping bird songs and adapting to complex avian soundscape separation requirements.

## Project Introduction
This model is specially designed for bird song separation tasks, with targeted improvements on the classic MultiDecoderDPRNN. It corely solves the problems of source number identification and accurate separation in multi-bird song overlapping scenarios:
- Introduces the **SENet (Squeeze-and-Excitation)** channel attention mechanism to recalibrate the channels of bird song time-frequency features and enhance the extraction of effective bird song features
- Designs an **EnhancedSelector** module, which captures the temporal context features of bird songs through parallel convolutional branches (standard convolution + dilated convolution) to accurately predict the number of overlapping bird songs
- Supports **dynamic decoder selection**, which can automatically match the corresponding number of decoders according to the predicted number of bird songs, breaking the limitation of fixed source numbers and adapting to multi-bird song overlapping scenarios
- Developed based on the Asteroid speech/audio separation framework, adapted to the feature characteristics of bird song audio, with good scalability and generalization

## Environmental Dependencies
```bash
# Basic core dependencies
pip install torch>=1.9.0 numpy>=1.21.0

# Core audio separation framework
pip install asteroid==0.5.4

# Audio processing tools (required for reading/writing and preprocessing bird song audio)
pip install soundfile librosa
```

## Model Architecture
### Core Modules
1. **SENet**: Channel attention mechanism that recalibrates each channel of bird song time-frequency features, suppresses background noise, and enhances bird song features
2. **EnhancedSelector**: Improved source number selector that captures the temporal context of bird songs through dual-branch convolution to accurately predict the number of overlapping bird songs
3. **DPRNN_MultiStage**: Multi-stage dual-path RNN, designed for the time-frequency features of bird songs, efficiently extracting spatio-temporal correlation features of multi-source bird songs
4. **Decoder_Select**: Dynamic decoder selection module that automatically matches the corresponding number of decoders to complete separation based on the prediction results of the EnhancedSelector
5. **SingleDecoder**: Single-source number decoder, designed for a specific number of bird songs to complete bird song source separation under that number

## Usage Instructions

### 1. Train the Model (train.py)
Train the model based on a custom bird song dataset, supporting configuration of training parameters, dataset paths, model hyperparameters, etc., to adapt to different bird song separation scenarios.
```bash
# Basic training command
python train.py \
  --config configs/birdsep_dprnn.yaml \
  --exp_dir exp/birdsep_multidecoder \
  --sample_rate 16000
```
**Key Parameter Explanations**:
- `--config`: Path to the model training configuration file (contains all parameters such as model, optimizer, dataset, training strategy)
- `--exp_dir`: Experimental result saving directory, which automatically stores model weights, training logs, configuration files, validation results, etc.
- `--sample_rate`: Sampling rate of bird song audio (must be consistent with the dataset, 16000/8000Hz is recommended)

### 2. Bird Song Separation (separate.py)
Use the trained pre-trained model to separate single/batch mixed bird song audio and output audio files of each independent bird song source.
```bash
# Batch mixed bird song separation command
python separate.py \
  --model_path exp/birdsep_multidecoder/checkpoints/best_model.ckpt \
  --input_dir data/mixed_bird_audio \
  --output_dir data/separated_bird_audio \
  --sample_rate 16000
```
**Key Parameter Explanations**:
- `--model_path`: Path to the trained model weight file (ckpt format)
- `--input_dir`: Directory of mixed bird song audio files (only .wav format is supported, must be consistent with the training sampling rate)
- `--output_dir`: Directory for saving separated independent bird song audio, automatically tracked/named by bird song source
- `--sample_rate`: Sampling rate of the input mixed bird song audio, must be consistent with that during model training

### 3. Model Evaluation (eval.py)
Evaluate the model performance on the annotated bird song separation test set, supporting multiple classic audio separation metrics to quantify the model's separation effect.
```bash
# Model performance evaluation command
python eval.py \
  --model_path exp/birdsep_multidecoder/checkpoints/best_model.ckpt \
  --test_dir data/birdsep_test_set \
  --metrics sisdr stoi sdr \
  --sample_rate 16000
```
**Key Parameter Explanations**:
- `--test_dir`: Directory of the bird song separation test set (must contain mixed bird song audio and corresponding annotated clean independent bird song audio)
- `--metrics`: Evaluation metrics, supporting common audio separation metrics such as SISDR, SDR, STOI, multiple metrics are separated by spaces
- `--sample_rate`: Sampling rate of the test set audio, must be consistent with that during model training

### 4. Load Pre-trained Model for Inference
Directly load the trained best model to perform separation inference on single/batch bird song audio, adapted to the integration of custom business code.
```python
import json
import torch
from model import load_best_model  # Import the loading function from the model file

# Load training configuration file
with open("exp/birdsep_multidecoder/config.json", "r") as f:
    train_conf = json.load(f)

# Load the trained best model
model = load_best_model(
    train_conf=train_conf,
    exp_dir="exp/birdsep_multidecoder",
    sample_rate=16000
)

# Model inference (eval mode turns off gradient calculation to improve speed)
model.eval()
with torch.no_grad():
    # Input shape: [batch_size, audio_time_length], example is 1 piece of 3-second 16k sampled mixed bird song
    mixed_bird_audio = torch.rand(1, 48000)
    separated_audio, selector_output = model(mixed_bird_audio)
    print(f"Shape of separated bird song audio: {separated_audio.shape}")  # [batch, number of sources, time length]
    print(f"Predicted number of overlapping bird songs: {selector_output.argmax(-1).item()}")
```

## Code Structure
The project adopts a clear modular structure, separating the core model from business scripts for easy maintenance, modification and expansion, adapting to different bird song separation datasets and scenario requirements.
```
├── configs/                  # Configuration file directory
│   └── birdsep_dprnn.yaml    # Bird song separation model training configuration file (contains all hyperparameters, dataset configurations)
├── model.py                  # Core model code (all modules such as SENet, EnhancedSelector, DPRNN)
├── train.py                  # Model training script (end-to-end training, supporting resume training, validation, log recording)
├── separate.py               # Bird song separation inference script (supports single/batch audio, outputs separated audio files)
├── eval.py                   # Model performance evaluation script (supports multi-metric quantification, outputs evaluation report)
└── README.md                 # Project description and usage documentation
```

## Summary
1. **Core Positioning**: A multi-source audio separation model specially designed for **bird song separation**, solving the problems of source number identification and accurate separation in multi-bird song overlapping scenarios, supporting multi-bird song separation not limited to 4;
2. **Core Advantages**: Dynamically identify the number of overlapping bird songs, adaptively select decoders, integrate attention mechanism to enhance bird song feature extraction, and adapt to complex avian soundscapes;
3. **Running Scripts**:
   - `train.py`: Train the model based on a custom bird song dataset, need to be used with a yaml configuration file;
   - `separate.py`: Use the pre-trained model to separate mixed bird song audio and output independent bird song sources;
   - `eval.py`: Quantitatively evaluate the model's separation performance on the annotated test set and generate a multi-metric evaluation report;
4. **Environmental Requirements**: The core dependencies are PyTorch and the Asteroid audio separation framework, combined with SoundFile and Librosa for audio processing. It is recommended to run with Python3.8+ version.