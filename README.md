# MiniGPT4-V: Multimodal Vision-Language Reasoning with Video and Physiological Signals

MiniGPT4-V extends MiniGPT-4 to support video-based reasoning tasks using Vision-Language Models (VLMs) such as LLaMA2 and Mistral. This project integrates both spatial-temporal video features and auxiliary physiological signals (e.g., heart rate, breathing rate) for tasks like emotion recognition,  and engagement analysis.

## Key Features

- Vision-language inference with LLaMA2 and Mistral backbones
- Integration of Toeplitz-structured physiological signals (HR/BR)
- Multi-stage training and zero-shot evaluation pipelines
- Support for multi-node distributed training

## Directory Structure

```
Minigpt4-V-main/
├── minigpt4/                    # Core models and processing modules
│   ├── models/                  # BLIP2, Q-Former, LLaMA2, Mistral
│   ├── tasks/                   # VQA, image-text pretraining
│   ├── datasets/                # Dataset handling and loaders
│   ├── processors/              # Tokenization, augmentation
├── train_configs/              # Training config files (YAML)
├── test_configs/               # Testing config files
├── train.py                    # Main training script
├── train_multinode.py          # Multi-node training script
├── eval_video.py               # Evaluation script for video-VQA
├── fine-tuning.md              # Fine-tuning instructions
├── accuracy-evaluation/        # LangChain/Logic-based evaluation
├── quantitative-evalauiton/    # Benchmark evaluation scripts
├── br-hr-batch-extract.py      # Toeplitz image generation from HR/BR signals
├── toeplitz-image-batch.py     # Processing pipeline for signal inputs
```

## Installation

Create a conda environment and install dependencies:

```
conda env create -f environment.yml
conda activate minigpt4-v
```

## Usage

### 1. Inference with pretrained VLM
```
python minigpt4_video_inference.py --cfg-path test_configs/mistral_test_config.yaml
```

### 2. Fine-tuning on custom video+signal dataset
```
python train.py --cfg-path train_configs/224_v2_mistral_video_stage_2.yaml
```

### 3. Evaluate video-based reasoning
```
python eval_video.py --cfg-path test_configs/mistral_engegenet_test_config.yaml
```

## Supported Datasets

- EngageNet
- DAiSEE
- SED (Student Engagement Dataset)
- UBFC-PHYSics

## Evaluation Options

- Zero-shot and few-shot evaluations (`quantitative-evalauiton/`)
- Logic-based LangChain reasoning evaluation (`accuracy-evaluation/`)

## License

- General codebase: See `LICENSE.md`
- Lavis-based modules: See `LICENSE_Lavis.md`

## Acknowledgements

This repository builds on the foundation of [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), extended to handle video and physiological signal fusion for advanced multimodal tasks.
