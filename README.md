# Diffusion Models for Denoising and Reverse Diffusion

This repository provides an implementation and resources related to diffusion models, specifically focusing on image denoising and reverse diffusion processes. Diffusion models have become prominent for their ability to generate high-quality images by progressively reversing a noisy image back to its clean state through learned iterative denoising.

## Overview

Diffusion models operate by defining a forward process (gradually adding noise to data) and learning a reverse process (removing noise) to reconstruct the original data. These models have shown impressive performance in various image restoration tasks, including denoising, deblurring, and image enhancement, especially under challenging conditions like low-light, foggy, or hazy environments.

## Key Features

- **Denoising Diffusion Probabilistic Models (DDPM)**: Implementation of standard diffusion-based denoising models.
- **Reverse Diffusion Process**: Algorithms for reconstructing clean images from noisy inputs.
- **Pretrained Models**: Ready-to-use pretrained weights for common benchmarks.
- **Customizable Training and Inference Pipelines**: Easy-to-use scripts for training and evaluating custom datasets.

## Repository Structure

```
.
├── datasets/
│   └── README.md
├── models/
│   └── diffusion_model.py
├── scripts/
│   ├── train.py
│   └── infer.py
├── notebooks/
│   └── diffusion_demo.ipynb
├── pretrained/
│   └── README.md
├── utils/
│   └── utils.py
└── requirements.txt
```

## Getting Started

### Installation

Clone the repository:
```bash
git clone https://github.com/islamfadl/diffusion-models.git
cd diffusion-models
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Datasets

This repository is designed to work with popular datasets:
- Kitti
- Kitti(MonoWAD) 
- Custom datasets

For instructions on setting up and using datasets, see [datasets/README.md](datasets/README.md).

## Usage

### Training

To train the diffusion model:
```bash
python scripts/train.py --dataset #dataset_name --epochs #num --batch-size #num```

### Inference

To denoise images using a trained model:
```bash
python scripts/infer.py --model-path pretrained/model.pth --input noisy_image.png --output denoised_image.png
```

## Notebooks

Check [notebooks/diffusion_demo.ipynb](notebooks/diffusion_demo.ipynb) for an interactive demonstration and visual exploration of diffusion models.

## References

Key foundational papers:
- Ho, J., Jain, A., & Abbeel, P. (2020). ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)
- Song, J., Meng, C., & Ermon, S. (2020). ["Denoising Diffusion Implicit Models"](https://arxiv.org/abs/2010.02502)
