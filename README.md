# Ultrasound Image Enhancement Thesis

## Overview

This repository contains the code, data pointers, and documentation for my M.Tech thesis on real-time ultrasound image enhancement. We propose a novel framework combining CycleGAN's unpaired domain translation with a Denoising Diffusion Probabilistic Model (DDPM)–based U-Net generator and an LPIPS perceptual loss for improved realism.

## Problem Statement

* Speckle noise in B-mode ultrasound images degrades diagnostic quality.
* Existing denoising techniques often blur fine anatomical details or require paired noisy/clean data.

This thesis addresses how to enhance ultrasound images in real time on portable devices by:

1. Learning unpaired mappings between noisy and clean domains (CycleGAN).
2. Iteratively refining generation via a diffusion-based U-Net model.
3. Maintaining anatomical fidelity using perceptual LPIPS.

## Approach

1. **Unpaired Domain Mapping (CycleGAN):** Learn mappings $G: A \to B$ and $F: B \to A$ with cycle-consistency constraints.
2. **DDPM U-Net Generator:** Replace the standard CycleGAN generators with a time-conditional U-Net diffusion model for iterative denoising.
3. **Perceptual Losses:** Combine adversarial, cycle, identity, LPIPS and DDPM noise losses for balanced training.

## Repository Structure

```
├── source/
│   ├── models/         # Model definitions ( CycleGAN based DDPM U-Net, Pix2Pix,CycleGAN Unet based)
│   ├── datasets.py     # Dataset loaders for unpaired images
│   ├── train.py        # Training loop with multi-loss optimization
│   ├── evaluate.py     # Evaluation scripts (PSNR, SSIM)
│   └── utils.py        # Helper functions and visualization
├── notebooks           # Main model ipynb file
├── results/            # Generated images, metrics logs, and plots
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sahith0610/ultrasound_image_enhancement.git
   cd ultrasound_image_enhancement
   ```
2. Create a Python environment (recommended `conda`):

   ```bash
   conda create -n ultrasound_env python=3.11
   conda activate ultrasound_env
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation

```bash
python source/datasets.py --preprocess
```

### 2. Training

```bash
python source/train.py \
```

### 3. Evaluation

```bash
python source/evaluate.py \
```

## Evaluation Metrics

* **PSNR:** Peak Signal-to-Noise Ratio for quantitative fidelity.
* **SSIM / MS-SSIM:** Structural similarity index for perceptual quality.


## Acknowledgements
* **Dr.Deepak Mishra**,Department of Avionics,IIST,Guide
* **Apurva Srivastava**, Project Scientist, for guidance and domain insights.
* **Ansh Fetal Care, Ahmedabad** for providing the ultrasound dataset.

---

*Developed by Sahith Arya Charan Patchigalla, Indian Institute of Space Science and Technology (IIST)*
