# TGMR-Det

**Visible-Thermal Counter-UAV Detection Under Natural Misalignment: A Benchmark Dataset and Target-Guided Multi-Scale Registration Framework**

## Overview

This repository provides the implementation of **TGMR-Det**, a target-guided multi-scale registration framework for visible-thermal UAV detection under natural cross-modal misalignment.

The framework uses a dual-stream architecture to extract visible and thermal features and integrates cross-modal target search, geometric feature registration, and feature correction and fusion to improve detection robustness.

## Method

TGMR-Det consists of three main modules:

- **ITSM (Infrared-Guided Target Search Module):** Establishes cross-modal target associations using infrared-guided attention.
- **MSFR-Net (Multi-Scale Feature Registration Network):** Performs multi-scale feature alignment through global geometric transformation and local residual flow.
- **SSFCFM (Sparse-Saliency Feature Correction and Fusion Module):** Enhances multimodal feature representations through adaptive correction, fusion, and structured sparse self-attention (S³A).

The detection framework is built upon dual YOLO11 backbones with a multi-scale anchor-free detection head.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Prepare the visible-thermal dataset and a corresponding experiment configuration file before running the following commands.

### Training

```bash
python train.py --config configs/aeromis_vt.yaml --device cuda
```

### Evaluation

```bash
python val.py --config configs/aeromis_vt.yaml --weights path/to/best.pt --device cuda
```

### Inference

```bash
python detect.py --config configs/aeromis_vt.yaml --weights path/to/best.pt --visible path/to/visible.jpg --thermal path/to/thermal.jpg --output prediction.png --device cuda
```

**Note:** The experiment configuration file must be supplied separately and adapted to the corresponding environment and dataset.

## Dataset

The **AeroMis-VT** dataset contains naturally misaligned visible-thermal UAV image pairs collected using a ground-based dual-sensor imaging system.

Due to restrictions associated with an ongoing research project, the original AeroMis-VT dataset is not currently included in this repository.

## Citation

If you find this work useful, please consider citing our paper:

*Visible-Thermal Counter-UAV Detection Under Natural Misalignment: A Benchmark Dataset and Target-Guided Multi-Scale Registration Framework.*
