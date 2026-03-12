# PTS-Quant

PyTorch implementation of **PTS-Quant**, a post-training quantization (PTQ) method that constrains quantization scales to **powers-of-two** for hardware-efficient neural network inference.

---

## Overview

Quantization is an important technique for deploying deep neural networks on resource-constrained hardware.  
By mapping floating-point weights and activations to low-bit integers, quantization can significantly reduce model size and computation cost.

However, conventional quantization methods typically use arbitrary floating-point scale factors, which still require floating-point multiplications during inference.

**PTS-Quant** addresses this issue by constraining quantization scales to **power-of-two values**, allowing multiplications to be replaced with efficient **bit-shift operations** on hardware.

The proposed method introduces a **learnable scale rounding value** and jointly optimizes:

- **Weight rounding**
- **Scale rounding**

within a rounding-based PTQ framework.

---

## Key Features

- Post-training quantization (PTQ)
- Power-of-two scale quantization
- Learnable scale rounding values
- Joint optimization of weight and scale rounding values
- Hardware-friendly quantization for efficient inference

---

## Method

PTS-Quant extends rounding-based PTQ methods such as **AdaRound** and **PD-Quant**.

The quantization scale is constrained as:

`scale = 2^k`

where `k` is an integer learned through a **scale rounding optimization process**.

During reconstruction, PTS-Quant jointly optimizes:

1. **Weight rounding values**
2. **Scale rounding values**

This reduces quantization error while preserving the power-of-two constraint.

---

## Quantization Pipeline

```
Full-Precision Model
        ↓
Calibration Dataset
        ↓
Scale and Weight Rounding Optimization
        ↓
Fix Scale to Power-of-two
        ↓
Fine-tune Weight Rounding Values
        ↓
Power-of-Two Quantized Model
```

---

## Repository Structure

```
PTS-Quant
│
├── quant/
│   ├── ptq.py
│   ├── quant_layer.py
│   ├── quant_block.py
│   ├── quant_model.py
│   ├── fold_bn.py
│   ├── data_utils.py
│   ├── layer_recon.py
│   ├── set_act_quantize_params.py
│   ├── set_weight_quantize_params.py
│   └── block_recon.py
│
├── models/
│   ├── Resnet.py
│   ├── regnet.py
│   └── MobileNetV2.py
│
├── configs/
│   └── quant.yaml
│
├── utils/
│
└── README.md
```


---

## Installation

### Clone the repository

```
git clone https://github.com/vcvc111222/PTS-Quant.git
cd PTS-Quant/
```


### Create environment

```
conda create -n ptsquant python=3.8
conda activate ptsquant
pip install -r requirements.txt
```


---

## Dataset

### ImageNet

Create a symbolic link to the ImageNet dataset and organize it as follows:
```
data/
└── ImageNet-1k/
    └── ILSVRC/
        ├── Annotations/
        └── Data/
            └── CLS-LOC/
                ├── train/
                ├── val/
                └── test/
```

---

## Running Experiments

Example: quantizing **ResNet-18**
Set the configuration in `config/quant.yaml`:
```
version: 0.1.0

models:
  - model_name: ResNet18
    weight_path: weights_cifar10/ResNet18.pth
    save_path: weights_cifar10/PTS-Quant/ResNet18-i4sc_PTS-Quant.pth
    wq_params: {'n_bits': 4, symmetric: True, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params: {'n_bits': 4, symmetric: True, 'channel_wise': False, 'scale_method': 'mse',
                    'leaf_param': True, 'prob': 0.5}
    recon: True
```
Then run:
`python quant/ptq.py`

