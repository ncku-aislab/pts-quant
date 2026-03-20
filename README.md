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

First clone the repository with `git clone`

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
    save_name: ResNet18-i4sc
    wq_params: {'n_bits': 4, symmetric: True, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params: {'n_bits': 4, symmetric: True, 'channel_wise': False, 'scale_method': 'mse',
                    'leaf_param': True, 'prob': 0.5}
    recon: True

```
Then run:
`python quant/ptq.py`

## Experimental Results

We evaluate PTS-Quant on ImageNet across multiple architectures under strict power-of-two (PoT) scale constraints. The proposed method is compared with both QAT-based approaches (e.g., TQT, HMQ) and reconstruction-based PTQ methods (e.g., PD-Quant).

Results show that PTS-Quant consistently improves accuracy over PD-Quant under PoT constraints, while narrowing the performance gap between PTQ and QAT methods.

### 4bit Quantization(W4A4)

#### ResNet-18

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| HAWQv3 | QAT | 4/4 | 68.45 | -3.02 |
| PD-Quant (PoT) | PTQ | 4/4 | 68.70 | -2.31 |
| **PTS-Quant** | PTQ | 4/4 | **68.77** | **-2.24** |

#### ResNet-50

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| TQT | QAT | 4/8 | 74.40 | -0.80 |
| HMQ | QAT | 3.55/8 | 76.30 | +0.15 |
| HAWQv3 | QAT | 4/4 | 74.24 | -3.48 |
| PD-Quant (PoT) | PTQ | 4/4 | 74.03 | -2.60 |
| **PTS-Quant** | PTQ | 4/4 | **74.59** | **-2.04** |

#### MobileNetV2

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| HMQ | QAT | 4.16/8 | 71.40 | -0.48 |
| PD-Quant (PoT) | PTQ | 4/4 | 65.57 | -7.05 |
| **PTS-Quant** | PTQ | 4/4 | **65.63** | **-6.99** |

#### RegNetX-600MF

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| PD-Quant (PoT) | PTQ | 4/4 | 69.33 | -4.19 |
| **PTS-Quant** | PTQ | 4/4 | **69.68** | **-3.84** |

#### RegNetX-3.2GF

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| PD-Quant (PoT) | PTQ | 4/4 | 75.24 | -3.22 |
| **PTS-Quant** | PTQ | 4/4 | **76.12** | **-2.34** |

### 2bit Quantization(W2A2)

#### ResNet-18

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| PD-Quant (PoT) | PTQ | 2/2 | 45.70 | -25.31 |
| **PTS-Quant** | PTQ | 2/2 | **51.24** | **-19.77** |

#### ResNet-50

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| HMQ | QAT | 2.04/8 | 75.00 | +0.15 |
| PD-Quant (PoT) | PTQ | 2/2 | 51.91 | -24.72 |
| **PTS-Quant** | PTQ | 2/2 | **53.20** | **-23.43** |

#### MobileNetV2

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| HMQ | QAT | 2.22/8 | 65.70 | -6.18 |
| PD-Quant (PoT) | PTQ | 2/2 | 3.43 | -69.19 |
| **PTS-Quant** | PTQ | 2/2 | **5.25** | **-67.37** |

#### RegNetX-600MF

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| PD-Quant (PoT) | PTQ | 2/2 | 27.70 | -45.82 |
| **PTS-Quant** | PTQ | 2/2 | **33.76** | **-39.76** |

#### RegNetX-3.2GF

| Method | Scheme | W/A | Top-1 | Δ |
|--------|--------|-----|------|----|
| PD-Quant (PoT) | PTQ | 2/2 | 42.00 | -36.46 |
| **PTS-Quant** | PTQ | 2/2 | **49.17** | **-29.29** |

Δ denotes the accuracy drop compared to the corresponding full-precision model.

**Key Observations**

- PTS-Quant consistently outperforms PD-Quant under power-of-two constraints across all evaluated models.
- The performance gap between PTQ and QAT is significantly reduced.
- The improvement is especially notable in lightweight models such as RegNet.