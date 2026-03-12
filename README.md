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
- Powers-of-two scale quantization
- Learnable scale rounding values
- Joint optimization of weight and scale rounding value
- Hardware-friendly quantization for efficient inference

---

## Method

PTS-Quant extends reconstruction-based PTQ methods such as **AdaRound** and **PD-Quant**.

The quantization scale is constrained as:

`scale = 2^k`

where `k` is an integer learned through a **scale rounding optimization process**.

During reconstruction, PTS-Quant simultaneously optimizes:

1. **Weight rounding values**
2. **Scale rounding values**

This reduces quantization error while preserving the power-of-two constraint.

---

## Quantization Pipeline

```
Full Precision Model
        ↓
Calibration Dataset
        ↓
Scale and Weight Rounding Optimization
        ↓
Fix Scale to Powers-of-two
        ↓
Fine-tune Weight Rounding Value
        ↓
Power-of-Two Quantized Model
```

---

## Repository Structure

```
PTS-Quant
│
├── quant/
│ ├── ptq.py
│ ├── quant_layer.py
│ ├── quant_block.py
│ ├── quant_model.py
│ ├── fold_bn.py
│ ├── data_utils.py
│ ├── layer_recon.py
│ ├── set_act_quantize_params.py
│ ├── set_weight_quantize_params.py
│ └── block_recon.py
│
├── models/
│ ├── Resnet.py
│ ├── regnet.py
│ └── MobileNetV2.py
│
├── data/
│ └── ImageNet-1k/
│
├── configs/
│ └── quant.yaml
│
├── utils/
│
└── README.md
```
