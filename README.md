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

within a reconstruction-based PTQ framework.

---

## Key Features

- Post-training quantization (PTQ)
- Power-of-two scale quantization
- Learnable scale rounding values
- Joint optimization of weight and scale rounding
- Hardware-friendly quantization for efficient inference

---

## Method

PTS-Quant extends reconstruction-based PTQ methods such as **AdaRound** and **PD-Quant**.

The quantization scale is constrained as:
