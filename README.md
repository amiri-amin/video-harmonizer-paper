# Video-Harmonizer

## Implementation

A from-scratch, architecture-faithful implementation of **Video-Harmonizer: A Fast-Learning UHD Video Tokenizer for Multimodal LLMs**, following the IEEE Transactions on Multimedia manuscript  
(`20260504_IEEE_Transaction_Video_Harmonizer_...pdf`).

> **Developed by Amin Amiri in 2025.**

Every module maps directly to the paper's equations. The default **tiny** configuration runs the complete architecture on a CPU in seconds, while the **paper** configuration preserves the same architecture at the UHD scale described in the manuscript and intended for training on two NVIDIA A100 GPUs.

---

## Architecture-to-Equation Map

| Stage | Module | Paper |
|---|---|---|
| Patch extraction | `models/windowing.py` · `patchify` | Eqs. 1–2 |
| Per-patch mean and variance | `AdaptiveHannTukeyWindow` | Eqs. 3–4 |
| Adaptive Hann–Tukey 3D windowing | `AdaptiveHannTukeyWindow` | Eqs. 5–11 |
| Weierstrass Gaussian low-pass transform | `transforms.TransformBank.weierstrass` | Eqs. 12–13 |
| Riesz directional high-pass transform ×3 | `transforms.TransformBank.riesz` | Eq. 14 |
| Gabor band-pass transform, 8 orientations | `transforms.TransformBank.gabor` | Eqs. 15–16 |
| Mellin log-radial spatial transform | `transforms.TransformBank.mellin` | Eq. 17 |
| Concatenation into 14·C channels | `TransformBank.forward` | Eqs. 18–19 |
| FluxHead streaming self-attention | `models/fluxhead.py` | Eq. 20 |
| Discrete FusionQuantizer, residual and hierarchical | `models/fusion_quantizer.py` | Eqs. 21–26 |
| Continuous Gumbel-Softmax quantization | `models/gumbel_quantizer.py` | Eq. 27 |
| Separate FluxFormer decoders | `models/fluxformer.py` | Eqs. 28–31 |
| MSE and SSIM multi-objective loss | `models/losses.py` | Eqs. 30–31 |
| Commitment loss, β | `fusion_quantizer.py` | Eq. 32 |
| Sequence-Level Normalized Error, SNE% | `test_harmonizer.py` | Eq. 36 |

---

## Reproduced Paper Settings

The implementation reproduces the principal architectural and training settings described in the paper:

- Gabor transform with **8 orientations**
- Gumbel-Softmax temperature annealed from **1.0 to 0.3**
- Reconstruction-loss weights:
  - **λ_MSE = 0.75**
  - **λ_SSIM = 0.25**
- Shared **FluxHead encoder** for both token streams
- Separate **FluxFormer decoders** for the discrete and continuous streams
- Dual discrete and continuous tokenization paths
- Direct emission of:
  - discrete codebook indices
  - continuous token embeddings
- No intermediate conversion layer between the discrete and continuous representations

---

## Quick Start

### 1. Create a Small Synthetic Dataset

The following command creates eight synthetic clips with a shape of `3 × 8 × 32 × 32`.

```bash
python make_dummy_data.py --out_dir data/dummy --num_clips 8
```

### 2. Run the End-to-End Smoke Test

This verifies the complete forward and backward passes of the architecture.

```bash
python smoke_test.py
```

### 3. Train the Tiny Configuration on CPU

```bash
python train_harmonizer.py \
    --data_dir data/dummy \
    --epochs 150 \
    --batch_size 2 \
    --lr 2e-3
```

### 4. Reconstruct Videos and Evaluate the Model

```bash
python test_harmonizer.py \
    --data_dir data/dummy \
    --checkpoint checkpoints/video_harmonizer_last.pth
```

The evaluation script reports:

- PSNR
- SSIM
- Sequence-Level Normalized Error, SNE%
- Discrete token indices
- Continuous token embeddings

---

## Tiny Configuration

The **tiny** configuration is designed as a functional, end-to-end demonstration of the complete architecture.

It uses:

- 8 synthetic video clips
- approximately 0.45 million parameters
- small spatial and temporal dimensions
- CPU-compatible training
- a few hundred training epochs

The tiny configuration is not intended to reproduce the full quantitative performance reported in the paper. Its purpose is to demonstrate that:

- the full architecture runs end-to-end
- the forward and backward passes are valid
- the reconstruction loss decreases
- the codebook commitment loss converges
- the Gumbel temperature anneals correctly
- the discrete token stream is generated successfully
- the continuous token stream is generated successfully
- both token streams can reconstruct the input video

Because the demonstration dataset is intentionally small, the resulting absolute PSNR and SSIM values are expected to remain modest.

---

## Scaling to the Paper Configuration

Use the `--config paper` argument with the training or evaluation scripts to activate the UHD-scale configuration.

### Training

```bash
python train_harmonizer.py \
    --config paper \
    --data_dir data/uhd \
    --epochs 150 \
    --batch_size 2
```

### Evaluation

```bash
python test_harmonizer.py \
    --config paper \
    --data_dir data/uhd \
    --checkpoint checkpoints/video_harmonizer_last.pth
```

The paper configuration includes:

- larger embedding dimensions
- deeper FluxHead attention blocks
- deeper FluxFormer decoder blocks
- 8,192-entry codebooks
- tubelet or temporal chunk size of 4
- mixed-precision training
- tiled spatial processing for UHD frames

The paper-scale model is intended for 4K and 8K video clips and was described as being trained from scratch using:

- **2 × NVIDIA A100 80 GB GPUs**
- mixed-precision computation
- approximately **672 GPU-hours**

Full-resolution 4K and 8K frames are processed using spatial tiles based on the configured `image_size`.

---

## Project Structure

```text
config.py
    tiny_config()
    paper_config()

models/
    windowing.py
        Patch extraction
        Adaptive Hann–Tukey windowing

    transforms.py
        Weierstrass transform
        Riesz transform
        Gabor transform
        Mellin transform

    fluxhead.py
        Shared streaming-attention encoder

    fusion_quantizer.py
        Discrete residual codebooks
        Hierarchical codebook quantization

    gumbel_quantizer.py
        Continuous Gumbel-Softmax quantization

    fluxformer.py
        Per-stream reconstruction decoders

    losses.py
        MSE reconstruction loss
        SSIM reconstruction loss
        Multi-objective loss

    video_harmonizer.py
        Complete model pipeline
        Encoder, quantizers, and decoders

datasets/
    video_dataset.py
        Video clip loader
        Supports .pt tensors
        Supports .npy tensors
        Supports standard video files

make_dummy_data.py
    Synthetic video-clip generator

train_harmonizer.py
    Training loop
    Gumbel-temperature annealing
    Commitment-loss optimization
    Checkpoint saving

test_harmonizer.py
    Video reconstruction
    PSNR evaluation
    SSIM evaluation
    SNE% evaluation
    Token-stream inspection

smoke_test.py
    End-to-end forward-pass test
    End-to-end backward-pass test
    Architecture self-check
```

---

## Core Pipeline

```text
Input Video
    ↓
Patch Extraction
    ↓
Adaptive Hann–Tukey 3D Windowing
    ↓
Weierstrass / Riesz / Gabor / Mellin Transform Bank
    ↓
Transform-Feature Concatenation
    ↓
Shared FluxHead Encoder
    ↓
┌───────────────────────────────┬───────────────────────────────┐
│ Discrete Token Stream         │ Continuous Token Stream       │
│                               │                               │
│ FusionQuantizer               │ Gumbel-Softmax Quantizer      │
│ Residual Codebooks            │ Continuous Embeddings         │
│ Hierarchical Quantization     │ Temperature Annealing         │
└───────────────────────────────┴───────────────────────────────┘
    ↓                                           ↓
Discrete FluxFormer Decoder          Continuous FluxFormer Decoder
    ↓                                           ↓
Discrete Reconstruction              Continuous Reconstruction
```

---

## Token Outputs

The model emits two complementary token representations.

### Discrete Tokens

The discrete stream produces hierarchical codebook indices through the FusionQuantizer.

These tokens are suitable for:

- compact symbolic representation
- codebook-based video compression
- discrete sequence modeling
- multimodal language-model integration
- token-level retrieval and reasoning

### Continuous Tokens

The continuous stream produces Gumbel-Softmax-based embeddings.

These tokens are suitable for:

- differentiable downstream learning
- continuous multimodal alignment
- reconstruction-sensitive representations
- representation transfer
- end-to-end optimization with multimodal models

Both representations are emitted directly, without requiring a conversion layer between the discrete and continuous streams.

---

## Training Objectives

The model uses a weighted reconstruction objective combining MSE and SSIM:

```text
L_reconstruction =
    λ_MSE × L_MSE
    +
    λ_SSIM × L_SSIM
```

with:

```text
λ_MSE = 0.75
λ_SSIM = 0.25
```

The discrete quantization stream also uses a commitment loss weighted by β.

The complete training objective supports:

- pixel-level reconstruction accuracy
- structural reconstruction quality
- codebook utilization
- latent-space commitment
- discrete-token stability
- continuous-token differentiability

---

## Evaluation Metrics

### Peak Signal-to-Noise Ratio

PSNR measures pixel-level reconstruction fidelity.

### Structural Similarity Index

SSIM measures perceptual and structural similarity between the original and reconstructed clips.

### Sequence-Level Normalized Error

SNE% measures normalized reconstruction error across the full video sequence, following Equation 36 of the paper.

---

## Supported Input Formats

The dataset loader supports:

```text
.pt
.npy
.mp4
.avi
.mov
.mkv
```

Tensor inputs should follow the expected video format:

```text
Channels × Frames × Height × Width
```

For example:

```text
3 × 8 × 32 × 32
```

---

## Reproducibility

For reproducible experiments, record:

- random seed
- selected configuration
- dataset version
- batch size
- learning rate
- number of epochs
- codebook size
- embedding dimension
- Gumbel temperature schedule
- spatial tile size
- temporal chunk size
- hardware configuration
- software-library versions

---

## Intended Use

This repository is intended for:

- research reproduction
- architecture validation
- video-tokenizer experimentation
- multimodal LLM research
- discrete and continuous video-token analysis
- UHD video-representation research
- educational demonstrations of paper-faithful implementation

The tiny configuration is intended primarily for debugging, demonstration, and architectural verification.

The paper configuration is intended for large-scale GPU experimentation on high-resolution video datasets.

---

## Authorship

This implementation was independently developed by **Amin Amiri** in 2025.

Development work included:

- architecture translation
- equation-to-module mapping
- implementation of the discrete and continuous tokenization streams
- end-to-end model integration
- construction of the training and evaluation pipelines
- testing and validation of model behavior

---

## Citation and Attribution

When using or adapting this implementation, please cite the original **Video-Harmonizer** paper and retain the copyright and license notices included with this repository.

Suggested implementation acknowledgment:

```text
This work uses an architecture-faithful implementation of Video-Harmonizer
developed by Amin Amiri in 2025.
```

---

## License

This project is released under the **MIT License**.

```text
MIT License

Copyright (c) 2025 Amin Amiri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Disclaimer

This repository is an independent implementation based on the architecture and equations described in the referenced manuscript.

It is not the official implementation of the original paper unless explicitly recognized as such by the paper's authors or publisher. Quantitative results may differ because of differences in:

- datasets
- preprocessing
- training duration
- random initialization
- hardware
- optimization settings
- implementation details not fully specified in the manuscript
