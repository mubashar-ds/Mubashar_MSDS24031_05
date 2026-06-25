# Diffusion Model for Animal Image Generation

## Overview
This assignment implements a Denoising Diffusion Probabilistic Model (DDPM) for unconditional animal image generation using PyTorch. Model learns to generate images by gradually adding Gaussian noise during forward diffusion process & learning to reverse this process using a U-Net based neural network.

## Dataset
Model is trained on an animal image dataset containing following classes:
* Cat
* Lion
* Tiger
* Horse
* Elephant

All images are resized to 64 × 64 pixels and normalized to the range [-1, 1] before training.

---

## Model Architecture
Denoising network is based on lightweight U-Net architecture consisting of:
* Double Convolution blocks
* Encoder (Down Blocks)
* Bottleneck
* Decoder (Up Blocks)
* Time Embedding module
* Final 1×1 Convolution layer

Network predicts Gaussian noise added during forward diffusion process.

---

## Diffusion Process

### Forward Diffusion
Forward process gradually corrupts an image with Gaussian noise using a linear variance schedule.

### Reverse Diffusion
Trained U-Net predicts added noise at each diffusion step, allowing model to progressively reconstruct image from random noise.

---

## Training Configuration

| Parameter | Value |
| ------------- | -----------------------: |
| Image Size | 64 × 64 |
| Noise Steps | 1000 |
| Optimizer | Adam |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-5 |
| Loss Function | MSE |
| Epochs | 100 |
| Model Saving | Best Training Loss |

---

## Results
Training loss decreases steadily throughout training, indicating successful optimization of denoising objective.  Project demonstrates:

* Forward diffusion
* Reverse diffusion
* Image generation from Gaussian noise
* Intermediate denoising visualization

Since implementation uses a compact U-Net and limited computational resources, so generated images capture coarse structures and textures rather than highly detailed animal appearances.

---

## Requirements
Install the required packages:

```bash
pip install -r requirements.txt
```
---

## How to Run

### Train the Model
```bash
cd codes
python train.py
```
Best performing model will be saved automatically.

### Generate Images

Execute the inference notebook from directory codes/

---

## Outputs
The project generates:
* Training loss curve
* Forward diffusion visualization
* Reverse diffusion visualization
* Generated sample images
* Best trained model checkpoint

---

## Future Improvements
Possible extensions include:
* Sinusoidal positional embeddings
* Residual U-Net blocks
* Self-attention layers
* Cosine noise scheduler
* Higher image resolution
* Larger training dataset
* Longer training duration
* Exponential Moving Average (EMA)

---

## Author
**Mubashar Hussain**  
MSDS24031  
Department of Artificial Intelligence, (ITU)