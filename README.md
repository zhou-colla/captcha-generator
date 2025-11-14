# CAPTCHA Generator

A Python project for generating CAPTCHA images using two different approaches: a rule-based arithmetic generator and a deep learning GAN-based generator.

## Overview

This project provides two methods for creating CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) images:

1. **Arithmetic CAPTCHA Generator** - Rule-based approach with customizable fonts and distortions
2. **GAN-based CAPTCHA Generator** - Deep learning approach using Generative Adversarial Networks

## Features

### Arithmetic Generator (`captcha_generator_arithmetic.py`)
- Simple rule-based CAPTCHA generation
- Customizable dimensions and font sizes
- Random text from alphanumeric characters
- Built-in noise and distortion effects
- Color randomization
- Easy integration for quick CAPTCHA needs

### GAN Generator (`captcha_generator_gan.py`)
- Deep learning based image synthesis using GANs
- Generator and Discriminator neural networks
- Conditional generation based on text labels
- Training support with checkpointing
- Generates realistic-looking CAPTCHA images
- GPU-accelerated training (CUDA support)

## Requirements

```
torch
torchvision
Pillow (PIL)
```

Install dependencies:
```bash
pip install torch torchvision pillow
```

## Usage

### Arithmetic Generator

```python
from captcha_generator_arithmetic import SimpleCaptcha

# Initialize generator
captcha = SimpleCaptcha(
    width=160,
    height=60,
    font_size=42,
    fonts=["/System/Library/Fonts/Supplemental/Arial.ttf"]  # Adjust for your OS
)

# Generate a single CAPTCHA
image, text = captcha.generate()
image.save("captcha_sample.png")
print(f"Generated CAPTCHA text: {text}")

# Generate multiple CAPTCHAs
for i in range(10):
    image, text = captcha.generate()
    image.save(f"captcha_{i:04d}.png")
```

### GAN Generator

#### Training

```bash
python captcha_generator_gan.py \
    --data_dir data/ \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.0002
```

**Arguments:**
- `--data_dir`: Directory containing training images (default: `data/`)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.0002)
- `--resume`: Resume training from checkpoint

#### Generating Images

After training, the model checkpoints are saved:
- `captcha_gan_G.pt` - Generator weights
- `captcha_gan_D.pt` - Discriminator weights
- `captcha_gan_ckpt.pth` - Full checkpoint with optimizer state

Generated samples are saved during training in the `generated/` folder.

## Project Structure

```
captcha-generator/
├── captcha_generator_arithmetic.py   # Simple rule-based generator
├── captcha_generator_gan.py          # GAN-based generator
├── data/                             # Training data for GAN
├── fonts/                            # Font files for arithmetic generator
├── generated/                        # Generated CAPTCHA images from GAN
├── README.md                         # This file
└── .gitignore                        # Git ignore file
```

## Data Format

The `data/` folder should contain PNG images with filenames in the format:
```
{label}-{index}.png
```

Example:
```
abc123-0.png
def456-1.png
xyz789-2.png
```

The label is extracted from the filename and used as the conditional input to the GAN.

## Model Architecture

### Generator
- Input: Random noise (100 dims) + embedded text label (128 dims)
- FC layer expanding to 256 × 8 × 16
- Three transposed convolution layers with upsampling
- Output: Grayscale CAPTCHA images (1 × 64 × 128)

### Discriminator
- Input: CAPTCHA image + embedded text label
- Three convolution layers with downsampling
- Label embedding concatenated with image features
- FC layer producing binary output (real/fake)

## Notes

- For macOS, default system fonts are used in the arithmetic generator. Adjust `fonts` parameter for other systems
- GAN training requires significant computational resources; GPU recommended
- Both generators produce grayscale images for simplicity
- Generated CAPTCHAs are 64 pixels tall and 128 pixels wide

## License

This project is open source.

## Contributing

Feel free to submit issues or pull requests for improvements.
