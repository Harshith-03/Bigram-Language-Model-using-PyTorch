# Bigram Language Model using PyTorch

This repository contains an implementation of a Transformer-based bigram language model built using PyTorch. The model is trained on a text corpus and can generate text based on learned patterns. This project is the work of Andrej Karpathy and has **only** been **optimised** by me. The link to his repo `https://github.com/karpathy/ng-video-lecture`

## Features
- Implements a Transformer-based language model with self-attention.
- Uses PyTorch's `torch.nn` module to build a multi-layer Transformer.
- Implements gradient checkpointing for memory efficiency.
- Supports mixed-precision training using `torch.cuda.amp` for better performance.
- Uses `AdamW` optimizer with learning rate scheduling.

## Installation
Ensure you have Python installed along with PyTorch:
```bash
pip install torch
```

## Dataset
Place your text data in `input.txt` in the project directory. The script will read this file, encode characters, and use it for training.

## Model Architecture
The model is based on a simplified Transformer with self-attention layers. Key components include:
- **Embedding Layers**: Converts input tokens into dense vectors.
- **Self-Attention Mechanism**: Allows the model to focus on relevant context.
- **Feed-Forward Layers**: Applies transformations to learned representations.
- **Layer Normalization**: Stabilizes training.

## Training
Run the script to start training:
```bash
python train.py
```

### Training Details:
- Uses a dataset split (90% training, 10% validation).
- Implements multi-head self-attention and feedforward layers.
- Optimized using `AdamW` with gradient clipping and learning rate scheduling.

## Text Generation
After training, the model can generate text. The script runs inference using the trained model and prints generated sequences.

## Usage
Modify `input.txt` to experiment with different text sources and retrain the model to see how it adapts to different writing styles.

## License
This project is licensed under the MIT License.
