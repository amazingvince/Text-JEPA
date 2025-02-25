# Text-JEPA: Joint-Embedding Predictive Architecture for NLP

This repository contains an implementation of the Text-JEPA (Joint-Embedding Predictive Architecture) for natural language processing, inspired by the [JEPA paper](https://arxiv.org/abs/2301.08243).

![Diagram](/images//text-jepa-final-v2.svg)

## Overview

Text-JEPA adapts the Joint-Embedding Predictive Architecture for text data. The core idea is to predict the representations of target spans from context spans in embedding space. This approach:

1. Doesn't rely on predicting exact token distributions (unlike MLM)
2. Learns more semantic representations by predicting in embedding space
3. Uses an EMA-updated target encoder for stable training

The architecture consists of:

- **Context Encoder**: Processes visible context tokens
- **Target Encoder**: Encodes target spans (updated via EMA of context encoder)
- **Predictor**: Predicts target representations from context representations
- **Loss Calculation**: L2 distance between predicted and actual representations

## Key Features

- Implemented in PyTorch with modular architecture
- Uses the AllenAI/C4 dataset for pretraining
- Supports streaming for efficient large-scale training
- Implements metrics for model evaluation (L2 loss, cosine similarity)
- Includes both training and evaluation scripts

## Installation

```bash
# Clone the repository
git clone https://github.com/amazingvince/Text-JEPA.git
cd Text-JEPA

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train the Text-JEPA model on the C4 dataset:

```bash
python train.py \
  --config config/default.yaml \
  --output_dir outputs \
  --log_dir logs \
  --subset en \
  --streaming
```

Options:

- `--config`: Path to configuration file
- `--output_dir`: Directory to save model checkpoints
- `--log_dir`: Directory to save logs and TensorBoard events
- `--subset`: C4 dataset subset (e.g., 'en', 'realnewslike')
- `--streaming`: Enable streaming mode for the dataset
- `--resume`: Path to checkpoint for resuming training

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py \
  --model_path outputs/best_model.pt \
  --output_dir eval_results \
  --subset en \
  --split validation \
  --num_samples 1000
```

Options:

- `--model_path`: Path to the model checkpoint
- `--output_dir`: Directory to save evaluation results
- `--subset`: C4 dataset subset to evaluate on
- `--split`: Dataset split to evaluate on
- `--num_samples`: Number of samples to evaluate
- `--batch_size`: Evaluation batch size

## Model Architecture

### Context Encoder

- 12-layer Transformer with position embeddings
- Encodes visible context tokens

### Target Encoder

- 12-layer Transformer with position embeddings (identical to Context Encoder)
- Weights updated via EMA of Context Encoder

### Predictor

- 6-layer Transformer with position information
- Predicts target representations from context representations

### Loss Calculation

- L2 loss between predicted and actual target representations
- Additional metric: cosine similarity between predicted and target representations

## Data Processing

The model uses the AllenAI/C4 dataset with a custom processing pipeline that:

1. Tokenizes input text
2. Randomly selects spans as targets
3. Creates inputs for context and target encoders
4. Provides span position information to the predictor

## Configuration

The default configuration is in `config/default.yaml`. You can customize:

- Model architecture (hidden size, number of layers, etc.)
- Training parameters (batch size, learning rate, etc.)
- Data processing (maximum sequence length, etc.)
- Evaluation settings (number of samples, batch size, etc.)

## Metrics

Text-JEPA includes several metrics for evaluating model performance:

- **L2 Loss**: Mean squared error between predicted and target representations
- **Cosine Similarity**: Similarity between predicted and target representations
