# Make Your Own LLM

A Python project for training custom Large Language Models (LLMs) from scratch using a Transformer-based architecture with PyTorch.

## Overview

This project provides a complete pipeline to:
- Download and preprocess datasets from Hugging Face
- Train a custom tokenizer (BPE-based)
- Train a Transformer-based language model from scratch
- Generate text using the trained model
- Resume training from checkpoints
- Track model performance with evaluation metrics

## Features

✨ **Key Features:**
- **Custom Transformer Architecture** - Causal self-attention with multi-head attention
- **Flexible Configuration** - Easily adjustable model hyperparameters via JSON
- **Dataset Management** - Support for multiple datasets from Hugging Face
- **Checkpoint System** - Save and resume training at any epoch
- **Distributed Training Ready** - Multi-GPU support with mixed precision (fp16)
- **Tokenizer Training** - BPE tokenizer training for both English and French
- **Evaluation Metrics** - Per-epoch validation and prompt generation tracking

## Project Structure

```
makeyourownllm/
├── train.py                 # Main training script
├── run.py                   # Inference/generation script
├── train_tokenizer.py       # Tokenizer training
├── custom_data.py           # Dataset handling utilities
├── dataset_manager.py       # Dataset management
├── config.json              # Model configuration
├── config-run.template.json # Template for inference config
├── datasets.json            # Dataset sources configuration
├── requirements.txt         # Python dependencies
├── logs/                    # Training logs directory
├── utils/                   # Utility functions
└── myllm0_1/               # Example trained model directory
```

## Requirements

```
Python 3.8+
PyTorch with CUDA support (recommended)
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### Main Dependencies

- **transformers** (4.43.0) - Hugging Face transformers library
- **torch** - PyTorch deep learning framework
- **accelerate** (1.11.0) - Training acceleration utilities
- **datasets** (2.15.0) - Hugging Face datasets library
- **tokenizers** (>=0.19) - Fast tokenizer implementation
- **bitsandbytes** (0.40.0) - 8-bit quantization support
- **numpy, scipy, tqdm** - Numerical and utility libraries
- **beautifulsoup4** - Web scraping utilities

## Configuration

### Model Configuration (config.json)

```json
{
  "embed_dim": 400,        # Embedding dimension
  "num_heads": 16,         # Number of attention heads
  "num_layers": 9,         # Number of transformer blocks
  "block_size": 256,       # Context window size
  "vocab_size": 30000,     # Vocabulary size
  "mlp_ratio": 5.0,        # MLP hidden dimension ratio
  "dropout": 0.2,          # Dropout probability
  "batch_size": 32,        # Training batch size
  "lr": 0.0001,            # Learning rate
  "epochs": 10             # Number of training epochs
}
```

### Dataset Configuration (datasets.json)

Define the datasets to download and use for training.

## Usage

### 1. Prepare Datasets and Tokenizer

```bash
python train.py mymodel \
    --generate-tokenizer \
    --import-datasets \
    --merge-datasets \
    --tokenizer-lang en
```

**Arguments:**
- `--generate-tokenizer` - Train a new BPE tokenizer
- `--import-datasets` - Download datasets from Hugging Face
- `--merge-datasets` - Merge datasets into train/val/test splits
- `--tokenizer-lang` - Language for tokenizer (en or fr)

### 2. Train the Model

```bash
python train.py mymodel \
    --tokenizer-path tokenizer.json \
    --config-path config.json \
    --eval-prompt "Once upon a time" \
    --train-seed 42
```

**Training Arguments:**
- `--tokenizer-path` - Path to the tokenizer file
- `--config-path` - Path to model configuration
- `--eval-prompt` - Prompt for evaluation during training
- `--eval-max-new-tokens` - Max tokens for eval generation (default: 50)
- `--train-seed` - Random seed (default: 42)
- `--workers` - DataLoader workers (default: 0)

### 3. Resume Training

Resume from the last checkpoint:
```bash
python train.py mymodel --resume-training last
```

Resume from a specific epoch:
```bash
python train.py mymodel --resume-training 5
```

Resume from a checkpoint directory:
```bash
python train.py mymodel --resume-training path/to/checkpoint
```

### 4. Generate Text

```bash
python run.py "Your prompt here" \
    -m mymodel \
    --device cuda \
    --tokenizer-path mymodel/tokenizer.json \
    -o output.txt
```

**Inference Arguments:**
- `prompt` - Text prompt for generation
- `-m, --model-path` - Path to trained model directory
- `--device` - Device to use (cuda or cpu)
- `-t, --tokenizer-path` - Path to tokenizer file
- `--use-auto-tokenizer` - Use HuggingFace auto tokenizer
- `-o, --output` - Output file path (optional)

**Generation Parameters (config-run.json):**
```json
{
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 50,
  "num_beams": 1,
  "max_new_tokens": 50,
  "do_sample": true,
  "repetition_penalty": 1.0,
  "length_penalty": 1.0,
  "early_stopping": false,
  "seed": 42
}
```

## Model Architecture

### Transformer Components

**CausalSelfAttention:**
- Multi-head causal attention mechanism
- Prevents tokens from attending to future positions
- Supports configurable number of attention heads

**TransformerBlock:**
- Layer normalization before attention and MLP
- Feed-forward network with GELU activation
- Residual connections and dropout

**myTransformer:**
- Token and positional embeddings
- Stack of transformer blocks
- Language modeling head for next-token prediction
- Generation method with temperature, top-k, and top-p sampling

## Training Features

### Mixed Precision Training
- Automatic mixed precision (fp16) using PyTorch's autocast
- Gradient scaling for stable training

### Learning Rate Scheduling
- Warmup phase with linear increase
- Cosine annealing decay
- Configurable warmup steps and total steps

### Checkpointing
- Save model, optimizer, and scheduler states
- Track best validation loss
- Save every N epochs
- Metadata tracking (epoch, global step, best loss)

### Data Pipeline
- Streaming dataset for memory efficiency
- Multi-worker data loading
- Per-worker seed management for reproducibility

## Outputs

### Training Logs
- `logs/{model_name}.log` - Training log file
- `{model_name}/eval_prompts.txt` - Generated text samples during training

### Model Checkpoints
- `{model_name}/checkpoints/{epoch}/` - Checkpoint directory containing:
  - `model.pt` - Model state dict
  - `optimizer.pt` - Optimizer state dict
  - `scheduler.pt` - Learning rate scheduler state dict
  - `meta.json` - Metadata (epoch, global_step, best_val_loss)

### Final Model
- `{model_name}/checkpoints/{model_name}.pt` - Final trained model weights

## Example Workflow

```bash
# 1. Create tokenizer and download datasets
python train.py my_custom_llm \
    --generate-tokenizer \
    --import-datasets \
    --merge-datasets \
    --tokenizer-lang en \
    --reset

# 2. Train for 10 epochs
python train.py my_custom_llm \
    --tokenizer-path tokenizer.json \
    --config-path config.json \
    --eval-prompt "The future of AI is"

# 3. Generate text
python run.py "The future of AI is" \
    -m my_custom_llm \
    --device cuda \
    -o generated_text.txt
```

## Performance Considerations

- **GPU Memory:** Adjust `batch_size` in config.json based on GPU VRAM
- **Sequence Length:** `block_size` affects memory; start small and increase
- **Num Heads:** Must evenly divide `embed_dim`
- **Multi-GPU:** Modify code to use `torch.nn.DataParallel` or `DistributedDataParallel`

## Troubleshooting

**NaN/Inf in loss:**
- Reduce learning rate
- Increase warmup steps
- Check for gradient overflow

**Out of Memory:**
- Reduce batch size
- Reduce block_size (sequence length)
- Use gradient accumulation

**Poor generation quality:**
- Train longer (increase epochs)
- Use larger model (increase embed_dim, num_layers)
- Increase dataset size/quality

## License

MIT

## Author

levashi

---

**Note:** This is a learning project for understanding transformer-based language models from first principles.