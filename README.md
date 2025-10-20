# Storytime SLM: Small Language Model for Story Generation

A compact, from-scratch implementation of a GPT-based language model trained on the TinyStories dataset to generate engaging children's stories. Inspired by nanoGPT, this project demonstrates efficient training and inference for small-scale language models using PyTorch.

## Features

- **Custom GPT Architecture**: Lightweight GPT model with configurable layers, heads, and embeddings (6 layers, 6 heads, 384 embeddings by default).
- **Efficient Training**: Supports mixed precision (bfloat16/float16), gradient accumulation, weight decay, and learning rate scheduling (warmup + cosine decay).
- **GPU Acceleration**: Optimized for CUDA with automatic device detection and CPU fallback.
- **Text Generation**: Generate coherent stories from prompts with temperature and top-k sampling.
- **Data Preparation**: Automated tokenization and binary file creation for fast training.
- **Visualization**: Built-in loss plotting for training monitoring.
- **Remarkable Efficiency**: Trains a functional story generator in under 20,000 iterations on modest hardware, showcasing the power of transformer architectures for creative tasks.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Storytime
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support (recommended for training), install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open `notebooks/Storytime.ipynb` to run the project.

## Usage

The project is contained in a single Jupyter notebook. Run cells sequentially for end-to-end execution.

### Data Preparation
- Downloads and tokenizes the TinyStories dataset using tiktoken (GPT-2 encoding).
- Creates binary files (`train.bin`, `validation.bin`) for efficient loading.

### Training
- Configures model hyperparameters (e.g., block size 128, batch size 32).
- Trains with AdamW optimizer, gradient clipping, and automatic model saving (best validation loss).
- Monitors training/validation loss every 500 iterations.

### Text Generation
- Load the trained model and generate stories from prompts like "Once upon a time there was a pumpkin."
- Adjustable parameters: max tokens, temperature, top-k.

Example output:
```
Once upon a time there was a pumpkin. It was very big and orange. One day, a little girl saw the pumpkin and said, "What a nice pumpkin!" She picked it up and took it home. Her mom made a pie with the pumpkin. It was delicious!
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU)
- datasets
- tiktoken
- numpy
- tqdm
- matplotlib

See `requirements.txt` for exact versions.

## Project Structure

- `notebooks/Storytime.ipynb`: Main notebook with data prep, model, training, and generation.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes virtual environments, model checkpoints, and data files.

## Contributing

Contributions welcome! Open issues or pull requests for improvements, bug fixes, or extensions (e.g., larger models, different datasets).

## License

This project is open-source. Please check for any dataset-specific licenses (TinyStories).