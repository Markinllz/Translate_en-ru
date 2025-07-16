# English to Russian Translation Model

A neural machine translation project that translates English text to Russian using transformer architecture. This project supports both custom-trained models and pre-trained Hugging Face models.

## Features

- **Dual Model Support**: Works with both custom PyTorch transformer models and pre-trained Hugging Face models
- **Hydra Configuration**: Flexible configuration management using Hydra
- **Easy Inference**: Simple command-line interface for translation
- **Multiple Output Formats**: Saves translations to text files
- **GPU/CPU Support**: Automatic device detection and support

## Project Structure

```
Model_English_to_Russian/
├── inference.py              # Main inference script
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── src/
│   ├── configs/             # Hydra configuration files
│   │   ├── inference.yaml   # Inference configuration
│   │   ├── model/           # Model configurations
│   │   └── dataset/         # Dataset configurations
│   ├── model/               # Model implementations
│   ├── trainer/             # Training logic
│   ├── datasets/            # Data loading utilities
│   └── utils/               # Utility functions
├── saved/                   # Saved model checkpoints
└── outputs/                 # Output files
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Model_English_to_Russian
   ```

2. **Create a virtual environment**:
   ```bash
   conda create -n myenv python=3.9
   conda activate myenv
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start with Hugging Face Model

The easiest way to use this project is with a pre-trained Hugging Face model:

```bash
python inference.py
```

This will:
- Load the Helsinki-NLP/opus-mt-en-ru model
- Translate the text specified in the configuration
- Save the result to `translation.txt`

### Configuration

The inference behavior is controlled by `src/configs/inference.yaml`:

```yaml
inferencer:
  seed: 42
  device: auto  # or "cpu", "cuda"
  text: "Hello, how are you?"  # Text to translate
  output_file: "translation.txt"  # Output file
```

### Customizing Translation

To translate different text, modify the configuration:

1. **Edit the config file** (`src/configs/inference.yaml`):
   ```yaml
   inferencer:
     text: "Your text here"
     output_file: "my_translation.txt"
   ```

2. **Run inference**:
   ```bash
   python inference.py
   ```

### Using Custom Models

If you have a custom-trained model:

1. **Place your model checkpoint** in `saved/testing/model_best.pth`
2. **Update the configuration** to use the custom model:
   ```yaml
   defaults:
     - model: transformer  # Use custom model instead of huggingface_transformer
   ```

## Model Options

### 1. Hugging Face Model (Recommended)

- **Model**: Helsinki-NLP/opus-mt-en-ru
- **Advantages**: Pre-trained, high quality, no training required
- **Configuration**: `src/configs/model/huggingface_transformer.yaml`

### 2. Custom Transformer Model

- **Architecture**: PyTorch Transformer (encoder-decoder)
- **Training**: Requires custom training with your dataset
- **Configuration**: `src/configs/model/transformer.yaml`

## Training (Optional)

If you want to train your own model:

1. **Prepare your dataset** in the required format
2. **Configure training parameters** in `src/configs/`
3. **Run training**:
   ```bash
   python train.py
   ```

## Configuration Files

### Inference Configuration (`src/configs/inference.yaml`)
- Device settings
- Text to translate
- Output file path
- Model selection

### Model Configurations (`src/configs/model/`)
- `huggingface_transformer.yaml`: Hugging Face model settings
- `transformer.yaml`: Custom transformer parameters

### Dataset Configuration (`src/configs/dataset/translation.yaml`)
- Batch size
- Maximum sequence length
- Model name for tokenizer

## Output

The inference script produces:
1. **Console output**: Translation progress and results
2. **Text file**: Saved translation with original and translated text

Example output file (`translation.txt`):
```
English: Hello, how are you?
Russian: Привет, как дела?
```

## Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Hydra**: Configuration management
- **Datasets**: Data loading utilities
- **WandB**: Experiment tracking (optional)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `device: "cpu"` in config
2. **Model not found**: Ensure model checkpoint exists in `saved/testing/`
3. **Import errors**: Check that all dependencies are installed

### Performance Tips

- Use GPU for faster inference: `device: "cuda"`
- Adjust `max_length` in configuration for longer texts
- Use beam search for better quality (configured in Hugging Face model)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Helsinki-NLP for the pre-trained translation model
- Hugging Face for the transformers library
- PyTorch team for the deep learning framework
