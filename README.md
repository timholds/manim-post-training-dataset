# Manim Code Generation with Fine-tuned Models

This project fine-tunes various code generation models on a curated Manim dataset to generate high-quality Manim animation code from natural language descriptions.

## Overview

- **Purpose**: Create a model-agnostic dataset for fine-tuning code generation models on Manim animations
- **Current Dataset**: 3,827 unique animation examples from 3 high-quality sources
- **Training Method**: QLoRA with Unsloth optimizations  
- **Supported Models**: QWEN2.5-Coder, CodeLlama, DeepSeek, CodeGemma, Stable Code
- **Deployment**: Ollama-compatible GGUF format

## Quick Start

### Prerequisites
- NVIDIA GPU with 16GB+ VRAM
- Python 3.8+
- Ollama (for deployment)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd manim-post-training

# Create and activate virtual environment
python -m venv manim-env
source manim-env/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Download and process all datasets with deduplication
python prepare_data_enhanced.py

# Or process specific datasets
python prepare_data_enhanced.py --datasets bespoke_manim thanks_dataset
```

### Train Model

```bash
# Fine-tune with default settings
python fine_tune.py --model "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Custom training parameters
python fine_tune.py --model "Qwen/Qwen2.5-Coder-1.5B-Instruct" --epochs 5 --batch-size 2
```

## Documentation

- **[Current Dataset State](docs/CURRENT_STATE.md)** - Latest statistics and quality metrics
- **[Data Pipeline Guide](docs/PIPELINE.md)** - How to use and extend the data pipeline
- **[Development Roadmap](docs/ROADMAP.md)** - Plans for adding 200-500+ more samples

## Key Features

- 🔍 Automatic deduplication across multiple data sources
- 📊 Model-agnostic dataset format (handles special tokens during training)
- 🚀 Efficient 4-bit quantized training with ~12GB VRAM usage
- 🧪 Comprehensive evaluation metrics with Weights & Biases integration
- 🔧 Easy deployment with Ollama

## Project Structure

```
manim-post-training/
├── prepare_data_enhanced.py  # Multi-dataset preparation pipeline
├── fine_tune.py             # Universal training script
├── docs/                    # Detailed documentation
│   ├── CURRENT_STATE.md    # Dataset statistics
│   ├── PIPELINE.md         # Pipeline usage guide
│   └── ROADMAP.md          # Future development plans
├── data_formatted/          # Processed datasets
├── models/                  # Trained model outputs
└── requirements.txt         # Python dependencies
```

## Contributing

See the [Development Roadmap](docs/ROADMAP.md) for priority datasets to add. The data pipeline is designed to easily integrate new Manim code sources.

## License

[Your license here]