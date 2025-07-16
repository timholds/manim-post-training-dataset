# Manim Code Generation with Fine-tuned Models

This project fine-tunes various code generation models on a curated Manim dataset to generate high-quality Manim animation code from natural language descriptions.

## Overview

- **Purpose**: Create a model-agnostic dataset for fine-tuning code generation models on Manim animations
- **Current Dataset**: 4,000+ unique animation examples from 11 high-quality sources
- **Architecture**: Plugin-based data pipeline for easy extensibility
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
git clone https://github.com/timholds/manim-post-training-dataset
cd manim-post-training-dataset

# Create and activate virtual environment
python -m venv manim-env
source manim-env/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Download and process all datasets with deduplication
python prepare_data.py

# List available data sources
python prepare_data.py --list-sources

# Process specific datasets
python prepare_data.py --sources bespoke_manim thanks_dataset

# Enable data augmentation (creates variations of prompts)
python prepare_data.py --augmentation
```

### Enhance with LLM Descriptions (Optional)

```bash
# Generate better descriptions using LLM (with caching)
python prepare_data_with_llm.py generate-descriptions \
    --input data_formatted/train.json \
    --output data_enhanced/train.json \
    --llm gemini

# Check LLM cache statistics
python prepare_data_with_llm.py cache-stats
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
- **[Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md)** - Decoupled pipeline design
- **[Data Pipeline Guide](docs/PIPELINE.md)** - How to use and extend the data pipeline
- **[Adding New Data Sources](docs/migration_guide.md)** - Guide for creating new extractors
- **[Development Roadmap](docs/ROADMAP.md)** - Plans for adding 200-500+ more samples

## Key Features

- 🔌 **Plugin-based architecture** - Add new data sources by creating a single extractor file
- 🔍 **Automatic deduplication** - Intelligent duplicate removal across multiple sources
- 📊 **Model-agnostic dataset** - Handles special tokens during training
- 🤖 **LLM-enhanced descriptions** - Optional LLM generation with smart caching
- 📐 **Decoupled pipeline** - Each stage (extract → enhance → train) is independent
- 🚀 **Efficient training** - 4-bit quantized training with ~12GB VRAM usage
- 🧪 **Comprehensive evaluation** - Weights & Biases integration for metrics
- 🔧 **Easy deployment** - Export to Ollama-compatible GGUF format

## Project Structure

```
manim-post-training/
├── prepare_data.py          # Main data preparation script
├── fine_tune.py             # Universal training script
├── extractors/              # Plugin-based data extractors
│   ├── base.py             # Base extractor interface
│   ├── registry.py         # Dynamic extractor registry
│   └── sources/            # Individual data source extractors
│       ├── kaggle.py       # Kaggle datasets (Manimbench)
│       ├── huggingface.py  # HuggingFace datasets
│       └── local.py        # Local file extractors
├── docs/                    # Detailed documentation
│   ├── CURRENT_STATE.md    # Dataset statistics
│   ├── PIPELINE.md         # Pipeline usage guide
│   ├── migration_guide.md  # Adding new extractors
│   └── ROADMAP.md          # Future development plans
├── data_formatted/          # Processed datasets
├── models/                  # Trained model outputs
└── requirements.txt         # Python dependencies
```

## Adding New Data Sources

Adding a new data source is as simple as creating a new extractor:

```python
# extractors/sources/your_source.py
from ..base import BaseExtractor
from ..registry import register_extractor

@register_extractor
class YourSourceExtractor(BaseExtractor):
    source_id = "your_source"
    source_name = "Your Data Source"
    priority = 3  # 1-5, higher = keep when deduplicating
    
    def extract(self):
        # Your extraction logic here
        yield {"description": "...", "code": "..."}
```

See the [Adding New Data Sources](docs/migration_guide.md) guide for detailed instructions.

## Contributing

See the [Development Roadmap](docs/ROADMAP.md) for priority datasets to add. The plugin-based architecture makes it easy to contribute new data sources.

## Deduplication Strategy
Besides rows that have a placeholder for the description that an LLM will fill in later, all descriptions must be unique. When we find two or more rows with the same description, we keep the one with the highest priority source. 