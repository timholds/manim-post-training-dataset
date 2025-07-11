# Data Pipeline Documentation

## Overview

The `prepare_data_enhanced.py` script provides a comprehensive data pipeline for preparing multiple Manim datasets for fine-tuning code generation models.

## Features

- **Multi-Dataset Support**: Automatically downloads and processes datasets from HuggingFace and Kaggle
- **Auto Field Detection**: Intelligently detects dataset schema variations
- **Data Augmentation**: Applies 2.5x augmentation through prompt variations
- **Quality Assurance**: Validates code syntax and ensures proper Scene class structure
- **Flexible Processing**: Choose specific datasets or process all available
- **Deduplication**: Removes duplicate descriptions across and within datasets (48.5% reduction achieved!)
  - Prioritizes high-quality sources (ManimBench > Bespoke > Thanks > ManimCodeGen)
  - Generates detailed reports of removed duplicates

## Usage

### Basic Usage
```bash
# Process all available datasets
python prepare_data_enhanced.py

# Process specific datasets
python prepare_data_enhanced.py --datasets bespoke_manim thanks_dataset

# Disable augmentation
python prepare_data_enhanced.py --no-augmentation

# Custom output directory
python prepare_data_enhanced.py --output-dir custom_data

# Enable deduplication (RECOMMENDED!)
python prepare_data_enhanced.py --deduplicate --output-dir data_formatted_deduplicated
```

### Dataset Configuration

The script supports the following datasets out of the box:

1. **Bespoke Manim** (HuggingFace)
   - 1,000 high-quality examples with rich descriptions
   - Auto-detects fields: `question` → description, `python_code` → code

2. **Thanks Dataset** (HuggingFace)
   - 4,400 examples with varied complexity
   - Fields: `input` → description, `output` → code

3. **ManimCodeGen** (HuggingFace)
   - 1,622 examples with queries
   - Fields: `query` → description, `answer` → code

4. **ManimBench** (Kaggle) ✅
   - 417 high-quality examples with detailed descriptions
   - Fields: `Reviewed Description` → description, `Code` → code
   - 100% unique content (no overlap with other datasets)

## Data Format

### Input Processing
- Automatically strips markdown code blocks
- Validates Python syntax
- Ensures proper imports and Scene class structure

### Output Format (JSON Lines)
```json
{
  "conversations": [
    {"from": "system", "value": "You are a Manim code generator..."},
    {"from": "user", "value": "Create animation that..."},
    {"from": "assistant", "value": "```python\nfrom manim import *\n...```"}
  ],
  "source": "dataset_name"  // Source tracking field
}
```

### Augmentation Strategy
The pipeline applies intelligent augmentation to training data:
- Original prompt
- "Create a Manim animation that..."
- "Write Manim code to..."
- "Generate a Manim scene that..."
- And more variations

## Output Files

Default output directory: `data_formatted_with_sources/` (or `data_formatted_deduplicated/` with --deduplicate)
- `train.json` - Augmented training data with source tracking
- `test.json` - Test split (10% of data) with source tracking
- `dataset_stats.json` - Detailed statistics including source distribution

### With Deduplication Enabled
Additional files created:
- `deduplication_report.json` - Comprehensive deduplication statistics
- `removed_duplicates.json` - Examples of removed duplicate entries

### Source Tracking
Each sample includes a `"source"` field indicating which dataset it came from:
- `"manimbench"` - From ManimBench (Kaggle)
- `"bespoke_manim"` - From Bespoke Manim (HuggingFace)
- `"thanks_dataset"` - From Thanks Dataset (HuggingFace)
- `"manim_codegen"` - From ManimCodeGen (HuggingFace)

## Adding New Datasets

To add a new dataset, edit the `DATASETS` dictionary in the script:

```python
"new_dataset": {
    "type": "huggingface",  # or "kaggle"
    "dataset_name": "org/dataset-name",
    "description_field": "instruction",
    "code_field": "output",
    "split": "train",
    "expected_samples": 1000
}
```

## Troubleshooting

### Kaggle Datasets
1. Install kagglehub: `uv pip install kagglehub`
2. Get API token from https://www.kaggle.com/account
3. Place `kaggle.json` in `~/.kaggle/`

### Field Detection
If a dataset has non-standard field names, the script will:
1. List available columns
2. Attempt auto-detection based on common patterns
3. Use the detected fields or skip if unable to detect

### Memory Issues
For large datasets, the script caches downloaded data in `~/.cache/manim_datasets/` to avoid re-downloading.