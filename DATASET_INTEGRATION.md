# Manim Dataset Integration Guide

## Overview

This guide documents the multi-dataset integration pipeline for creating high-quality Manim training data. The system now supports combining multiple datasets with intelligent filtering, deduplication, and quality control.

## Current Dataset Status

### Integrated Datasets

1. **ManimBench** (417 samples → 135 used)
   - Source: Kaggle (local parquet file)
   - High-quality curated examples with descriptions
   - 100% valid syntax and structure

2. **Bespoke Manim** (1000 samples → 135 used)
   - Source: HuggingFace `bespokelabs/bespoke-manim`
   - Educational animations with rich metadata
   - Includes questions, titles, and narration

3. **ManimCodeGen** (1622 samples → 135 used)
   - Source: HuggingFace `generaleoley/manim-codegen`
   - Query-response format
   - Good for diverse animation types

4. **Thanks Manim** (4400 samples → 45 used)
   - Source: HuggingFace `thanhkt/manim_code`
   - Large dataset but lower quality
   - Many samples filtered out during quality checks

### Combined Dataset Statistics

- **Total raw samples**: 7,439
- **After quality filtering**: 4,219 (57%)
- **After deduplication**: 2,237 (30%)
- **After balancing**: 450 unique samples
- **Final training set**: 826 samples (with 2x augmentation)
- **Final test set**: 37 samples

## Adding New Datasets

### 1. Define Dataset Configuration

Add a new `DatasetConfig` to the `DATASET_CONFIGS` list in `prepare_multi_dataset.py`:

```python
DatasetConfig(
    name="your-dataset-name",
    path="path/to/dataset",  # Local path or HuggingFace ID
    format="huggingface",     # Options: parquet, json, jsonl, csv, huggingface
    description_col="prompt", # Column containing the description/instruction
    code_col="completion",    # Column containing the Manim code
    split_col="split"         # Optional: column with train/test split
)
```

### 2. Handle Special Cases

For datasets with non-standard formats, add special handling in `load_dataset_from_config()`:

```python
if config.name == "your-dataset-name":
    # Custom processing logic
    description = extract_description(item)
    code = clean_code(item)
```

### 3. Quality Criteria

All samples must pass these checks:
- Valid Python syntax (AST parsing)
- Contains Manim imports or can be wrapped
- Has Scene class or animation method calls
- Code length: 50-5000 characters
- No exact or near duplicates (85% similarity threshold)

### 4. Running the Pipeline

```bash
source manim-env/bin/activate
python prepare_multi_dataset.py
```

## Quality Control Features

### 1. Code Cleaning
- Removes markdown formatting
- Extracts code from code blocks
- Handles escaped newlines (`\n`)

### 2. Code Wrapping
- Automatically adds missing imports
- Wraps standalone code in Scene class
- Ensures proper `construct` method

### 3. Deduplication
- Exact match detection via MD5 hashing
- Near-duplicate detection using normalized code
- Sequence matching for similarity scoring

### 4. Dataset Balancing
- Prevents single dataset dominance
- Max 3x ratio between datasets
- Random sampling for over-represented sources

## Output Format

### Conversation Structure
```json
{
  "conversations": [
    {"from": "system", "value": "System prompt..."},
    {"from": "user", "value": "User instruction"},
    {"from": "assistant", "value": "```python\n[code]\n```"}
  ],
  "metadata": {
    "source": "dataset_name",
    "original_description": "..."
  }
}
```

### Augmentation
- Training data includes 2x variations:
  - Original format
  - Template variations ("Create a Manim animation that...")
- Test data uses original format only

## Next Steps

1. **Add More High-Quality Sources**:
   - Reducible animations (GitHub + YouTube)
   - Dan4Life AoC2024 videos
   - Official Manim CE examples

2. **Implement Compilation Validation**:
   - Test Scene instantiation
   - Render first frame
   - Catch runtime errors

3. **Synthetic Data Generation**:
   - Use GPT-4 to create variations
   - Mathematical concept coverage
   - Difficulty progression

4. **Curriculum Learning**:
   - Organize by complexity
   - Create learning progression
   - Track concept coverage

## Troubleshooting

### Common Issues

1. **Memory errors during processing**:
   - Process datasets in chunks
   - Reduce batch size in deduplication

2. **Poor quality filtering**:
   - Check regex patterns for your dataset
   - Adjust minimum animation call threshold

3. **Imbalanced results**:
   - Adjust `max_ratio` parameter
   - Consider dataset-specific weights

### Dataset-Specific Notes

- **Bespoke Manim**: Best quality, use all fields (question, title, narration)
- **ManimCodeGen**: May have markdown in responses, needs cleaning
- **Thanks Manim**: Variable quality, aggressive filtering recommended
- **ManimBench**: Pre-reviewed, minimal filtering needed