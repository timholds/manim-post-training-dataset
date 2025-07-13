# Data Pipeline Documentation

## Overview

The `prepare_data.py` script provides a plugin-based data pipeline for preparing multiple Manim datasets for fine-tuning code generation models. The architecture allows easy addition of new data sources without modifying core code.

## Architecture

### Plugin-Based System
- **Extractors**: Each data source is a self-contained plugin in `extractors/sources/`
- **Auto-Discovery**: New extractors are automatically discovered - no manual registration
- **Standardized Interface**: All extractors inherit from `BaseExtractor`
- **Priority System**: Sources have priorities (1-5) for intelligent deduplication

### Core Components
1. **prepare_data.py**: Main orchestration script (~300 lines)
2. **extractors/base.py**: Abstract base class defining the extractor interface
3. **extractors/registry.py**: Dynamic registry for extractor discovery
4. **extractors/sources/**: Individual extractor plugins

## Features

- **Multi-Dataset Support**: Automatically downloads and processes datasets from HuggingFace, Kaggle, and local files
- **Intelligent Deduplication**: Removes duplicates based on source priorities
- **Data Augmentation**: Optional prompt variations for increased diversity
- **Quality Validation**: Ensures code has proper Scene class structure
- **Flexible Processing**: Choose specific datasets or process all available
- **Detailed Reporting**: Generates statistics and deduplication reports

## Usage

### Basic Commands
```bash
# List all available data sources
python prepare_data.py --list-sources

# Process all available datasets with deduplication (default)
python prepare_data.py

# Process specific datasets
python prepare_data.py --sources bespoke_manim manimbench

# Enable augmentation (creates prompt variations)
python prepare_data.py --augmentation

# Custom output directory
python prepare_data.py --output-dir custom_data

# Disable deduplication (not recommended)
python prepare_data.py --no-deduplicate
```

### Available Data Sources

Run `python prepare_data.py --list-sources` to see all available sources:

1. **manimbench** (Priority: 4)
   - Highest quality dataset with reviewed descriptions
   - ~400 examples from Kaggle

2. **bespoke_manim** (Priority: 3)
   - 1,000 examples with rich context and transcripts
   - From HuggingFace

3. **thanks_dataset** (Priority: 2)
   - 4,400 examples with varied complexity
   - From HuggingFace

4. **dan4life_aoc2024** (Priority: 2)
   - Advent of Code 2024 animations
   - Local dataset

5. **szymon_ozog** (Priority: 2)
   - Educational Manim examples
   - Local dataset

## Adding New Data Sources

### Step 1: Create Extractor

Create a new file in `extractors/sources/your_source.py`:

```python
from typing import Iterator, Dict, Any, Optional
from ..base import BaseExtractor
from ..registry import register_extractor

@register_extractor
class YourSourceExtractor(BaseExtractor):
    # Required attributes
    source_id = "your_source"
    source_name = "Your Data Source Name"
    priority = 3  # 1-5, higher = keep when deduplicating
    
    def _validate_config(self) -> None:
        """Validate and set default configuration."""
        self.data_path = self.config.get("path", "default/path.jsonl")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 1000  # Or None if unknown
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from your source."""
        # Your extraction logic here
        for item in your_data_source:
            yield {
                "description": item["prompt"],
                "code": item["manim_code"],
                "metadata": {  # Optional
                    "source_file": "example.json",
                    "extra_info": "value"
                }
            }
```

### Step 2: Test Your Extractor

```python
# Test script
from extractors import get_registry

registry = get_registry()
registry.auto_discover()

# Test your extractor
extractor = registry.get_extractor("your_source")
for i, sample in enumerate(extractor):
    print(f"Sample {i}: {sample['description'][:50]}...")
    if i >= 5:
        break
```

### Step 3: Run Pipeline

Your source is now available:
```bash
python prepare_data.py --sources your_source
```

## Output Structure

```
output_dir/
├── train.json          # Training data (JSONL format)
├── test.json           # Test data (10% split)
├── dataset_stats.json  # Statistics and metadata
├── deduplication_report.json  # Deduplication details
└── removed_duplicates.json    # Examples of removed duplicates
```

### Data Format

Each line in train.json/test.json is a JSON object:
```json
{
  "conversations": [
    {"from": "system", "value": "You are a Manim code generator..."},
    {"from": "user", "value": "Create an animation that shows..."},
    {"from": "assistant", "value": "```python\n...\n```"}
  ],
  "source": "bespoke_manim"
}
```

## Deduplication Strategy

The pipeline uses an advanced deduplication system that considers both code and descriptions:

1. **Code-First Approach**: Deduplication primarily focuses on code similarity to preserve unique implementations
2. **Fast Hybrid Algorithm**:
   - **Hash Comparison**: Instant detection of exact code matches
   - **Token-Based Similarity**: Jaccard similarity for fuzzy matching
   - **Early Termination**: Skips comparison for obviously different code sizes
3. **Conservative Removal**: Only removes when very confident (>95% similarity for both code AND description, or exact code match)
4. **Priority-Based Selection**: When duplicates found, keeps the highest priority source

### Implementation Details

- **Code Normalization**: Removes comments and normalizes whitespace for comparison only (original code is preserved)
- **Performance**: ~600,000 comparisons per second using optimized token-based matching
- **Similarity Thresholds**:
  - Exact code match (normalized): Always remove
  - >95% code AND >95% description similarity: Remove
  - Everything else: Keep both to preserve diversity

### Source Priorities
- vivek3141_dl: 5 (deep learning focused, highest quality)
- manimbench: 4 (manually reviewed)
- bespoke_manim, manim_ce_examples, kutuzova: 3 (high quality)
- thanks_dataset, dan4life_aoc2024, szymon_ozog: 2 (good quality)
- Others: 1 (default)

## Configuration

### Extractor Configuration

Pass configuration when processing:
```python
# In your extractor
config = {
    "file": "custom/path/data.jsonl",
    "cache_dir": "/tmp/cache",
    "max_samples": 1000
}

# When running
python prepare_data.py --sources your_source --config your_source:config.json
```

### Global Options

- `--seed`: Random seed for reproducibility
- `--test-ratio`: Test set split ratio (default: 0.1)
- `--augmentation`: Enable prompt variations
- `--no-deduplicate`: Disable deduplication

## Best Practices

1. **Always Use Deduplication**: Ensures dataset quality
2. **Test New Extractors**: Validate output before full processing
3. **Set Appropriate Priority**: Based on data quality
4. **Include Metadata**: Helps with debugging and analysis
5. **Handle Errors Gracefully**: Log warnings, don't crash
6. **Validate Early**: Check data quality in the extractor

## Troubleshooting

### Common Issues

1. **Extractor Not Found**
   - Ensure file is in `extractors/sources/`
   - Check `@register_extractor` decorator
   - Verify `source_id` is set

2. **Import Errors**
   - Use relative imports: `from ..base import`
   - Ensure `__init__.py` exists in directories

3. **Empty Output**
   - Check validation logic in extractor
   - Verify data paths are correct
   - Look for logged warnings

### Debug Mode

```bash
# Verbose logging
python prepare_data.py --sources your_source --verbose

# Test single source
python prepare_data.py --sources your_source --output-dir test_output
```

## Performance

- Processes ~5,000 samples in ~5 seconds
- Memory efficient with iterator-based extraction
- Caches downloaded datasets to avoid re-downloading
- Parallel processing for multiple sources

## Migration from Old Pipeline

If you have custom scripts using the old `prepare_data_enhanced.py`:

1. Create an extractor for your custom data
2. Move extraction logic to the `extract()` method
3. Update scripts to use `prepare_data.py`
4. See `docs/migration_guide.md` for detailed instructions