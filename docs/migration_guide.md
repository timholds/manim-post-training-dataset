# Migration Guide: Plugin-Based Extractor Architecture

## Overview

The new plugin-based architecture makes it easy to add, modify, and manage data sources independently. Each source is now a self-contained plugin that can be developed and tested in isolation.

## Key Benefits

1. **Scalability**: Add new sources without modifying core code
2. **Maintainability**: Each source is isolated in its own file
3. **Testability**: Test extractors independently
4. **Flexibility**: Different sources can have completely different logic
5. **Clean Git History**: Changes to one source don't affect others

## Architecture Components

### 1. Base Extractor (`extractors/base.py`)
- Abstract base class defining the interface all extractors must follow
- Provides common functionality like validation and transformation
- Key methods: `extract()`, `validate_sample()`, `transform_sample()`

### 2. Registry (`extractors/registry.py`)
- Manages all extractors dynamically
- Auto-discovers extractors in the `sources/` directory
- No manual imports needed

### 3. Individual Extractors (`extractors/sources/*.py`)
- One file per data source
- Self-contained logic for that source
- Registered automatically via decorator

## Adding a New Data Source

### Step 1: Create Extractor File

Create a new file in `extractors/sources/your_source.py`:

```python
from typing import Iterator, Dict, Any, Optional
from ..base import BaseExtractor
from ..registry import register_extractor

@register_extractor
class YourSourceExtractor(BaseExtractor):
    # Required class attributes
    source_id = "your_source_id"
    source_name = "Human Readable Name"
    priority = 2  # 1-5, higher = keep when deduplicating
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        # Set default config values, validate paths, etc.
        self.data_path = self.config.get("path", "default/path")
    
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
                "metadata": {  # Optional metadata
                    "source_file": "example.json",
                    "custom_field": "value"
                }
            }
```

### Step 2: Test Your Extractor

```python
from extractors import get_registry

# Get your extractor
registry = get_registry()
registry.auto_discover()
extractor = registry.get_extractor("your_source_id")

# Test extraction
for i, sample in enumerate(extractor):
    print(f"Sample {i}: {sample['description'][:50]}...")
    if i > 5:
        break
```

### Step 3: Run Data Preparation

```bash
# Process only your new source
python prepare_data_v2.py --sources your_source_id

# Or process all sources including yours
python prepare_data_v2.py
```

## Migrating Existing Sources

### Example: Migrating a Local JSONL Source

Old way (in `prepare_data_enhanced.py`):
```python
DATASETS = {
    "my_dataset": {
        "type": "local",
        "file": "data/my_dataset.jsonl",
        "description_field": "prompt",
        "code_field": "code"
    }
}
```

New way (create `extractors/sources/my_dataset.py`):
```python
import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from ..base import BaseExtractor
from ..registry import register_extractor

@register_extractor
class MyDatasetExtractor(BaseExtractor):
    source_id = "my_dataset"
    source_name = "My Custom Dataset"
    priority = 2
    
    def _validate_config(self) -> None:
        self.file_path = Path(self.config.get("file", "data/my_dataset.jsonl"))
    
    def estimate_sample_count(self) -> Optional[int]:
        return 500  # Your estimate
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        with open(self.file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                yield {
                    "description": item["prompt"],
                    "code": item["code"]
                }
```

## Common Patterns

### 1. Handling Different File Formats

```python
def extract(self) -> Iterator[Dict[str, Any]]:
    if self.file_path.suffix == '.json':
        with open(self.file_path) as f:
            data = json.load(f)
            for item in data:
                yield self.process_item(item)
    
    elif self.file_path.suffix == '.csv':
        df = pd.read_csv(self.file_path)
        for _, row in df.iterrows():
            yield {
                "description": row["prompt"],
                "code": row["code"]
            }
```

### 2. Handling API/Remote Sources

```python
def extract(self) -> Iterator[Dict[str, Any]]:
    response = requests.get(self.api_url)
    data = response.json()
    
    for item in data["results"]:
        yield {
            "description": item["question"],
            "code": self.clean_code(item["answer"])
        }
```

### 3. Custom Validation

```python
def validate_sample(self, sample: Dict[str, Any]) -> bool:
    # Call parent validation first
    if not super().validate_sample(sample):
        return False
    
    # Add custom validation
    code = sample["code"]
    if "manim" not in code.lower():
        return False
    
    if len(sample["description"].split()) < 3:
        return False
    
    return True
```

## Configuration

Pass configuration when creating extractors:

```python
config = {
    "file": "custom/path/to/data.jsonl",
    "api_key": "your_api_key",
    "max_samples": 1000
}

extractor = registry.get_extractor("your_source", config)
```

## Testing

Create unit tests for your extractor:

```python
# tests/test_your_extractor.py
import pytest
from extractors.sources.your_source import YourSourceExtractor

def test_extractor_basics():
    extractor = YourSourceExtractor()
    assert extractor.source_id == "your_source"
    assert extractor.priority > 0

def test_extraction():
    extractor = YourSourceExtractor({"file": "test_data.jsonl"})
    samples = list(extractor)
    assert len(samples) > 0
    assert all("description" in s for s in samples)
    assert all("code" in s for s in samples)
```

## Best Practices

1. **Keep extractors focused**: One extractor per data source
2. **Handle errors gracefully**: Log warnings, don't crash
3. **Validate early**: Check data quality in the extractor
4. **Document metadata**: Include useful metadata for debugging
5. **Test thoroughly**: Each extractor should have tests
6. **Use type hints**: Makes code more maintainable
7. **Follow naming conventions**: `source_id` should be lowercase with underscores

## Troubleshooting

### Extractor not found
- Check that the file is in `extractors/sources/`
- Ensure the `@register_extractor` decorator is used
- Verify `source_id` is set

### Import errors
- Make sure `__init__.py` exists in all directories
- Check relative imports (use `..base` not `extractors.base`)

### Data not loading
- Check file paths in `_validate_config()`
- Add logging to debug extraction process
- Verify data format matches expectations