# Migration to Registry System

## Why the Registry System?

As we scale to more datasets (Dan4Life, Szymon Ozog, Reducible, Kilacoda, etc.), individual extraction scripts become unmaintainable. The registry system provides:

1. **Unified Interface**: All extractors follow the same pattern
2. **Centralized Management**: Single place to register and configure datasets
3. **Shared Infrastructure**: Caching, LLM processing, metadata tracking
4. **Extensibility**: Easy to add new datasets by implementing `DatasetExtractor`

## Key Components

### 1. Base Extractor Class
```python
class DatasetExtractor(ABC):
    @abstractmethod
    def extract(self) -> List[Dict[str, Any]]:
        """Extract samples from the dataset source."""
        pass
```

### 2. Specialized Extractors
- `LocalCodeExtractor`: Base for GitHub repositories
- `Dan4LifeExtractor`: Adds AoC-specific metadata
- `SzymonOzogExtractor`: Handles multiple repos with YouTube mappings

### 3. LLM Description Generator
- Built-in caching by code hash
- Batch processing support
- Transcript integration ready

### 4. Dataset Registry
- Central registration point
- Unified extraction command
- Integrated LLM processing

## Migration Steps

### From Individual Scripts:
```bash
# Old approach
python extract_dan4life_data.py
python extract_szymon_ozog.py
python extract_reducible.py
# ... many scripts
```

### To Registry System:
```bash
# New approach - extract all
python dataset_registry.py

# Or specific datasets
python dataset_registry.py --datasets dan4life_aoc2024 szymon_ozog
```

## Adding New Datasets

### 1. Create Extractor
```python
class ReducibleExtractor(LocalCodeExtractor):
    def __init__(self):
        super().__init__(
            name="reducible",
            repo_path=Path("Reducible"),
            metadata_enricher=self._add_video_metadata
        )
    
    def _add_video_metadata(self, sample):
        # Add YouTube URLs, episode info, etc.
        return sample
```

### 2. Register in Setup
```python
def setup_registry():
    registry = DatasetRegistry()
    registry.register("reducible", ReducibleExtractor())
    return registry
```

### 3. Extract and Process
```bash
python dataset_registry.py --datasets reducible
```

## Integration with Pipeline

The registry outputs JSONL files compatible with `prepare_data_enhanced.py`:

```python
# In prepare_data_enhanced.py
"new_dataset": {
    "type": "local",
    "file": "data_extracted/processed_samples.jsonl",
    "description_field": "conversations[1].value",
    "code_field": "conversations[2].value",
    "expected_samples": 100
}
```

## Benefits Over Individual Scripts

1. **Code Reuse**: Common extraction logic in base classes
2. **Consistent Format**: All extractors produce the same output structure
3. **Metadata Standards**: Unified metadata schema across datasets
4. **Cache Sharing**: LLM descriptions cached across all datasets
5. **Batch Operations**: Extract/process multiple datasets in one command

## Example: Full Workflow

```bash
# 1. Extract all registered datasets
python dataset_registry.py --output-dir data_extracted

# 2. Generate LLM descriptions (with caching)
python generate_llm_descriptions.py \
    --input data_extracted/samples_for_llm.jsonl \
    --output data_extracted/samples_with_descriptions.jsonl

# 3. Run main pipeline
python prepare_data_enhanced.py \
    --datasets dan4life_aoc2024 szymon_ozog reducible
```

This approach scales cleanly as we add more datasets!