# Integration Summary: LLM Support for prepare_data.py

## What We Added

### 1. LLM Description Generator (`extractors/llm_description_generator.py`)
- **SHA256-based caching** to avoid redundant LLM calls
- **Code feature analysis** (3D, LaTeX, visual elements, etc.)
- **Batch processing** support
- **Cache management** with statistics

### 2. Enhanced Pipeline (`prepare_data_with_llm.py`)
Extends the existing `prepare_data.py` with:
- `generate-descriptions` command for LLM processing
- `cache-stats` command to view cache usage
- Integration with gemini/claude backends

## How It Works with Existing System

The existing plugin-based extractor system remains unchanged. We add LLM support as a separate layer:

```bash
# Step 1: Extract data using existing system
python prepare_data.py prepare --sources dan4life_aoc2024 szymon_ozog

# Step 2: Generate descriptions for samples needing them
python prepare_data_with_llm.py generate-descriptions \
    --input data_formatted/train.json \
    --output data_formatted/train_with_descriptions.json \
    --llm gemini

# Step 3: View cache statistics
python prepare_data_with_llm.py cache-stats
```

## Key Improvements

### 1. Code Feature Analysis
```python
features = {
    "has_3d": any(x in code for x in ["ThreeDScene", "ThreeDAxes"]),
    "has_latex": "MathTex" in code,
    "visual_elements": ["circles", "squares", "arrows", ...]
}
```

### 2. Smart Caching
- Cache key includes: code + source + context + youtube_url
- Prevents redundant LLM calls across runs
- Tracks which model generated each description

### 3. Metadata Enhancement
Extractors can now add:
- `needs_description: True` - flags samples for LLM processing
- `youtube_url` - for transcript fetching
- `original_context` - e.g., "Advent of Code Day 22"
- `code_features` - analyzed visual elements

## Workflow Example

### For New Dataset (e.g., Reducible):

1. **Create Extractor** (using existing plugin system):
```python
# extractors/sources/reducible.py
@register_extractor("reducible")
class ReducibleExtractor(LocalExtractor):
    def transform_sample(self, sample):
        sample["metadata"]["youtube_url"] = self.get_video_url(sample)
        sample["metadata"]["needs_description"] = True
        return sample
```

2. **Extract Data**:
```bash
python prepare_data.py prepare --sources reducible
```

3. **Generate Descriptions**:
```bash
python prepare_data_with_llm.py generate-descriptions \
    --input data_formatted/train.json \
    --output data_formatted/train_final.json
```

## Benefits

1. **No changes to existing extractors** - LLM layer is separate
2. **Efficient caching** - descriptions generated once, reused
3. **Flexible** - can process any JSONL file, not tied to extraction
4. **Backwards compatible** - existing workflow unchanged

## Next Steps

1. Implement actual gemini/claude API calls in `generate_descriptions_with_gemini()`
2. Add transcript fetching for YouTube URLs
3. Consider adding this to existing extractors that need it:
   ```python
   sample["metadata"]["needs_description"] = True
   sample["metadata"]["youtube_url"] = "..."
   ```

This approach keeps the clean plugin architecture while adding powerful LLM capabilities!