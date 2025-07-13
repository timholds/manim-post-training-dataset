## Overview
Our overall goal is to construct a perfect manim fine-tuning dataset from several sources using a plugin-based architecture for easy extensibility.

## Development Environment Setup

- Run `source manim-env/bin/activate` and use `uv pip` to install packages

## Development Style
- When creating diagnostic or testing scripts, it's better and cleaner to just run the python command directly instead of creating a new file and clogging up the repo. 

- After we have accomplished our goals, we should update the PROJECT_PLAN.md file

## Data Pipeline Architecture

We use a plugin-based extractor system:

1. **Adding New Data Sources**: Create a file in `extractors/sources/` with your extractor class
2. **Base Class**: All extractors inherit from `BaseExtractor` 
3. **Auto-discovery**: New extractors are automatically discovered - no manual registration needed
4. **Deduplication**: Sources have priorities (1-5) to determine which to keep when duplicates are found

Example of adding a new source:
```python
# extractors/sources/my_source.py
@register_extractor
class MySourceExtractor(BaseExtractor):
    source_id = "my_source"
    priority = 3
    
    def extract(self):
        # yield samples
``` 


## Analyzing Large Datasets

When you need to analyze entire datasets, validate quality across thousands of samples, or debug deduplication issues that exceed your context limits:
- Use `gemini -p` with instructions from GEMINI.md
- This is especially useful for:
  - Analyzing the full training dataset (16,000+ samples)
  - Comparing datasets before/after deduplication
  - Validating code quality across all sources
  - Finding patterns in removed duplicates
  - Assessing dataset diversity and coverage

See GEMINI.md for specific commands and examples.


## Known Data Quality Issues

### thanks_dataset (thanhkt/manim_code)
- **SEVERE QUALITY ISSUES**: 47.3% of entries have mismatched code-description pairs
- Same code is used for completely unrelated descriptions
- Use `thanks_dataset_cleaned` extractor instead (removes problematic entries)
- See `docs/THANKS_DATASET_FIX_REPORT.md` for full analysis
