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

## Default Behavior of prepare_data.py

When you run `./prepare_data.py` with no arguments, it:
1. Processes ALL available data sources (11 sources)
2. Quality validation is DISABLED by default (per quality_config.json)
3. Performs deduplication (removes exact code matches and >95% similar pairs)
4. Creates 90/10 train/test split
5. Saves output to `data_formatted_v2/` directory
6. Generates `data_sources.png` visualization showing source statistics
7. Does NOT apply augmentation or rendering validation by default

## Quality Validation Modes

The `--quality-strict` flag controls validation strictness:

**Default (lenient mode)**:
- Only rejects samples with CRITICAL issues:
  - Syntax errors
  - No Scene class found
  - Empty construct method
  - Code too short (<50 chars)

**Strict mode (`--quality-strict`)**:
- Rejects samples with CRITICAL or HIGH issues:
  - All critical issues above, PLUS:
  - Missing imports
  - Placeholder content (TODO, FIXME, etc.)
  - Description too short (<20 chars)
  - Missing construct method

**Disabled (`--no-quality-validation`)**:
- Skips all quality checks
- Only basic validation (code/description exist)


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
- **SEVERE QUALITY ISSUES**: 47.2% of entries have mismatched code-description pairs
- Same code is used for completely unrelated descriptions
- **NEW APPROACH**: Use `thanks_dataset_raw` extractor which ignores descriptions entirely
  - Treats it as code-only dataset (like manim_ce_examples)
  - Deduplicates based on code alone (~2,441 unique code blocks from 4,400 entries)
  - LLM descriptions will be generated later
- Alternative: Use `thanks_dataset_cleaned` for pre-cleaned version (removes problematic entries)
- See `docs/THANKS_DATASET_FIX_REPORT.md` and `THANKS_DATASET_VERIFICATION.md` for full analysis
