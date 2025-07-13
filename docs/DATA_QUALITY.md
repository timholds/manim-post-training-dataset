# Data Quality Guidelines for Manim Dataset

## Overview

This document outlines the data quality standards and validation framework for the Manim fine-tuning dataset. Quality validation is critical to ensure the model learns from high-quality, correct, and useful examples.

## Quality Validation Framework

### Architecture

1. **Quality Validator** (`extractors/quality_validator.py`)
   - Validates individual samples during extraction
   - Configurable strictness levels
   - Detailed issue reporting

2. **Quality Configuration** (`quality_config.json`)
   - Global and source-specific settings
   - Customizable validation rules
   - Quality thresholds

3. **Integration Points**
   - Base extractor automatically applies validation
   - Per-source configuration overrides
   - Real-time filtering during extraction

### Validation Levels

#### Critical Issues (Reject in all modes)
- **Syntax errors**: Code must be valid Python
- **Missing Scene structure**: Must have a Scene class
- **Empty construct method**: Scene must have implementation
- **Code too short**: Minimum 50 characters

#### High Issues (Reject in strict mode)
- **Missing imports**: Code must import Manim
- **Placeholder content**: No TODO, FIXME, or ellipsis
- **Incomplete code**: No implementation markers
- **Description too short**: Minimum 20-30 characters

#### Medium Issues (Warnings)
- **Missing animation methods**: Should use play(), wait(), etc.
- **Missing math objects**: Should create visual elements
- **Generic descriptions**: Should be specific
- **Description-code mismatch**: Should align

#### Low Issues (Informational)
- **Formatting issues**: Parentheses, capitalization
- **Class naming**: Should reflect content

## Source-Specific Guidelines

### High-Quality Sources (Strict Validation)
- **bespoke_manim**: Professional dataset, maintain standards
- **manimbench**: Curated examples, high quality expected
- **manim_ce_examples**: Official examples, exemplary code
- **reducible**: Educational content, clarity important

### Medium-Quality Sources (Lenient Validation)
- **vivek3141**: Older Manim style, auto-generated descriptions
- **thanks_dataset**: High error rate, needs cleanup
- **Other community sources**: Variable quality

## Quality Metrics

### Current Dataset Status (as of analysis)
- **Overall pass rate (strict)**: 71.6%
- **Overall pass rate (lenient)**: 79.2%
- **Best sources**: bespoke_manim (99.3%), manimbench (98.1%)
- **Problem sources**: thanks_dataset (54.1% strict pass rate)

### Target Metrics
- **Syntax error rate**: < 5%
- **Empty constructs**: < 1%
- **Missing imports**: < 10%
- **Animation presence**: > 80%
- **Math object presence**: > 70%

## Usage Instructions

### Running with Quality Validation

```bash
# Default (uses quality_config.json)
python prepare_data.py

# With custom config
python prepare_data.py --quality-config my_config.json

# Without validation (not recommended)
python prepare_data.py --quality-config none
```

### Analyzing Data Quality

```bash
# Analyze existing data
python analyze_data_quality.py --detailed

# Test validation impact
python test_quality_validation.py --all
```

### Configuration Examples

#### Strict validation for all sources:
```json
{
  "global_settings": {
    "enable_quality_validation": true,
    "quality_strict_mode": true
  }
}
```

#### Lenient for specific source:
```json
{
  "source_overrides": {
    "problematic_source": {
      "quality_strict_mode": false,
      "min_code_length": 50
    }
  }
}
```

## Best Practices for Extractors

1. **Generate Meaningful Descriptions**
   - Include mathematical concepts mentioned
   - Specify what the animation demonstrates
   - Avoid generic phrases

2. **Validate Code Completeness**
   - Ensure all imports are present
   - Check for actual animation logic
   - No placeholder or stub code

3. **Test Extraction**
   ```python
   # Test with validation
   extractor = registry.get_extractor("my_source", {
       "enable_quality_validation": true,
       "quality_strict_mode": true
   })
   
   for sample in extractor:
       print(f"Valid sample: {sample['description'][:50]}...")
   ```

4. **Monitor Quality Reports**
   - Check extractor logs for validation stats
   - Review filtered samples
   - Adjust source-specific rules as needed

## Troubleshooting

### High Rejection Rate
1. Check syntax errors in source data
2. Adjust min_code_length for short examples
3. Consider lenient mode for community sources

### Missing Animations
1. Ensure construct() method has content
2. Check for animation method calls
3. Verify math object creation

### Description Issues
1. Generate more specific descriptions
2. Include mathematical context
3. Avoid placeholders and TODOs

## Future Improvements

1. **LLM-Enhanced Descriptions**: Use GPT to improve auto-generated descriptions
2. **Automatic Code Fixes**: Fix common syntax errors
3. **Smart Deduplication**: Consider quality when deduplicating
4. **Quality Scoring**: Rank samples by quality score
5. **Progressive Validation**: Start lenient, increase strictness over time

## Summary

Data quality is crucial for training effective models. This framework provides:
- Automated quality validation during extraction
- Configurable rules per data source
- Detailed reporting and metrics
- Clear path to improvement

By maintaining high standards while being practical about source limitations, we can build a dataset that balances quality with quantity.