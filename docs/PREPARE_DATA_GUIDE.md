# Manim Dataset Pipeline Quick Reference

## Basic Usage

```bash
# Default: Process all sources, no quality validation, with deduplication
./prepare_data.py

# List available data sources
./prepare_data.py --list-sources

# Process specific sources only
./prepare_data.py --sources thanks_dataset manim_ce_examples

# Enable data augmentation (2-3x training samples)
./prepare_data.py --augmentation
```

## Data Augmentation

When `--augmentation` is enabled, each training sample gets 1-2 additional variations with rephrased prompts:
- Original: "Draw a circle"
- Variation 1: "Create a Manim animation that draws a circle"
- Variation 2: "Write Manim code to draw a circle"

This increases training data diversity without changing the code. Only applies to training set, not test set.

## Quality Validation Options

```bash
# Enable strict quality validation (recommended for production)
./prepare_data.py --quality-strict

# Disable all quality validation
./prepare_data.py --no-quality-validation

# Use custom quality config
./prepare_data.py --quality-config quality_config_lenient.json
```

### What Quality Validation Checks

**CRITICAL Issues (always rejected when validation enabled):**
- Syntax errors
- No Scene class found
- Empty construct method
- Code too short (<50 chars)

**HIGH Issues (rejected only in strict mode):**
- Missing imports
- Placeholder content (TODO, FIXME, etc.)
- Description too short (<20 chars)
- Missing construct method

**MEDIUM/LOW Issues (logged but not rejected):**
- Generic descriptions
- Missing animation methods
- Code-description misalignment

## Rendering Validation (NEW!)

```bash
# Enable rendering validation - only keep code that produces videos
./prepare_data.py --enable-rendering

# Customize rendering timeout (default: 30 seconds per sample)
./prepare_data.py --enable-rendering --rendering-timeout 60

# Disable auto-fixing during rendering
./prepare_data.py --enable-rendering --no-rendering-fixes

# Analyze rendering failures
./analyze_rendering_failures.py data_formatted_v2/rendering_validation_report.json
```

### What Rendering Validation Auto-Fixes

- Missing imports (`from manim import *`)
- Scene inheritance issues (`class MyScene():` → `class MyScene(Scene):`)
- Missing self in method signatures (`def construct():` → `def construct(self):`)
- Missing self in animation methods (`play(Create(c))` → `self.play(Create(c))`)
- Tab/space indentation issues

## Complete Pipeline Examples

```bash
# Production-ready dataset with all validations
./prepare_data.py --quality-strict --enable-rendering --augmentation

# Quick test on one problematic source
./prepare_data.py --sources thanks_dataset --enable-rendering --output-dir test_output

# Process without any validation (fastest, lowest quality)
./prepare_data.py --no-quality-validation --no-deduplicate

# Custom configuration with specific sources
./prepare_data.py --sources manim_ce_examples vivek3141_dl --quality-strict --enable-rendering
```

## Output Structure

```
data_formatted_v2/
├── train.json              # Training data (JSONL format)
├── test.json               # Test data (JSONL format)
├── dataset_stats.json      # Comprehensive statistics
├── deduplication_report.json     # (if deduplication enabled)
├── removed_duplicates.json       # (if deduplication enabled)
└── rendering_validation_report.json  # (if rendering enabled)
```

## Key Statistics to Check

After running, check `dataset_stats.json` for:
- `total_samples.raw` - Original count before processing
- `total_samples.after_deduplication` - After removing duplicates
- `total_samples.after_rendering_validation` - After rendering checks
- `rendering_validation.pass_rate` - Percentage that rendered successfully
- `rendering_validation.auto_fixed` - How many were fixed automatically

## Common Issues and Solutions

**Problem**: Too many samples rejected
- **Solution**: Use default mode (no `--quality-strict`) or `--no-quality-validation`

**Problem**: Rendering validation takes too long
- **Solution**: Process specific sources first, reduce timeout, or run on subset

**Problem**: Need to debug why samples fail
- **Solution**: Check the JSON report files for detailed error messages

## Data Source Priorities

When duplicates are found, we keep the highest priority source:
- Priority 5: `manim_ce_examples`, `vivek3141_dl` (official/highest quality)
- Priority 4: `manimbench`
- Priority 3: Most sources (`benjamin_hackl`, `bespoke_manim`, `kutuzova`, `reducible`)
- Priority 2: `dan4life_aoc2024`, `szymon_ozog`
- Priority 1: `thanks_dataset`, `vivek3141` (known issues)