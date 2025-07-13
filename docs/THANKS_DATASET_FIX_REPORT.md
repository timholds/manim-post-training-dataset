# Thanks Dataset Quality Issues - Investigation and Fix Report

## Executive Summary

We discovered severe data quality issues in the `thanhkt/manim_code` dataset where approximately **47.3% of entries had mismatched code-description pairs**. The same code was being used for completely unrelated descriptions, making nearly half the dataset unusable for training.

## Key Findings

### 1. Duplicate Analysis Results

From our analysis of the original dataset:
- **Total samples**: 4,400
- **Unique code blocks**: 2,441 (only 55.5% unique)
- **Codes with multiple different descriptions**: 1,025 (42% of unique codes!)

### 2. Examples of Mismatches

The mismatches were severe:
- A neural network animation code was used for descriptions about:
  - "animating_the_forward_pass.py"
  - "first_neural_network.py"
  - "example.py"
- A chemical bond animation code was paired with a description requesting a probability tree diagram
- An integral calculation code was paired with a description about animating colored shapes

### 3. Pattern of Issues

- Many entries had identical descriptions with identical code (pure duplicates)
- Many entries had the same code paired with completely unrelated descriptions
- Descriptions often started with "Generate accurate and correct ManimCE Python code for..." suggesting synthetic generation
- The code rarely matched what the description requested

### 4. Comparison with Other Datasets

All other datasets showed 100% code uniqueness:
- `vivek3141/manimbench`: 100% unique codes
- `bespoke_manim`: 100% unique codes
- `dan4life`: 100% unique codes
- `szymon_ozog`: 100% unique codes

## Root Cause Analysis

The issues suggest the dataset was constructed incorrectly, possibly by:
1. Taking code samples from various sources
2. Generating or collecting descriptions separately
3. Pairing them randomly or with flawed logic
4. Not validating that the code actually implements what the description requests

## Solution Implemented

### 1. Created Cleaning Script (`fix_thanks_dataset.py`)

The script:
- Identifies all codes that appear with multiple different descriptions
- Removes ALL entries using these problematic codes
- Keeps only entries where code is unique to one description

### 2. Cleaning Results

- **Original entries**: 4,395
- **Clean entries**: 2,318
- **Removed entries**: 2,077 (47.3%)
- **Problematic codes identified**: 1,025

### 3. Created New Extractor (`thanks_cleaned.py`)

- Uses the cleaned dataset instead of the original
- Higher priority than original but lower than high-quality sources
- Includes metadata indicating entries are from the cleaned version

### 4. Final Dataset Statistics

After preparing the dataset with all sources including cleaned thanks_dataset:
- Total samples after deduplication: 2,889
- thanks_dataset_cleaned contributed: 1,276 samples
- Further reduced from 2,318 due to cross-source deduplication

## Recommendations

1. **Use cleaned dataset**: Always use `thanks_dataset_cleaned` instead of the original
2. **Quality validation**: The cleaning only removes obvious mismatches; further quality validation may be needed
3. **Consider exclusion**: Given the severe issues, consider excluding this dataset entirely if training quality is paramount
4. **Report upstream**: These issues should be reported to the dataset maintainer

## Files Created/Modified

1. `analyze_code_reuse.py` - Initial analysis script
2. `find_true_mismatches.py` - Deep analysis of mismatches
3. `fix_thanks_dataset.py` - Cleaning script
4. `data/thanks_dataset_cleaned/` - Cleaned dataset directory
5. `extractors/sources/thanks_cleaned.py` - New extractor for cleaned data
6. Updated warning in `extractors/sources/huggingface.py`

## Conclusion

The thanks_dataset had severe quality issues with nearly half its entries containing mismatched code-description pairs. We successfully identified and removed these problematic entries, reducing the dataset size by 47.3% but significantly improving its quality. The cleaned dataset can now be used for training, though users should be aware of its history and potentially lower quality compared to other sources.