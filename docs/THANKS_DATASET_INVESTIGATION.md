# Thanks Dataset Investigation Report

## Summary

After thorough investigation, I can confirm that **the quality issues are inherent in the original `thanhkt/manim_code` dataset**, not caused by our processing.

## Evidence

### 1. Direct Verification
I loaded the raw dataset from HuggingFace and found:
- The same `ChemicalBondAnimation` code is paired with 3 different types of requests in the ORIGINAL data:
  - "Create a probability tree diagram" (3 times)
  - "Create pie slice fraction animations" (3 times)
  - Never actually paired with a chemical bond request!

### 2. Dataset Statistics (Original)
- Total samples: 4,400
- Unique codes: 2,432 (only 55.3% unique)
- Exact duplicates: 1,920 (43.6%)
- Mismatched pairs: 60 samples with different descriptions
- **Total problematic: 45.0% of the dataset**

### 3. Our Processing Is Correct
Our extractor simply:
- Takes the 'input' field as the description
- Takes the 'output' field as the code
- No transformation that could cause mismatches

## Root Cause

The dataset appears to have been constructed by:
1. Collecting Manim code samples from various sources
2. Collecting or generating description prompts separately  
3. **Incorrectly pairing them** - possibly through some flawed automated process

## Our Solution Was Appropriate

Our cleaning approach:
1. Identifies all codes that appear with multiple different descriptions
2. Removes ALL instances of these problematic codes
3. Keeps only codes that have a single, consistent description

This removes 47.3% of the dataset but ensures quality.

## Minor Issue Found

The dataset has two input formats:
- 42% have wrapper: "Generate accurate and correct ManimCE Python code... Here is the user's request: [actual request]"
- 58% are direct requests

Our extractor keeps the full text including wrapper. This is cosmetic and doesn't affect the core mismatch issue.

## Recommendations

1. **Continue using the cleaned dataset** - The quality issues are severe and inherent
2. **Report to dataset maintainer** - These issues should be fixed at the source
3. **Consider wrapper handling** - Optionally update extractor to strip the "Generate accurate..." wrapper for cleaner descriptions
4. **Document thoroughly** - Make sure users know about these quality issues

## Files Created

- `GEMINI_THANKS_DATASET_ANALYSIS.md` - Detailed analysis for Gemini review
- `thanks_dataset_final_analysis.json` - Comprehensive statistics
- `data/thanks_dataset_cleaned/` - Cleaned dataset (2,318 samples, down from 4,400)