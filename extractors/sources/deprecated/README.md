# Deprecated Extractors

This folder contains data source extractors that have been deprecated because they use ManimGL instead of ManimCE.

## Excluded Sources

### vivek3141.py
- **Reason**: 97% ManimGL code (35/36 animation files use `from manimlib.imports import *`)
- **Issue**: The extractor has a critical bug where it strips imports from classes and incorrectly adds ManimCE imports to ~40% of them, creating non-functional mixed code
- **Failure Rate**: 99.7% rendering failure when processed

### vivek3141_dl.py
- **Reason**: 100% ManimGL code
- **Source**: https://github.com/vivek3141/dl-visualization
- **Issue**: Uses old 3b1b manim version, incompatible with ManimCE

## Why These Are Preserved

These extractors are kept for reference in case someone wants to:
1. Convert ManimGL code to ManimCE properly
2. Study the differences between ManimGL and ManimCE
3. Create a separate ManimGL dataset

## Note

For a pure ManimCE fine-tuning dataset, these sources should NOT be used.