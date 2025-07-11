# Current Dataset State

Last Updated: July 11, 2025

## Overview

This document provides the authoritative source for the current state of the Manim fine-tuning dataset. All statistics are derived from the latest data preparation run.

## Active Datasets

### Successfully Integrated
- **Thanks Dataset**: 4,395 samples collected → 2,413 unique (45.1% duplicates removed)
- **Bespoke Manim**: 1,000 samples collected → 1,000 unique (0% duplicates)
- **ManimBench**: 417 samples collected → 414 unique (0.7% duplicates removed)

### Removed Due to Duplication
- **ManimCodeGen**: Previously 1,622 samples → 100% were duplicates of other datasets

## Current Statistics (No Augmentation)

### Raw Data
- **Total samples collected**: 5,812
- **After deduplication**: 3,827 (34.2% reduction)
- **Unique descriptions**: 3,827

### Train/Test Split
- **Training set**: 3,445 samples
- **Test set**: 382 samples
- **Split ratio**: 90/10
- **Augmentation**: Disabled (1.0x factor)

### Source Distribution (Training Set)
| Dataset | Samples | Percentage |
|---------|---------|------------|
| Thanks Dataset | 2,167 | 62.9% |
| Bespoke Manim | 903 | 26.2% |
| ManimBench | 375 | 10.9% |

## Data Quality Metrics

### Deduplication Impact
- **Total duplicates removed**: 1,985
- **Cross-dataset duplicates**: Unknown (included in total)
- **Within-dataset duplicates**: Unknown (included in total)
- **Priority order**: ManimBench > Bespoke > Thanks

### Code Validation
- All samples have valid Python syntax
- All samples include proper Scene class structure
- Minimum code length: 20 characters
- Minimum description length: 5 characters

## Data Locations

### Primary Dataset
- **Location**: `data_formatted/`
- **Statistics**: `data_formatted/dataset_stats.json`
- **Training data**: `data_formatted/train.json`
- **Test data**: `data_formatted/test.json`

### Reports
- **Deduplication report**: `data_formatted/deduplication_report.json`
- **Removed examples**: `data_formatted/removed_duplicates.json`

## Output Format

Each sample follows this structure:
```json
{
  "conversations": [
    {"from": "system", "value": "You are a Manim code generator..."},
    {"from": "user", "value": "<animation description>"},
    {"from": "assistant", "value": "```python\n<manim code>\n```"}
  ],
  "source": "<dataset_name>"
}
```

## Next Steps

See [Development Roadmap](ROADMAP.md) for planned dataset additions that could add 200-500+ additional samples.

## Historical Data

For detailed deduplication history and the full list of originally considered datasets, see [DATASETS.md](../DATASETS.md) in the root directory.