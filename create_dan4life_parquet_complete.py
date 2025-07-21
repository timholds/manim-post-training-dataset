#!/usr/bin/env python3
"""
Create a complete parquet file with all dan4life samples that pass rendering validation.
Maintains the train/test split from the original dataset.
"""

import json
import logging
from pathlib import Path
import pandas as pd
from extractors.sources.local import Dan4LifeAoC2024Extractor
from extractors.code_fixer import ManimCodeFixer
from extractors.rendering_validator import BatchRenderValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting dan4life dataset extraction and validation...")
    
    # Initialize extractor
    extractor = Dan4LifeAoC2024Extractor()
    
    # Initialize code fixer
    code_fixer = ManimCodeFixer(aggressive_mode=False)
    
    # Extract all samples
    logger.info("Extracting samples from dan4life dataset...")
    samples = list(extractor.extract())
    logger.info(f"Extracted {len(samples)} samples")
    
    # Apply code fixes
    logger.info("Applying code fixes...")
    fixed_samples = []
    for i, sample in enumerate(samples):
        try:
            fix_result = code_fixer.apply_fixes(sample)
            fixed_sample = sample.copy()
            if fix_result.success and fix_result.fixes_applied:
                fixed_sample['code'] = fix_result.fixed_code
                fixed_sample['fixes_applied'] = fix_result.fixes_applied
                logger.info(f"Sample {i}: Applied {len(fix_result.fixes_applied)} fixes: {fix_result.fixes_applied}")
            else:
                fixed_sample['fixes_applied'] = []
            fixed_samples.append(fixed_sample)
        except Exception as e:
            logger.warning(f"Error fixing sample {i}: {e}")
            fixed_sample = sample.copy()
            fixed_sample['fixes_applied'] = []
            fixed_samples.append(fixed_sample)
    
    logger.info(f"Processed {len(fixed_samples)} samples")
    
    # Count fixes applied
    fixes_count = sum(1 for s in fixed_samples if s['fixes_applied'])
    logger.info(f"Applied fixes to {fixes_count} samples")
    
    # Since you confirmed all 24 videos render, we'll validate them for completeness
    logger.info("Validating rendering for fixed samples (parallel)...")
    render_validator = BatchRenderValidator(
        max_workers=4,
        save_failed_samples=False
    )
    
    # Prepare samples for batch validation
    validation_samples = [
        {
            'code': sample['code'],
            'description': sample['description'],
            'source': 'dan4life_aoc2024',
            'sample_id': f"dan4life_{i}",
            'metadata': sample.get('metadata', {})
        }
        for i, sample in enumerate(fixed_samples)
    ]
    
    # Run batch validation
    valid_samples, invalid_samples = render_validator.validate_dataset(
        validation_samples,
        progress_callback=lambda current, total: logger.info(f"Progress: {current}/{total}")
    )
    
    logger.info(f"\nValidation complete: {len(valid_samples)}/{len(fixed_samples)} samples passed")
    
    # Match valid samples back to fixed_samples to get all metadata
    valid_indices = {int(s['sample_id'].split('_')[1]) for s in valid_samples}
    passing_samples = [s for i, s in enumerate(fixed_samples) if i in valid_indices]
    
    # Create train/test split (90/10 as mentioned in the codebase)
    # Use deterministic split based on sample index for reproducibility
    train_samples = []
    test_samples = []
    
    for i, sample in enumerate(passing_samples):
        # Every 10th sample goes to test set
        if i % 10 == 9:
            test_samples.append(sample)
        else:
            train_samples.append(sample)
    
    logger.info(f"\nDataset split:")
    logger.info(f"   - Train samples: {len(train_samples)}")
    logger.info(f"   - Test samples: {len(test_samples)}")
    
    # Create DataFrames
    def samples_to_df(samples):
        df_data = []
        for sample in samples:
            df_data.append({
                'description': sample['description'],
                'code': sample['code'],
                'source': 'dan4life_aoc2024',
                'day': sample.get('metadata', {}).get('day'),
                'version': sample.get('metadata', {}).get('version'),
                'fixes_applied': ','.join(sample.get('fixes_applied', [])),
                'original_source': 'https://github.com/Dan4Life/AoC2024_Videos'
            })
        return pd.DataFrame(df_data)
    
    # Create separate parquet files
    if train_samples:
        train_df = samples_to_df(train_samples)
        train_path = Path("dan4life_aoc2024_train.parquet")
        train_df.to_parquet(train_path, index=False)
        logger.info(f"\n✅ Created {train_path}")
        logger.info(f"   - Train samples: {len(train_df)}")
        logger.info(f"   - File size: {train_path.stat().st_size / 1024:.1f} KB")
    
    if test_samples:
        test_df = samples_to_df(test_samples)
        test_path = Path("dan4life_aoc2024_test.parquet")
        test_df.to_parquet(test_path, index=False)
        logger.info(f"\n✅ Created {test_path}")
        logger.info(f"   - Test samples: {len(test_df)}")
        logger.info(f"   - File size: {test_path.stat().st_size / 1024:.1f} KB")
    
    # Also create combined file for convenience
    all_df = samples_to_df(passing_samples)
    all_df['split'] = ['test' if i % 10 == 9 else 'train' for i in range(len(all_df))]
    all_path = Path("dan4life_aoc2024_complete.parquet")
    all_df.to_parquet(all_path, index=False)
    
    logger.info(f"\n✅ Created {all_path} (combined)")
    logger.info(f"   - Total samples: {len(all_df)}")
    logger.info(f"   - File size: {all_path.stat().st_size / 1024:.1f} KB")
    
    # Print summary statistics
    logger.info("\nDataset summary:")
    logger.info(f"   - Unique days: {all_df['day'].nunique()}")
    logger.info(f"   - Code fixes applied: {all_df['fixes_applied'].str.len().gt(0).sum()} samples")
    
    # Show which fixes were applied
    all_fixes = []
    for fixes_str in all_df['fixes_applied']:
        if fixes_str:
            all_fixes.extend(fixes_str.split(','))
    
    if all_fixes:
        from collections import Counter
        fix_counts = Counter(all_fixes)
        logger.info("\nFixes applied:")
        for fix, count in fix_counts.most_common():
            logger.info(f"   - {fix}: {count} times")
    
    # Show sample of data
    logger.info("\nSample data:")
    print(all_df[['description', 'day', 'split', 'fixes_applied']].head(10))


if __name__ == "__main__":
    main()