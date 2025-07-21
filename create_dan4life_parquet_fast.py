#!/usr/bin/env python3
"""
Create a clean parquet file with dan4life samples that pass rendering validation.
Fast version using parallel validation.
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
            if fix_result.success and fix_result.fixes_applied:
                fixed_sample = sample.copy()
                fixed_sample['code'] = fix_result.fixed_code
                fixed_sample['fixes_applied'] = fix_result.fixes_applied
                fixed_samples.append(fixed_sample)
                logger.info(f"Sample {i}: Applied {len(fix_result.fixes_applied)} fixes: {fix_result.fixes_applied}")
            else:
                # Keep the original sample even if no fixes were needed
                fixed_sample = sample.copy()
                fixed_sample['fixes_applied'] = []
                fixed_samples.append(fixed_sample)
        except Exception as e:
            logger.warning(f"Error fixing sample {i}: {e}")
            # Keep the original sample even if fixing failed
            fixed_sample = sample.copy()
            fixed_sample['fixes_applied'] = []
            fixed_samples.append(fixed_sample)
    
    logger.info(f"Processed {len(fixed_samples)} samples")
    
    # Count fixes applied
    fixes_count = sum(1 for s in fixed_samples if s['fixes_applied'])
    logger.info(f"Applied fixes to {fixes_count} samples")
    
    # Validate rendering using batch validator
    logger.info("Validating rendering for fixed samples (parallel)...")
    render_validator = BatchRenderValidator(
        max_workers=4,  # Run 4 validations in parallel
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
    
    if not valid_samples:
        logger.error("No samples passed validation!")
        return
    
    # Match valid samples back to fixed_samples to get all metadata
    valid_indices = {int(s['sample_id'].split('_')[1]) for s in valid_samples}
    passing_samples = [s for i, s in enumerate(fixed_samples) if i in valid_indices]
    
    # Create DataFrame for parquet
    df_data = []
    for sample in passing_samples:
        df_data.append({
            'description': sample['description'],
            'code': sample['code'],
            'source': 'dan4life_aoc2024',
            'day': sample.get('metadata', {}).get('day'),
            'version': sample.get('metadata', {}).get('version'),
            'fixes_applied': ','.join(sample.get('fixes_applied', [])),
            'original_source': 'https://github.com/Dan4Life/AoC2024_Videos'
        })
    
    df = pd.DataFrame(df_data)
    
    # Save to parquet
    output_path = Path("dan4life_aoc2024_validated.parquet")
    df.to_parquet(output_path, index=False)
    
    logger.info(f"\n✅ Successfully created {output_path}")
    logger.info(f"   - Total samples: {len(df)}")
    logger.info(f"   - File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print summary statistics
    logger.info("\nDataset summary:")
    logger.info(f"   - Unique days: {df['day'].nunique()}")
    logger.info(f"   - Code fixes applied: {df['fixes_applied'].str.len().gt(0).sum()} samples")
    
    # Show which fixes were applied
    all_fixes = []
    for fixes_str in df['fixes_applied']:
        if fixes_str:
            all_fixes.extend(fixes_str.split(','))
    
    if all_fixes:
        from collections import Counter
        fix_counts = Counter(all_fixes)
        logger.info("\nFixes applied:")
        for fix, count in fix_counts.most_common():
            logger.info(f"   - {fix}: {count} times")
    
    # Show sample of data
    logger.info("\nFirst few samples:")
    print(df[['description', 'day', 'fixes_applied']].head())


if __name__ == "__main__":
    main()