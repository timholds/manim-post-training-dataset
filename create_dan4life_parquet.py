#!/usr/bin/env python3
"""
Create a clean parquet file with dan4life samples that pass rendering validation.
Only includes samples that successfully render after code fixes are applied.
"""

import json
import logging
from pathlib import Path
import pandas as pd
from extractors.sources.local import Dan4LifeAoC2024Extractor
from extractors.code_fixer import ManimCodeFixer
from extractors.rendering_validator import RenderingValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting dan4life dataset extraction and validation...")
    
    # Initialize extractor
    extractor = Dan4LifeAoC2024Extractor()
    
    # Initialize code fixer
    code_fixer = ManimCodeFixer(aggressive_mode=False)
    
    # Initialize rendering validator
    validator = RenderingValidator(timeout=30)
    
    # Extract all samples
    logger.info("Extracting samples from dan4life dataset...")
    samples = list(extractor.extract())
    logger.info(f"Extracted {len(samples)} samples")
    
    # Apply code fixes
    logger.info("Applying code fixes...")
    fixed_samples = []
    for sample in samples:
        try:
            fix_result = code_fixer.apply_fixes(sample)
            if fix_result.success:
                fixed_sample = sample.copy()
                fixed_sample['code'] = fix_result.fixed_code
                fixed_sample['fixes_applied'] = fix_result.fixes_applied
                fixed_samples.append(fixed_sample)
                logger.debug(f"Fixed sample with {len(fix_result.fixes_applied)} fixes: {fix_result.fixes_applied}")
            else:
                # Even if no fixes were needed, keep the sample
                fixed_sample = sample.copy()
                fixed_sample['fixes_applied'] = []
                fixed_samples.append(fixed_sample)
        except Exception as e:
            logger.warning(f"Error fixing sample Day {sample.get('metadata', {}).get('day', 'unknown')}: {e}")
            # Keep the original sample even if fixing failed
            fixed_sample = sample.copy()
            fixed_sample['fixes_applied'] = []
            fixed_samples.append(fixed_sample)
    
    logger.info(f"Processed {len(fixed_samples)} samples")
    
    # Validate rendering
    logger.info("Validating rendering for fixed samples...")
    passing_samples = []
    
    for i, sample in enumerate(fixed_samples):
        logger.info(f"Validating sample {i+1}/{len(fixed_samples)}: Day {sample.get('metadata', {}).get('day', 'unknown')}")
        
        success, validation_result = validator.validate_render(
            sample['code'],
            sample_id=f"dan4life_{i}"
        )
        
        if success:
            passing_samples.append(sample)
            logger.info(f"✓ Sample passed validation")
        else:
            logger.warning(f"✗ Sample failed validation: {validation_result.get('error', 'Unknown error')}")
    
    logger.info(f"\nValidation complete: {len(passing_samples)}/{len(fixed_samples)} samples passed")
    
    if not passing_samples:
        logger.error("No samples passed validation!")
        return
    
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
    
    # Show sample of data
    logger.info("\nFirst few samples:")
    print(df[['description', 'day', 'fixes_applied']].head())


if __name__ == "__main__":
    main()