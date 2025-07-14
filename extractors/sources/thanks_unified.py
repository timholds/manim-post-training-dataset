"""Unified extractor for thanhkt/manim_code dataset.

This combines the best features from all previous implementations:
- Ignores descriptions due to 47.2% mismatch rate
- Removes duplicate code blocks  
- Applies quality fixes (imports, syntax)
- Uses simple, maintainable approach
"""

import logging
from typing import Iterator, Dict, Any, Optional
from datasets import load_dataset

from ..base import BaseExtractor
from ..registry import register_extractor
from ..utils import fix_missing_imports, fix_code_syntax_issues
from ..constants import PLACEHOLDER_DESCRIPTION

logger = logging.getLogger(__name__)


@register_extractor
class ThanksManimExtractor(BaseExtractor):
    """Unified extractor for thanhkt/manim_code dataset (code-only).
    
    IMPORTANT: This dataset has severe quality issues:
    - 47.2% of entries have mismatched code-description pairs
    - ~50% of the dataset contains duplicate code
    - Many syntax errors and missing imports
    
    We ignore descriptions completely and treat this as a code-only dataset.
    Descriptions can be generated later using LLM analysis of the code.
    
    See docs/THANKS_DATASET_ANALYSIS.md for detailed analysis.
    """
    
    source_id = "thanks_dataset"
    source_name = "ThanhKT Manim Code Dataset (Unified)"
    priority = 1  # Lowest priority due to quality issues
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.dataset_name = self.config.get("dataset_name", "thanhkt/manim_code")
        self.split = self.config.get("split", "train")
        self.min_code_length = self.config.get("min_code_length", 50)
        # Note: We ignore any placeholder_description from config and use the standardized one
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        # Dataset has 4,400 total but only ~2,200 unique after deduplication
        return 2200
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract unique code samples from dataset, ignoring descriptions."""
        try:
            logger.info(f"Loading HuggingFace dataset: {self.dataset_name}")
            logger.warning("Ignoring descriptions due to 47.2% mismatch rate - treating as code-only")
            
            dataset = load_dataset(self.dataset_name, split=self.split)
            logger.info(f"Dataset loaded with {len(dataset)} examples")
            
            # Track unique codes to avoid duplicates within this source
            seen_codes = set()
            duplicates_skipped = 0
            too_short = 0
            total_processed = 0
            
            for idx, item in enumerate(dataset):
                total_processed += 1
                
                try:
                    # Extract code from 'output' field
                    code = str(item.get("output", "")).strip()
                    
                    # Skip if code is too short
                    if len(code) < self.min_code_length:
                        too_short += 1
                        continue
                    
                    # Skip exact duplicates
                    if code in seen_codes:
                        duplicates_skipped += 1
                        continue
                    
                    seen_codes.add(code)
                    
                    # Clean up code blocks
                    if code.startswith("```python"):
                        code = code[9:]
                    if code.endswith("```"):
                        code = code[:-3]
                    code = code.strip()
                    
                    # Fix literal \n at start (common in this dataset)
                    if code.startswith('\\n'):
                        code = code[2:].lstrip()
                    
                    # Apply comprehensive code fixes
                    code = fix_code_syntax_issues(code)
                    
                    yield {
                        "description": f"{PLACEHOLDER_DESCRIPTION} - Source: thanks_dataset",
                        "code": code,
                        "metadata": {
                            "needs_description": True,
                            "original_index": idx,
                            "source_note": "Code-only due to description quality issues"
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing item {idx}: {e}")
                    continue
            
            # Log final stats
            logger.info(f"Thanks dataset extraction complete:")
            logger.info(f"  Total processed: {total_processed}")
            logger.info(f"  Unique samples: {len(seen_codes)}")
            logger.info(f"  Duplicates skipped: {duplicates_skipped} ({duplicates_skipped/total_processed*100:.1f}% if total_processed > 0 else 0)")
            logger.info(f"  Too short: {too_short}")
                    
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {self.dataset_name}: {e}")
            return