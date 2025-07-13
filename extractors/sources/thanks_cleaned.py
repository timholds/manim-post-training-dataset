"""Cleaned version of thanhkt dataset extractor."""

import json
import logging
from typing import Iterator, Dict, Any, Optional
from pathlib import Path

from ..base import BaseExtractor
from ..registry import register_extractor
from ..utils import fix_missing_imports

logger = logging.getLogger(__name__)


@register_extractor
class ThanksCleanedExtractor(BaseExtractor):
    """Extractor for cleaned version of thanhkt/manim_code dataset."""
    
    source_id = "thanks_dataset_cleaned"
    source_name = "ThanhKT Manim Code Dataset (Cleaned)"
    priority = 2  # Higher than original but lower than high-quality sources
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.data_path = Path(self.config.get("data_path", "data/thanks_dataset_cleaned/train.json"))
        
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 2318  # Based on cleaning results
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from cleaned dataset."""
        if not self.data_path.exists():
            logger.error(f"Cleaned dataset not found at {self.data_path}")
            logger.info("Run fix_thanks_dataset.py to generate the cleaned dataset")
            return
            
        try:
            logger.info(f"Loading cleaned dataset from {self.data_path}")
            
            with open(self.data_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                        
                    try:
                        item = json.loads(line)
                        
                        # Extract from conversation format
                        description = None
                        code = None
                        
                        for conv in item.get('conversations', []):
                            if conv['from'] == 'user':
                                description = conv['value'].strip()
                            elif conv['from'] == 'assistant':
                                # Extract code from markdown block
                                value = conv['value']
                                if '```python' in value:
                                    start = value.find('```python') + len('```python')
                                    end = value.find('```', start)
                                    if end != -1:
                                        code = value[start:end].strip()
                        
                        if not description or not code:
                            continue
                        
                        # Fix missing imports
                        code = fix_missing_imports(code)
                        
                        yield {
                            "description": description,
                            "code": code,
                            "metadata": {
                                "cleaned": True,
                                "line_number": line_num
                            }
                        }
                        
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to load cleaned dataset from {self.data_path}: {e}")
            return