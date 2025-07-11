"""Local file dataset extractors."""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class Dan4LifeAoC2024Extractor(BaseExtractor):
    """Extractor for Dan4Life's Advent of Code 2024 dataset."""
    
    source_id = "dan4life_aoc2024"
    source_name = "Dan4Life AoC 2024 Videos"
    priority = 2  # Good quality but limited samples
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.file_path = Path(self.config.get("file", "data_dan4life/dan4life_aoc2024.jsonl"))
        if not self.file_path.exists():
            logger.warning(f"Data file not found: {self.file_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 24
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Dan4Life dataset."""
        if not self.file_path.exists():
            logger.error(f"Data file not found: {self.file_path}")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract from conversation format
                        conversations = item.get("conversations", [])
                        if len(conversations) >= 3:
                            description = conversations[1].get("value", "")
                            code = conversations[2].get("value", "")
                            
                            # Clean code from markdown if needed
                            if code.startswith("```python"):
                                code = code.split("```python")[1].split("```")[0].strip()
                            
                            yield {
                                "description": description,
                                "code": code,
                                "metadata": {
                                    "source_file": str(self.file_path),
                                    "line_number": line_num,
                                    "day": item.get("metadata", {}).get("day"),
                                    "version": item.get("metadata", {}).get("version")
                                }
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            return


@register_extractor
class SzymonOzogExtractor(BaseExtractor):
    """Extractor for Szymon Ozog's dataset."""
    
    source_id = "szymon_ozog"
    source_name = "Szymon Ozog Manim Examples"
    priority = 2
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.file_path = Path(self.config.get("file", "data_szymon_ozog/szymon_ozog_processed.jsonl"))
        if not self.file_path.exists():
            logger.warning(f"Data file not found: {self.file_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 29
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Szymon Ozog dataset."""
        if not self.file_path.exists():
            logger.error(f"Data file not found: {self.file_path}")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract from conversation format
                        conversations = item.get("conversations", [])
                        if len(conversations) >= 3:
                            description = conversations[1].get("value", "")
                            code = conversations[2].get("value", "")
                            
                            # Clean code from markdown if needed
                            if code.startswith("```python"):
                                code = code.split("```python")[1].split("```")[0].strip()
                            
                            yield {
                                "description": description,
                                "code": code,
                                "metadata": {
                                    "source_file": str(self.file_path),
                                    "line_number": line_num
                                }
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            return