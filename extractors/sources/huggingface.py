"""HuggingFace dataset extractors."""

import logging
from typing import Iterator, Dict, Any, Optional
from pathlib import Path

from datasets import load_dataset
from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class BespokeManimExtractor(BaseExtractor):
    """Extractor for Bespoke Labs Manim dataset."""
    
    source_id = "bespoke_manim"
    source_name = "Bespoke Labs Manim Dataset"
    priority = 3  # Rich context, transcripts
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.dataset_name = self.config.get("dataset_name", "bespokelabs/bespoke-manim")
        self.split = self.config.get("split", "train")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 1000
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Bespoke Manim dataset."""
        try:
            logger.info(f"Loading HuggingFace dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name, split=self.split)
            
            for idx, item in enumerate(dataset):
                try:
                    # Field names based on actual dataset schema
                    description = str(item.get("question", ""))
                    code = str(item.get("python_code", ""))
                    
                    if not description or not code:
                        continue
                    
                    # Clean up code if needed
                    if code.startswith("```python"):
                        code = code[9:]
                    if code.endswith("```"):
                        code = code[:-3]
                    code = code.strip()
                    
                    yield {
                        "description": description,
                        "code": code,
                        "metadata": {
                            "transcript": item.get("transcript", ""),
                            "video_id": item.get("video_id", ""),
                            "row_index": idx
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing item {idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {self.dataset_name}: {e}")
            return


@register_extractor
class ThanhktManimExtractor(BaseExtractor):
    """Extractor for thanhkt/manim_code dataset."""
    
    source_id = "thanks_dataset"
    source_name = "ThanhKT Manim Code Dataset"
    priority = 2  # Large dataset
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.dataset_name = self.config.get("dataset_name", "thanhkt/manim_code")
        self.split = self.config.get("split", "train")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 4400
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from thanhkt dataset."""
        try:
            logger.info(f"Loading HuggingFace dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name, split=self.split)
            
            for idx, item in enumerate(dataset):
                try:
                    description = str(item.get("input", ""))
                    code = str(item.get("output", ""))
                    
                    if not description or not code:
                        continue
                    
                    # Clean up code
                    if code.startswith("```python"):
                        code = code[9:]
                    if code.endswith("```"):
                        code = code[:-3]
                    code = code.strip()
                    
                    # Remove literal \n from the beginning (common in this dataset)
                    if code.startswith('\\n'):
                        code = code[2:].lstrip()
                    
                    yield {
                        "description": description,
                        "code": code,
                        "metadata": {
                            "row_index": idx
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing item {idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {self.dataset_name}: {e}")
            return