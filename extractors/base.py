"""Base extractor interface for all data sources."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Abstract base class for all data extractors."""
    
    # Source identifier (must be unique)
    source_id: str = None
    
    # Human-readable name
    source_name: str = None
    
    # Priority for deduplication (higher = keep when duplicates found)
    priority: int = 1
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize extractor with optional configuration."""
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for this extractor."""
        pass
    
    @abstractmethod
    def extract(self) -> Iterator[Dict[str, Any]]:
        """
        Extract samples from the source.
        
        Yields:
            Dict with keys:
            - description: str
            - code: str  
            - metadata: Dict[str, Any] (optional)
        """
        pass
    
    @abstractmethod
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples (for progress tracking)."""
        pass
    
    def transform_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a raw sample into standardized format.
        Override this for custom transformations.
        """
        return {
            "description": sample.get("description", ""),
            "code": sample.get("code", ""),
            "source": self.source_id,
            "metadata": sample.get("metadata", {})
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample meets quality requirements.
        Override for custom validation logic.
        """
        # Basic validation
        if not sample.get("description") or not sample.get("code"):
            return False
        if len(sample["code"]) < 20 or len(sample["description"]) < 5:
            return False
        return True
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Make extractor iterable for convenience."""
        for sample in self.extract():
            transformed = self.transform_sample(sample)
            if self.validate_sample(transformed):
                yield transformed
            else:
                logger.debug(f"Skipped invalid sample from {self.source_id}")