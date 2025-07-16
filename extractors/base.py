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
        
        # Quality validation settings
        self.enable_quality_validation = self.config.get("enable_quality_validation", True)
        self.quality_strict_mode = self.config.get("quality_strict_mode", True)
        self._quality_validator = None
        
        # Track extraction statistics
        self.extraction_stats = {
            "total_extracted": 0,
            "passed_validation": 0,
            "failed_basic_validation": 0,
            "failed_quality_validation": 0
        }
    
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
        # Get the code
        code = sample.get("code", "")
        
        # Animation timing fix disabled - we want to preserve original behavior
        # for training data authenticity
        pass
        
        return {
            "description": sample.get("description", ""),
            "code": code,
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
        # Lazy import to avoid circular dependency
        if self.enable_quality_validation and self._quality_validator is None:
            from .quality_validator import QualityValidator
            # Use full quality config if available, otherwise use extractor config
            quality_config = self.config.get("_quality_config", self.config)
            self._quality_validator = QualityValidator(
                strict_mode=self.quality_strict_mode,
                config=quality_config
            )
        
        for sample in self.extract():
            self.extraction_stats["total_extracted"] += 1
            transformed = self.transform_sample(sample)
            
            # Basic validation
            if not self.validate_sample(transformed):
                self.extraction_stats["failed_basic_validation"] += 1
                logger.debug(f"Skipped invalid sample from {self.source_id}")
                continue
            
            # Quality validation if enabled
            if self.enable_quality_validation:
                is_valid, issues = self._quality_validator.validate_sample(
                    transformed, 
                    source_id=self.source_id
                )
                if not is_valid:
                    self.extraction_stats["failed_quality_validation"] += 1
                    logger.debug(f"Quality validation failed for {self.source_id}: {issues[:2]}")
                    continue
            
            self.extraction_stats["passed_validation"] += 1
            yield transformed
        
        # Log quality report if validation was used
        if self.enable_quality_validation and self._quality_validator:
            logger.info(f"\n{self.source_id} Quality Report:\n{self._quality_validator.get_validation_report()}")