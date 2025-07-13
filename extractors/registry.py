"""Registry for dynamically discovering and managing extractors."""

from typing import Dict, Type, List, Any
from pathlib import Path
import importlib
import logging

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """Registry for managing data source extractors."""
    
    def __init__(self):
        self._extractors: Dict[str, Type[BaseExtractor]] = {}
        self._instances: Dict[str, BaseExtractor] = {}
    
    def register(self, extractor_class: Type[BaseExtractor]) -> None:
        """Register an extractor class."""
        if not extractor_class.source_id:
            raise ValueError(f"Extractor {extractor_class.__name__} must have source_id")
        
        if extractor_class.source_id in self._extractors:
            logger.warning(f"Overwriting existing extractor for {extractor_class.source_id}")
        
        self._extractors[extractor_class.source_id] = extractor_class
        logger.info(f"Registered extractor: {extractor_class.source_id}")
    
    def get_extractor(self, source_id: str, config: Dict[str, Any] = None) -> BaseExtractor:
        """Get an extractor instance by source ID."""
        if source_id not in self._extractors:
            raise ValueError(f"No extractor registered for source: {source_id}")
        
        # Create new instance with config
        return self._extractors[source_id](config)
    
    def list_sources(self) -> List[str]:
        """List all registered source IDs."""
        return list(self._extractors.keys())
    
    def get_all_extractors(self, configs: Dict[str, Dict[str, Any]] = None) -> Dict[str, BaseExtractor]:
        """Get instances of all registered extractors."""
        configs = configs or {}
        extractors = {}
        
        for source_id, extractor_class in self._extractors.items():
            config = configs.get(source_id, {})
            extractors[source_id] = extractor_class(config)
        
        return extractors
    
    def auto_discover(self, package_path: Path = None) -> None:
        """
        Auto-discover extractors in the extractors/sources directory.
        
        Looks for Python files that contain classes inheriting from BaseExtractor.
        """
        if package_path is None:
            package_path = Path(__file__).parent / "sources"
        
        if not package_path.exists():
            logger.warning(f"Sources directory not found: {package_path}")
            return
        
        for py_file in package_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            module_name = f"extractors.sources.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
                
                # Find all BaseExtractor subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseExtractor) and 
                        attr is not BaseExtractor and
                        hasattr(attr, 'source_id') and 
                        attr.source_id):
                        # Only register if not already registered
                        if attr.source_id not in self._extractors:
                            self.register(attr)
                        
            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")


# Global registry instance
_registry = ExtractorRegistry()


def register_extractor(extractor_class: Type[BaseExtractor]) -> Type[BaseExtractor]:
    """Decorator to register an extractor class."""
    _registry.register(extractor_class)
    return extractor_class


def get_registry() -> ExtractorRegistry:
    """Get the global registry instance."""
    return _registry