"""Extractor package for handling multiple data sources."""

from .base import BaseExtractor
from .registry import ExtractorRegistry, register_extractor, get_registry

__all__ = ['BaseExtractor', 'ExtractorRegistry', 'register_extractor', 'get_registry']