"""Manim Community Edition documentation examples extractor."""

import json
import re
import logging
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from ..base import BaseExtractor
from ..registry import register_extractor
from ..constants import PLACEHOLDER_DESCRIPTION

logger = logging.getLogger(__name__)


@register_extractor
class ManimCEExamplesExtractor(BaseExtractor):
    """Extractor for official Manim CE documentation examples."""
    
    source_id = "manim_ce_examples"
    source_name = "Manim CE Documentation Examples"
    priority = 5  # Highest quality - official examples
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.url = self.config.get("url", "https://docs.manim.community/en/stable/examples.html")
        self.cache_file = Path(self.config.get("cache_file", "data_manim_ce_examples.jsonl"))
        self.use_cache = self.config.get("use_cache", True)
        self.skip_validation = self.config.get("skip_validation", True)  # Skip validation for samples without descriptions
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 27  # Based on initial scraping
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Manim CE documentation."""
        # If cache exists and we're using cache, read from it
        if self.use_cache and self.cache_file.exists():
            logger.info(f"Reading from cache: {self.cache_file}")
            yield from self._read_from_cache()
            return
        
        # Otherwise scrape from web
        logger.info(f"Scraping examples from: {self.url}")
        examples = self._scrape_examples()
        
        # Cache the results if we have a cache file configured
        if self.cache_file:
            self._save_to_cache(examples)
        
        # Yield the examples
        for example in examples:
            yield example
    
    def _scrape_examples(self) -> list[Dict[str, Any]]:
        """Scrape examples from the documentation website."""
        try:
            response = requests.get(self.url)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch examples: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all code examples
        code_blocks = soup.find_all('div', class_='highlight-python')
        logger.info(f"Found {len(code_blocks)} code blocks")
        
        examples = []
        
        # Map of section IDs to category names
        category_map = {
            "basic-concepts": "Basic Concepts",
            "animations": "Animations", 
            "plotting-with-manim": "Plotting with Manim",
            "special-camera-settings": "Special Camera Settings"
        }
        
        for code_block in code_blocks:
            # Extract the code
            pre_tag = code_block.find('pre')
            if not pre_tag:
                continue
                
            code_text = pre_tag.text.strip()
            
            # Skip if code is empty or too short
            if len(code_text) < 50:
                continue
                
            # Extract class name from the code
            class_match = re.search(r'class\s+(\w+)\s*\(', code_text)
            if not class_match:
                continue
                
            class_name = class_match.group(1)
            
            # Find the category by looking at parent sections
            category = "General"
            parent = code_block.parent
            while parent:
                if parent.name == 'section' and parent.get('id'):
                    section_id = parent.get('id')
                    category = category_map.get(section_id, category)
                    break
                parent = parent.parent
            
            # Create the sample with placeholder description
            # We'll generate descriptions later using LLM with code analysis
            example = {
                "description": f"{PLACEHOLDER_DESCRIPTION} - Source: manim_ce_examples, Class: {class_name}",
                "code": code_text,
                "metadata": {
                    "example_name": class_name,
                    "category": category,
                    "needs_description": True,
                    "original_url": self.url
                }
            }
            
            examples.append(example)
            logger.debug(f"Extracted example: {class_name} ({category})")
        
        logger.info(f"Total examples extracted: {len(examples)}")
        return examples
    
    def _read_from_cache(self) -> Iterator[Dict[str, Any]]:
        """Read examples from cache file."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    
                    # Extract from conversation format if needed
                    if "conversations" in item:
                        conversations = item.get("conversations", [])
                        if len(conversations) >= 3:
                            code = conversations[2].get("value", "")
                            # Remove markdown code blocks
                            code = re.sub(r'^```python\n', '', code)
                            code = re.sub(r'\n```$', '', code)
                            
                            # Extract placeholder description from user message
                            description = conversations[1].get("value", "")
                            
                            yield {
                                "description": description,
                                "code": code,
                                "metadata": item.get("metadata", {})
                            }
                    else:
                        yield item
                        
        except Exception as e:
            logger.error(f"Failed to read cache: {e}")
    
    def _save_to_cache(self, examples: list[Dict[str, Any]]) -> None:
        """Save examples to cache file in JSONL format."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    # Save in the conversation format for consistency
                    formatted = {
                        "conversations": [
                            {
                                "from": "system",
                                "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
                            },
                            {
                                "from": "user",
                                "value": f"{PLACEHOLDER_DESCRIPTION} - Source: manim_ce_examples"
                            },
                            {
                                "from": "assistant",
                                "value": f"```python\n{example['code']}\n```"
                            }
                        ],
                        "source": self.source_id,
                        "metadata": example.get("metadata", {})
                    }
                    f.write(json.dumps(formatted) + '\n')
            logger.info(f"Saved {len(examples)} examples to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Override validation to allow samples without descriptions.
        Per TRANSCRIPT_STRATEGY.md, we'll generate descriptions later.
        """
        if self.skip_validation:
            # Only validate that we have code
            return bool(sample.get("code") and len(sample["code"]) > 20)
        return super().validate_sample(sample)