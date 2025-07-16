"""
Extractor for Vivek3141's Deep Learning Visualization series.
Source: https://github.com/vivek3141/dl-visualization

WARNING: This source is EXCLUDED in quality_config.json because:
- 100% ManimGL code (uses old 3b1b manim version)
- Incompatible with ManimCE rendering
- See vivek3141.py for detailed explanation of issues
"""

import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import subprocess
import tempfile
import shutil
import re

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class Vivek3141DLExtractor(BaseExtractor):
    """Extract Manim animations from Vivek3141's Deep Learning visualization repository."""
    
    source_id = "vivek3141_dl"
    source_name = "Vivek3141 Deep Learning Visualizations"
    priority = 5  # Highest priority for specialized DL content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_url = self.config.get("repo_url", "https://github.com/vivek3141/dl-visualization.git")
        self.temp_dir = None
    
    def _clone_repository(self) -> Path:
        """Clone the repository to a temporary directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="vivek3141_dl_")
        logger.info(f"Cloning repository to {self.temp_dir}")
        
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, self.temp_dir],
                check=True,
                capture_output=True,
                text=True
            )
            return Path(self.temp_dir)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            raise
    
    def _generate_dl_description(self, filename: str, content: str) -> str:
        """Generate a deep learning focused description."""
        # Remove .py extension and clean filename
        name = filename.replace('.py', '').replace('_', ' ')
        
        # Common DL concepts mapping
        dl_concepts = {
            'gradient': 'gradient descent optimization',
            'backprop': 'backpropagation algorithm',
            'neural': 'neural network architecture',
            'convolution': 'convolutional neural networks',
            'attention': 'attention mechanism',
            'transformer': 'transformer architecture',
            'loss': 'loss function visualization',
            'activation': 'activation functions',
            'optimization': 'optimization algorithms',
            'regularization': 'regularization techniques',
            'dropout': 'dropout regularization',
            'batch': 'batch normalization',
            'lstm': 'LSTM networks',
            'rnn': 'recurrent neural networks',
            'gan': 'generative adversarial networks'
        }
        
        # Check content for specific concepts
        content_lower = content.lower()
        found_concepts = []
        
        for concept, full_name in dl_concepts.items():
            if concept in content_lower or concept in name.lower():
                found_concepts.append(full_name)
        
        if found_concepts:
            concept_str = found_concepts[0]
            if len(found_concepts) > 1:
                concept_str = ', '.join(found_concepts[:-1]) + f" and {found_concepts[-1]}"
            return f"Create a Manim animation that visualizes {concept_str} in deep learning"
        else:
            # Generic deep learning description
            words = name.split()
            topic = ' '.join(words).title()
            return f"Create a Manim animation explaining {topic} concepts in deep learning"
    
    def _is_manim_file(self, file_path: Path) -> bool:
        """Check if a Python file contains Manim animations."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check for Manim imports
            has_manim = any(pattern in content for pattern in [
                'from manim import',
                'import manim',
                'from manimlib',
                'import manimlib'
            ])
            
            # Check for Scene classes
            has_scene = 'class ' in content and 'Scene' in content
            
            # Also check for common DL visualization patterns
            has_dl_viz = any(pattern in content for pattern in [
                'neuron', 'layer', 'network', 'gradient', 'weight', 'bias'
            ])
            
            return has_manim and has_scene
            
        except Exception as e:
            logger.debug(f"Could not read file {file_path}: {e}")
            return False
    
    def _extract_from_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Extract animation from a single Python file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Skip if not a Manim file
            if not self._is_manim_file(file_path):
                return
            
            # Generate specialized DL description
            description = self._generate_dl_description(file_path.name, content)
            
            # Extract specific DL concepts for metadata
            dl_topics = []
            content_lower = content.lower()
            
            if 'gradient' in content_lower:
                dl_topics.append('optimization')
            if 'backprop' in content_lower:
                dl_topics.append('backpropagation')
            if 'convol' in content_lower:
                dl_topics.append('cnn')
            if 'neural' in content_lower or 'network' in content_lower:
                dl_topics.append('neural_networks')
            if 'transformer' in content_lower or 'attention' in content_lower:
                dl_topics.append('transformers')
            
            yield {
                "description": description,
                "code": content.strip(),
                "metadata": {
                    "source_file": file_path.name,
                    "topic": "deep_learning",
                    "subtopics": dl_topics,
                    "specialized": True,
                    "playlist": "Visualizing Deep Learning"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Vivek3141's DL visualization repository."""
        repo_path = self._clone_repository()
        
        try:
            # Find all Python files
            python_files = list(repo_path.rglob("*.py"))
            
            logger.info(f"Found {len(python_files)} Python files in DL visualization repo")
            
            # Process each file
            for file_path in python_files:
                # Skip setup files
                if file_path.name in ['__init__.py', 'setup.py', 'test.py']:
                    continue
                    
                logger.debug(f"Processing DL file: {file_path.name}")
                yield from self._extract_from_file(file_path)
                
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    def estimate_sample_count(self) -> Optional[int]:
        """Estimate number of samples."""
        return 5  # Based on expected specialized DL animations