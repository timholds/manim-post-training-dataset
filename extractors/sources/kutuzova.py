"""
Extractor for Kutuzova's Deep Learning That Works animations.
Source: https://github.com/sgalkina/animations
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import subprocess
import tempfile
import shutil

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class KutuzovaExtractor(BaseExtractor):
    """Extract Manim animations from Kutuzova's Jupyter notebooks."""
    
    source_id = "kutuzova"
    source_name = "Deep Learning That Works (Kutuzova)"
    priority = 3  # Medium-high priority for quality content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_url = self.config.get("repo_url", "https://github.com/sgalkina/animations.git")
        self.notebooks_dir = self.config.get("notebooks_dir", "notebooks")
        self.temp_dir = None
    
    def _clone_repository(self) -> Path:
        """Clone the repository to a temporary directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="kutuzova_")
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
    
    def _parse_notebook(self, notebook_path: Path) -> Iterator[Dict[str, Any]]:
        """Parse a Jupyter notebook and extract Manim code cells."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            cells = notebook.get('cells', [])
            current_description = None
            
            # Extract imports from early cells
            imports = []
            for i, cell in enumerate(cells[:5]):  # Check first 5 cells for imports
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Look for import statements
                    for line in source.split('\n'):
                        line = line.strip()
                        if (line.startswith('import ') or line.startswith('from ')) and line not in imports:
                            imports.append(line)
            
            # Create import block
            import_block = '\n'.join(imports) if imports else 'from manim import *'
            
            for cell in cells:
                if cell.get('cell_type') == 'markdown':
                    # Extract description from markdown cells
                    source = ''.join(cell.get('source', []))
                    if source.strip():
                        current_description = source.strip()
                
                elif cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Check if this is a Manim Scene cell
                    if 'class' in source and 'Scene' in source:
                        # Generate description based on class name and context
                        class_name = None
                        for line in source.split('\n'):
                            if 'class ' in line and 'Scene' in line:
                                class_name = line.split('class ')[1].split('(')[0].strip()
                                break
                        
                        if class_name:
                            # Create a description
                            if current_description:
                                description = f"Create a Manim animation for {current_description}"
                            else:
                                # Generate description from class name
                                words = []
                                current_word = []
                                for char in class_name:
                                    if char.isupper() and current_word:
                                        words.append(''.join(current_word))
                                        current_word = [char]
                                    else:
                                        current_word.append(char)
                                if current_word:
                                    words.append(''.join(current_word))
                                
                                topic = ' '.join(words).lower()
                                description = f"Create a Manim animation that visualizes {topic} concepts"
                            
                            # Check if code already has imports
                            has_imports = any(line.strip().startswith(('import ', 'from ')) for line in source.split('\n'))
                            
                            # Prepend imports if needed
                            if not has_imports:
                                final_code = f"{import_block}\n\n{source.strip()}"
                            else:
                                final_code = source.strip()
                            
                            yield {
                                "description": description,
                                "code": final_code,
                                "metadata": {
                                    "source_file": notebook_path.name,
                                    "class_name": class_name,
                                    "topic": "deep_learning",
                                    "imports_added": not has_imports
                                }
                            }
                            
                            # Reset description after use
                            current_description = None
                            
        except Exception as e:
            logger.error(f"Error parsing notebook {notebook_path}: {e}")
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Kutuzova's notebooks."""
        repo_path = self._clone_repository()
        notebooks_path = repo_path / self.notebooks_dir
        
        try:
            if not notebooks_path.exists():
                logger.warning(f"Notebooks directory not found: {notebooks_path}")
                return
            
            # Process all notebook files
            for notebook_file in notebooks_path.glob("*.ipynb"):
                logger.info(f"Processing notebook: {notebook_file.name}")
                yield from self._parse_notebook(notebook_file)
                
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    def estimate_sample_count(self) -> Optional[int]:
        """Estimate number of samples."""
        return 5  # Based on expected number of animations