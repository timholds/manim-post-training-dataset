"""
Extractor for Benjamin Hackl's Manim animations.
Sources:
- manim-with-ease: Tutorial notebooks and examples
- manim-content: Mathematical visualizations
"""

import ast
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple
import re

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class BenjaminHacklExtractor(BaseExtractor):
    """Extract Manim animations from Benjamin Hackl's tutorials and mathematical content."""
    
    source_id = "benjamin_hackl"
    source_name = "Benjamin Hackl Manim Tutorials"
    priority = 3  # High quality educational content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.manim_ease_path = Path(self.config.get("manim_ease_path", "data/manim-with-ease"))
        self.manim_content_path = Path(self.config.get("manim_content_path", "data/manim-content"))
        
        if not self.manim_ease_path.exists():
            logger.warning(f"manim-with-ease path not found: {self.manim_ease_path}")
        if not self.manim_content_path.exists():
            logger.warning(f"manim-content path not found: {self.manim_content_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 54  # ~31 from manim-with-ease + ~23 from manim-content
    
    def _parse_notebook(self, notebook_path: Path) -> Iterator[Dict[str, Any]]:
        """Parse a Jupyter notebook and extract Manim code cells."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            cells = notebook.get('cells', [])
            current_description = None
            
            # Extract episode number from filename
            episode_match = re.search(r'[Ee](\d+)', notebook_path.stem)
            episode_num = episode_match.group(1) if episode_match else None
            
            # Extract title from filename
            title_parts = notebook_path.stem.split('-', 1)
            episode_title = title_parts[1].strip() if len(title_parts) > 1 else notebook_path.stem
            episode_title = episode_title.replace('-', ' ').replace('_', ' ').title()
            
            for i, cell in enumerate(cells):
                if cell.get('cell_type') == 'markdown':
                    # Extract description from markdown cells
                    source = ''.join(cell.get('source', []))
                    if source.strip():
                        # Clean markdown formatting
                        source = source.strip()
                        # Remove headers
                        source = re.sub(r'^#+\s*', '', source, flags=re.MULTILINE)
                        current_description = source
                
                elif cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Check if this is a Manim code cell
                    # Look for %%manim magic command or Scene classes
                    if ('%%manim' in source or 
                        ('class' in source and 'Scene' in source)):
                        
                        # Extract class name
                        class_name = None
                        for line in source.split('\n'):
                            if 'class ' in line and 'Scene' in line:
                                class_name = line.split('class ')[1].split('(')[0].strip()
                                break
                        
                        if class_name or '%%manim' in source:
                            # Clean code - remove %%manim line
                            code_lines = source.split('\n')
                            if code_lines and '%%manim' in code_lines[0]:
                                code_lines = code_lines[1:]
                            code = '\n'.join(code_lines).strip()
                            
                            # Check if we need imports (same logic as in Python file processing)
                            if 'from manim import' not in code and 'import' not in code:
                                code = "from manim import *\n\n" + code
                            
                            # Generate description
                            if current_description:
                                description = f"Tutorial Episode {episode_num} - {episode_title}: {current_description}"
                            elif class_name:
                                # Convert CamelCase to readable
                                readable_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name).lower()
                                description = f"Tutorial Episode {episode_num} - {episode_title}: Create a Manim animation demonstrating {readable_name}"
                            else:
                                description = f"Tutorial Episode {episode_num} - {episode_title}: Manim animation example"
                            
                            yield {
                                "description": description,
                                "code": code,
                                "metadata": {
                                    "source_file": notebook_path.name,
                                    "class_name": class_name or "ManimScene",
                                    "episode": episode_num,
                                    "episode_title": episode_title,
                                    "type": "tutorial"
                                }
                            }
                            
                            # Reset description after use
                            current_description = None
                            
        except Exception as e:
            logger.error(f"Error parsing notebook {notebook_path}: {e}")
    
    def _extract_scenes_from_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """Extract Scene classes from a Python file."""
        scenes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Find all Scene classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Scene or any Scene-like class
                    base_names = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_names.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_names.append(base.attr)
                    
                    # Check if it's a Scene class
                    scene_bases = ['Scene', 'ThreeDScene', 'VectorScene', 'GraphScene', 
                                  'MovingCameraScene', 'ZoomedScene', 'LinearTransformationScene']
                    if any(base in scene_bases for base in base_names):
                        # Extract class code
                        class_start = node.lineno - 1
                        class_end = node.end_lineno
                        class_lines = content.split('\n')[class_start:class_end]
                        class_code = '\n'.join(class_lines)
                        
                        # Clean up indentation
                        min_indent = float('inf')
                        for line in class_lines:
                            if line.strip():
                                indent = len(line) - len(line.lstrip())
                                min_indent = min(min_indent, indent)
                        
                        if min_indent < float('inf'):
                            class_lines = [line[min_indent:] if len(line) > min_indent else line 
                                         for line in class_lines]
                            class_code = '\n'.join(class_lines)
                        
                        scenes.append((node.name, class_code))
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
        
        return scenes
    
    def _generate_description_from_class(self, class_name: str, file_path: Path) -> str:
        """Generate a description based on class name and file context."""
        # Extract topic from file name
        file_stem = file_path.stem
        
        # Handle date-prefixed files (e.g., "2022-04_partitions")
        topic_match = re.search(r'\d{4}-\d{2}_(.+)', file_stem)
        if topic_match:
            topic = topic_match.group(1).replace('_', ' ').replace('-', ' ')
        else:
            topic = file_stem.replace('_', ' ').replace('-', ' ')
        
        # Convert CamelCase class name to readable format
        readable_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name).lower()
        
        # Generate description based on known patterns
        if 'partition' in topic.lower():
            desc = f"Mathematical animation demonstrating {readable_name} related to integer partitions"
        elif 'gf' in topic.lower() or 'generating' in topic.lower():
            desc = f"Mathematical animation showing {readable_name} for generating functions"
        elif 'four' in topic.lower() and 'problems' in topic.lower():
            desc = f"Mathematical problem visualization: {readable_name}"
        else:
            desc = f"Benjamin Hackl's mathematical animation: {readable_name} from {topic}"
        
        return desc
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Benjamin Hackl's content."""
        
        # Process manim-with-ease tutorials
        if self.manim_ease_path.exists():
            logger.info(f"Processing manim-with-ease tutorials from {self.manim_ease_path}")
            
            # Process notebooks
            for notebook_file in self.manim_ease_path.glob("*.ipynb"):
                if notebook_file.name.startswith('.'):
                    continue
                logger.info(f"Processing notebook: {notebook_file.name}")
                yield from self._parse_notebook(notebook_file)
            
            # Process Python files
            for py_file in self.manim_ease_path.glob("*.py"):
                if py_file.name.startswith('_') or py_file.name.startswith('.'):
                    continue
                logger.info(f"Processing Python file: {py_file.name}")
                scenes = self._extract_scenes_from_file(py_file)
                
                for class_name, class_code in scenes:
                    # Check if we need imports
                    full_code = class_code
                    if 'from manim import' not in class_code and 'import' not in class_code:
                        full_code = "from manim import *\n\n" + class_code
                    
                    # Extract episode info from filename
                    episode_match = re.search(r'[Ee](\d+)', py_file.stem)
                    episode_num = episode_match.group(1) if episode_match else None
                    
                    # Generate description
                    readable_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name).lower()
                    if episode_num:
                        description = f"Tutorial Episode {episode_num}: Manim animation demonstrating {readable_name}"
                    else:
                        description = f"Manim tutorial: {readable_name}"
                    
                    yield {
                        "description": description,
                        "code": full_code,
                        "metadata": {
                            "source_file": py_file.name,
                            "class_name": class_name,
                            "episode": episode_num,
                            "type": "tutorial"
                        }
                    }
        
        # Process manim-content mathematical animations
        if self.manim_content_path.exists():
            logger.info(f"Processing manim-content from {self.manim_content_path}")
            
            for py_file in self.manim_content_path.glob("*.py"):
                if py_file.name.startswith('_') or py_file.name.startswith('.'):
                    continue
                logger.info(f"Processing Python file: {py_file.name}")
                scenes = self._extract_scenes_from_file(py_file)
                
                for class_name, class_code in scenes:
                    # Check if we need imports
                    full_code = class_code
                    if 'from manim import' not in class_code and 'import' not in class_code:
                        full_code = "from manim import *\n\n" + class_code
                    
                    # Generate description
                    description = self._generate_description_from_class(class_name, py_file)
                    
                    yield {
                        "description": description,
                        "code": full_code,
                        "metadata": {
                            "source_file": py_file.name,
                            "class_name": class_name,
                            "type": "mathematical",
                            "needs_description_update": True  # Mark for potential LLM enhancement
                        }
                    }