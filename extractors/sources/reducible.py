"""Reducible YouTube channel dataset extractor."""

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
class ReducibleExtractor(BaseExtractor):
    """Extractor for Reducible's YouTube channel manim animations."""
    
    source_id = "reducible"
    source_name = "Reducible YouTube Channel"
    priority = 3  # High quality educational content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_path = Path(self.config.get("repo_path", "Reducible/Reducible"))
        if not self.repo_path.exists():
            logger.warning(f"Repository not found: {self.repo_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 285  # Based on our analysis
    
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
    
    def _generate_description(self, class_name: str, file_path: Path) -> str:
        """Generate a description based on class name and file context."""
        # Extract topic from file path
        parts = file_path.parts
        year = None
        topic = None
        
        for i, part in enumerate(parts):
            if part in ['2019', '2020', '2021', '2022']:
                year = part
                if i + 1 < len(parts):
                    topic = parts[i + 1]
                break
        
        # Clean up topic name
        if topic:
            topic = topic.replace('_', ' ').replace('-', ' ').title()
        
        # Generate description
        desc_parts = []
        
        # Add topic context
        if topic:
            desc_parts.append(f"Create a Manim animation for {topic}")
        else:
            desc_parts.append("Create a Manim animation")
        
        # Add class name context
        # Convert CamelCase to readable format
        readable_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
        desc_parts.append(f"that demonstrates {readable_name}")
        
        # Add year context if available
        if year:
            desc_parts.append(f"(from Reducible's {year} video series)")
        
        return ' '.join(desc_parts) + '.'
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Reducible repository."""
        # Check if we have a pre-extracted JSONL file first
        jsonl_file = Path("data_reducible.jsonl")
        if jsonl_file.exists():
            logger.info(f"Using pre-extracted data from {jsonl_file}")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Extract from conversation format
                    if "conversations" in data and len(data["conversations"]) >= 3:
                        description = data["conversations"][1]["value"]
                        code = data["conversations"][2]["value"]
                        # Clean code from markdown
                        if code.startswith("```python"):
                            code = code.split("```python")[1].split("```")[0].strip()
                        
                        yield {
                            "description": description,
                            "code": code,
                            "source": "reducible",
                            "metadata": data.get("metadata", {})
                        }
            return
        
        if not self.repo_path.exists():
            logger.error(f"Repository not found: {self.repo_path}")
            return
        
        # Find all Python files in year directories
        year_dirs = ['2019', '2020', '2021', '2022']
        python_files = []
        
        for year in year_dirs:
            year_path = self.repo_path / year
            if year_path.exists():
                python_files.extend(year_path.rglob('*.py'))
        
        # Process each file
        for file_path in python_files:
            # Skip test files and __init__.py
            if 'test' in file_path.name.lower() or file_path.name == '__init__.py':
                continue
            
            # Skip manimlib directory
            if 'manimlib' in file_path.parts:
                continue
            
            scenes = self._extract_scenes_from_file(file_path)
            
            for class_name, class_code in scenes:
                # Check if we need imports
                full_code = class_code
                if 'from manim import' not in class_code and 'import' not in class_code:
                    # Add default imports
                    full_code = "from manim import *\n\n" + class_code
                
                # Generate description
                description = self._generate_description(class_name, file_path)
                
                yield {
                    "description": description,
                    "code": full_code,
                    "metadata": {
                        "source_file": str(file_path.relative_to(self.repo_path)),
                        "class_name": class_name,
                        "year": file_path.parts[-3] if len(file_path.parts) > 2 else None,
                        "topic": file_path.parts[-2] if len(file_path.parts) > 1 else None,
                        "needs_description_update": True  # Mark for LLM enhancement later
                    }
                }