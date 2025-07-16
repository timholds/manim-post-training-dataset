"""Vivek3141 YouTube channel dataset extractor.

WARNING: This source is EXCLUDED in quality_config.json because:
1. The repository is 97% ManimGL code (35/36 animation files)
2. This extractor has a critical bug: it extracts classes without their imports,
   then incorrectly adds ManimCE imports to ~40% of them
3. This creates non-functional code mixing ManimGL syntax with ManimCE imports
4. Results in 99.7% rendering failure rate

To use this source, it would need to be rewritten to either:
- Preserve original ManimGL imports and convert everything to ManimCE
- Extract classes with their original imports intact
"""

import ast
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple
import re

from ..base import BaseExtractor
from ..registry import register_extractor
from ..utils import fix_missing_imports

logger = logging.getLogger(__name__)


@register_extractor
class Vivek3141Extractor(BaseExtractor):
    """Extractor for Vivek3141's YouTube channel manim animations."""
    
    source_id = "vivek3141"
    source_name = "Vivek3141 YouTube Channel"
    priority = 1  # Lower priority - high duplicate rate, old manim style
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_path = Path(self.config.get("repo_path", "videos"))
        if not self.repo_path.exists():
            logger.warning(f"Repository not found: {self.repo_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 300  # Estimated based on ~45 files with multiple scenes each
    
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
                    
                    # Check if it's a Scene class or has construct method
                    scene_bases = ['Scene', 'ThreeDScene', 'VectorScene', 'GraphScene', 
                                  'MovingCameraScene', 'ZoomedScene', 'LinearTransformationScene',
                                  'SpecialThreeDScene', 'PartScene']
                    
                    is_scene = any(base in scene_bases for base in base_names)
                    
                    # Also check if it has a construct method (manim pattern)
                    has_construct = any(
                        isinstance(item, ast.FunctionDef) and item.name == 'construct'
                        for item in node.body
                    )
                    
                    if is_scene or has_construct:
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
    
    def _generate_description(self, class_name: str, file_name: str, class_code: str) -> str:
        """Generate a description based on class name, file name, and code content."""
        # Topic mapping based on file names
        topic_map = {
            'gaussian': 'Gaussian integrals and probability',
            'prime': 'prime numbers and number theory',
            'complex_derivative': 'complex analysis and derivatives',
            'green_theorem': "Green's theorem and vector calculus",
            'maxwell': 'Maxwell equations and electromagnetism',
            'coin_flip': 'probability and coin flip analysis',
            'coupons': 'coupon collector problem',
            'tic_tac_toe': 'game theory and tic-tac-toe',
            'pacman': 'Pac-Man game mechanics',
            'navier_stokes': 'Navier-Stokes equations and fluid dynamics',
            'line_integral': 'line integrals in calculus',
            'bezier': 'Bezier curves and animations',
            'ai': 'artificial intelligence concepts',
            'rubiks': "Rubik's cube algorithms",
            'buffon': "Buffon's needle problem",
            'sorting': 'sorting algorithms visualization',
            'dft': 'discrete Fourier transform',
            'fft': 'fast Fourier transform',
            'basel': 'Basel problem and series',
            'mandelbrot': 'Mandelbrot set and fractals',
            'taylor': 'Taylor series expansion'
        }
        
        # Get base topic from file name
        file_stem = file_name.replace('.py', '').lower()
        topic = None
        for key, value in topic_map.items():
            if key in file_stem:
                topic = value
                break
        
        if not topic:
            # Try to infer from class name
            readable_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name).lower()
            if 'intro' in readable_name:
                topic = 'introduction or title sequence'
            elif 'part' in readable_name or 'chapter' in readable_name:
                topic = 'multi-part educational content'
            else:
                topic = 'mathematical visualization'
        
        # Check for specific mathematical concepts in code
        code_lower = class_code.lower()
        if 'derivative' in code_lower:
            topic += ' involving derivatives'
        elif 'integral' in code_lower:
            topic += ' involving integrals'
        elif 'matrix' in code_lower:
            topic += ' involving matrices'
        elif 'vector' in code_lower:
            topic += ' involving vectors'
        elif 'graph' in code_lower and 'axes' in code_lower:
            topic += ' with graphical visualization'
        
        # Generate final description
        readable_class = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
        
        if 'Part' in class_name and any(char.isdigit() for char in class_name):
            # Extract part number
            part_num = ''.join(filter(str.isdigit, class_name))
            return f"Create a Manim animation for {topic} - Part {part_num} ({readable_class})"
        else:
            return f"Create a Manim animation demonstrating {topic} ({readable_class})"
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Vivek3141 repository."""
        if not self.repo_path.exists():
            logger.error(f"Repository not found: {self.repo_path}")
            return
        
        # Find all Python files
        python_files = []
        
        # Root directory files
        python_files.extend(self.repo_path.glob('*.py'))
        
        # Subdirectories (like alg1/lineq/)
        python_files.extend(self.repo_path.rglob('*.py'))
        
        # Process each file
        for file_path in python_files:
            # Skip utility files and non-animation files
            skip_files = ['render_all.py', 'storage.py', '__init__.py', 'manim.py']
            if file_path.name in skip_files:
                continue
            
            # Skip directories we don't want
            if 'img' in file_path.parts or 'images' in file_path.parts or 'shaders' in file_path.parts:
                continue
            
            scenes = self._extract_scenes_from_file(file_path)
            
            for class_name, class_code in scenes:
                # Skip utility classes
                if class_name in ['PartScene', 'Introduction', 'Outro', 'Reference']:
                    continue
                
                # Fix missing imports (handle both old and new manim)
                if 'import' not in class_code:
                    # Check if it's likely old manim style
                    if any(old_style in class_code for old_style in ['TexMobject', 'TextMobject', 'VMobject']):
                        full_code = "from manimlib.imports import *\n\n" + class_code
                    else:
                        full_code = fix_missing_imports(class_code)
                else:
                    full_code = class_code
                
                # Generate description
                description = self._generate_description(
                    class_name, 
                    file_path.name,
                    class_code
                )
                
                yield {
                    "description": description,
                    "code": full_code,
                    "metadata": {
                        "source_file": str(file_path.relative_to(self.repo_path)),
                        "class_name": class_name,
                        "file_topic": file_path.stem,
                        "is_multi_part": 'Part' in class_name,
                        "needs_description_update": True  # Mark for LLM enhancement later
                    }
                }