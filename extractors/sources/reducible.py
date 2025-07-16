"""Reducible YouTube channel dataset extractor - ManimCE content only."""

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
    """Extractor for Reducible's YouTube channel manim animations - ManimCE only."""
    
    source_id = "reducible"
    source_name = "Reducible YouTube Channel (ManimCE)"
    priority = 3  # High quality educational content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_path = Path(self.config.get("repo_path", "data/Reducible"))
        if not self.repo_path.exists():
            logger.warning(f"Repository not found: {self.repo_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        # Based on analysis: 2022 directory + MarchingSquares from 2021
        return 150  # Reduced estimate for ManimCE content only
    
    def _is_manim_ce_file(self, file_path: Path) -> bool:
        """Check if a file uses ManimCE imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for ManimCE imports
            if "from manim import" in content:
                # Make sure it's not also importing ManimGL
                if "from manimlib" not in content:
                    return True
            
            return False
        except Exception as e:
            logger.debug(f"Error checking file {file_path}: {e}")
            return False
    
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
                    
                    # Common Scene base classes in Manim
                    scene_bases = ['Scene', 'MovingCameraScene', 'ThreeDScene', 
                                   'GraphScene', 'LinearTransformationScene', 
                                   'SampleSpaceScene', 'ZoomedScene']
                    
                    if any(base in scene_bases for base in base_names):
                        # Extract the class code
                        class_code = ast.get_source_segment(content, node)
                        if class_code:
                            scenes.append((node.name, class_code))
                        else:
                            # Fallback: extract manually
                            start_line = node.lineno - 1
                            end_line = node.end_lineno
                            lines = content.split('\n')[start_line:end_line]
                            class_code = '\n'.join(lines)
                            scenes.append((node.name, class_code))
            
        except Exception as e:
            logger.error(f"Error extracting scenes from {file_path}: {e}")
        
        return scenes
    
    def _clean_code(self, code: str, file_path: Path) -> str:
        """Clean and prepare code for training."""
        # Remove sys.path.insert workarounds
        lines = code.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for line in lines:
            if skip_next:
                skip_next = False
                continue
                
            if 'sys.path.insert' in line:
                skip_next = True
                continue
            
            # Skip import sys if it was only for path manipulation
            if line.strip() == 'import sys':
                # Check if sys is used elsewhere
                remaining_code = '\n'.join(lines[lines.index(line)+1:])
                if 'sys.' not in remaining_code or only_sys_path_usage(remaining_code):
                    continue
            
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
        
        # Add necessary imports if not present
        if 'from manim import *' not in code:
            code = 'from manim import *\n\n' + code
        
        # Handle local imports from Reducible's common modules
        code = code.replace('from reducible_colors import *', 
                          '# Note: Custom colors from Reducible theme')
        code = code.replace('from functions import *', 
                          '# Note: Helper functions from Reducible')
        code = code.replace('from classes import *', 
                          '# Note: Custom classes from Reducible')
        
        return code.strip()
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract ManimCE samples from Reducible repository."""
        if not self.repo_path.exists():
            logger.error(f"Repository path does not exist: {self.repo_path}")
            return
        
        # Only look in directories with ManimCE content
        manim_ce_dirs = [
            self.repo_path / "2022",  # All 2022 content uses ManimCE
            self.repo_path / "2021" / "MarchingSquares"  # This specific 2021 video uses ManimCE
        ]
        
        total_scenes = 0
        
        for search_dir in manim_ce_dirs:
            if not search_dir.exists():
                logger.warning(f"Directory not found: {search_dir}")
                continue
                
            # Find all Python files
            py_files = list(search_dir.rglob("*.py"))
            
            for py_file in py_files:
                # Skip test files, setup files, and non-scene files
                if any(skip in py_file.name.lower() for skip in 
                       ['test', 'setup', '__pycache__', 'config', 'utils', 'solver_utils', 'lz77']):
                    continue
                
                # Verify it's actually ManimCE
                if not self._is_manim_ce_file(py_file):
                    logger.debug(f"Skipping non-ManimCE file: {py_file}")
                    continue
                
                # Extract scenes from this file
                scenes = self._extract_scenes_from_file(py_file)
                
                for scene_name, scene_code in scenes:
                    total_scenes += 1
                    
                    # Clean the code
                    cleaned_code = self._clean_code(scene_code, py_file)
                    
                    # Generate description based on file path and scene name
                    relative_path = py_file.relative_to(self.repo_path)
                    video_topic = str(relative_path.parts[1]) if len(relative_path.parts) > 1 else "General"
                    
                    description = f"Animation of {scene_name} from Reducible's {video_topic} video"
                    
                    # Special handling for known topics
                    topic_descriptions = {
                        "PageRank": "PageRank algorithm visualization",
                        "PNGvsQOI": "PNG vs QOI image compression comparison",
                        "JPEGImageCompression": "JPEG image compression visualization",
                        "TSPProblem": "Traveling Salesman Problem visualization",
                        "MarchingSquares": "Marching Squares algorithm visualization"
                    }
                    
                    if video_topic in topic_descriptions:
                        description = f"{topic_descriptions[video_topic]} - {scene_name}"
                    
                    yield {
                        "description": description,
                        "code": cleaned_code,
                        "source": self.source_id,
                        "metadata": {
                            "file": str(relative_path),
                            "scene_name": scene_name,
                            "year": relative_path.parts[0] if relative_path.parts else "unknown",
                            "topic": video_topic
                        }
                    }
        
        logger.info(f"Extracted {total_scenes} ManimCE scenes from Reducible")


def only_sys_path_usage(code: str) -> bool:
    """Check if sys is only used for path manipulation."""
    sys_usages = re.findall(r'sys\.\w+', code)
    return all('path' in usage for usage in sys_usages)