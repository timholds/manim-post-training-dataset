"""
Extractor for xiaoxiae/videos repository.
Educational video content with sophisticated mathematical visualizations.
"""

import ast
import logging
import subprocess
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import re

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class XiaoxiaeVideosExtractor(BaseExtractor):
    """Extract educational mathematical animations from xiaoxiae/videos repository."""
    
    source_id = "xiaoxiae_videos"
    source_name = "xiaoxiae Educational Videos"
    priority = 4  # High quality educational content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_url = self.config.get("repo_url", "https://github.com/xiaoxiae/videos.git")
        self.repo_path = Path(self.config.get("repo_path", "xiaoxiae-videos"))
        self.cache_dir = Path(self.config.get("cache_dir", ".cache"))
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 120  # Based on 25+ video projects with 3-8 scenes each
    
    def _clone_or_update_repo(self) -> bool:
        """Clone or update the repository."""
        try:
            if self.repo_path.exists():
                logger.info(f"Repository already exists at {self.repo_path}")
                return True
            
            logger.info(f"Cloning {self.repo_url} to {self.repo_path}")
            result = subprocess.run(
                ["git", "clone", self.repo_url, str(self.repo_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info("Repository cloned successfully")
                return True
            else:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return False
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    
    def _get_video_directories(self) -> List[Path]:
        """Get list of video project directories to process."""
        if not self.repo_path.exists():
            return []
        
        video_dirs = []
        
        # Find numbered directories (01-xx pattern)
        for item in self.repo_path.iterdir():
            if item.is_dir():
                # Match numbered directories like 01-lopt, 22-delaunay, etc.
                if re.match(r'^\d{2}-', item.name):
                    video_dirs.append(item)
                # Also include special directories
                elif item.name.startswith('ksp-') and item.name != 'ksp-intro':
                    video_dirs.append(item)
        
        # Sort by directory name
        video_dirs.sort(key=lambda x: x.name)
        logger.info(f"Found {len(video_dirs)} video directories")
        
        return video_dirs
    
    def _read_description(self, video_dir: Path) -> str:
        """Read video description from DESCRIPTION.md or directory name."""
        desc_file = video_dir / "DESCRIPTION.md"
        if desc_file.exists():
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Extract first meaningful line
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            return line
            except Exception as e:
                logger.debug(f"Error reading description from {desc_file}: {e}")
        
        # Fall back to directory name
        dir_name = video_dir.name
        # Remove number prefix and convert to readable format
        if re.match(r'^\d{2}-', dir_name):
            topic = dir_name[3:]  # Remove "01-" prefix
        else:
            topic = dir_name
        
        # Convert dashes to spaces and title case
        topic = topic.replace('-', ' ').replace('_', ' ').title()
        return f"Educational video about {topic}"
    
    def _extract_scene_classes(self, scenes_file: Path) -> List[tuple]:
        """Extract Scene classes from scenes.py file."""
        scenes = []
        
        try:
            with open(scenes_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find Scene classes
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Scene
                    base_names = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_names.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_names.append(base.attr)
                    
                    if 'Scene' in base_names:
                        # Extract class code
                        class_start = node.lineno - 1
                        class_end = node.end_lineno if node.end_lineno else len(content.split('\n'))
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
                        
                        # Extract docstring for additional context
                        docstring = ast.get_docstring(node) or ""
                        
                        scenes.append((node.name, class_code, docstring))
                        logger.debug(f"Extracted Scene class: {node.name}")
            
        except Exception as e:
            logger.error(f"Error extracting scenes from {scenes_file}: {e}")
        
        return scenes
    
    def _read_utilities(self, video_dir: Path) -> str:
        """Read utilities.py file if it exists."""
        utilities_file = video_dir / "utilities.py"
        if utilities_file.exists():
            try:
                with open(utilities_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Filter out imports to avoid conflicts
                    lines = content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not (line.strip().startswith('from manim import') or 
                               line.strip().startswith('import manim')):
                            filtered_lines.append(line)
                    return '\n'.join(filtered_lines)
            except Exception as e:
                logger.debug(f"Error reading utilities from {utilities_file}: {e}")
        return ""
    
    def _bundle_code_with_utilities(self, scene_code: str, utilities: str) -> str:
        """Bundle Scene code with utilities if needed."""
        # Start with standard imports
        bundled_code = "from manim import *\n"
        
        # Add common imports that xiaoxiae uses
        bundled_code += "import numpy as np\n"
        bundled_code += "import networkx as nx\n"
        bundled_code += "from typing import *\n\n"
        
        # Add utilities if they exist
        if utilities.strip():
            bundled_code += "# === Utility functions ===\n"
            bundled_code += utilities + "\n\n"
        
        # Add the Scene class
        bundled_code += "# === Scene class ===\n"
        bundled_code += scene_code
        
        return bundled_code
    
    def _generate_description(self, class_name: str, video_description: str, docstring: str) -> str:
        """Generate description for the Scene class."""
        # Start with video context
        base_desc = f"Educational animation from xiaoxiae's videos: {video_description}"
        
        # Add class-specific context
        class_context = ""
        if docstring:
            class_context = f" - {docstring.strip()}"
        elif class_name.lower() in ['intro', 'introduction']:
            class_context = " - Introduction to the concept"
        elif class_name.lower() in ['example', 'examples']:
            class_context = " - Example demonstration"
        elif class_name.lower() in ['theorem', 'proof']:
            class_context = " - Mathematical theorem and proof"
        elif class_name.lower() in ['outro', 'conclusion']:
            class_context = " - Summary and conclusion"
        elif class_name.lower() == 'thumbnail':
            class_context = " - Thumbnail visualization"
        else:
            # Convert CamelCase to readable format
            readable_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name).lower()
            class_context = f" - {readable_name}"
        
        return base_desc + class_context
    
    def _get_video_topic(self, video_dir: Path) -> str:
        """Extract topic from video directory name."""
        dir_name = video_dir.name
        if re.match(r'^\d{2}-', dir_name):
            topic = dir_name[3:]  # Remove "01-" prefix
        else:
            topic = dir_name
        return topic.replace('-', '_').replace(' ', '_')
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from xiaoxiae/videos repository."""
        # Clone repository if needed
        if not self._clone_or_update_repo():
            logger.error("Failed to clone repository")
            return
        
        # Get video directories
        video_dirs = self._get_video_directories()
        if not video_dirs:
            logger.error("No video directories found")
            return
        
        total_scenes = 0
        
        # Process each video directory
        for video_dir in video_dirs:
            logger.info(f"Processing video directory: {video_dir.name}")
            
            # Look for scenes.py file
            scenes_file = video_dir / "scenes.py"
            if not scenes_file.exists():
                logger.debug(f"No scenes.py found in {video_dir.name}")
                continue
            
            # Read video description
            video_description = self._read_description(video_dir)
            
            # Read utilities if they exist
            utilities = self._read_utilities(video_dir)
            
            # Extract scenes
            scenes = self._extract_scene_classes(scenes_file)
            
            for class_name, scene_code, docstring in scenes:
                # Bundle with utilities
                bundled_code = self._bundle_code_with_utilities(scene_code, utilities)
                
                # Generate description
                description = self._generate_description(class_name, video_description, docstring)
                
                # Get topic for metadata
                topic = self._get_video_topic(video_dir)
                
                yield {
                    "description": description,
                    "code": bundled_code,
                    "metadata": {
                        "class_name": class_name,
                        "video_directory": video_dir.name,
                        "topic": topic,
                        "has_docstring": bool(docstring),
                        "has_utilities": bool(utilities.strip()),
                        "complexity": "high",  # xiaoxiae's content is sophisticated
                        "educational_level": "advanced",
                        "content_type": "mathematical_visualization"
                    }
                }
                
                total_scenes += 1
            
            logger.info(f"Extracted {len(scenes)} scenes from {video_dir.name}")
        
        logger.info(f"Total scenes extracted: {total_scenes}")