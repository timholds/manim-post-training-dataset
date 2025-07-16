"""
Extractor for Likey00/manim-data-structures repository.
Focuses on educational data structure visualizations with bundled dependencies.
"""

import ast
import logging
import subprocess
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import tempfile
import shutil

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class LikeyDataStructuresExtractor(BaseExtractor):
    """Extract data structure visualizations from Likey00/manim-data-structures."""
    
    source_id = "likey_data_structures"
    source_name = "Likey00 Data Structures"
    priority = 3  # High quality educational content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_url = self.config.get("repo_url", "https://github.com/Likey00/manim-data-structures.git")
        self.repo_path = Path(self.config.get("repo_path", "data/Likey00-manim-data-structures"))
        self.cache_dir = Path(self.config.get("cache_dir", ".cache"))
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 4  # Based on repository analysis
    
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
                timeout=60
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
    
    def _read_utility_files(self, bst_dir: Path) -> Dict[str, str]:
        """Read utility files and return their content."""
        utility_files = {}
        
        # Files to include as dependencies
        files_to_read = ["bst.py", "get_bst.py", "insert_bst.py"]
        
        for filename in files_to_read:
            file_path = bst_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        utility_files[filename] = content
                        logger.debug(f"Read utility file: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to read {filename}: {e}")
        
        return utility_files
    
    def _extract_scene_classes(self, visuals_file: Path) -> List[tuple]:
        """Extract Scene classes from bst-visuals.py file."""
        scenes = []
        
        try:
            with open(visuals_file, 'r', encoding='utf-8') as f:
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
                        
                        # Extract docstring for description
                        docstring = ast.get_docstring(node) or ""
                        
                        scenes.append((node.name, class_code, docstring))
                        logger.debug(f"Extracted Scene class: {node.name}")
            
        except Exception as e:
            logger.error(f"Error extracting scenes from {visuals_file}: {e}")
        
        return scenes
    
    def _bundle_code_with_dependencies(self, scene_code: str, utility_files: Dict[str, str]) -> str:
        """Bundle Scene code with its dependencies."""
        # Start with standard Manim import
        bundled_code = "from manim import *\nfrom random import sample\n\n"
        
        # Add utility code, removing their imports and converting to inline functions
        for filename, content in utility_files.items():
            # Special handling for bst.py to fix duplicate __init__ and incomplete delete
            if filename == "bst.py":
                content = self._fix_bst_code(content)
            
            # Remove imports from utility files
            lines = content.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip import statements
                if (line.strip().startswith('import ') or 
                    line.strip().startswith('from ') or
                    line.strip().startswith('#') or
                    not line.strip()):
                    if not line.strip():
                        filtered_lines.append(line)  # Keep empty lines for formatting
                    continue
                filtered_lines.append(line)
            
            # Add utility code with a comment header
            if filtered_lines:
                bundled_code += f"# === {filename} utility code ===\n"
                bundled_code += '\n'.join(filtered_lines) + '\n\n'
        
        # Add the Scene class
        bundled_code += "# === Scene class ===\n"
        bundled_code += scene_code
        
        return bundled_code
    
    def _fix_bst_code(self, content: str) -> str:
        """Fix the BST class issues: merge duplicate __init__ and remove incomplete delete method."""
        lines = content.split('\n')
        fixed_lines = []
        in_delete = False
        skip_first_init = False
        
        for i, line in enumerate(lines):
            # Skip the first parameterless __init__ method
            if "def __init__(self):" in line and "BST" in lines[i-2]:
                skip_first_init = True
                continue
            
            # Skip the body of first init
            if skip_first_init:
                if line.strip() == "self.root = None":
                    continue
                else:
                    skip_first_init = False
            
            # Fix the second __init__ to be the only one
            if "def __init__(self, keys):" in line:
                # Change it to accept optional keys parameter
                fixed_lines.append("    def __init__(self, keys=None):")
                fixed_lines.append("        self.root = None")
                fixed_lines.append("        if keys:")
                fixed_lines.append("            self.insert(keys)")
                # Skip the original body
                continue
            
            # Skip original body of second init
            if i > 0 and "def __init__(self, keys):" in lines[i-1]:
                continue
            if i > 1 and "def __init__(self, keys):" in lines[i-2] and line.strip().startswith("self."):
                continue
            
            # Remove incomplete delete method
            if "def delete(self, key):" in line:
                in_delete = True
                continue
            
            # Skip delete method body
            if in_delete:
                # Check if we've reached the end of the class or another method
                if line.strip() and not line.startswith(' '):
                    in_delete = False
                    fixed_lines.append(line)
                else:
                    continue
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _generate_description(self, class_name: str, docstring: str) -> str:
        """Generate description based on class name and docstring."""
        if docstring:
            return f"Data structure visualization: {docstring.strip()}"
        
        # Generate based on class name
        if "DrawOne" in class_name:
            return "Create a Manim animation that draws a single binary search tree with random numbers"
        elif "DrawMany" in class_name:
            return "Create a Manim animation showing transitions between multiple binary search trees"
        elif "InsertOne" in class_name:
            return "Create a Manim animation demonstrating the insertion of a single element into a BST"
        elif "InsertAll" in class_name:
            return "Create a Manim animation building a BST by progressively inserting elements"
        else:
            return f"Create a Manim animation for binary search tree operations: {class_name}"
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Likey00/manim-data-structures repository."""
        # Clone repository if needed
        if not self._clone_or_update_repo():
            logger.error("Failed to clone repository")
            return
        
        # Navigate to binary-search-tree directory
        bst_dir = self.repo_path / "binary-search-tree"
        if not bst_dir.exists():
            logger.error(f"Binary search tree directory not found: {bst_dir}")
            return
        
        # Read utility files
        utility_files = self._read_utility_files(bst_dir)
        if not utility_files:
            logger.warning("No utility files found - scenes may not work correctly")
        
        # Extract scenes from bst-visuals.py
        visuals_file = bst_dir / "bst-visuals.py"
        if not visuals_file.exists():
            logger.error(f"Main visuals file not found: {visuals_file}")
            return
        
        scenes = self._extract_scene_classes(visuals_file)
        logger.info(f"Found {len(scenes)} Scene classes")
        
        # Process each scene
        for class_name, scene_code, docstring in scenes:
            # Bundle with dependencies
            bundled_code = self._bundle_code_with_dependencies(scene_code, utility_files)
            
            # Generate description
            description = self._generate_description(class_name, docstring)
            
            yield {
                "description": description,
                "code": bundled_code,
                "metadata": {
                    "class_name": class_name,
                    "data_structure": "binary_search_tree",
                    "bundled_dependencies": list(utility_files.keys()),
                    "has_docstring": bool(docstring),
                    "complexity": "high",  # Due to custom utilities
                    "educational_focus": "data_structures"
                }
            }
        
        logger.info(f"Extracted {len(scenes)} data structure visualization samples")