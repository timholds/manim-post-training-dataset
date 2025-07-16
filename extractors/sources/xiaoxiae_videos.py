"""
Extractor for xiaoxiae/videos repository.
Educational video content with sophisticated mathematical visualizations.
"""

import ast
import logging
import subprocess
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Set, Tuple
import re

from ..base import BaseExtractor
from ..registry import register_extractor
from ..asset_replacer import replace_assets_in_code

logger = logging.getLogger(__name__)


class FileInliner(ast.NodeTransformer):
    """AST transformer that inlines file read operations."""
    
    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.inlined_files = []
        self.failed_files = []
    
    def visit_Call(self, node):
        """Visit Call nodes to find file operations."""
        # Check if this is a .read() call on an open() call
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr in ['read', 'readlines'] and
            isinstance(node.func.value, ast.Call) and
            self._is_open_call(node.func.value)):
            
            filename = self._extract_filename(node.func.value)
            if filename:
                content_node = self._create_inlined_string(filename)
                if content_node:
                    # Handle .read().strip() or other chained calls
                    parent = getattr(node, '_parent', None)
                    if (parent and isinstance(parent, ast.Attribute) and 
                        parent.attr == 'strip' and 
                        isinstance(getattr(parent, '_parent', None), ast.Call)):
                        # Return stripped content
                        return ast.Call(
                            func=ast.Attribute(
                                value=content_node,
                                attr='strip',
                                ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[]
                        )
                    return content_node
        
        # Check for variable-based file operations like f.read() where f = open(...)
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr in ['read', 'readlines'] and
            isinstance(node.func.value, ast.Name)):
            
            var_name = node.func.value.id
            filename = self._resolve_file_variable(var_name)
            if filename:
                content_node = self._create_inlined_string(filename)
                if content_node:
                    return content_node
        
        # Continue visiting children
        self.generic_visit(node)
        return node
    
    def visit_With(self, node):
        """Handle with open(...) as f: patterns."""
        if len(node.items) == 1:
            item = node.items[0]
            if item.context_expr and self._is_open_call(item.context_expr):
                filename = self._extract_filename(item.context_expr)
                if filename and item.optional_vars:
                    # Look for f.read() in the body
                    var_name = item.optional_vars.id if isinstance(item.optional_vars, ast.Name) else None
                    if var_name and self._body_contains_read(node.body, var_name):
                        # Replace the entire with block with inlined content
                        return self._create_with_replacement(filename, node.body, var_name)
        
        self.generic_visit(node)
        return node
    
    def _is_open_call(self, node):
        """Check if node is an open() call."""
        return (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == 'open' and 
                len(node.args) >= 1)
    
    def _extract_filename(self, node):
        """Extract filename from open() call."""
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                return arg.value
            elif isinstance(arg, ast.Str):  # Python 3.7 compatibility
                return arg.s
            elif isinstance(arg, ast.Name):
                # Handle variable filename like: filename = "data.txt"; open(filename)
                return self._resolve_variable_filename(arg.id)
            elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div):
                # Handle Path operations like: Path("data") / "file.txt"
                return self._resolve_path_operation(arg)
        return None
    
    def _resolve_variable_filename(self, var_name):
        """Try to resolve a variable filename from the AST context."""
        # This is a simplified approach - in a full implementation, we'd need
        # to track variable assignments throughout the AST
        logger.debug(f"Variable filename '{var_name}' detected - manual resolution needed")
        return None
    
    def _resolve_path_operation(self, binop_node):
        """Resolve Path operations like Path("data") / "file.txt"."""
        try:
            if isinstance(binop_node.left, ast.Call):
                # Check if it's Path("data") / "file.txt"
                if (isinstance(binop_node.left.func, ast.Name) and 
                    binop_node.left.func.id == 'Path' and
                    binop_node.left.args and
                    isinstance(binop_node.left.args[0], (ast.Constant, ast.Str))):
                    
                    left_part = (binop_node.left.args[0].value if isinstance(binop_node.left.args[0], ast.Constant) 
                                else binop_node.left.args[0].s)
                    
                    if isinstance(binop_node.right, (ast.Constant, ast.Str)):
                        right_part = (binop_node.right.value if isinstance(binop_node.right, ast.Constant) 
                                     else binop_node.right.s)
                        
                        return f"{left_part}/{right_part}"
        except Exception as e:
            logger.debug(f"Failed to resolve path operation: {e}")
        return None
    
    def _resolve_file_variable(self, var_name):
        """Try to resolve a file variable from previous assignments."""
        # This is a simplified approach - would need full AST analysis for complete resolution
        # For now, we'll just log and return None
        logger.debug(f"File variable '{var_name}' detected - enhanced resolution needed")
        return None
    
    def _body_contains_read(self, body, var_name):
        """Check if body contains var.read() call."""
        for stmt in body:
            if self._stmt_contains_read(stmt, var_name):
                return True
        return False
    
    def _stmt_contains_read(self, stmt, var_name):
        """Recursively check if statement contains var.read()."""
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call):
                if (isinstance(stmt.value.func, ast.Attribute) and
                    isinstance(stmt.value.func.value, ast.Name) and
                    stmt.value.func.value.id == var_name and
                    stmt.value.func.attr == 'read'):
                    return True
        
        # Check all child nodes
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call):
                if (isinstance(child.func, ast.Attribute) and
                    isinstance(child.func.value, ast.Name) and
                    child.func.value.id == var_name and
                    child.func.attr in ['read', 'readlines']):
                    return True
        return False
    
    def _create_inlined_string(self, filename):
        """Create an AST node with inlined file content."""
        file_path = self.video_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.inlined_files.append(filename)
                logger.info(f"Inlined file: {filename}")
                return ast.Constant(value=content)
            except Exception as e:
                logger.warning(f"Failed to inline {filename}: {e}")
                self.failed_files.append(filename)
        return None
    
    def _create_with_replacement(self, filename, body, var_name):
        """Replace with open() block with direct assignment."""
        file_path = self.video_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.inlined_files.append(filename)
                logger.info(f"Inlined file from with statement: {filename}")
                
                # Find the read() assignment in the body
                new_body = []
                for stmt in body:
                    if isinstance(stmt, ast.Assign) and self._stmt_contains_read(stmt, var_name):
                        # Replace with direct content assignment
                        new_assign = ast.Assign(
                            targets=stmt.targets,
                            value=ast.Constant(value=content)
                        )
                        new_body.append(new_assign)
                    elif not self._stmt_contains_read(stmt, var_name):
                        new_body.append(stmt)
                
                return new_body[0] if len(new_body) == 1 else ast.Module(body=new_body)
                
            except Exception as e:
                logger.warning(f"Failed to inline {filename}: {e}")
                self.failed_files.append(filename)
        return None
    
    def transform_code(self, code: str) -> str:
        """Transform code to inline file operations."""
        try:
            # For xiaoxiae videos, check for common file patterns and inline them
            # This is more aggressive than AST transformation but necessary
            transformed_code = code
            
            # Pattern 1: open("filename").read()
            import re
            pattern1 = r'open\s*\(\s*["\']([^"\']+)["\']\s*\)\s*\.read\s*\(\s*\)'
            for match in re.finditer(pattern1, transformed_code):
                filename = match.group(1)
                file_path = self.video_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Escape the content for Python string
                        escaped_content = repr(content)
                        transformed_code = transformed_code.replace(match.group(0), escaped_content)
                        self.inlined_files.append(filename)
                        logger.info(f"Inlined file via regex: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to inline {filename}: {e}")
                        self.failed_files.append(filename)
            
            # Now also do AST-based transformation for complex cases
            tree = ast.parse(transformed_code)
            
            # Add parent references for context
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    child._parent = parent
            
            # Transform the tree
            new_tree = self.visit(tree)
            
            # Generate code
            return ast.unparse(new_tree)
            
        except Exception as e:
            logger.error(f"Failed to transform code: {e}")
            return code


class DependencyAnalyzer:
    """Analyzes dependencies in Python code to build a complete dependency graph."""
    
    def __init__(self):
        self.dependencies = set()
        self.visited = set()
    
    def analyze_scene_code(self, scene_code: str, utilities_code: str) -> Tuple[Set[str], Set[str]]:
        """Analyze scene code to find all dependencies."""
        try:
            # Parse both ASTs
            scene_tree = ast.parse(scene_code)
            util_tree = ast.parse(utilities_code) if utilities_code else None
            
            # Find all names used in scene
            scene_names = self._collect_names(scene_tree)
            
            # Build utilities index
            util_index = self._build_util_index(util_tree) if util_tree else {}
            
            # Find all required utilities (with transitive dependencies)
            required_utils = set()
            for name in scene_names:
                if name in util_index:
                    self._collect_dependencies(name, util_index, required_utils)
            
            # Find all required imports
            required_imports = self._determine_required_imports(scene_names, required_utils, util_tree)
            
            return required_imports, required_utils
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            return set(), set()
    
    def _collect_names(self, tree: ast.AST) -> Set[str]:
        """Collect all names used in the code."""
        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.names = set()
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.names.add(node.id)
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Collect module names from module.function calls
                    if isinstance(node.func.value, ast.Name):
                        self.names.add(node.func.value.id)
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    self.names.add(node.value.id)
                self.generic_visit(node)
        
        collector = NameCollector()
        collector.visit(tree)
        return collector.names
    
    def _build_util_index(self, tree: ast.AST) -> Dict[str, ast.AST]:
        """Build index of all utilities definitions."""
        index = {}
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                index[node.name] = node
            elif isinstance(node, ast.Assign):
                # Handle variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        index[target.id] = node
        
        return index
    
    def _collect_dependencies(self, name: str, util_index: Dict[str, ast.AST], required: Set[str]):
        """Recursively collect all dependencies for a given name."""
        if name in self.visited or name in required:
            return
        
        self.visited.add(name)
        
        if name in util_index:
            required.add(name)
            node = util_index[name]
            
            # Find dependencies of this node
            deps = self._collect_names(node)
            
            # For xiaoxiae's code, also add common helper functions that are often used together
            if name in ['create_code', 'myCode']:
                for helper in ['create_code', 'myCode', 'create_output', 'myOutput']:
                    if helper in util_index and helper not in required:
                        required.add(helper)
            
            # Recursively collect their dependencies
            for dep in deps:
                if dep in util_index and dep not in required:
                    self._collect_dependencies(dep, util_index, required)
    
    def _determine_required_imports(self, scene_names: Set[str], util_names: Set[str], util_tree: ast.AST) -> Set[str]:
        """Determine which imports are required."""
        all_names = scene_names | util_names
        
        imports = {"from manim import *"}  # Always needed
        
        # Module imports based on usage
        if 'np' in all_names or 'numpy' in all_names:
            imports.add("import numpy as np")
        
        if 'nx' in all_names or 'networkx' in all_names:
            imports.add("import networkx as nx")
        
        # Standard library imports
        math_funcs = {'sin', 'cos', 'tan', 'sqrt', 'pi', 'exp', 'log', 'floor', 'ceil'}
        if any(func in all_names for func in math_funcs):
            imports.add("import math")
            imports.add("from math import *")
        
        random_funcs = {'random', 'choice', 'randint', 'uniform', 'shuffle', 'seed'}
        if any(func in all_names for func in random_funcs):
            imports.add("import random")
            imports.add("from random import *")
        
        # Check for specific library usage in utilities
        if util_tree:
            util_imports = self._extract_imports(util_tree)
            
            # Add imports that are actually used
            for imp in util_imports:
                if 'pulp' in imp and any(name in all_names for name in ['LpProblem', 'LpVariable', 'lpSum']):
                    imports.add(imp)
                elif 'yaml' in imp and any(name in all_names for name in ['load', 'dump']):
                    imports.add(imp)
                elif 'functools' in imp and any(name in all_names for name in ['partial', 'reduce']):
                    imports.add(imp)
                elif 'itertools' in imp and any(name in all_names for name in ['combinations', 'permutations', 'product']):
                    imports.add(imp)
                elif 'typing' in imp:
                    imports.add(imp)
        
        return imports
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all import statements from the tree."""
        imports = set()
        
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [alias.name for alias in node.names]
                    if '*' in names:
                        imports.add(f"from {node.module} import *")
                    else:
                        imports.add(f"from {node.module} import {', '.join(names)}")
        
        return imports


@register_extractor
class XiaoxiaeVideosExtractor(BaseExtractor):
    """Extract educational mathematical animations from xiaoxiae/videos repository."""
    
    source_id = "xiaoxiae_videos"
    source_name = "xiaoxiae Educational Videos"
    priority = 4  # High quality educational content
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.repo_url = self.config.get("repo_url", "https://github.com/xiaoxiae/videos.git")
        self.repo_path = Path(self.config.get("repo_path", "data/xiaoxiae-videos"))
        self.cache_dir = Path(self.config.get("cache_dir", ".cache"))
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 170  # Based on 26 video projects with all Scene types extracted
    
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
                    
                    # Check if it inherits from Scene or any Scene subclass
                    scene_classes = {'Scene', 'MovingCameraScene', 'ThreeDScene', 'GraphScene', 'ZoomedScene'}
                    if any(base_name in scene_classes for base_name in base_names):
                        # Extract class code
                        class_start = node.lineno - 1
                        class_end = node.end_lineno if node.end_lineno else len(content.split('\n'))
                        class_lines = content.split('\n')[class_start:class_end]
                        
                        # Clean up indentation of class part only
                        min_indent = float('inf')
                        for line in class_lines:
                            if line.strip():
                                indent = len(line) - len(line.lstrip())
                                min_indent = min(min_indent, indent)
                        
                        if min_indent < float('inf'):
                            class_lines = [line[min_indent:] if len(line) > min_indent else line 
                                         for line in class_lines]
                            class_code = '\n'.join(class_lines)
                        else:
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
                    return f.read()
            except Exception as e:
                logger.debug(f"Error reading utilities from {utilities_file}: {e}")
        return ""
    
    def _bundle_code_with_utilities(self, scene_code: str, utilities: str, video_dir: Path) -> Tuple[str, Dict[str, Any]]:
        """Bundle Scene code with minimal required utilities and return statistics."""
        stats = {
            "files_inlined": 0,
            "assets_replaced": 0,
            "asset_types": {}
        }
        
        # First inline any file dependencies
        inliner = FileInliner(video_dir)
        inlined_scene = inliner.transform_code(scene_code)
        
        # Log inlining results
        if inliner.inlined_files:
            stats["files_inlined"] = len(inliner.inlined_files)
            logger.info(f"Successfully inlined files: {', '.join(inliner.inlined_files)}")
        if inliner.failed_files:
            logger.warning(f"Failed to inline files: {', '.join(inliner.failed_files)}")
        
        # Replace binary assets with Manim primitives
        asset_replaced_scene, asset_stats = replace_assets_in_code(inlined_scene)
        if asset_stats.get('replaced', 0) > 0:
            stats["assets_replaced"] = asset_stats['replaced']
            stats["asset_types"] = {
                "characters": asset_stats.get('characters', 0),
                "buildings": asset_stats.get('buildings', 0),
                "symbols": asset_stats.get('symbols', 0),
                "other": asset_stats.get('other', 0)
            }
            logger.info(f"Replaced {asset_stats['replaced']} assets: "
                       f"{asset_stats.get('characters', 0)} characters, "
                       f"{asset_stats.get('buildings', 0)} buildings, "
                       f"{asset_stats.get('symbols', 0)} symbols")
            inlined_scene = asset_replaced_scene
        
        # Analyze dependencies to get ONLY what's needed
        analyzer = DependencyAnalyzer()
        required_imports, required_utils = analyzer.analyze_scene_code(inlined_scene, utilities)
        
        # Build bundled code
        parts = []
        
        # First check if the scene imports from utilities
        imports_from_utilities = "from utilities import" in inlined_scene
        
        if imports_from_utilities:
            # If scene already imports from utilities, we need to inline ALL utilities
            # Remove the import line first
            lines = inlined_scene.split('\n')
            filtered_lines = [line for line in lines if 'from utilities import' not in line]
            inlined_scene = '\n'.join(filtered_lines)
            
            # Add all imports from utilities.py first
            parts.append("from manim import *")
            if utilities:
                try:
                    util_tree = ast.parse(utilities)
                    for node in util_tree.body:
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                parts.append(f"import {alias.name}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                names = [alias.name for alias in node.names]
                                if '*' in names:
                                    parts.append(f"from {node.module} import *")
                                else:
                                    parts.append(f"from {node.module} import {', '.join(names)}")
                except Exception as e:
                    logger.warning(f"Failed to extract imports: {e}")
            
            parts.append("")
            
            # Add ALL utility functions/classes (not just required ones)
            if utilities:
                try:
                    util_tree = ast.parse(utilities)
                    for node in util_tree.body:
                        # Skip import statements (already handled)
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            continue
                        # Include all functions, classes, and assignments
                        parts.append(ast.unparse(node))
                        parts.append("")
                except Exception as e:
                    logger.warning(f"Failed to include all utilities: {e}")
                    # Fallback: include raw utilities
                    parts.extend(utilities.split('\n'))
                    parts.append("")
        else:
            # Original logic for scenes that don't import from utilities
            parts.append("from manim import *")
            parts.append("")
        
        # If we didn't import from utilities, only add required utilities
        if not imports_from_utilities and required_utils and utilities:
            included_utils = []
            try:
                util_tree = ast.parse(utilities)
                for node in util_tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if node.name in required_utils:
                            parts.append(ast.unparse(node))
                            parts.append("")
                            included_utils.append(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in required_utils:
                                parts.append(ast.unparse(node))
                                parts.append("")
                                included_utils.append(target.id)
                                break
            except Exception as e:
                logger.warning(f"Failed to extract specific utilities: {e}")
            
            if included_utils:
                logger.debug(f"Included utilities for {video_dir.name}: {', '.join(included_utils)}")
        
        # Add the scene class (with files already inlined)
        parts.append(inlined_scene)
        
        return '\n'.join(parts), stats
    
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
                bundled_code, bundle_stats = self._bundle_code_with_utilities(
                    scene_code, utilities, video_dir
                )
                
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
                        "content_type": "mathematical_visualization",
                        "files_inlined": bundle_stats.get("files_inlined", 0),
                        "assets_replaced": bundle_stats.get("assets_replaced", 0),
                        "asset_types_replaced": bundle_stats.get("asset_types", {})
                    }
                }
                
                total_scenes += 1
            
            logger.info(f"Extracted {len(scenes)} scenes from {video_dir.name}")
        
        logger.info(f"Total scenes extracted: {total_scenes}")