"""Smart filtering of utilities.py imports based on actual usage"""
import ast
from typing import Set, List, Tuple

class DependencyAnalyzer(ast.NodeVisitor):
    """Analyze which imports are actually needed by the code"""
    
    def __init__(self):
        self.used_names = set()
        self.used_modules = set()
        self.defined_names = set()
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        # Track module usage like nx.Graph
        if isinstance(node.value, ast.Name):
            self.used_modules.add(node.value.id)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Track function calls
        if isinstance(node.func, ast.Name):
            self.used_names.add(node.func.id)
        self.generic_visit(node)


def filter_utilities_smart(utilities_code: str, scene_code: str) -> str:
    """Filter utilities to only include what's actually used"""
    
    # Parse scene code to find dependencies
    scene_tree = ast.parse(scene_code)
    analyzer = DependencyAnalyzer()
    analyzer.visit(scene_tree)
    
    # Parse utilities
    util_tree = ast.parse(utilities_code)
    
    # Determine which imports are needed
    needed_imports = []
    filtered_nodes = []
    
    for node in util_tree.body:
        if isinstance(node, ast.ImportFrom):
            # Check specific imports
            if node.module == 'manim':
                # Always include manim imports
                needed_imports.append(node)
            elif node.module == 'pulp':
                # Only include if LP functions are used
                if any(name in analyzer.used_names for name in ['LpProblem', 'LpVariable', 'lpSum', 'PULP_CBC_CMD']):
                    needed_imports.append(node)
            elif node.module == 'yaml':
                # Only if yaml functions used
                if 'load' in analyzer.used_names or 'dump' in analyzer.used_names:
                    needed_imports.append(node)
            else:
                # For other imports, include if the module is referenced
                needed_imports.append(node)
                
        elif isinstance(node, ast.Import):
            # Check if imported module is used
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name in analyzer.used_modules:
                    needed_imports.append(node)
                    break
                    
        elif isinstance(node, ast.FunctionDef):
            # Include function if it's called
            if node.name in analyzer.used_names:
                filtered_nodes.append(node)
                # Recursively check what this function uses
                func_analyzer = DependencyAnalyzer()
                func_analyzer.visit(node)
                analyzer.used_names.update(func_analyzer.used_names)
                analyzer.used_modules.update(func_analyzer.used_modules)
                
        elif isinstance(node, ast.ClassDef):
            # Include class if it's used
            if node.name in analyzer.used_names:
                filtered_nodes.append(node)
    
    # Rebuild filtered utilities
    filtered_tree = ast.Module(body=needed_imports + filtered_nodes, type_ignores=[])
    return ast.unparse(filtered_tree)


def create_minimal_utilities(scene_code: str) -> str:
    """Create minimal utilities based on what the scene actually needs"""
    
    # Analyze scene dependencies
    tree = ast.parse(scene_code)
    analyzer = DependencyAnalyzer()
    analyzer.visit(tree)
    
    # Build minimal imports
    imports = ["from manim import *"]
    
    # Add other imports only if needed
    if any(name in analyzer.used_names for name in ['sin', 'cos', 'sqrt', 'pi']):
        imports.append("from math import *")
    
    if any(name in analyzer.used_names for name in ['random', 'choice', 'seed']):
        imports.append("from random import *")
        
    if 'nx' in analyzer.used_modules:
        imports.append("import networkx as nx")
        
    if any(name in analyzer.used_names for name in ['np', 'numpy']):
        imports.append("import numpy as np")
    
    return '\n'.join(imports)