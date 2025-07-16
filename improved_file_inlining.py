"""Improved file inlining using AST parsing"""
import ast
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileInliner(ast.NodeTransformer):
    """AST transformer that inlines file reads"""
    
    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.inlined_files = {}
        self.imports_to_add = set()
        
    def visit_Call(self, node):
        """Detect and replace file read operations"""
        # Check for open().read() pattern
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'read' and
            isinstance(node.func.value, ast.Call) and
            isinstance(node.func.value.func, ast.Name) and
            node.func.value.func.id == 'open'):
            
            # Extract filename from open() call
            open_call = node.func.value
            if open_call.args and isinstance(open_call.args[0], ast.Constant):
                filename = open_call.args[0].value
                return self._create_inlined_reference(filename)
                
        return self.generic_visit(node)
    
    def visit_With(self, node):
        """Handle with open() as f: patterns"""
        for item in node.items:
            if (isinstance(item.context_expr, ast.Call) and
                isinstance(item.context_expr.func, ast.Name) and
                item.context_expr.func.id == 'open' and
                item.context_expr.args and
                isinstance(item.context_expr.args[0], ast.Constant)):
                
                filename = item.context_expr.args[0].value
                var_name = item.optional_vars.id if item.optional_vars else 'f'
                
                # Check if the body just reads the file
                if self._is_simple_read(node.body, var_name):
                    # Replace entire with block with inlined content
                    return self._create_inlined_assignment(filename, node.body[0].targets[0].id)
                    
        return self.generic_visit(node)
    
    def _is_simple_read(self, body, file_var):
        """Check if with block just reads file"""
        if len(body) == 1 and isinstance(body[0], ast.Assign):
            if (isinstance(body[0].value, ast.Call) and
                isinstance(body[0].value.func, ast.Attribute) and
                body[0].value.func.attr == 'read' and
                isinstance(body[0].value.func.value, ast.Name) and
                body[0].value.func.value.id == file_var):
                return True
        return False
        
    def _create_inlined_reference(self, filename):
        """Create reference to inlined content"""
        var_name = self._inline_file(filename)
        return ast.Name(id=var_name, ctx=ast.Load())
    
    def _create_inlined_assignment(self, filename, target_var):
        """Create assignment of inlined content"""
        var_name = self._inline_file(filename)
        return ast.Assign(
            targets=[ast.Name(id=target_var, ctx=ast.Store())],
            value=ast.Name(id=var_name, ctx=ast.Load())
        )
    
    def _inline_file(self, filename):
        """Actually inline the file content"""
        if filename not in self.inlined_files:
            file_path = self.video_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    var_name = f"_inlined_{filename.replace('.', '_').replace('-', '_')}"
                    self.inlined_files[filename] = (var_name, content)
                    logger.info(f"Inlined file: {filename}")
                except Exception as e:
                    logger.error(f"Failed to inline {filename}: {e}")
                    raise
            else:
                logger.warning(f"File not found: {filename}")
                raise FileNotFoundError(f"Cannot inline {filename}")
        
        return self.inlined_files[filename][0]
    
    def get_inlined_content_nodes(self):
        """Get AST nodes for inlined content definitions"""
        nodes = []
        for filename, (var_name, content) in self.inlined_files.items():
            # Create: var_name = '''content'''
            nodes.append(ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Constant(value=content)
            ))
        return nodes