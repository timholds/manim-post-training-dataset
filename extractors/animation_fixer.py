"""
Animation unpacking fixer for Manim code.

Fixes the critical timing issue where self.play(*animations, run_time=X) 
runs animations simultaneously instead of as a group.
"""

import ast
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class AnimationUnpackingFixer(ast.NodeTransformer):
    """Fixes self.play(*animations, ...) patterns by wrapping in AnimationGroup."""
    
    def __init__(self):
        self.fixes_applied = 0
        
    def visit_Call(self, node):
        """Find and fix self.play(*...) calls."""
        # Check if this is a self.play() call
        if (isinstance(node.func, ast.Attribute) and
            node.func.attr == 'play' and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'self'):
            
            # Look for starred arguments with keywords (especially run_time)
            has_starred = any(isinstance(arg, ast.Starred) for arg in node.args)
            has_keywords = bool(node.keywords)
            
            if has_starred and has_keywords:
                # Check if any keyword is timing-related
                timing_keywords = ['run_time', 'rate_func', 'lag_ratio']
                has_timing = any(kw.arg in timing_keywords for kw in node.keywords if kw.arg)
                
                if has_timing:
                    # Collect all starred expressions
                    starred_expressions = []
                    regular_args = []
                    
                    for arg in node.args:
                        if isinstance(arg, ast.Starred):
                            starred_expressions.append(arg.value)
                        else:
                            regular_args.append(arg)
                    
                    if starred_expressions:
                        # Create AnimationGroup with all starred content
                        if len(starred_expressions) == 1:
                            # Single starred expression: AnimationGroup(*expr)
                            anim_group = ast.Call(
                                func=ast.Name(id='AnimationGroup', ctx=ast.Load()),
                                args=[ast.Starred(value=starred_expressions[0], ctx=ast.Load())],
                                keywords=[]
                            )
                        else:
                            # Multiple starred: AnimationGroup(*expr1, *expr2, ...)
                            anim_group = ast.Call(
                                func=ast.Name(id='AnimationGroup', ctx=ast.Load()),
                                args=[ast.Starred(value=expr, ctx=ast.Load()) for expr in starred_expressions],
                                keywords=[]
                            )
                        
                        # Create new call with AnimationGroup first, then other args
                        new_node = ast.Call(
                            func=node.func,
                            args=[anim_group] + regular_args,
                            keywords=node.keywords
                        )
                        
                        self.fixes_applied += 1
                        return new_node
        
        # Continue visiting
        self.generic_visit(node)
        return node


def fix_animation_unpacking(code: str) -> Tuple[str, int]:
    """
    Fix animation unpacking issues in code.
    
    Args:
        code: Python code containing Manim animations
        
    Returns:
        Tuple of (fixed_code, number_of_fixes)
    """
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # Apply fixes
        fixer = AnimationUnpackingFixer()
        new_tree = fixer.visit(tree)
        
        # Convert back to code
        fixed_code = ast.unparse(new_tree)
        
        if fixer.fixes_applied > 0:
            logger.info(f"Fixed {fixer.fixes_applied} animation unpacking issues")
        
        return fixed_code, fixer.fixes_applied
        
    except SyntaxError as e:
        logger.warning(f"Syntax error in code, skipping animation fix: {e}")
        return code, 0
    except Exception as e:
        logger.error(f"Failed to fix animation unpacking: {e}")
        return code, 0