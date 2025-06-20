#!/usr/bin/env python3
"""
Simple and robust Manim code extraction from LLM outputs.

Handles extraction, validation, and basic sanitization of Manim code
from various output formats (chat templates, markdown blocks, etc).
"""

import re
import ast
import subprocess
import tempfile
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Simple container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ManimCodeExtractor:
    """Extract and validate Manim code from LLM outputs."""
    
    def extract(self, model_output: str) -> str:
        """
        Extract Manim code from model output.
        
        Handles:
        1. Chat template markers (QWEN format)
        2. Markdown code blocks
        3. Plain code
        """
        # Step 1: Remove chat template markers if present
        clean_output = self._remove_chat_markers(model_output)
        
        # Step 2: Extract code from markdown blocks
        code = self._extract_from_markdown(clean_output)
        
        # Step 3: If no markdown blocks, assume entire output is code
        if not code:
            code = clean_output.strip()
        
        # Step 4: Sanitize the extracted code
        return self.sanitize(code)
    
    def _remove_chat_markers(self, text: str) -> str:
        """Remove QWEN chat template markers."""
        # Extract assistant response if template markers exist
        if "<|im_start|>assistant" in text:
            # Get everything after the assistant marker
            parts = text.split("<|im_start|>assistant")
            if len(parts) > 1:
                assistant_text = parts[-1]
                # Remove end marker if present
                if "<|im_end|>" in assistant_text:
                    assistant_text = assistant_text.split("<|im_end|>")[0]
                return assistant_text.strip()
        return text
    
    def _extract_from_markdown(self, text: str) -> Optional[str]:
        """Extract code from markdown code blocks."""
        # Look for ```python blocks first
        python_pattern = r'```python\s*(.*?)```'
        matches = re.findall(python_pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            # Return the first match (most likely the main code)
            return matches[0].strip()
        
        # Fallback to generic code blocks
        generic_pattern = r'```\s*(.*?)```'
        matches = re.findall(generic_pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            return matches[0].strip()
        
        return None
    
    def validate(self, code: str) -> ValidationResult:
        """
        Validate extracted Manim code.
        
        Checks:
        1. Valid Python syntax
        2. Has required imports
        3. Has Scene class
        4. Has construct method
        5. Enhanced Manim-specific checks
        """
        errors = []
        warnings = []
        
        # Check 1: Valid Python syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
            return ValidationResult(False, errors, warnings)
        
        # Check 2: Has manim imports
        has_imports = self._check_imports(tree)
        if not has_imports:
            errors.append("Missing manim imports (e.g., 'from manim import *')")
        
        # Check 3: Has Scene class
        scene_class = self._find_scene_class(tree)
        if not scene_class:
            errors.append("No class inheriting from Scene found")
        else:
            # Check 4: Has construct method
            has_construct = self._check_construct_method(scene_class)
            if not has_construct:
                errors.append("Scene class missing 'construct' method")
            else:
                # Check 5: Enhanced validation for construct method
                construct_issues = self._validate_construct_method(scene_class)
                errors.extend(construct_issues["errors"])
                warnings.extend(construct_issues["warnings"])
        
        # Check 6: Common Manim pattern validation
        pattern_issues = self._check_common_patterns(code, tree)
        warnings.extend(pattern_issues["warnings"])
        
        # Additional checks that are warnings
        if "TODO" in code or "FIXME" in code:
            warnings.append("Code contains TODO/FIXME comments")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings)
    
    def _check_imports(self, tree: ast.AST) -> bool:
        """Check if code has manim imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "manim":
                    return True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "manim" in alias.name:
                        return True
        return False
    
    def _find_scene_class(self, tree: ast.AST) -> Optional[ast.ClassDef]:
        """Find a class that inherits from Scene."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from Scene
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "Scene":
                        return node
        return None
    
    def _check_construct_method(self, class_node: ast.ClassDef) -> bool:
        """Check if class has a construct method."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "construct":
                return True
        return False
    
    def _validate_construct_method(self, class_node: ast.ClassDef) -> dict:
        """Perform detailed validation of the construct method."""
        errors = []
        warnings = []
        
        # Find construct method
        construct_method = None
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "construct":
                construct_method = node
                break
        
        if not construct_method:
            return {"errors": errors, "warnings": warnings}
        
        # Check if construct has self parameter
        if not construct_method.args.args or construct_method.args.args[0].arg != "self":
            errors.append("construct method must have 'self' as first parameter")
        
        # Check if construct has any content
        if not construct_method.body:
            errors.append("construct method is empty")
        elif len(construct_method.body) == 1 and isinstance(construct_method.body[0], ast.Pass):
            warnings.append("construct method only contains 'pass'")
        
        # Check for common animation patterns
        has_play_call = False
        has_wait_call = False
        has_manim_objects = False
        
        for node in ast.walk(construct_method):
            # Check for self.play() calls
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and 
                node.func.value.id == "self" and 
                node.func.attr == "play"):
                has_play_call = True
            
            # Check for self.wait() calls
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Name) and 
                node.func.value.id == "self" and 
                node.func.attr == "wait"):
                has_wait_call = True
            
            # Check for Manim object creation
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func_name = node.func.id
                common_objects = ['Circle', 'Square', 'Text', 'Line', 'Dot', 'Arrow', 
                                'Rectangle', 'Polygon', 'Ellipse', 'Arc', 'NumberPlane', 
                                'Axes', 'Graph', 'VGroup', 'VMobject']
                if func_name in common_objects:
                    has_manim_objects = True
        
        if not has_play_call:
            warnings.append("No self.play() calls found - animation might be static")
        
        if not has_manim_objects:
            warnings.append("No Manim objects created in construct method")
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_common_patterns(self, code: str, tree: ast.AST) -> dict:
        """Check for common Manim patterns and potential issues."""
        warnings = []
        
        # Check for common misspellings or case issues
        if "scene" in code and "Scene" in code:
            # Check if someone wrote 'scene' instead of 'Scene'
            if re.search(r'class\s+\w+\s*\(\s*scene\s*\)', code):
                warnings.append("Class inherits from 'scene' (lowercase) - should be 'Scene'")
        
        # Check for missing numpy import when using mathematical functions
        if any(func in code for func in ['np.', 'numpy.', 'sin', 'cos', 'pi', 'sqrt']):
            has_numpy = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "numpy":
                            has_numpy = True
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "numpy":
                        has_numpy = True
            
            if not has_numpy and 'np.' in code:
                warnings.append("Code uses 'np.' but numpy might not be imported")
        
        # Check for animate syntax issues
        if '.animate' in code:
            # Check for common mistake: obj.animate.method() instead of obj.animate.method()
            if re.search(r'\.animate\s*\.\s*\w+\s*\(\s*\)\s*\)', code):
                warnings.append("Possible issue with animate syntax - check parentheses")
        
        # Check for self.add without self.play
        if 'self.add(' in code and 'self.play(' not in code:
            warnings.append("Using self.add() without self.play() - objects will appear instantly")
        
        return {"errors": [], "warnings": warnings}
    
    def sanitize(self, code: str) -> str:
        """
        Apply basic sanitization to extracted code.
        
        Fixes:
        1. Trailing whitespace
        2. Ensure newline at end
        3. Fix common indentation issues
        """
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in code.splitlines()]
        
        # Remove common leading whitespace (dedent)
        if lines:
            # Find minimum indentation (excluding empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                lines = [line[min_indent:] if line else line for line in lines]
        
        # Ensure proper spacing
        clean_code = "\n".join(lines)
        
        # Ensure newline at end
        if not clean_code.endswith("\n"):
            clean_code += "\n"
        
        return clean_code
    
    def extract_and_validate(self, model_output: str, compile_check: bool = False) -> Tuple[str, ValidationResult]:
        """
        Convenience method to extract and validate in one call.
        
        Args:
            model_output: The raw model output
            compile_check: Whether to also test compilation with Manim
            
        Returns: (extracted_code, validation_result)
        """
        code = self.extract(model_output)
        code = self.sanitize(code)
        validation = self.validate(code)
        
        # Optionally perform compilation check
        if compile_check and validation.is_valid:
            compile_result = self.test_compilation(code)
            if not compile_result["success"]:
                validation.errors.append(f"Compilation failed: {compile_result['error']}")
                validation.is_valid = False
        
        return code, validation
    
    def test_compilation(self, code: str, timeout: int = 30) -> dict:
        """
        Test if the code compiles with Manim.
        
        Args:
            code: The Manim code to test
            timeout: Maximum time to wait for compilation (seconds)
            
        Returns: dict with 'success' and 'error' keys
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Try to compile with manim (dry run)
            cmd = ["manim", "-ql", temp_file, "--dry_run"]
            
            # Check if we're in a virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                # Use the manim from the virtual environment
                manim_path = os.path.join(venv_path, 'bin', 'manim')
                if os.path.exists(manim_path):
                    cmd[0] = manim_path
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            error = result.stderr if not success else ""
            
            # Extract meaningful error message
            if error:
                # Try to extract the actual Python error
                lines = error.split('\n')
                error_lines = []
                for i, line in enumerate(lines):
                    if 'Error' in line or 'error' in line:
                        error_lines.append(line.strip())
                        # Include a few lines of context
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            if j != i and lines[j].strip():
                                error_lines.append(lines[j].strip())
                
                if error_lines:
                    error = '\n'.join(error_lines[:5])  # Limit to 5 lines
                else:
                    error = error.split('\n')[0]  # Just first line if no specific error found
            
            return {"success": success, "error": error}
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Compilation timeout"}
        except Exception as e:
            return {"success": False, "error": f"Compilation check failed: {str(e)}"}
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass


def simple_extract(model_output: str) -> str:
    """
    Simple one-line extraction for basic use cases.
    
    Just extracts the code without validation.
    """
    extractor = ManimCodeExtractor()
    return extractor.sanitize(extractor.extract(model_output))


if __name__ == "__main__":
    # Example usage
    sample_output = """<|im_start|>assistant
    Here's a Manim animation for a sine wave:
    
    ```python
    from manim import *
    
    class SineWaveScene(Scene):
        def construct(self):
            axes = Axes()
            sine_curve = axes.plot(lambda x: np.sin(x))
            self.play(Create(axes))
            self.play(Create(sine_curve))
            self.wait()
    ```
    <|im_end|>"""
    
    extractor = ManimCodeExtractor()
    code, validation = extractor.extract_and_validate(sample_output)
    
    print("Extracted code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    print(f"Valid: {validation.is_valid}")
    if validation.errors:
        print("Errors:", validation.errors)
    if validation.warnings:
        print("Warnings:", validation.warnings)