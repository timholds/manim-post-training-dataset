#!/usr/bin/env python3
"""
Simple pipeline for rendering Manim animations from LLM outputs.

Integrates extraction, validation, and rendering into a single workflow.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from manim_code_extractor import ManimCodeExtractor, ValidationResult


@dataclass
class RenderResult:
    """Result of a render attempt."""
    success: bool
    output_path: Optional[str]
    error_message: Optional[str]
    extracted_code: str
    validation: ValidationResult


class ManimRenderPipeline:
    """
    End-to-end pipeline for rendering Manim animations from LLM outputs.
    
    Simple workflow:
    1. Extract code from LLM output
    2. Validate the code structure
    3. Save to temporary file
    4. Render with manim
    5. Return result
    """
    
    def __init__(self, output_dir: str = "./renders"):
        self.extractor = ManimCodeExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def render_from_output(self, 
                          model_output: str, 
                          scene_name: Optional[str] = None,
                          quality: str = "medium_quality") -> RenderResult:
        """
        Render Manim animation from model output.
        
        Args:
            model_output: Raw output from the model
            scene_name: Name for the output file (auto-generated if None)
            quality: Manim quality setting (low_quality, medium_quality, high_quality)
            
        Returns:
            RenderResult with status and paths
        """
        # Step 1: Extract and validate code
        code, validation = self.extractor.extract_and_validate(model_output)
        
        # Step 2: Check if valid
        if not validation.is_valid:
            return RenderResult(
                success=False,
                output_path=None,
                error_message=f"Validation failed: {'; '.join(validation.errors)}",
                extracted_code=code,
                validation=validation
            )
        
        # Step 3: Find scene class name
        class_name = self._find_scene_class_name(code)
        if not class_name:
            return RenderResult(
                success=False,
                output_path=None,
                error_message="Could not find Scene class name",
                extracted_code=code,
                validation=validation
            )
        
        # Step 4: Render the code
        try:
            output_path = self._render_code(code, class_name, scene_name, quality)
            return RenderResult(
                success=True,
                output_path=str(output_path),
                error_message=None,
                extracted_code=code,
                validation=validation
            )
        except Exception as e:
            return RenderResult(
                success=False,
                output_path=None,
                error_message=f"Render failed: {str(e)}",
                extracted_code=code,
                validation=validation
            )
    
    def _find_scene_class_name(self, code: str) -> Optional[str]:
        """Extract the Scene class name from the code."""
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "Scene":
                            return node.name
        except:
            pass
        return None
    
    def _render_code(self, code: str, class_name: str, 
                     scene_name: Optional[str], quality: str) -> Path:
        """
        Render the Manim code to video.
        
        Returns path to the output video.
        """
        # Create temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Prepare output name
            if not scene_name:
                scene_name = class_name
            
            # Run manim command  
            quality_map = {
                "low_quality": "l",
                "medium_quality": "m", 
                "high_quality": "h",
                "production_quality": "p",
                "fourk_quality": "k"
            }
            q_flag = quality_map.get(quality, "m")
            
            cmd = [
                "manim", 
                "-q", q_flag,
                "-o", scene_name,
                temp_file,
                class_name
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.output_dir
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Manim failed: {result.stderr}")
            
            # Find the output file - Manim creates videos in media/videos/{filename}/{resolution}/
            base_dir = self.output_dir / "media" / "videos" / Path(temp_file).stem
            
            # Find the actual quality directory (it varies by version)
            if base_dir.exists():
                quality_dirs = list(base_dir.iterdir())
                if quality_dirs:
                    video_dir = quality_dirs[0]  # Use the first (and likely only) quality dir
                    video_files = list(video_dir.glob("*.mp4"))
                    
                    if video_files:
                        return video_files[0]
            
            raise RuntimeError(f"No output video found in {base_dir}")
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
    
    def render_code_directly(self, code: str, scene_name: str = "output",
                           quality: str = "medium_quality") -> RenderResult:
        """
        Render Manim code directly without extraction.
        
        Useful when you already have clean Manim code.
        """
        # Validate first
        validation = self.extractor.validate(code)
        
        if not validation.is_valid:
            return RenderResult(
                success=False,
                output_path=None,
                error_message=f"Validation failed: {'; '.join(validation.errors)}",
                extracted_code=code,
                validation=validation
            )
        
        # Find class name and render
        class_name = self._find_scene_class_name(code)
        if not class_name:
            return RenderResult(
                success=False,
                output_path=None,
                error_message="Could not find Scene class name",
                extracted_code=code,
                validation=validation
            )
        
        try:
            output_path = self._render_code(code, class_name, scene_name, quality)
            return RenderResult(
                success=True,
                output_path=str(output_path),
                error_message=None,
                extracted_code=code,
                validation=validation
            )
        except Exception as e:
            return RenderResult(
                success=False,
                output_path=None,
                error_message=f"Render failed: {str(e)}",
                extracted_code=code,
                validation=validation
            )


def quick_render(model_output: str) -> str:
    """
    Quick one-liner to render Manim from model output.
    
    Returns path to video or raises exception.
    """
    pipeline = ManimRenderPipeline()
    result = pipeline.render_from_output(model_output)
    
    if not result.success:
        raise RuntimeError(result.error_message)
    
    return result.output_path


if __name__ == "__main__":
    # Test with sample output
    sample = """
    Here's a simple Manim animation:
    
    ```python
    from manim import *
    
    class TestScene(Scene):
        def construct(self):
            circle = Circle()
            self.play(Create(circle))
            self.wait()
    ```
    """
    
    pipeline = ManimRenderPipeline()
    result = pipeline.render_from_output(sample, "test_animation")
    
    if result.success:
        print(f"✓ Rendered successfully: {result.output_path}")
    else:
        print(f"✗ Render failed: {result.error_message}")
        if result.validation.errors:
            print("  Validation errors:", result.validation.errors)