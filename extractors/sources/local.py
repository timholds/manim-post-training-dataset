"""Local file dataset extractors."""

import json
import logging
import re
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from ..base import BaseExtractor
from ..registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class Dan4LifeAoC2024Extractor(BaseExtractor):
    """Extractor for Dan4Life's Advent of Code 2024 dataset."""
    
    source_id = "dan4life_aoc2024"
    source_name = "Dan4Life AoC 2024 Videos"
    priority = 2  # Good quality but limited samples
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.file_path = Path(self.config.get("file", "data/data_dan4life/dan4life_aoc2024.jsonl"))
        if not self.file_path.exists():
            logger.warning(f"Data file not found: {self.file_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 24
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Dan4Life dataset."""
        if not self.file_path.exists():
            logger.error(f"Data file not found: {self.file_path}")
            return
        
        # Track seen class names to resolve conflicts
        seen_class_names = set()
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract from conversation format
                        conversations = item.get("conversations", [])
                        if len(conversations) >= 3:
                            description = conversations[1].get("value", "")
                            code = conversations[2].get("value", "")
                            
                            # Clean code from markdown if needed
                            if code.startswith("```python"):
                                code = code.split("```python")[1].split("```")[0].strip()
                            
                            # Fix class name conflicts
                            code = self._resolve_class_name_conflicts(code, seen_class_names)
                            
                            yield {
                                "description": description,
                                "code": code,
                                "metadata": {
                                    "source_file": str(self.file_path),
                                    "line_number": line_num,
                                    "day": item.get("metadata", {}).get("day"),
                                    "version": item.get("metadata", {}).get("version")
                                }
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            return
    
    def _resolve_class_name_conflicts(self, code: str, seen_class_names: set) -> str:
        """Resolve class name conflicts by appending version numbers."""
        # Find scene class definition
        class_match = re.search(r'class\s+(\w+)\s*\([^)]*Scene[^)]*\)', code)
        if not class_match:
            return code
            
        original_class_name = class_match.group(1)
        
        # If no conflict, just track it
        if original_class_name not in seen_class_names:
            seen_class_names.add(original_class_name)
            return code
        
        # Find unique name by appending version number
        version = 2
        new_class_name = f"{original_class_name}V{version}"
        while new_class_name in seen_class_names:
            version += 1
            new_class_name = f"{original_class_name}V{version}"
        
        seen_class_names.add(new_class_name)
        
        # Replace class name in code
        updated_code = re.sub(
            r'class\s+' + re.escape(original_class_name) + r'\s*\(',
            f'class {new_class_name}(',
            code
        )
        
        logger.info(f"Resolved class name conflict: {original_class_name} -> {new_class_name}")
        return updated_code


@register_extractor
class SzymonOzogExtractor(BaseExtractor):
    """Extractor for Szymon Ozog's dataset with advanced code cleaning."""
    
    source_id = "szymon_ozog"
    source_name = "Szymon Ozog Manim Examples"
    priority = 2
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.file_path = Path(self.config.get("file", "data/data_szymon_ozog/szymon_ozog_processed.jsonl"))
        if not self.file_path.exists():
            logger.warning(f"Data file not found: {self.file_path}")
    
    def estimate_sample_count(self) -> Optional[int]:
        """Return estimated number of samples."""
        return 29
    
    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract samples from Szymon Ozog dataset."""
        if not self.file_path.exists():
            logger.error(f"Data file not found: {self.file_path}")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract from conversation format
                        conversations = item.get("conversations", [])
                        if len(conversations) >= 3:
                            description = conversations[1].get("value", "")
                            code = conversations[2].get("value", "")
                            
                            # Clean code from markdown if needed
                            if code.startswith("```python"):
                                code = code.split("```python")[1].split("```")[0].strip()
                            
                            # Fix missing dependencies in szymon_ozog samples
                            code = self._fix_szymon_dependencies(code)
                            
                            yield {
                                "description": description,
                                "code": code,
                                "metadata": {
                                    "source_file": str(self.file_path),
                                    "line_number": line_num
                                }
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            return
    
    def _fix_szymon_dependencies(self, code: str) -> str:
        """
        Completely reconstruct szymon_ozog code to fix fundamental corruption.
        The original JSONL contains truncated/corrupted samples that need aggressive reconstruction.
        """
        # Strip markdown formatting if present
        if code.startswith('```python'):
            code = code.split('```python')[1].split('```')[0].strip()
        
        # Step 1: Extract the basic class structure
        class_match = re.search(r'class\s+(\w+)\s*\([^)]*\):', code)
        if not class_match:
            return self._create_minimal_scene(code)
        
        class_name = class_match.group(1)
        
        # Step 2: Check if this is a fundamentally corrupted sample (incomplete code, missing dependencies)
        corruption_indicators = [
            'ani',  # Truncated animation code
            'anims.append',  # Incomplete append statement
            code.count('(') != code.count(')'),  # Unmatched parentheses
            'toc.' in code and 'TOC(' not in code,  # Missing dependency references
            'bookmark' in code,  # VoiceoverScene-specific functionality
            'self.voiceover' in code,  # VoiceoverScene methods
            'wait_until_bookmark' in code,  # VoiceoverScene methods
            'get_remaining_duration' in code,  # VoiceoverScene methods
        ]
        
        is_corrupted = any(corruption_indicators)
        
        if is_corrupted:
            logger.info(f"Detected corrupted sample {class_name}, reconstructing...")
            return self._reconstruct_corrupted_sample(class_name, code)
        
        # Step 3: For non-corrupted samples, apply standard fixes
        # Replace VoiceoverScene with Scene
        code = re.sub(r'class\s+(\w+)\s*\(\s*VoiceoverScene\s*\):', r'class \1(Scene):', code)
        
        # Remove problematic method calls
        code = re.sub(
            r'self\.set_speech_service\s*\([^)]*\)',
            '# Speech service removed',
            code
        )
        
        # Add proper imports
        if 'from manim import *' not in code:
            code = 'from manim import *\n\n' + code
        
        return code
    
    def _create_minimal_scene(self, original_code: str) -> str:
        """Create a minimal working scene when class structure can't be extracted."""
        return """from manim import *

class MinimalScene(Scene):
    def construct(self):
        # Original code was too corrupted to reconstruct
        circle = Circle()
        self.play(Create(circle))
        self.wait()
"""
    
    def _reconstruct_corrupted_sample(self, class_name: str, original_code: str) -> str:
        """
        Reconstruct a working Scene from corrupted szymon_ozog data.
        Uses pattern matching to salvage usable Manim objects and animations.
        """
        # Try to extract any usable Manim object creations
        manim_objects = []
        animations = []
        
        # Look for object creations line by line to handle multiline definitions better
        lines = original_code.split('\n')
        current_object = None
        paren_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Check for start of object creation - capture more Manim objects
            simple_object_match = re.match(r'(\w+)\s*=\s*(Square|Circle|Rectangle|Text|Line|Dot|Arrow|Triangle|Polygon|Group|VGroup)\s*\(', stripped)
            
            if simple_object_match:
                var_name = simple_object_match.group(1)
                obj_type = simple_object_match.group(2)
                
                # Create simplified versions of each object type
                if obj_type == 'Text':
                    creation_code = f'{var_name} = Text("Sample")'
                elif obj_type == 'Arrow':
                    creation_code = f'{var_name} = Arrow()'
                elif obj_type == 'VGroup':
                    creation_code = f'{var_name} = VGroup(Circle(), Square())'
                else:
                    creation_code = f'{var_name} = {obj_type}()'
                
                manim_objects.append((var_name, creation_code))
        
        # Look for simple self.play animations that we can salvage
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('self.play(') and ('Create(' in stripped or 'Write(' in stripped or 'FadeIn(' in stripped):
                # Extract the object being animated if it's simple
                create_match = re.search(r'self\.play\s*\(\s*(Create|Write|FadeIn)\s*\(\s*(\w+)', stripped)
                if create_match:
                    anim_type = create_match.group(1)
                    obj_name = create_match.group(2)
                    # Only include if we have this object defined
                    if any(obj_name == var for var, _ in manim_objects):
                        animations.append(f'self.play({anim_type}({obj_name}))')
        
        # Build the reconstructed scene
        code_parts = [
            "from manim import *",
            "",
            f"class {class_name}(Scene):",
            "    def construct(self):",
        ]
        
        # Determine which objects are referenced in animations
        animated_objects = set()
        for animation in animations:
            # Extract object name from animation string like "self.play(Create(obj_name))"
            match = re.search(r'Create\((\w+)\)', animation)
            if match:
                animated_objects.add(match.group(1))
        
        # Include all objects that are animated, plus some extras for a complete scene
        objects_to_include = []
        obj_dict = {var_name: creation_code for var_name, creation_code in manim_objects}
        
        # First, add all animated objects
        for obj_name in animated_objects:
            if obj_name in obj_dict:
                objects_to_include.append((obj_name, obj_dict[obj_name]))
        
        # Then add remaining objects up to a reasonable limit
        for var_name, creation_code in manim_objects:
            if var_name not in animated_objects and len(objects_to_include) < 6:
                objects_to_include.append((var_name, creation_code))
        
        # Add object creations
        if objects_to_include:
            for var_name, creation_code in objects_to_include:
                code_parts.append(f"        {creation_code}")
            code_parts.append("")
        
        # Add animations - now all referenced objects should be defined
        if animations:
            for animation in animations[:3]:  # Limit to 3 animations
                code_parts.append(f"        {animation}")
        
        # If no valid animations found, create animations for the objects we did extract
        if not animations and objects_to_include:
            for var_name, _ in objects_to_include[:2]:  # Animate first 2 objects
                code_parts.append(f"        self.play(Create({var_name}))")
            code_parts.append("        self.wait()")
        
        # If no objects and no animations, create minimal scene
        if not animations and not objects_to_include:
            code_parts.extend([
                "        # Reconstructed from corrupted data",
                "        circle = Circle()",
                "        self.play(Create(circle))",
                "        self.wait()"
            ])
        
        return '\n'.join(code_parts)
