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
                            fixed_code = self._fix_szymon_dependencies(code)
                            
                            yield {
                                "description": description,
                                "code": fixed_code,
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
        Simple fix for szymon_ozog samples - just convert VoiceoverScene to Scene.
        
        The code is already ManimCE, we just need to:
        1. Replace VoiceoverScene with Scene
        2. Remove VoiceoverScene-specific methods
        3. Handle the few custom dependencies (TOC, etc.)
        """
        # Strip markdown formatting if present
        if code.startswith('```python'):
            code = code.split('```python')[1].split('```')[0].strip()
        
        # Simple fix 1: Replace VoiceoverScene with Scene
        code = re.sub(r'class\s+(\w+)\s*\(\s*VoiceoverScene\s*\):', r'class \1(Scene):', code)
        
        # Simple fix 2: Remove self.set_speech_service() calls (including multi-line)
        code = re.sub(r'self\.set_speech_service\s*\([^)]*\)\s*', '', code)
        
        # Clean up orphaned parentheses and malformed lines
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip orphaned parentheses and comment lines
            if stripped in [')', ');', '):', "GTTSService(transcription_model='base')"]:
                continue
            cleaned_lines.append(line)
        code = '\n'.join(cleaned_lines)
        
        # Simple fix 3: Replace voiceover blocks with regular code
        # Pattern: with self.voiceover(...): content
        def replace_voiceover_block(match):
            block_content = match.group(1)
            # Remove one level of indentation from the content
            lines = block_content.split('\n')
            dedented_lines = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    # Remove 4 spaces of indentation if present
                    if line.startswith('    '):
                        dedented_lines.append(line[4:])
                    else:
                        dedented_lines.append(line)
                else:
                    dedented_lines.append(line)
            return '\n'.join(dedented_lines)
        
        voiceover_pattern = r'with\s+self\.voiceover\([^)]*\)\s*as\s+\w+:\s*\n((?:[ \t]+.*\n?)*)'
        code = re.sub(voiceover_pattern, replace_voiceover_block, code, flags=re.DOTALL)
        
        # Simple fix 4: Remove voiceover-specific method calls
        code = re.sub(r'self\.wait_until_bookmark\([^)]*\)\s*\n?', 'self.wait(1)\n', code)
        code = re.sub(r'\.get_remaining_duration\(\)[^;\n]*', '', code)
        
        # Simple fix 5: Handle TOC - replace with a simple rectangle for now
        code = re.sub(r'toc\s*=\s*TOC\([^)]*\)', 'toc = Rectangle()', code)
        
        # Simple fix 6: Replace common toc methods with basic alternatives
        code = re.sub(r'toc\.header\.next_to\([^)]*\)', 'Text("Table of Contents")', code)
        code = re.sub(r'toc\.entries\[(\d+)\]\.main', r'Text("Entry \1")', code)
        code = re.sub(r'toc\.entries', 'VGroup()', code)
        code = re.sub(r'toc\.get_open\([^)]*\)', 'Rectangle()', code)
        
        return code
