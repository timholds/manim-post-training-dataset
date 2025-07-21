#!/usr/bin/env python3
"""
Enhanced converter for szymon_ozog manim_voiceover animations to pure ManimCE

This converter handles:
- VoiceoverScene to Scene conversion
- Complex bookmark synchronization
- Custom class replacements (TOC, BSC, Entry, etc.)
- Multi-class inheritance
- Timing estimation based on text length
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SzymonOzogConverter:
    def __init__(self):
        # Timing configuration
        self.words_per_minute = 150  # Average speaking rate
        self.min_wait_time = 0.5
        self.max_wait_time = 5.0
        self.bookmark_wait_time = 0.8  # Default wait at bookmarks
        
        # Track conversions for reporting
        self.conversion_stats = {
            'voiceover_blocks': 0,
            'bookmarks_replaced': 0,
            'custom_classes_replaced': 0,
            'files_converted': 0,
            'errors': []
        }
    
    def estimate_speech_duration(self, text: str) -> float:
        """Estimate speech duration based on text length"""
        # Remove bookmark tags and clean text
        clean_text = re.sub(r'<bookmark mark=[\'"][^"\']+[\'"]/?>', '', text)
        
        # Count words
        words = len(clean_text.split())
        
        # Calculate duration (words per minute to seconds)
        duration = (words / self.words_per_minute) * 60
        
        # Apply min/max constraints
        return max(self.min_wait_time, min(duration, self.max_wait_time))
    
    def replace_custom_classes(self, code: str) -> str:
        """Replace custom classes with standard ManimCE equivalents"""
        
        # Replace TOC class instantiation with simple text
        def replace_toc(match):
            self.conversion_stats['custom_classes_replaced'] += 1
            return """# Table of Contents (simplified from TOC class)
        toc_title = Text("Information Theory", font_size=72)
        toc_items = VGroup(
            Text("1. Information"),
            Text("2. Entropy"),
            Text("3. Entropy with multiple events"),
            Text("4. Communication System"),
            Text("5. Noiseless Channel Theorem"),
            Text("6. Noisy Channel Theorem")
        ).arrange(DOWN, aligned_edge=LEFT)"""
        
        code = re.sub(r'toc = TOC\(\d*\)', replace_toc, code)
        
        # Replace BSC class with inline definition
        bsc_replacement = """# Binary Symmetric Channel (simplified from BSC class)
        bsc_in0 = Circle(radius=0.5, color=BLUE).shift(3*LEFT + 2*UP)
        bsc_in1 = Circle(radius=0.5, color=BLUE).shift(3*LEFT + 2*DOWN)
        bsc_out0 = Circle(radius=0.5, color=BLUE).shift(3*RIGHT + 2*UP)
        bsc_out1 = Circle(radius=0.5, color=BLUE).shift(3*RIGHT + 2*DOWN)
        bsc_arrows = VGroup(
            Arrow(bsc_in0.get_right(), bsc_out0.get_left(), color=GREEN),
            Arrow(bsc_in0.get_right(), bsc_out1.get_left(), color=RED),
            Arrow(bsc_in1.get_right(), bsc_out0.get_left(), color=RED),
            Arrow(bsc_in1.get_right(), bsc_out1.get_left(), color=GREEN)
        )"""
        
        if 'BSC()' in code:
            code = code.replace('bsc = BSC()', bsc_replacement)
            self.conversion_stats['custom_classes_replaced'] += 1
        
        # Replace Entry class references
        code = re.sub(r'Entry\("([^"]+)",\s*\[([^\]]+)\]\)',
                      r'VGroup(Text("\1"), BulletedList(\2))',
                      code)
        
        # Replace TOC methods
        code = re.sub(r'toc\.entries\[(\d+)\]\.main', r'toc_items[\1]', code)
        code = re.sub(r'toc\.get_open\(\d+\)', 'VGroup(toc_title, toc_items)', code)
        
        # Replace EntropyBoxRepresentation (simplified)
        if 'EntropyBoxRepresentation' in code:
            code = re.sub(r'ebr = EntropyBoxRepresentation\([^)]*\)',
                          '# Entropy visualization (simplified)\n        ebr_box = Rectangle(width=4, height=2, fill_opacity=0.8)',
                          code)
            code = re.sub(r'ebr\.update\([^)]+\)', 'self.play(ebr_box.animate.set_fill(BLUE))', code)
            code = re.sub(r'ebr\.whole', 'ebr_box', code)
            self.conversion_stats['custom_classes_replaced'] += 1
        
        return code
    
    def convert_voiceover_block(self, match) -> str:
        """Convert a voiceover block to standard Manim animations"""
        self.conversion_stats['voiceover_blocks'] += 1
        
        indent = match.group(1)
        text = match.group(2)
        var_name = match.group(3) if match.lastindex >= 3 else "trk"
        block_content = match.group(4) if match.lastindex >= 4 else match.group(3)
        
        # Clean up the text
        clean_text = text.replace('\n', ' ').strip()
        
        # Extract bookmarks
        bookmarks = re.findall(r'<bookmark mark=[\'"](\w+)[\'"]/?>', clean_text)
        clean_text_no_bookmarks = re.sub(r'<bookmark mark=[\'"][^"\']+[\'"]/?>', '', clean_text)
        
        # Estimate total duration
        total_duration = self.estimate_speech_duration(clean_text)
        
        # Process the block content
        lines = block_content.split('\n')
        result_lines = []
        
        # Add comment about original voiceover
        if len(clean_text_no_bookmarks) > 100:
            result_lines.append(f'{indent}# Voiceover: "{clean_text_no_bookmarks[:100]}..."')
        else:
            result_lines.append(f'{indent}# Voiceover: "{clean_text_no_bookmarks}"')
        
        # Track sections between bookmarks
        bookmark_sections = []
        current_section = []
        bookmark_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # Check for wait_until_bookmark
            if 'wait_until_bookmark' in line:
                # Save current section
                if current_section:
                    bookmark_sections.append(current_section)
                    current_section = []
                bookmark_count += 1
                self.conversion_stats['bookmarks_replaced'] += 1
            else:
                # Regular animation line - remove extra indentation
                if line.startswith(indent + '    '):
                    current_section.append(line[4:])  # Remove 4 spaces
                elif line.startswith(indent):
                    current_section.append(line[len(indent):])
                else:
                    current_section.append(line)
        
        # Add final section
        if current_section:
            bookmark_sections.append(current_section)
        
        # Calculate timing for each section
        if bookmarks and bookmark_sections:
            # Distribute time proportionally
            time_per_section = total_duration / max(len(bookmark_sections), 1)
            
            for i, section in enumerate(bookmark_sections):
                # Add the animation lines
                result_lines.extend([indent + line for line in section])
                
                # Add wait time if not the last section
                if i < len(bookmark_sections) - 1:
                    wait_time = min(time_per_section, self.bookmark_wait_time)
                    result_lines.append(f'{indent}self.wait({wait_time:.1f})  # Bookmark {i+1}')
        else:
            # No bookmarks - just add all lines with single wait at end
            for line in lines:
                if line.strip() and 'wait_until_bookmark' not in line:
                    if line.startswith(indent + '    '):
                        result_lines.append(line[4:])
                    else:
                        result_lines.append(line)
            
            # Add wait for speech duration
            result_lines.append(f'{indent}self.wait({total_duration:.1f})')
        
        return '\n'.join(result_lines)
    
    def convert_class_inheritance(self, code: str) -> str:
        """Fix class inheritance from VoiceoverScene"""
        # Pattern 1: VoiceoverScene only
        code = re.sub(
            r'class\s+(\w+)\s*\(\s*VoiceoverScene\s*\)\s*:',
            r'class \1(Scene):',
            code
        )
        
        # Pattern 2: VoiceoverScene with other scenes
        code = re.sub(
            r'class\s+(\w+)\s*\(\s*VoiceoverScene\s*,\s*(\w+Scene)\s*\)\s*:',
            r'class \1(\2):',
            code
        )
        
        # Pattern 3: Other scene with VoiceoverScene
        code = re.sub(
            r'class\s+(\w+)\s*\(\s*(\w+Scene)\s*,\s*VoiceoverScene\s*\)\s*:',
            r'class \1(\2):',
            code
        )
        
        return code
    
    def convert_imports(self, code: str) -> str:
        """Clean up imports"""
        # Remove manim_voiceover imports
        code = re.sub(r'from manim_voiceover.*\n', '', code)
        
        # Add numpy import if using entropy functions
        if any(func in code for func in ['HX', 'HY', 'HXY', 'I(', 'make_probs']):
            if 'import numpy' not in code:
                code = 'import numpy as np\n' + code
        
        return code
    
    def add_helper_functions(self, code: str) -> str:
        """Add helper functions that were imported from entropy.py"""
        if any(func in code for func in ['HX(', 'HY(', 'HXY(', 'I(', 'make_probs(']):
            helper_code = '''
# Helper functions from entropy.py
def make_probs(p, q):
    """Create probability matrix for binary symmetric channel"""
    return [[p*q, (1-p)*q], [(1-q)*p, (1-q)*(1-p)]]

def HX(p):
    """Entropy of X"""
    import math
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if np.sum(p[i,:]) > 0:
                ret -= p[i,j] * math.log2(np.sum(p[i,:]))
    return ret

def HY(p):
    """Entropy of Y"""
    import math
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if np.sum(p[:,j]) > 0:
                ret -= p[i,j] * math.log2(np.sum(p[:,j]))
    return ret

def HXY(p):
    """Joint entropy"""
    import math
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i,j] > 0:
                ret -= p[i,j] * math.log2(p[i,j])
    return ret

def HX_g_Y(p):
    """Conditional entropy H(X|Y)"""
    import math
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if np.sum(p[:,j]) > 0 and p[i,j] > 0:
                pi_g_j = p[i,j]/np.sum(p[:,j])
                ret -= p[i,j] * math.log2(pi_g_j)
    return ret

def HY_g_X(p):
    """Conditional entropy H(Y|X)"""
    import math
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if np.sum(p[i,:]) > 0 and p[i,j] > 0:
                pj_g_i = p[i,j]/np.sum(p[i,:])
                ret -= p[i,j] * math.log2(pj_g_i)
    return ret

def I(p):
    """Mutual information"""
    return HY(p) - HY_g_X(p)

'''
            # Insert after imports
            import_end = max(
                code.find('from manim import'),
                code.find('import numpy')
            )
            if import_end > 0:
                # Find the end of the import section
                lines = code[:import_end + 100].split('\n')
                for i, line in enumerate(lines):
                    if line and not line.startswith(('import', 'from')) and not line.strip() == '':
                        insert_point = '\n'.join(lines[:i])
                        rest = '\n'.join(lines[i:]) + code[import_end + 100:]
                        code = insert_point + '\n' + helper_code + rest
                        break
            else:
                code = helper_code + '\n' + code
                
        return code
    
    def convert_code(self, code: str) -> str:
        """Main conversion function"""
        logger.info("Starting conversion...")
        
        # Step 1: Clean imports
        code = self.convert_imports(code)
        
        # Step 2: Fix class inheritance
        code = self.convert_class_inheritance(code)
        
        # Step 3: Remove set_speech_service
        code = re.sub(
            r'self\.set_speech_service\s*\([^)]*\)\s*\n',
            '',
            code,
            flags=re.MULTILINE
        )
        
        # Step 4: Replace custom classes
        code = self.replace_custom_classes(code)
        
        # Step 5: Convert voiceover blocks
        # Pattern for text="..." syntax
        pattern1 = r'^(\s*)with self\.voiceover\s*\(\s*text\s*=\s*"([^"]+)"\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n?)*)'
        code = re.sub(pattern1, self.convert_voiceover_block, code, flags=re.MULTILINE)
        
        # Pattern for """...""" syntax
        pattern2 = r'^(\s*)with self\.voiceover\s*\(\s*"""([^"]+)"""\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n?)*)'
        code = re.sub(pattern2, self.convert_voiceover_block, code, flags=re.MULTILINE)
        
        # Pattern for f"..." syntax
        pattern3 = r'^(\s*)with self\.voiceover\s*\(\s*f"([^"]+)"\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n?)*)'
        code = re.sub(pattern3, self.convert_voiceover_block, code, flags=re.MULTILINE)
        
        # Step 6: Clean up any remaining wait_until_bookmark
        code = re.sub(
            r'self\.wait_until_bookmark\([^)]*\)',
            f'self.wait({self.bookmark_wait_time})',
            code
        )
        
        # Step 7: Add helper functions if needed
        code = self.add_helper_functions(code)
        
        # Step 8: Clean up
        # Remove empty service calls
        code = re.sub(r'RecorderService\([^)]*\)', '', code)
        code = re.sub(r'GTTSService\([^)]*\)', '', code)
        
        # Remove orphaned imports
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if not any(service in line for service in ['RecorderService', 'GTTSService', 'manim_voiceover']):
                # Skip orphaned parentheses
                if line.strip() not in [')', ');', '):']:
                    cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
        
        logger.info(f"Conversion complete. Stats: {self.conversion_stats}")
        
        return code
    
    def convert_file(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Convert a single file"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # Skip if not a voiceover file
            if 'VoiceoverScene' not in original_code:
                logger.info(f"Skipping {input_path} - not a VoiceoverScene")
                return False
            
            converted_code = self.convert_code(original_code)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(converted_code)
                logger.info(f"Converted {input_path} -> {output_path}")
            else:
                print(converted_code)
            
            self.conversion_stats['files_converted'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            self.conversion_stats['errors'].append(f"{input_path}: {str(e)}")
            return False
    
    def get_report(self) -> str:
        """Get conversion report"""
        return f"""
Conversion Report:
==================
Files converted: {self.conversion_stats['files_converted']}
Voiceover blocks: {self.conversion_stats['voiceover_blocks']}
Bookmarks replaced: {self.conversion_stats['bookmarks_replaced']}
Custom classes replaced: {self.conversion_stats['custom_classes_replaced']}
Errors: {len(self.conversion_stats['errors'])}
"""


if __name__ == "__main__":
    # Test the converter
    converter = SzymonOzogConverter()
    
    # Test with a sample
    test_code = '''from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService

class TestScene(VoiceoverScene, ThreeDScene):
    def construct(self):
        self.set_speech_service(RecorderService())
        
        toc = TOC(1)
        
        with self.voiceover(text="Hello <bookmark mark='1'/> world") as trk:
            self.play(Write(toc.header))
            self.wait_until_bookmark("1")
            self.play(Create(Square()))
'''
    
    print("Original:")
    print(test_code)
    print("\n" + "="*60 + "\n")
    print("Converted:")
    converted = converter.convert_code(test_code)
    print(converted)
    print("\n" + converter.get_report())