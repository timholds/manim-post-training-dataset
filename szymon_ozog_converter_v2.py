#!/usr/bin/env python3
"""
Enhanced converter for szymon_ozog manim_voiceover animations to pure ManimCE
Version 2 - Fixed indentation and improved replacements
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
        
        # Track TOC replacements
        toc_replaced = False
        
        # Replace TOC class instantiation
        if 'TOC(' in code:
            toc_replaced = True
            self.conversion_stats['custom_classes_replaced'] += 1
            
            # Replace TOC instantiation
            code = re.sub(r'toc = TOC\(\d*\)', 
                         """# Table of Contents (simplified from TOC class)
        toc_title = Text("Information Theory", font_size=72)
        toc_items = VGroup(
            Text("1. Information"),
            Text("2. Entropy"),
            Text("3. Entropy with multiple events"),
            Text("4. Communication System"),
            Text("5. Noiseless Channel Theorem"),
            Text("6. Noisy Channel Theorem")
        ).arrange(DOWN, aligned_edge=LEFT)
        toc_header = toc_title  # Alias for compatibility""", code)
            
            # Replace TOC attribute access
            code = re.sub(r'toc\.header', 'toc_header', code)
            code = re.sub(r'toc\.entries\[(\d+)\]\.main', r'toc_items[\1]', code)
            code = re.sub(r'toc\.get_open\(\d+\)', 'VGroup(toc_title, toc_items)', code)
        
        # Replace BSC class
        if 'BSC()' in code:
            self.conversion_stats['custom_classes_replaced'] += 1
            bsc_replacement = """# Binary Symmetric Channel (simplified from BSC class)
        bsc = lambda: None  # Create namespace
        bsc.input_bit_0 = Circle(radius=0.5, color=BLUE).shift(3*LEFT + 2*UP)
        bsc.input_bit_1 = Circle(radius=0.5, color=BLUE).shift(3*LEFT + 2*DOWN)
        bsc.output_bit_0 = Circle(radius=0.5, color=BLUE).shift(3*RIGHT + 2*UP)
        bsc.output_bit_1 = Circle(radius=0.5, color=BLUE).shift(3*RIGHT + 2*DOWN)
        bsc.arrow_00 = Arrow(bsc.input_bit_0.get_right(), bsc.output_bit_0.get_left(), color=GREEN)
        bsc.arrow_01 = Arrow(bsc.input_bit_0.get_right(), bsc.output_bit_1.get_left(), color=RED)
        bsc.arrow_10 = Arrow(bsc.input_bit_1.get_right(), bsc.output_bit_0.get_left(), color=RED)
        bsc.arrow_11 = Arrow(bsc.input_bit_1.get_right(), bsc.output_bit_1.get_left(), color=GREEN)
        bsc.bits = VGroup(bsc.input_bit_0, bsc.input_bit_1, bsc.output_bit_0, bsc.output_bit_1)
        bsc.full_channel = VGroup(bsc.bits, bsc.arrow_00, bsc.arrow_01, bsc.arrow_10, bsc.arrow_11)"""
            
            code = code.replace('bsc = BSC()', bsc_replacement)
        
        # Replace Entry class
        entry_pattern = r'(\w+)\s*=\s*Entry\("([^"]+)",\s*\[([^\]]+)\]\)'
        def replace_entry(match):
            var_name = match.group(1)
            title = match.group(2)
            items = match.group(3)
            return f"""{var_name} = lambda: None  # Entry namespace
        {var_name}.main = Text("{title}")
        {var_name}.list = BulletedList({items})"""
        
        code = re.sub(entry_pattern, replace_entry, code)
        
        # Replace EntropyBoxRepresentation
        if 'EntropyBoxRepresentation' in code:
            self.conversion_stats['custom_classes_replaced'] += 1
            code = re.sub(r'ebr = EntropyBoxRepresentation\([^)]*\)',
                          """# Entropy visualization (simplified)
        ebr = lambda: None
        ebr.whole = VGroup(
            Rectangle(width=4, height=1, fill_opacity=0.8, color=GREEN),
            Text("H(X,Y)", color=GREEN).scale(0.8)
        )""", code)
            code = re.sub(r'ebr\.update\([^)]+\)', 'self.play(ebr.whole.animate.set_fill(BLUE))', code)
        
        return code
    
    def convert_voiceover_block(self, match) -> str:
        """Convert a voiceover block to standard Manim animations"""
        self.conversion_stats['voiceover_blocks'] += 1
        
        indent = match.group(1)
        text = match.group(2)
        var_name = match.group(3) if match.lastindex >= 3 else "trk"
        
        # For pattern with block content
        if match.lastindex >= 4:
            block_content = match.group(4)
        else:
            # For simpler patterns, find the indented block after
            block_content = match.group(3) if match.lastindex >= 3 else ""
        
        # Clean up the text
        clean_text = text.replace('\n', ' ').strip()
        
        # Extract bookmarks
        bookmarks = re.findall(r'<bookmark mark=[\'"](\w+)[\'"]/?>', clean_text)
        clean_text_no_bookmarks = re.sub(r'<bookmark mark=[\'"][^"\']+[\'"]/?>', '', clean_text).strip()
        
        # Estimate total duration
        total_duration = self.estimate_speech_duration(clean_text)
        
        # Process the block content
        if not block_content:
            # Empty block
            return f'{indent}# Voiceover: "{clean_text_no_bookmarks}"\n{indent}self.wait({total_duration:.1f})'
        
        lines = block_content.split('\n')
        result_lines = []
        
        # Add comment about original voiceover
        if len(clean_text_no_bookmarks) > 100:
            result_lines.append(f'{indent}# Voiceover: "{clean_text_no_bookmarks[:100]}..."')
        else:
            result_lines.append(f'{indent}# Voiceover: "{clean_text_no_bookmarks}"')
        
        # Process lines and handle bookmarks
        current_lines = []
        bookmark_count = 0
        
        for line in lines:
            if not line.strip():
                continue
                
            # Check for wait_until_bookmark
            if 'wait_until_bookmark' in line:
                # Add current lines first
                for curr_line in current_lines:
                    # Fix indentation
                    if curr_line.startswith(indent + '    '):
                        result_lines.append(indent + curr_line[len(indent)+4:])
                    else:
                        result_lines.append(curr_line)
                current_lines = []
                
                # Add wait for bookmark
                bookmark_count += 1
                self.conversion_stats['bookmarks_replaced'] += 1
                result_lines.append(f'{indent}self.wait({self.bookmark_wait_time})')
            else:
                current_lines.append(line)
        
        # Add remaining lines
        for line in current_lines:
            if line.strip():
                # Fix indentation
                if line.startswith(indent + '    '):
                    result_lines.append(indent + line[len(indent)+4:])
                else:
                    result_lines.append(line)
        
        # Add final wait if no bookmarks or if there's content after last bookmark
        if bookmark_count == 0 or current_lines:
            result_lines.append(f'{indent}self.wait({total_duration:.1f})')
        
        return '\n'.join(result_lines)
    
    def convert_code(self, code: str) -> str:
        """Main conversion function"""
        logger.info("Starting conversion...")
        
        # Step 1: Clean imports
        code = re.sub(r'from manim_voiceover.*\n', '', code)
        
        # Step 2: Fix class inheritance
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
        
        # Step 3: Remove set_speech_service (match multi-line)
        code = re.sub(
            r'self\.set_speech_service\s*\([^)]*\)(?:\s*#[^\n]*)?\s*\n',
            '',
            code,
            flags=re.MULTILINE | re.DOTALL
        )
        
        # Step 4: Replace custom classes
        code = self.replace_custom_classes(code)
        
        # Step 5: Convert voiceover blocks
        # Need to handle indented blocks properly
        
        # Pattern 1: text="..." with explicit as variable
        def convert_with_content(match):
            return self.convert_voiceover_block(match)
        
        pattern1 = r'^(\s*)with self\.voiceover\s*\(\s*text\s*=\s*"([^"]+)"\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n)*)'
        code = re.sub(pattern1, convert_with_content, code, flags=re.MULTILINE)
        
        # Pattern 2: """...""" with triple quotes
        pattern2 = r'^(\s*)with self\.voiceover\s*\(\s*"""((?:[^"]|"(?!""))+)"""\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n)*)'
        code = re.sub(pattern2, convert_with_content, code, flags=re.MULTILINE)
        
        # Pattern 3: f-strings
        pattern3 = r'^(\s*)with self\.voiceover\s*\(\s*f"([^"]+)"\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n)*)'
        code = re.sub(pattern3, convert_with_content, code, flags=re.MULTILINE)
        
        # Step 6: Clean up any remaining wait_until_bookmark
        code = re.sub(
            r'self\.wait_until_bookmark\([^)]*\)',
            f'self.wait({self.bookmark_wait_time})',
            code
        )
        
        # Step 7: Add helper functions if needed
        if any(func in code for func in ['HX(', 'HY(', 'HXY(', 'I(', 'make_probs(']):
            if 'import numpy' not in code:
                code = 'import numpy as np\n' + code
            
            # Add helper functions after imports
            helper_code = '''
# Helper functions from entropy.py
def make_probs(p, q):
    """Create probability matrix for binary symmetric channel"""
    return [[p*q, (1-p)*q], [(1-q)*p, (1-q)*(1-p)]]

def probs_to_str(probs):
    """Convert probability matrix to string representation"""
    return [[f"{p:.2f}" for p in row] for row in probs]

# Simplified entropy functions (add full implementations if needed)
def HX(p): return 1.0  # Placeholder
def HY(p): return 1.0  # Placeholder
def HXY(p): return 1.8  # Placeholder
def HX_g_Y(p): return 0.5  # Placeholder
def HY_g_X(p): return 0.5  # Placeholder
def I(p): return 0.3  # Placeholder

'''
            # Find where to insert
            lines = code.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(('from ', 'import ')):
                    import_end = i
                    break
            
            lines.insert(import_end, helper_code)
            code = '\n'.join(lines)
        
        # Step 8: Final cleanup
        lines = code.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            # Skip service-related lines
            if any(x in line for x in ['RecorderService', 'GTTSService', '# GTTSService']):
                # Check if it's a multi-line statement
                if line.strip().endswith('('):
                    skip_next = True
                continue
                
            # Skip orphaned parentheses
            if line.strip() in [')', ');', '):', '# )']:
                continue
                
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


def test_converter():
    """Test the converter with various examples"""
    converter = SzymonOzogConverter()
    
    # Test 1: Simple example
    test_code1 = '''from manim import *
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
    
    print("Test 1 - Simple example:")
    print("="*60)
    converted1 = converter.convert_code(test_code1)
    print(converted1)
    
    # Test 2: Real example from GPU programming
    test_code2 = '''from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService

class GPUIntro(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            RecorderService(trim_buffer_end=50)
        )
        
        title = Text("GPU Programming", font_size=72)
        
        with self.voiceover(text="Welcome to GPU programming") as trk:
            self.play(Write(title))
        
        with self.voiceover(text="Let's explore <bookmark mark='1'/> the architecture") as trk:
            self.play(title.animate.shift(UP))
            self.wait_until_bookmark("1")
            gpu = Square(color=BLUE)
            self.play(Create(gpu))
'''
    
    print("\n\nTest 2 - GPU example:")
    print("="*60)
    converter.conversion_stats = {
        'voiceover_blocks': 0,
        'bookmarks_replaced': 0,
        'custom_classes_replaced': 0,
        'files_converted': 0,
        'errors': []
    }
    converted2 = converter.convert_code(test_code2)
    print(converted2)
    
    print("\n" + converter.get_report())


if __name__ == "__main__":
    test_converter()