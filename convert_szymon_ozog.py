#!/usr/bin/env python3
"""
Final converter for szymon_ozog manim_voiceover animations to pure ManimCE

Usage:
    python convert_szymon_ozog.py <input_file> [output_file]
    python convert_szymon_ozog.py --batch <input_dir> <output_dir>
    
This converter handles:
- VoiceoverScene to Scene conversion
- Complex bookmark synchronization
- Custom class replacements (TOC, BSC, Entry, etc.)
- Multi-class inheritance
- Timing estimation based on text length
- Proper indentation preservation
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class VoiceoverToManimConverter:
    def __init__(self):
        # Timing configuration
        self.words_per_minute = 150  # Average speaking rate
        self.min_wait_time = 0.5
        self.max_wait_time = 5.0
        self.bookmark_wait_time = 1.0  # Default wait at bookmarks
        
        # Track conversions for reporting
        self.stats = {
            'files_processed': 0,
            'files_converted': 0,
            'voiceover_blocks': 0,
            'bookmarks_replaced': 0,
            'custom_classes_replaced': 0,
            'errors': []
        }
    
    def estimate_duration(self, text: str) -> float:
        """Estimate speech duration based on text length"""
        # Clean text
        clean_text = re.sub(r'<bookmark mark=[\'"][^"\']+[\'"]/?>', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Count words
        words = len(clean_text.split())
        
        # Calculate duration
        duration = (words / self.words_per_minute) * 60
        
        # Apply constraints
        return round(max(self.min_wait_time, min(duration, self.max_wait_time)), 1)
    
    def convert_voiceover_block(self, match) -> str:
        """Convert a voiceover block to standard Manim animations"""
        self.stats['voiceover_blocks'] += 1
        
        indent = match.group(1)
        text = match.group(2)
        var_name = match.group(3)
        block_content = match.group(4)
        
        # Clean text
        clean_text = text.replace('\\n', ' ').strip()
        bookmarks = re.findall(r'<bookmark mark=[\'"](\w+)[\'"]/?>', clean_text)
        text_no_bookmarks = re.sub(r'<bookmark mark=[\'"][^"\']+[\'"]/?>', '', clean_text).strip()
        
        # Start building output
        result = []
        
        # Add voiceover comment
        if len(text_no_bookmarks) > 80:
            result.append(f'{indent}# Voiceover: "{text_no_bookmarks[:80]}..."')
        else:
            result.append(f'{indent}# Voiceover: "{text_no_bookmarks}"')
        
        # Process block content
        lines = block_content.split('\n')
        animation_lines = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # Handle wait_until_bookmark
            if 'wait_until_bookmark' in line:
                # First, add any pending animation lines
                for anim_line in animation_lines:
                    # Remove extra indentation
                    clean_line = anim_line[len(indent)+4:] if anim_line.startswith(indent + '    ') else anim_line
                    result.append(indent + clean_line)
                animation_lines = []
                
                # Add wait
                self.stats['bookmarks_replaced'] += 1
                result.append(f'{indent}self.wait({self.bookmark_wait_time})')
            else:
                animation_lines.append(line)
        
        # Add remaining animation lines
        for line in animation_lines:
            if line.strip():
                # Remove extra indentation
                clean_line = line[len(indent)+4:] if line.startswith(indent + '    ') else line
                result.append(indent + clean_line)
        
        # Add final wait if needed
        if not bookmarks or animation_lines:
            duration = self.estimate_duration(text)
            result.append(f'{indent}self.wait({duration})')
        
        return '\n'.join(result)
    
    def replace_custom_classes(self, code: str) -> str:
        """Replace custom classes with ManimCE equivalents"""
        
        # TOC replacement
        if 'TOC(' in code:
            self.stats['custom_classes_replaced'] += 1
            
            # Find TOC variable name
            toc_match = re.search(r'(\w+)\s*=\s*TOC\((\d*)\)', code)
            if toc_match:
                var_name = toc_match.group(1)
                episode = toc_match.group(2) or "0"
                
                replacement = f"""# Table of Contents
        {var_name}_title = Text("Information Theory", font_size=72)
        {var_name}_items = VGroup(
            Text("1. Information"),
            Text("2. Entropy"), 
            Text("3. Entropy with multiple events"),
            Text("4. Communication System"),
            Text("5. Noiseless Channel Theorem"),
            Text("6. Noisy Channel Theorem")
        ).arrange(DOWN, aligned_edge=LEFT)"""
                
                code = code.replace(toc_match.group(0), replacement)
                
                # Replace attribute access
                code = re.sub(rf'{var_name}\.header', f'{var_name}_title', code)
                code = re.sub(rf'{var_name}\.entries\[(\d+)\]\.main', rf'{var_name}_items[\1]', code)
                code = re.sub(rf'{var_name}\.get_open\(\d+\)', f'VGroup({var_name}_title, {var_name}_items)', code)
        
        # BSC replacement
        if 'BSC()' in code:
            self.stats['custom_classes_replaced'] += 1
            
            bsc_match = re.search(r'(\w+)\s*=\s*BSC\(\)', code)
            if bsc_match:
                var_name = bsc_match.group(1)
                
                replacement = f"""# Binary Symmetric Channel
        class BSCNamespace: pass
        {var_name} = BSCNamespace()
        {var_name}.input_bit_0 = Circle(radius=0.5, color=BLUE).shift(3*LEFT + 2*UP)
        {var_name}.input_bit_1 = Circle(radius=0.5, color=BLUE).shift(3*LEFT + 2*DOWN)
        {var_name}.output_bit_0 = Circle(radius=0.5, color=BLUE).shift(3*RIGHT + 2*UP)
        {var_name}.output_bit_1 = Circle(radius=0.5, color=BLUE).shift(3*RIGHT + 2*DOWN)
        {var_name}.arrow_00 = Arrow({var_name}.input_bit_0.get_right(), {var_name}.output_bit_0.get_left(), color=GREEN)
        {var_name}.arrow_01 = Arrow({var_name}.input_bit_0.get_right(), {var_name}.output_bit_1.get_left(), color=RED)
        {var_name}.arrow_10 = Arrow({var_name}.input_bit_1.get_right(), {var_name}.output_bit_0.get_left(), color=RED)
        {var_name}.arrow_11 = Arrow({var_name}.input_bit_1.get_right(), {var_name}.output_bit_1.get_left(), color=GREEN)
        {var_name}.bits = VGroup({var_name}.input_bit_0, {var_name}.input_bit_1, {var_name}.output_bit_0, {var_name}.output_bit_1)
        {var_name}.texts = VGroup(Text("0"), Text("1"), Text("0"), Text("1"))
        {var_name}.arrows = VGroup({var_name}.arrow_00, {var_name}.arrow_01, {var_name}.arrow_10, {var_name}.arrow_11)
        {var_name}.labels = VGroup(Text("p"), Text("1-p"), Text("1-p"), Text("p"))
        {var_name}.q_label = Text("q")
        {var_name}.one_minus_q_label = Text("1-q")
        {var_name}.full_channel = VGroup({var_name}.bits, {var_name}.arrows)"""
                
                code = code.replace(bsc_match.group(0), replacement)
        
        # Entry class replacement
        def replace_entry(match):
            var_name = match.group(1)
            title = match.group(2)
            items = match.group(3)
            
            return f"""# Entry
        class EntryNamespace: pass
        {var_name} = EntryNamespace()
        {var_name}.main = Text("{title}")
        {var_name}.list = BulletedList({items})
        {var_name}.open = lambda: Transform({var_name}.main.copy(), {var_name}.list.next_to({var_name}.main, DOWN, aligned_edge=LEFT))"""
        
        code = re.sub(r'(\w+)\s*=\s*Entry\("([^"]+)",\s*\[([^\]]+)\]\)', replace_entry, code)
        
        # EntropyBoxRepresentation
        if 'EntropyBoxRepresentation' in code:
            self.stats['custom_classes_replaced'] += 1
            
            ebr_match = re.search(r'(\w+)\s*=\s*EntropyBoxRepresentation\([^)]*\)', code)
            if ebr_match:
                var_name = ebr_match.group(1)
                
                replacement = f"""# Entropy Box Visualization
        class EBRNamespace: pass
        {var_name} = EBRNamespace()
        {var_name}.whole = VGroup(
            Rectangle(width=4, height=1, fill_opacity=0.8, color=GREEN).shift(UP),
            Text("H(X,Y)", color=GREEN).scale(0.8)
        )
        {var_name}.update = lambda self, probs: self.play({var_name}.whole.animate.set_fill(BLUE))
        {var_name}.set_scale = lambda x: {var_name}.whole.scale(x)"""
                
                code = code.replace(ebr_match.group(0), replacement)
        
        return code
    
    def add_entropy_functions(self, code: str) -> str:
        """Add entropy helper functions if needed"""
        if any(func in code for func in ['HX(', 'HY(', 'HXY(', 'HX_g_Y(', 'HY_g_X(', 'I(', 'make_probs(']):
            # Add numpy import if missing
            if 'import numpy' not in code:
                code = 'import numpy as np\n' + code
            
            # Add helper functions after imports
            helper_code = '''
# Entropy calculation functions
import math

def make_probs(p, q):
    """Create probability matrix for BSC"""
    return [[p*q, (1-p)*q], [(1-q)*p, (1-q)*(1-p)]]

def probs_to_str(probs):
    """Convert probabilities to string format"""
    return [[f"{p:.2f}" for p in row] for row in probs]

def HX(p):
    """Entropy of X"""
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            row_sum = np.sum(p[i,:])
            if row_sum > 1e-10:
                ret -= p[i,j] * math.log2(row_sum)
    return ret

def HY(p):
    """Entropy of Y"""
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            col_sum = np.sum(p[:,j])
            if col_sum > 1e-10:
                ret -= p[i,j] * math.log2(col_sum)
    return ret

def HXY(p):
    """Joint entropy H(X,Y)"""
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i,j] > 1e-10:
                ret -= p[i,j] * math.log2(p[i,j])
    return ret

def HX_g_Y(p):
    """Conditional entropy H(X|Y)"""
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            col_sum = np.sum(p[:,j])
            if col_sum > 1e-10 and p[i,j] > 1e-10:
                pi_g_j = p[i,j] / col_sum
                ret -= p[i,j] * math.log2(pi_g_j)
    return ret

def HY_g_X(p):
    """Conditional entropy H(Y|X)"""
    p = np.array(p)
    ret = 0.0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            row_sum = np.sum(p[i,:])
            if row_sum > 1e-10 and p[i,j] > 1e-10:
                pj_g_i = p[i,j] / row_sum
                ret -= p[i,j] * math.log2(pj_g_i)
    return ret

def I(p):
    """Mutual information I(X;Y)"""
    return HX(p) + HY(p) - HXY(p)

'''
            # Find insertion point
            lines = code.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(('from ', 'import ', '#')):
                    insert_idx = i
                    break
            
            lines.insert(insert_idx, helper_code)
            code = '\n'.join(lines)
        
        return code
    
    def convert_code(self, code: str) -> str:
        """Main conversion function"""
        
        # Step 1: Remove voiceover imports
        code = re.sub(r'from manim_voiceover.*\n', '', code)
        
        # Step 2: Fix class inheritance
        code = re.sub(r'class\s+(\w+)\s*\(\s*VoiceoverScene\s*\)\s*:', r'class \1(Scene):', code)
        code = re.sub(r'class\s+(\w+)\s*\(\s*VoiceoverScene\s*,\s*(\w+)\s*\)\s*:', r'class \1(\2):', code)
        code = re.sub(r'class\s+(\w+)\s*\(\s*(\w+)\s*,\s*VoiceoverScene\s*\)\s*:', r'class \1(\2):', code)
        
        # Step 3: Remove set_speech_service
        code = re.sub(r'self\.set_speech_service\s*\([^)]*?\)(?:\s*#[^\n]*)?\s*\n', '', code, flags=re.DOTALL)
        
        # Step 4: Replace custom classes
        code = self.replace_custom_classes(code)
        
        # Step 5: Convert voiceover blocks
        # Pattern for all voiceover syntaxes
        patterns = [
            # text="..."
            r'^(\s*)with self\.voiceover\s*\(\s*text\s*=\s*"([^"]+)"\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n)*)',
            # """..."""
            r'^(\s*)with self\.voiceover\s*\(\s*"""((?:[^"]|"(?!""))+)"""\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n)*)',
            # f"..."
            r'^(\s*)with self\.voiceover\s*\(\s*f"([^"]+)"\s*\)\s*as\s*(\w+):\s*\n((?:\1\s+.*\n)*)',
        ]
        
        for pattern in patterns:
            code = re.sub(pattern, self.convert_voiceover_block, code, flags=re.MULTILINE)
        
        # Step 6: Clean up remaining wait_until_bookmark
        code = re.sub(r'self\.wait_until_bookmark\([^)]*\)', f'self.wait({self.bookmark_wait_time})', code)
        
        # Step 7: Add helper functions
        code = self.add_entropy_functions(code)
        
        # Step 8: Final cleanup
        lines = code.split('\n')
        cleaned = []
        skip_next = False
        
        for line in lines:
            if skip_next:
                skip_next = False
                continue
                
            # Skip service-related lines
            if any(x in line for x in ['RecorderService', 'GTTSService', 'from entropy import']):
                if line.strip().endswith('('):
                    skip_next = True
                continue
                
            # Skip orphaned syntax
            if line.strip() in [')', ');', '):', '# )']:
                continue
                
            cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def convert_file(self, input_path: Path, output_path: Optional[Path] = None) -> bool:
        """Convert a single file"""
        self.stats['files_processed'] += 1
        
        try:
            # Read file
            with open(input_path, 'r', encoding='utf-8') as f:
                original = f.read()
            
            # Check if it's a voiceover file
            if 'VoiceoverScene' not in original:
                logger.info(f"Skipping {input_path.name} - not a VoiceoverScene")
                return False
            
            # Convert
            logger.info(f"Converting {input_path.name}...")
            converted = self.convert_code(original)
            
            # Write output
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(converted)
                logger.info(f"  -> Saved to {output_path}")
            else:
                print(converted)
            
            self.stats['files_converted'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            self.stats['errors'].append(f"{input_path.name}: {str(e)}")
            return False
    
    def convert_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Convert all Python files in a directory"""
        py_files = list(input_dir.glob("**/*.py"))
        logger.info(f"Found {len(py_files)} Python files")
        
        for py_file in py_files:
            relative_path = py_file.relative_to(input_dir)
            output_path = output_dir / relative_path
            self.convert_file(py_file, output_path)
    
    def print_report(self) -> None:
        """Print conversion report"""
        print("\n" + "="*60)
        print("CONVERSION REPORT")
        print("="*60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files converted: {self.stats['files_converted']}")
        print(f"Voiceover blocks: {self.stats['voiceover_blocks']}")
        print(f"Bookmarks replaced: {self.stats['bookmarks_replaced']}")
        print(f"Custom classes replaced: {self.stats['custom_classes_replaced']}")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description="Convert szymon_ozog voiceover animations to ManimCE")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", nargs="?", help="Output file or directory")
    parser.add_argument("--batch", action="store_true", help="Convert entire directory")
    
    args = parser.parse_args()
    
    converter = VoiceoverToManimConverter()
    
    if args.batch:
        if not args.output:
            print("Error: Output directory required for batch conversion")
            sys.exit(1)
        
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        
        if not input_dir.is_dir():
            print(f"Error: {input_dir} is not a directory")
            sys.exit(1)
        
        converter.convert_directory(input_dir, output_dir)
    else:
        input_file = Path(args.input)
        output_file = Path(args.output) if args.output else None
        
        if not input_file.is_file():
            print(f"Error: {input_file} is not a file")
            sys.exit(1)
        
        converter.convert_file(input_file, output_file)
    
    converter.print_report()


if __name__ == "__main__":
    # If no arguments, run test
    if len(sys.argv) == 1:
        print("Running test conversion...")
        
        test_code = '''from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService

class Example(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService())
        
        title = Text("Test Scene")
        
        with self.voiceover(text="Hello world, <bookmark mark='1'/> this is a test") as trk:
            self.play(Write(title))
            self.wait_until_bookmark("1")
            self.play(title.animate.scale(2))
'''
        
        converter = VoiceoverToManimConverter()
        print("Original:")
        print(test_code)
        print("\n" + "="*60 + "\nConverted:")
        print(converter.convert_code(test_code))
        converter.print_report()
    else:
        main()