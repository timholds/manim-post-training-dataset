#!/usr/bin/env python3
"""
Analyze all voiceover patterns in szymon_ozog dataset
"""
import re
import os
from pathlib import Path
from collections import defaultdict

def analyze_voiceover_patterns():
    """Analyze all patterns used in szymon_ozog voiceover code"""
    
    patterns = defaultdict(list)
    voiceover_texts = []
    bookmark_usages = []
    custom_classes = set()
    inheritance_patterns = []
    
    # Directories to analyze
    dirs = [
        "data/data_szymon_ozog/GPU_Programming/manim_scripts",
        "data/data_szymon_ozog/InformationTheory"
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            continue
            
        for file_path in Path(dir_path).glob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Skip non-manim files
            if "from manim import" not in content:
                continue
                
            print(f"\nAnalyzing: {file_path.name}")
            
            # 1. Analyze class inheritance patterns
            class_matches = re.findall(r'class\s+(\w+)\s*\(([^)]+)\):', content)
            for class_name, inheritance in class_matches:
                inheritance_patterns.append((class_name, inheritance.strip()))
                print(f"  Class: {class_name}({inheritance.strip()})")
            
            # 2. Find all voiceover blocks with different syntaxes
            # Pattern 1: text="..."
            vo_pattern1 = re.findall(
                r'with self\.voiceover\s*\(\s*text\s*=\s*"([^"]+)"\s*\)\s*as\s*(\w+):',
                content, re.DOTALL
            )
            # Pattern 2: """..."""
            vo_pattern2 = re.findall(
                r'with self\.voiceover\s*\(\s*"""([^"]+)"""\s*\)\s*as\s*(\w+):',
                content, re.DOTALL
            )
            # Pattern 3: f-strings
            vo_pattern3 = re.findall(
                r'with self\.voiceover\s*\(\s*f"([^"]+)"\s*\)\s*as\s*(\w+):',
                content, re.DOTALL
            )
            
            all_voiceovers = vo_pattern1 + vo_pattern2 + vo_pattern3
            print(f"  Voiceover blocks: {len(all_voiceovers)}")
            
            for text, var_name in all_voiceovers:
                voiceover_texts.append(text)
                # Count bookmarks in this voiceover
                bookmarks = re.findall(r'<bookmark mark=[\'"](\w+)[\'"]/?>', text)
                if bookmarks:
                    bookmark_usages.append((text[:50], bookmarks))
            
            # 3. Find wait_until_bookmark patterns
            wait_patterns = re.findall(r'self\.wait_until_bookmark\([\'"](\w+)[\'"]\)', content)
            print(f"  wait_until_bookmark calls: {len(wait_patterns)}")
            
            # 4. Find custom class usage
            # Look for TOC, BSC, Entry, etc.
            for class_name in ['TOC', 'BSC', 'Entry', 'EntropyBoxRepresentation', 
                               'RecorderService', 'GTTSService']:
                if class_name + '(' in content:
                    custom_classes.add(class_name)
                    patterns[class_name].append(file_path.name)
            
            # 5. Analyze voiceover block complexity
            # Find multi-line voiceover blocks
            complex_blocks = re.findall(
                r'with self\.voiceover.*?\n((?:\s+.*\n)+?)(?=\n\S|\Z)',
                content, re.DOTALL
            )
            if complex_blocks:
                max_lines = max(len(block.strip().split('\n')) for block in complex_blocks)
                print(f"  Max lines in voiceover block: {max_lines}")
    
    # Summary
    print("\n" + "="*60)
    print("PATTERN ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal voiceover texts found: {len(voiceover_texts)}")
    print(f"Unique inheritance patterns: {len(set(inheritance_patterns))}")
    for class_name, inheritance in sorted(set(inheritance_patterns)):
        print(f"  - class {class_name}({inheritance})")
    
    print(f"\nCustom classes used:")
    for class_name in sorted(custom_classes):
        print(f"  - {class_name}: used in {len(patterns[class_name])} files")
    
    print(f"\nBookmark usage examples:")
    for text, bookmarks in bookmark_usages[:5]:
        print(f"  Text: '{text}...'")
        print(f"  Bookmarks: {bookmarks}")
    
    # Analyze text lengths for timing estimation
    text_lengths = [len(text.split()) for text in voiceover_texts]
    if text_lengths:
        avg_words = sum(text_lengths) / len(text_lengths)
        max_words = max(text_lengths)
        print(f"\nVoiceover text statistics:")
        print(f"  Average words per voiceover: {avg_words:.1f}")
        print(f"  Max words in a voiceover: {max_words}")
        print(f"  Suggested timing: ~{avg_words * 0.15:.1f}s average (150ms per word)")
    
    return {
        'voiceover_texts': voiceover_texts,
        'custom_classes': custom_classes,
        'inheritance_patterns': inheritance_patterns,
        'bookmark_usages': bookmark_usages
    }

if __name__ == "__main__":
    results = analyze_voiceover_patterns()