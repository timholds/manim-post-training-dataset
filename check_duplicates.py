#!/usr/bin/env python3
"""Check for duplicates across datasets."""

import json
from collections import defaultdict
import hashlib
from pathlib import Path

def normalize_text(text):
    """Normalize text for comparison."""
    return ' '.join(text.lower().split())

def extract_code_content(code):
    """Extract just the code content, removing markdown blocks."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def check_duplicates():
    # Load current dataset
    train_file = Path("data_formatted/train.json")
    
    descriptions = []
    code_samples = []
    desc_hashes = defaultdict(list)
    code_hashes = defaultdict(list)
    
    print("Loading dataset...")
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            desc = data['conversations'][1]['value']  # User message
            code = extract_code_content(data['conversations'][2]['value'])  # Assistant response
            
            # Store original descriptions (before augmentation)
            if not any(prefix in desc for prefix in ["Create a Manim animation that", "Write Manim code to", "Generate a Manim scene"]):
                descriptions.append(desc)
                
                # Hash for exact matches
                desc_hash = hashlib.md5(normalize_text(desc).encode()).hexdigest()
                desc_hashes[desc_hash].append((i, desc))
                
                # Code hash
                code_normalized = ' '.join(code.split())
                code_hash = hashlib.md5(code_normalized.encode()).hexdigest()
                code_hashes[code_hash].append((i, desc, code[:100]))
    
    print(f"\nTotal samples: {i+1}")
    print(f"Unique descriptions (excluding augmented): {len(descriptions)}")
    
    # Find duplicates
    desc_duplicates = {k: v for k, v in desc_hashes.items() if len(v) > 1}
    code_duplicates = {k: v for k, v in code_hashes.items() if len(v) > 1}
    
    print(f"\nDuplicate descriptions found: {len(desc_duplicates)}")
    if desc_duplicates:
        print("\nFirst 5 duplicate descriptions:")
        for i, (hash, items) in enumerate(list(desc_duplicates.items())[:5]):
            print(f"\n{i+1}. Hash: {hash[:8]}... ({len(items)} occurrences)")
            print(f"   Example: {items[0][1][:100]}...")
    
    print(f"\nDuplicate code patterns found: {len(code_duplicates)}")
    if code_duplicates:
        print("\nFirst 5 duplicate code patterns:")
        for i, (hash, items) in enumerate(list(code_duplicates.items())[:5]):
            print(f"\n{i+1}. Hash: {hash[:8]}... ({len(items)} occurrences)")
            print(f"   Description: {items[0][1][:60]}...")
            print(f"   Code start: {items[0][2][:60]}...")
    
    # Check for near-duplicates (similar descriptions)
    print("\n\nChecking for near-duplicates...")
    similar_pairs = []
    for i in range(min(100, len(descriptions))):  # Check first 100 for efficiency
        for j in range(i+1, min(100, len(descriptions))):
            desc1_words = set(normalize_text(descriptions[i]).split())
            desc2_words = set(normalize_text(descriptions[j]).split())
            
            if len(desc1_words) > 3 and len(desc2_words) > 3:
                overlap = len(desc1_words & desc2_words)
                similarity = overlap / max(len(desc1_words), len(desc2_words))
                
                if similarity > 0.8:  # 80% similar
                    similar_pairs.append((i, j, similarity, descriptions[i], descriptions[j]))
    
    if similar_pairs:
        print(f"\nFound {len(similar_pairs)} similar description pairs (>80% word overlap):")
        for i, (idx1, idx2, sim, desc1, desc2) in enumerate(similar_pairs[:3]):
            print(f"\n{i+1}. Similarity: {sim:.1%}")
            print(f"   Desc 1: {desc1[:80]}...")
            print(f"   Desc 2: {desc2[:80]}...")

if __name__ == "__main__":
    check_duplicates()