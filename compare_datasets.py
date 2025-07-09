#!/usr/bin/env python3
"""Compare ManimBench with existing datasets for overlap."""

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

def normalize_text(text):
    """Normalize text for comparison."""
    return ' '.join(text.lower().split())

def load_existing_descriptions():
    """Load descriptions from current dataset."""
    descriptions = set()
    with open("data_formatted/train.json", 'r') as f:
        for line in f:
            data = json.loads(line)
            desc = data['conversations'][1]['value']
            # Get base description (remove augmentation prefixes)
            for prefix in ["Create a Manim animation that ", "Write Manim code to ", "Generate a Manim scene that ", "Implement a Manim animation for: ", "Using Manim, "]:
                if desc.startswith(prefix):
                    desc = desc[len(prefix):]
                    break
            descriptions.add(normalize_text(desc))
    return descriptions

def check_manimbench_overlap():
    """Check overlap between ManimBench and existing datasets."""
    # Load existing descriptions
    existing_descs = load_existing_descriptions()
    print(f"Existing unique descriptions: {len(existing_descs)}")
    
    # Load ManimBench
    manimbench_path = Path.home() / ".cache/manim_datasets/ravidussilva_manim-sft_manim_sft_dataset.parquet"
    df = pd.read_parquet(manimbench_path)
    
    print(f"\nManimBench samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check overlaps
    exact_matches = 0
    similar_matches = 0
    unique_samples = []
    
    for idx, row in df.iterrows():
        desc = normalize_text(row['Reviewed Description'])
        
        if desc in existing_descs:
            exact_matches += 1
        else:
            # Check for similarity
            desc_words = set(desc.split())
            is_similar = False
            
            for existing in list(existing_descs)[:1000]:  # Check first 1000 for efficiency
                existing_words = set(existing.split())
                if len(desc_words) > 3 and len(existing_words) > 3:
                    overlap = len(desc_words & existing_words)
                    similarity = overlap / max(len(desc_words), len(existing_words))
                    if similarity > 0.8:
                        similar_matches += 1
                        is_similar = True
                        break
            
            if not is_similar:
                unique_samples.append(row['Reviewed Description'])
    
    print(f"\nOverlap Analysis:")
    print(f"- Exact matches: {exact_matches} ({exact_matches/len(df)*100:.1f}%)")
    print(f"- Similar matches (>80% word overlap): {similar_matches}")
    print(f"- Potentially unique: {len(unique_samples)} ({len(unique_samples)/len(df)*100:.1f}%)")
    
    print(f"\nFirst 5 unique ManimBench descriptions:")
    for i, desc in enumerate(unique_samples[:5]):
        print(f"{i+1}. {desc[:100]}...")
    
    # Show split distribution
    print(f"\nManimBench split distribution:")
    print(df['Split'].value_counts())

if __name__ == "__main__":
    check_manimbench_overlap()