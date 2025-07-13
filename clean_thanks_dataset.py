#!/usr/bin/env python3
"""
Improved cleaning script for thanks_dataset with adjustable filtering.

This version is less strict and more intelligent about what to remove:
1. Only removes entries where code is COMPLETELY unrelated to description
2. Keeps valid short animations
3. Uses similarity checking to identify true mismatches
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse

from datasets import load_dataset
from extractors.utils import normalize_code, normalize_description, calculate_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_manim_concepts(text: str) -> Set[str]:
    """Extract Manim-related concepts from text."""
    concepts = set()
    
    # Common Manim objects and concepts
    keywords = [
        'circle', 'square', 'rectangle', 'triangle', 'polygon', 'line', 'arrow',
        'text', 'tex', 'equation', 'graph', 'axes', 'plot', 'curve',
        'transform', 'move', 'shift', 'rotate', 'scale', 'fade', 'create',
        'integral', 'derivative', 'matrix', 'vector', 'angle',
        'animation', 'color', 'red', 'blue', 'green', 'yellow',
        '3d', 'surface', 'sphere', 'cube', 'cone', 'torus'
    ]
    
    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            concepts.add(keyword)
    
    return concepts


def is_code_relevant_to_description(code: str, description: str, threshold: float = 0.3) -> bool:
    """
    Check if code is relevant to description using multiple heuristics.
    
    Returns True if code seems relevant, False if it's clearly mismatched.
    """
    # Extract concepts from both
    desc_concepts = extract_manim_concepts(description)
    code_concepts = extract_manim_concepts(code)
    
    # If description mentions specific objects, code should contain at least some of them
    if desc_concepts:
        overlap = desc_concepts & code_concepts
        if len(overlap) / len(desc_concepts) < threshold:
            return False
    
    # Check for obvious mismatches
    mismatch_patterns = [
        # Description asks for one thing, code does something completely different
        ("integral", "class.*Ball"),  # Asking for integral, getting bouncing ball
        ("chemical", "integral"),      # Asking for chemistry, getting math
        ("molecule", "graph.*plot"),   # Asking for molecule, getting graph
        ("equation", "molecule"),      # Asking for equation, getting molecule
    ]
    
    desc_lower = description.lower()
    code_lower = code.lower()
    
    for desc_pattern, code_pattern in mismatch_patterns:
        if desc_pattern in desc_lower and code_pattern in code_lower:
            return False
    
    return True


def clean_dataset_intelligent(
    min_code_length: int = 50,
    max_duplicates_per_code: int = 3,
    similarity_threshold: float = 0.85,
    relevance_threshold: float = 0.3,
    output_dir: str = "data/thanks_dataset_cleaned"
) -> Dict[str, any]:
    """
    Clean the dataset with more intelligent filtering.
    
    Args:
        min_code_length: Minimum code length to keep (very short code is often generic)
        max_duplicates_per_code: Maximum times the same code can appear
        similarity_threshold: Threshold for considering descriptions similar
        relevance_threshold: Threshold for concept overlap
        output_dir: Output directory for cleaned dataset
    """
    logger.info("Loading thanhkt/manim_code dataset...")
    dataset = load_dataset("thanhkt/manim_code", split="train")
    
    # Group entries by code
    code_to_entries = defaultdict(list)
    all_entries = []
    
    wrapper_prefix = "Generate accurate and correct ManimCE Python code for the animation requested by the user. Here is the user's request:"
    
    for idx, item in enumerate(dataset):
        description = str(item.get("input", "")).strip()
        code = str(item.get("output", ""))
        
        # Remove wrapper text if present
        if description.startswith(wrapper_prefix):
            description = description[len(wrapper_prefix):].strip()
        elif "Here is the user's request:" in description:
            start = description.find("Here is the user's request:") + len("Here is the user's request:")
            description = description[start:].strip()
        
        # Clean up code
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        # Remove literal \n from the beginning
        if code.startswith('\\n'):
            code = code[2:].lstrip()
        
        if code and description:
            entry = {
                'description': description,
                'code': code,
                'index': idx,
                'norm_desc': normalize_description(description),
                'norm_code': normalize_code(code)
            }
            code_to_entries[code].append(entry)
            all_entries.append(entry)
    
    logger.info(f"Original dataset size: {len(all_entries)}")
    logger.info(f"Unique codes: {len(code_to_entries)}")
    
    # Identify problematic entries
    entries_to_remove = set()
    removal_reasons = defaultdict(int)
    
    for code, entries in code_to_entries.items():
        # Skip if code is too short (likely generic)
        if len(code) < min_code_length:
            for entry in entries:
                entries_to_remove.add(entry['index'])
                removal_reasons['code_too_short'] += 1
            continue
        
        # If same code appears too many times, check relevance
        if len(entries) > 1:
            # Group similar descriptions
            description_groups = []
            for entry in entries:
                # Check if this description belongs to an existing group
                added_to_group = False
                for group in description_groups:
                    # Compare with first entry in group
                    similarity = calculate_similarity(
                        entry['norm_desc'], 
                        group[0]['norm_desc']
                    )
                    if similarity > similarity_threshold:
                        group.append(entry)
                        added_to_group = True
                        break
                
                if not added_to_group:
                    description_groups.append([entry])
            
            # If we have multiple distinct description groups for same code
            if len(description_groups) > max_duplicates_per_code:
                # Keep the largest group, check relevance for others
                description_groups.sort(key=len, reverse=True)
                
                for i, group in enumerate(description_groups[1:], 1):
                    for entry in group:
                        # Check if code is relevant to description
                        if not is_code_relevant_to_description(
                            entry['code'], 
                            entry['description'],
                            relevance_threshold
                        ):
                            entries_to_remove.add(entry['index'])
                            removal_reasons['irrelevant_code'] += 1
                        elif i >= max_duplicates_per_code:
                            entries_to_remove.add(entry['index'])
                            removal_reasons['too_many_duplicates'] += 1
            
            # Also check relevance for entries with mismatched code
            elif len(description_groups) > 1:
                for group in description_groups:
                    for entry in group:
                        if not is_code_relevant_to_description(
                            entry['code'], 
                            entry['description'],
                            relevance_threshold
                        ):
                            entries_to_remove.add(entry['index'])
                            removal_reasons['irrelevant_code'] += 1
    
    # Build cleaned dataset
    clean_entries = [e for e in all_entries if e['index'] not in entries_to_remove]
    removed_entries = [e for e in all_entries if e['index'] in entries_to_remove]
    
    logger.info(f"\nCleaning complete!")
    logger.info(f"Clean entries: {len(clean_entries)}")
    logger.info(f"Removed entries: {len(removed_entries)} ({len(removed_entries)/len(all_entries)*100:.1f}%)")
    logger.info(f"\nRemoval reasons:")
    for reason, count in removal_reasons.items():
        logger.info(f"  {reason}: {count}")
    
    # Save the cleaned dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Format entries for output
    formatted_entries = []
    for entry in clean_entries:
        formatted_entry = {
            "description": entry['description'],
            "code": entry['code'],
            "metadata": {
                "row_index": entry['index'],
                "source": "thanhkt_cleaned"
            }
        }
        formatted_entries.append(formatted_entry)
    
    # Save as JSONL
    with open(output_path / "train.json", "w") as f:
        for entry in formatted_entries:
            f.write(json.dumps(entry) + "\n")
    
    # Save statistics
    stats = {
        "original_size": len(all_entries),
        "cleaned_size": len(clean_entries),
        "removed_size": len(removed_entries),
        "removal_percentage": f"{len(removed_entries)/len(all_entries)*100:.1f}%",
        "removal_reasons": dict(removal_reasons),
        "unique_codes_original": len(code_to_entries),
        "parameters": {
            "min_code_length": min_code_length,
            "max_duplicates_per_code": max_duplicates_per_code,
            "similarity_threshold": similarity_threshold,
            "relevance_threshold": relevance_threshold
        }
    }
    
    with open(output_path / "cleaning_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save sample of removed entries for inspection
    sample_removed = removed_entries[:100]
    with open(output_path / "removed_samples.json", "w") as f:
        json.dump([{
            'description': e['description'],
            'code': e['code'][:200] + '...' if len(e['code']) > 200 else e['code'],
            'index': e['index']
        } for e in sample_removed], f, indent=2)
    
    logger.info(f"\nSaved clean dataset to {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean thanks_dataset with intelligent filtering")
    parser.add_argument("--min-code-length", type=int, default=50, 
                        help="Minimum code length to keep")
    parser.add_argument("--max-duplicates", type=int, default=3,
                        help="Maximum times same code can appear")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Threshold for description similarity")
    parser.add_argument("--relevance-threshold", type=float, default=0.3,
                        help="Threshold for code-description relevance")
    parser.add_argument("--output-dir", default="data/thanks_dataset_cleaned",
                        help="Output directory")
    
    args = parser.parse_args()
    
    stats = clean_dataset_intelligent(
        min_code_length=args.min_code_length,
        max_duplicates_per_code=args.max_duplicates,
        similarity_threshold=args.similarity_threshold,
        relevance_threshold=args.relevance_threshold,
        output_dir=args.output_dir
    )
    
    print("\nFinal statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()