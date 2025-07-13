"""
Fix the thanks_dataset by removing entries with mismatched code-description pairs.

Strategy:
1. Identify all codes that appear with multiple different descriptions
2. Remove ALL entries that use these problematic codes
3. Keep only entries where code is unique to one description
"""

import json
from collections import defaultdict
from datasets import load_dataset
import os

def fix_thanks_dataset():
    print("Loading thanhkt/manim_code dataset...")
    dataset = load_dataset("thanhkt/manim_code", split="train")
    
    # First pass: identify problematic codes
    code_to_descriptions = defaultdict(set)
    all_entries = []
    
    for idx, item in enumerate(dataset):
        description = str(item.get("input", "")).strip()
        code = str(item.get("output", ""))
        
        # Clean up code
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        if code and description:
            code_to_descriptions[code].add(description)
            all_entries.append({
                'description': description,
                'code': code,
                'index': idx
            })
    
    # Identify problematic codes (used for multiple descriptions)
    problematic_codes = set()
    for code, descriptions in code_to_descriptions.items():
        if len(descriptions) > 1:
            problematic_codes.add(code)
    
    print(f"\nOriginal dataset size: {len(all_entries)}")
    print(f"Unique codes: {len(code_to_descriptions)}")
    print(f"Problematic codes (multiple descriptions): {len(problematic_codes)}")
    
    # Filter out entries with problematic codes
    clean_entries = []
    removed_entries = []
    
    for entry in all_entries:
        if entry['code'] not in problematic_codes:
            clean_entries.append(entry)
        else:
            removed_entries.append(entry)
    
    print(f"\nClean entries: {len(clean_entries)}")
    print(f"Removed entries: {len(removed_entries)}")
    
    # Save the cleaned dataset
    output_dir = "data/thanks_dataset_cleaned"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to conversation format to match other datasets
    formatted_entries = []
    for entry in clean_entries:
        formatted_entry = {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
                },
                {
                    "from": "user",
                    "value": entry['description']
                },
                {
                    "from": "assistant",
                    "value": f"```python\n{entry['code']}\n```"
                }
            ],
            "source": "thanks_dataset_cleaned"
        }
        formatted_entries.append(formatted_entry)
    
    # Save as JSONL
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        for entry in formatted_entries:
            f.write(json.dumps(entry) + "\n")
    
    # Save statistics
    stats = {
        "original_size": len(all_entries),
        "cleaned_size": len(clean_entries),
        "removed_size": len(removed_entries),
        "unique_codes_original": len(code_to_descriptions),
        "problematic_codes": len(problematic_codes),
        "cleaning_percentage": f"{len(removed_entries) / len(all_entries) * 100:.1f}%"
    }
    
    with open(os.path.join(output_dir, "cleaning_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nCleaning complete!")
    print(f"Removed {stats['cleaning_percentage']} of entries")
    print(f"Saved clean dataset to {output_dir}")
    
    # Also save a sample of removed entries for inspection
    sample_removed = removed_entries[:100]
    with open(os.path.join(output_dir, "removed_samples.json"), "w") as f:
        json.dump(sample_removed, f, indent=2)
    
    return stats

if __name__ == "__main__":
    stats = fix_thanks_dataset()