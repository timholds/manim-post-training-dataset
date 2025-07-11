#!/usr/bin/env python3
"""
Process LLM-generated descriptions and create final dataset.
This shows how to integrate LLM descriptions back into the dataset.
"""

import json
from pathlib import Path
from typing import List, Dict

# Example LLM-generated descriptions (in practice, these would come from actual LLM calls)
EXAMPLE_LLM_DESCRIPTIONS = {
    1: "I need an animation that visualizes comparing two lists of numbers side by side. Show the lists as columns of numbered squares, then draw lines connecting corresponding elements after sorting. Include running counters at the top showing the sum of differences. In the second part, show how many times each number from the first list appears in the second list, highlighting matches in yellow.",
    
    22: "Create an animation that demonstrates a pseudorandom number generator algorithm. Start by showing how XOR operations work with binary representations of numbers. Then illustrate the three-step process: multiply by 64 and XOR, divide by 32 and XOR, multiply by 2048 and XOR, each followed by modulo 16777216. Show the pseudocode on the left and step through actual calculations on the right. Include a section that generates the first 10 numbers in the sequence with a pointer tracking progress.",
    
    3: "I'd like an animation showing how to parse multiplication instructions from corrupted text. Display the text with regex patterns, highlight valid 'mul(X,Y)' instructions in color, and show a running total of the products. Include visual indicators for when instructions are enabled or disabled by 'do()' and 'don't()' commands.",
}

def load_raw_samples(input_path: Path) -> List[Dict]:
    """Load raw samples with placeholder descriptions"""
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def apply_llm_descriptions(samples: List[Dict], descriptions: Dict[int, str]) -> List[Dict]:
    """Apply LLM-generated descriptions to samples"""
    updated_samples = []
    
    for sample in samples:
        day = sample["metadata"]["day"]
        
        if day in descriptions:
            # Update the user prompt with LLM-generated description
            sample["conversations"][1]["value"] = descriptions[day]
            
            # Update metadata to indicate LLM generation
            sample["metadata"]["description_generated_by"] = "llm_example"
            sample["metadata"]["description_quality"] = "high"  # Can be used for augmentation decisions
            
            # Remove the placeholder marker
            if "[TO BE GENERATED:" in sample["conversations"][1]["value"]:
                continue  # Skip if description wasn't updated
        
        updated_samples.append(sample)
    
    return updated_samples

def create_augmentation_config(sample: Dict) -> Dict:
    """Create augmentation configuration based on metadata"""
    metadata = sample.get("metadata", {})
    
    if metadata.get("description_generated_by", "").startswith("llm"):
        # LLM-generated descriptions can have more aggressive augmentation
        return {
            "augmentation_factor": 3.0,  # More variations
            "variation_templates": [
                "{original}",
                "Could you create a Manim animation that {description_lower}",
                "I need a visualization that {description_lower}",
                "Please generate an animation showing {description_lower}",
                "Help me create an educational animation that {description_lower}",
                "Can you make an animated demonstration of {description_lower}"
            ],
            "preserve_original": True
        }
    else:
        # Human-written descriptions: more conservative augmentation
        return {
            "augmentation_factor": 1.5,
            "variation_templates": [
                "{original}",
                "Create an animation that {description_lower}"
            ],
            "preserve_original": True
        }

def save_final_dataset(samples: List[Dict], output_path: Path):
    """Save the final dataset with LLM descriptions"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Separate samples by description source
    llm_generated = []
    human_written = []
    
    with open(output_path, 'w') as f:
        for sample in samples:
            # Remove metadata from final output (keep it separate)
            output_sample = {
                "conversations": sample["conversations"],
                "source": sample["metadata"]["source"]
            }
            f.write(json.dumps(output_sample) + '\n')
            
            if sample["metadata"].get("description_generated_by", "").startswith("llm"):
                llm_generated.append(sample)
            else:
                human_written.append(sample)
    
    # Save metadata separately
    metadata_path = output_path.parent / "dataset_metadata.json"
    metadata = {
        "total_samples": len(samples),
        "llm_generated_descriptions": len(llm_generated),
        "human_written_descriptions": len(human_written),
        "samples_metadata": [s["metadata"] for s in samples]
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {len(samples)} samples to {output_path}")
    print(f"  - LLM-generated descriptions: {len(llm_generated)}")
    print(f"  - Human-written descriptions: {len(human_written)}")
    print(f"✓ Metadata saved to {metadata_path}")

def main():
    # Load raw samples
    input_path = Path("data_dan4life_enhanced/dan4life_raw_with_placeholders.jsonl")
    samples = load_raw_samples(input_path)
    print(f"Loaded {len(samples)} raw samples")
    
    # Apply LLM descriptions (using examples for demonstration)
    updated_samples = apply_llm_descriptions(samples, EXAMPLE_LLM_DESCRIPTIONS)
    print(f"Updated {len(updated_samples)} samples with descriptions")
    
    # Save final dataset
    output_path = Path("data_dan4life_enhanced/dan4life_with_llm_descriptions.jsonl")
    save_final_dataset(updated_samples, output_path)
    
    # Show example of augmentation config
    print("\nExample augmentation configs:")
    for sample in updated_samples[:2]:
        day = sample["metadata"]["day"]
        config = create_augmentation_config(sample)
        print(f"\nDay {day} ({sample['metadata'].get('description_generated_by', 'manual')}):")
        print(f"  Augmentation factor: {config['augmentation_factor']}")
        print(f"  Template variations: {len(config['variation_templates'])}")

if __name__ == "__main__":
    main()