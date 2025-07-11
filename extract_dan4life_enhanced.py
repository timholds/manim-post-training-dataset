#!/usr/bin/env python3
"""
Enhanced extraction for Dan4Life's AoC2024 dataset.
Extracts code first, then uses LLM to generate proper descriptions.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess

def extract_manim_code(file_path: Path) -> Optional[str]:
    """Extract Manim code from a scene.py file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_class_name(code: str) -> Optional[str]:
    """Extract the main Scene class name from the code"""
    match = re.search(r'class\s+(\w+)\s*\(Scene\)', code)
    return match.group(1) if match else None

def analyze_code_content(code: str) -> Dict[str, any]:
    """Analyze code to extract key features for description generation"""
    features = {
        "has_binary_operations": "binary" in code.lower() or "0b" in code or "texttt{" in code,
        "has_xor": "^" in code or "XOR" in code or "oplus" in code,
        "has_modulo": "%" in code or "mod" in code,
        "has_graphs": "axes" in code.lower() or "graph" in code.lower(),
        "has_trees": "tree" in code.lower() or "branch" in code.lower(),
        "has_grids": "grid" in code.lower() or "table" in code.lower(),
        "has_counters": "counter" in code.lower() or "Variable(" in code,
        "has_animations": ".animate" in code,
        "has_colors": any(color in code for color in ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE"]),
        "main_elements": []
    }
    
    # Extract main visual elements
    if "Square(" in code: features["main_elements"].append("squares")
    if "Circle(" in code: features["main_elements"].append("circles")
    if "Triangle(" in code: features["main_elements"].append("triangles")
    if "Line(" in code: features["main_elements"].append("lines")
    if "Text(" in code or "MathTex(" in code: features["main_elements"].append("text/math")
    if "pointer" in code.lower(): features["main_elements"].append("pointers")
    if "VGroup(" in code: features["main_elements"].append("grouped objects")
    
    return features

def create_placeholder_conversation(day: int, code: str, version: Optional[str] = None) -> Dict:
    """Create a conversation with placeholder description to be filled by LLM"""
    version_str = f" Version {version}" if version else ""
    
    # Analyze code to provide context
    features = analyze_code_content(code)
    class_name = get_class_name(code) or f"Day{day}"
    
    # Create metadata about this sample
    metadata = {
        "source": "dan4life_aoc2024",
        "day": day,
        "version": version,
        "class_name": class_name,
        "code_features": features,
        "description_generated_by": "llm_pending",  # Will be updated after LLM generation
        "original_context": f"Advent of Code 2024 Day {day}{version_str}"
    }
    
    # Placeholder that will be replaced by LLM
    placeholder_description = f"[TO BE GENERATED: Analyze the following Manim code for AoC 2024 Day {day}{version_str} and create a natural user request that would result in this specific animation]"
    
    return {
        "conversations": [
            {
                "from": "system",
                "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
            },
            {
                "from": "user", 
                "value": placeholder_description
            },
            {
                "from": "assistant",
                "value": f"```python\n{code}```"
            }
        ],
        "metadata": metadata
    }

def get_youtube_transcript(day: int) -> Optional[str]:
    """Attempt to get YouTube transcript for a given day (placeholder for now)"""
    # This would require YouTube API or youtube-transcript-api
    # For now, return None
    return None

def process_repository(repo_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Process repository and return samples needing LLM processing"""
    samples_for_llm = []
    completed_samples = []
    
    # Process each day
    for day in range(1, 26):
        day_dir = repo_path / f"Day_{day:02d}"
        
        if not day_dir.exists():
            continue
            
        # Check for scene.py directly in day directory
        scene_file = day_dir / "scene.py"
        if scene_file.exists():
            code = extract_manim_code(scene_file)
            if code and get_class_name(code):
                sample = create_placeholder_conversation(day, code)
                # Add YouTube transcript if available
                transcript = get_youtube_transcript(day)
                if transcript:
                    sample["metadata"]["youtube_transcript"] = transcript
                samples_for_llm.append(sample)
                print(f"✓ Extracted Day {day} (pending LLM description)")
        
        # Check for version subdirectories
        for version_dir in sorted(day_dir.glob("Version*")):
            if version_dir.is_dir():
                version_num = version_dir.name.replace("Version", "")
                scene_file = version_dir / "scene.py"
                if scene_file.exists():
                    code = extract_manim_code(scene_file)
                    if code and get_class_name(code):
                        sample = create_placeholder_conversation(day, code, version_num)
                        samples_for_llm.append(sample)
                        print(f"✓ Extracted Day {day} Version {version_num} (pending LLM description)")
    
    return samples_for_llm, completed_samples

def save_for_llm_processing(samples: List[Dict], output_path: Path):
    """Save samples that need LLM processing"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Include full code and metadata for LLM to analyze
            llm_input = {
                "code": sample["conversations"][2]["value"],
                "metadata": sample["metadata"],
                "task": "Generate a natural user request that would result in this Manim animation"
            }
            f.write(json.dumps(llm_input) + '\n')
    
    print(f"\n✓ Saved {len(samples)} samples for LLM processing to {output_path}")

def generate_llm_prompt_for_batch(samples: List[Dict]) -> str:
    """Generate a prompt for LLM to create descriptions for multiple samples"""
    prompt = """You are an expert at analyzing Manim animation code and creating natural user requests.

For each of the following Manim animations, create a natural user request that someone might ask to get this specific animation as output. The request should:
1. Be conversational and natural
2. Describe the visual elements and behavior they want to see
3. Be specific enough that this code would be a good response
4. NOT mention implementation details like class names or specific Manim functions
5. Focus on what the animation shows visually

Here are the animations to analyze:

"""
    
    for i, sample in enumerate(samples):
        code = sample["conversations"][2]["value"]
        metadata = sample["metadata"]
        prompt += f"\n---\nAnimation {i+1} (AoC 2024 Day {metadata['day']}"
        if metadata.get('version'):
            prompt += f" Version {metadata['version']}"
        prompt += "):\n"
        prompt += f"Code features: {metadata['code_features']['main_elements']}\n"
        prompt += f"Code preview (first 500 chars):\n{code[:500]}...\n"
    
    prompt += "\nProvide the user requests in a JSON array format."
    
    return prompt

def main():
    # Setup paths
    repo_path = Path("AoC2024_Videos")
    output_dir = Path("data_dan4life_enhanced")
    
    # Extract samples
    print("Extracting Dan4Life AoC2024 dataset for LLM processing...")
    samples_for_llm, _ = process_repository(repo_path)
    
    # Save for LLM processing
    llm_input_path = output_dir / "samples_for_llm.jsonl"
    save_for_llm_processing(samples_for_llm, llm_input_path)
    
    # Also save the raw samples with placeholders
    raw_path = output_dir / "dan4life_raw_with_placeholders.jsonl"
    with open(raw_path, 'w', encoding='utf-8') as f:
        for sample in samples_for_llm:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Saved raw samples with placeholders to {raw_path}")
    
    # Generate batch prompt for LLM
    if len(samples_for_llm) <= 5:
        # For small batches, include full code
        prompt = generate_llm_prompt_for_batch(samples_for_llm)
        prompt_path = output_dir / "llm_batch_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        print(f"✓ Generated LLM batch prompt at {prompt_path}")
    
    print(f"\nNext steps:")
    print(f"1. Use an LLM to generate descriptions from {llm_input_path}")
    print(f"2. Update the samples with generated descriptions")
    print(f"3. Mark metadata with 'description_generated_by': 'llm_gpt4' or similar")
    print(f"4. These LLM-generated samples can have more aggressive augmentation")

if __name__ == "__main__":
    main()