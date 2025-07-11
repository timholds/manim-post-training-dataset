#!/usr/bin/env python3
"""
Extract and format Dan4Life's Advent of Code 2024 Manim animations
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# Advent of Code 2024 problem descriptions (simplified)
AOC_DESCRIPTIONS = {
    1: "Visualize the solution to Advent of Code 2024 Day 1: Finding differences between two lists and calculating similarity scores",
    2: "Animate the solution to Advent of Code 2024 Day 2: Checking if sequences of numbers are safe based on increasing/decreasing patterns",
    3: "Show the solution to Advent of Code 2024 Day 3: Parsing and executing multiplication instructions from corrupted text",
    4: "Demonstrate solving Advent of Code 2024 Day 4: Finding occurrences of 'XMAS' in a word search grid",
    5: "Illustrate Advent of Code 2024 Day 5: Validating and fixing page ordering based on rules",
    6: "Visualize Advent of Code 2024 Day 6: Tracking a guard's path through a grid with obstacles",
    7: "Animate Advent of Code 2024 Day 7: Finding valid operator combinations to reach target values",
    8: "Show Advent of Code 2024 Day 8: Locating antinodes created by antenna frequencies",
    9: "Demonstrate Advent of Code 2024 Day 9: Compacting disk space by moving file blocks",
    10: "Visualize Advent of Code 2024 Day 10: Finding hiking trails and their ratings on a topographic map",
    11: "Animate Advent of Code 2024 Day 11: Simulating stone transformations based on engraved numbers",
    12: "Show Advent of Code 2024 Day 12: Calculating fencing costs for garden regions",
    13: "Illustrate Advent of Code 2024 Day 13: Finding optimal button presses for claw machines",
    14: "Visualize Advent of Code 2024 Day 14: Simulating robot movements and finding patterns",
    15: "Demonstrate Advent of Code 2024 Day 15: Moving boxes in a warehouse with a robot",
    16: "Animate Advent of Code 2024 Day 16: Finding optimal paths through a reindeer maze",
    17: "Show Advent of Code 2024 Day 17: Simulating a simple computer with jump instructions",
    18: "Visualize Advent of Code 2024 Day 18: Navigating through falling bytes in memory space",
    19: "Illustrate Advent of Code 2024 Day 19: Counting ways to create towel patterns",
    20: "Demonstrate Advent of Code 2024 Day 20: Finding shortcuts in a race track maze",
    21: "Animate Advent of Code 2024 Day 21: Controlling robots to type on keypads",
    22: "Show Advent of Code 2024 Day 22: Generating and analyzing pseudorandom numbers",
    23: "Visualize Advent of Code 2024 Day 23: Finding computer network cliques",
    24: "Illustrate Advent of Code 2024 Day 24: Debugging boolean gate circuits",
    25: "Demonstrate Advent of Code 2024 Day 25: Matching keys and locks based on pin heights"
}

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

def create_conversation(day: int, code: str, version: Optional[str] = None) -> Dict:
    """Create a conversation in the required format"""
    base_description = AOC_DESCRIPTIONS.get(day, f"Visualize the solution to Advent of Code 2024 Day {day}")
    
    if version:
        base_description += f" (Version {version})"
    
    # Convert to a proper request format
    # Remove "Visualize" or "Show" from the beginning to avoid redundancy
    cleaned_description = base_description
    for prefix in ["Visualize ", "Show ", "Demonstrate ", "Illustrate ", "Animate "]:
        if cleaned_description.startswith(prefix):
            cleaned_description = cleaned_description[len(prefix):]
            break
    
    user_prompt = f"Could you create an animation that visualizes {cleaned_description[0].lower()}{cleaned_description[1:]}? Please show the algorithm step by step with visual elements that make it easy to understand."
    
    return {
        "conversations": [
            {
                "from": "system",
                "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
            },
            {
                "from": "user", 
                "value": user_prompt
            },
            {
                "from": "assistant",
                "value": f"```python\n{code}```"
            }
        ],
        "source": "dan4life_aoc2024"
    }

def process_repository(repo_path: Path) -> List[Dict]:
    """Process the entire repository and extract all animations"""
    dataset = []
    
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
                dataset.append(create_conversation(day, code))
                print(f"✓ Extracted Day {day}")
        
        # Check for version subdirectories
        for version_dir in sorted(day_dir.glob("Version*")):
            if version_dir.is_dir():
                version_num = version_dir.name.replace("Version", "")
                scene_file = version_dir / "scene.py"
                if scene_file.exists():
                    code = extract_manim_code(scene_file)
                    if code and get_class_name(code):
                        dataset.append(create_conversation(day, code, version_num))
                        print(f"✓ Extracted Day {day} Version {version_num}")
    
    return dataset

def save_dataset(dataset: List[Dict], output_path: Path):
    """Save dataset in JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✓ Saved {len(dataset)} samples to {output_path}")

def generate_stats(dataset: List[Dict]) -> Dict:
    """Generate statistics about the dataset"""
    stats = {
        "total_samples": len(dataset),
        "days_covered": set(),
        "versions": 0,
        "total_code_lines": 0
    }
    
    for item in dataset:
        code = item["conversations"][2]["value"]
        stats["total_code_lines"] += len(code.split('\n'))
        
        # Extract day number from description
        day_match = re.search(r'Day (\d+)', item["conversations"][1]["value"])
        if day_match:
            stats["days_covered"].add(int(day_match.group(1)))
        
        if "Version" in item["conversations"][1]["value"]:
            stats["versions"] += 1
    
    stats["days_covered"] = sorted(list(stats["days_covered"]))
    stats["unique_days"] = len(stats["days_covered"])
    
    return stats

def main():
    # Setup paths
    repo_path = Path("AoC2024_Videos")
    output_dir = Path("data_dan4life")
    output_path = output_dir / "dan4life_aoc2024.jsonl"
    
    # Extract dataset
    print("Extracting Dan4Life AoC2024 dataset...")
    dataset = process_repository(repo_path)
    
    # Generate and print statistics
    stats = generate_stats(dataset)
    print("\nDataset Statistics:")
    print(f"- Total samples: {stats['total_samples']}")
    print(f"- Unique days: {stats['unique_days']}/25")
    print(f"- Additional versions: {stats['versions']}")
    print(f"- Total code lines: {stats['total_code_lines']:,}")
    print(f"- Average lines per sample: {stats['total_code_lines'] // stats['total_samples']}")
    
    # Save dataset
    save_dataset(dataset, output_path)
    
    # Save statistics
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✓ Saved statistics to {stats_path}")

if __name__ == "__main__":
    main()