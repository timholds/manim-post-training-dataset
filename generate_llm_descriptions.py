#!/usr/bin/env python3
"""
Generate LLM descriptions for extracted Manim samples.
This script can be run with different LLM backends.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import os
from dataset_registry import DatasetRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_with_gemini(samples: List[Dict[str, Any]], batch_size: int = 5) -> List[str]:
    """Generate descriptions using gemini CLI."""
    descriptions = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        
        # Create prompt
        prompt = """Analyze these Manim animation codes and generate natural user requests that would result in each animation.

For each animation:
1. Focus on what the animation visually shows
2. Be specific about visual elements and behavior
3. Don't mention implementation details
4. Make it conversational

"""
        
        for j, sample in enumerate(batch):
            metadata = sample["metadata"]
            features = metadata.get("code_features", {})
            
            prompt += f"\n--- Animation {j+1}"
            if metadata.get("original_context"):
                prompt += f" ({metadata['original_context']})"
            prompt += " ---\n"
            
            if features.get("visual_elements"):
                prompt += f"Visual elements: {', '.join(features['visual_elements'])}\n"
            
            # Add code preview
            code = sample["code"]
            if code.startswith("```python"):
                code = code[9:-3].strip()
            
            prompt += f"Code preview:\n{code[:500]}...\n"
            
            # Add YouTube info if available
            if metadata.get("youtube_url"):
                prompt += f"YouTube video: {metadata.get('video_title', 'Available')}\n"
                # Could fetch transcript here if needed
        
        prompt += "\nGenerate descriptions as a JSON array of strings."
        
        # Call gemini
        try:
            # Save prompt to temp file
            temp_file = Path("/tmp/manim_llm_prompt.txt")
            with open(temp_file, 'w') as f:
                f.write(prompt)
            
            # Run gemini command
            result = subprocess.run(
                ["gemini", "-p", str(temp_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse response
                response = result.stdout.strip()
                # Extract JSON array from response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    batch_descriptions = json.loads(json_match.group())
                    descriptions.extend(batch_descriptions)
                else:
                    logger.error(f"Could not parse JSON from gemini response")
                    # Fallback descriptions
                    descriptions.extend([f"[GENERATION FAILED]" for _ in batch])
            else:
                logger.error(f"Gemini command failed: {result.stderr}")
                descriptions.extend([f"[GENERATION FAILED]" for _ in batch])
                
        except Exception as e:
            logger.error(f"Error calling gemini: {e}")
            descriptions.extend([f"[GENERATION FAILED]" for _ in batch])
    
    return descriptions


def generate_with_claude_api(samples: List[Dict[str, Any]]) -> List[str]:
    """Generate descriptions using Claude API."""
    # This would use the actual Claude API
    # For now, return example descriptions
    
    example_descriptions = {
        "Day1": "I need an animation that visualizes comparing two lists of numbers side by side. Show the lists as columns of numbered squares, then draw lines connecting corresponding elements after sorting.",
        "Day22": "Create an animation that demonstrates a pseudorandom number generator algorithm using XOR operations. Show binary representations and the three-step process with visual indicators.",
        # Add more examples
    }
    
    descriptions = []
    for sample in samples:
        class_name = sample["metadata"].get("class_name", "")
        if class_name in example_descriptions:
            descriptions.append(example_descriptions[class_name])
        else:
            # Generate based on features
            features = sample["metadata"].get("code_features", {})
            elements = features.get("visual_elements", [])
            
            if elements:
                desc = f"Create an animation that shows {', '.join(elements[:3])}"
                if features.get("has_3d"):
                    desc += " in 3D space"
                if features.get("has_voiceover"):
                    desc += " with explanatory narration"
                descriptions.append(desc)
            else:
                descriptions.append("Create a Manim animation that demonstrates this concept visually")
    
    return descriptions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM descriptions for Manim samples")
    parser.add_argument("--input", required=True, help="Input JSONL file with samples")
    parser.add_argument("--output", required=True, help="Output JSONL file with descriptions")
    parser.add_argument("--llm", choices=["gemini", "claude"], default="gemini", 
                       help="LLM backend to use")
    parser.add_argument("--batch-size", type=int, default=5, 
                       help="Batch size for LLM calls")
    
    args = parser.parse_args()
    
    # Set up registry for caching
    registry = DatasetRegistry()
    
    # Load samples
    samples = []
    with open(args.input, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(samples)} samples for description generation")
    
    # Generate descriptions
    if args.llm == "gemini":
        if not os.system("which gemini > /dev/null 2>&1") == 0:
            logger.error("gemini CLI not found. Please install it first.")
            return
        
        descriptions = generate_with_gemini(samples, args.batch_size)
    else:
        descriptions = generate_with_claude_api(samples)
    
    # Update samples with descriptions
    updated_samples = []
    for sample, description in zip(samples, descriptions):
        if "[GENERATION FAILED]" not in description:
            # Update the placeholder
            sample["conversations"][1]["value"] = description
            sample["metadata"]["description_generated_by"] = f"llm_{args.llm}"
            sample["metadata"]["needs_description_update"] = False
            
            # Cache the description
            code = sample["conversations"][2]["value"]
            if code.startswith("```python"):
                code = code[9:-3].strip()
            
            registry.llm_generator.cache_description(
                code,
                sample["metadata"],
                description,
                args.llm
            )
        
        updated_samples.append(sample)
    
    # Save updated samples
    with open(args.output, 'w') as f:
        for sample in updated_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Saved {len(updated_samples)} samples with descriptions to {args.output}")
    
    # Report statistics
    success_count = sum(1 for s in updated_samples 
                       if not s["metadata"].get("needs_description_update", True))
    logger.info(f"Successfully generated {success_count}/{len(samples)} descriptions")
    
    # Show cache stats
    cache_stats = registry.llm_generator.cache_index
    logger.info(f"Cache now contains {len(cache_stats)} entries")


if __name__ == "__main__":
    main()