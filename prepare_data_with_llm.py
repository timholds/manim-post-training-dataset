#!/usr/bin/env python3
"""
Enhanced data preparation pipeline with LLM description generation.
Extends prepare_data.py with caching and LLM support.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess

from prepare_data import prepare_datasets
from extractors.llm_description_generator import get_llm_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_descriptions_with_gemini(samples: List[Dict[str, Any]], 
                                    batch_size: int = 5) -> List[str]:
    """Generate descriptions using gemini CLI."""
    descriptions = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        
        # Create prompt
        prompt = """Analyze these Manim animation codes and generate natural user requests.

For each animation:
1. Focus on what the animation visually shows
2. Be specific about visual elements and behavior  
3. Don't mention implementation details
4. Make it conversational - as if a user is asking for this animation

"""
        
        for j, sample in enumerate(batch):
            metadata = sample.get("metadata", {})
            features = metadata.get("code_features", {})
            
            prompt += f"\n--- Animation {j+1}"
            if metadata.get("original_context"):
                prompt += f" ({metadata['original_context']})"
            prompt += " ---\n"
            
            if features.get("visual_elements"):
                prompt += f"Visual elements: {', '.join(features['visual_elements'])}\n"
            
            # Add code preview
            code = sample.get("code", "")
            prompt += f"Code preview:\n{code[:500]}...\n"
            
            # Add YouTube info if available
            if metadata.get("youtube_url"):
                prompt += f"YouTube: {metadata.get('video_title', 'Available')}\n"
        
        prompt += "\nProvide descriptions as a JSON array of strings."
        
        # Call gemini (example - adjust for your setup)
        try:
            # Save prompt to temp file
            temp_file = Path("/tmp/manim_llm_prompt.txt")
            with open(temp_file, 'w') as f:
                f.write(prompt)
            
            # This is a placeholder - implement actual gemini call
            logger.info(f"Would call gemini with batch of {len(batch)} samples")
            
            # For now, generate example descriptions based on features
            for sample in batch:
                features = sample.get("metadata", {}).get("code_features", {})
                elements = features.get("visual_elements", [])
                
                if elements:
                    desc = f"Create an animation that shows {', '.join(elements[:3])}"
                    if features.get("has_3d"):
                        desc += " in 3D space"
                    if features.get("has_latex"):
                        desc += " with mathematical equations"
                    descriptions.append(desc)
                else:
                    descriptions.append("Create a Manim animation for this concept")
                    
        except Exception as e:
            logger.error(f"Error generating descriptions: {e}")
            descriptions.extend(["[GENERATION FAILED]" for _ in batch])
    
    return descriptions


def process_jsonl_with_llm(input_file: Path, output_file: Path, 
                          llm_backend: str = "gemini",
                          batch_size: int = 5):
    """Process JSONL file and add LLM descriptions."""
    
    llm_generator = get_llm_generator()
    
    # Load samples
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(samples)} samples from {input_file}")
    
    # Define LLM function based on backend
    if llm_backend == "gemini":
        llm_function = lambda s: generate_descriptions_with_gemini(s, batch_size)
    else:
        # Add other backends as needed
        llm_function = None
    
    # Process with LLM generator (uses cache)
    updated_samples = llm_generator.process_batch(
        samples, 
        llm_function=llm_function,
        use_cache=True
    )
    
    # Save updated samples
    with open(output_file, 'w') as f:
        for sample in updated_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Saved {len(updated_samples)} samples to {output_file}")
    
    # Show cache stats
    stats = llm_generator.get_stats()
    logger.info(f"Cache stats: {stats['total_entries']} entries, {stats['cache_size_mb']:.2f} MB")


def main():
    """Enhanced main function with LLM support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Manim datasets with LLM descriptions")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Standard prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare datasets')
    prepare_parser.add_argument("--output-dir", default="data_formatted", help="Output directory")
    prepare_parser.add_argument("--sources", nargs="+", help="Specific sources to process")
    prepare_parser.add_argument("--augmentation", action="store_true", help="Enable augmentation")
    prepare_parser.add_argument("--no-deduplicate", action="store_true", help="Disable deduplication")
    
    # LLM description generation command
    llm_parser = subparsers.add_parser('generate-descriptions', help='Generate LLM descriptions')
    llm_parser.add_argument("--input", required=True, help="Input JSONL file")
    llm_parser.add_argument("--output", required=True, help="Output JSONL file")
    llm_parser.add_argument("--llm", choices=["gemini", "claude"], default="gemini", help="LLM backend")
    llm_parser.add_argument("--batch-size", type=int, default=5, help="Batch size for LLM calls")
    
    # Cache stats command
    cache_parser = subparsers.add_parser('cache-stats', help='Show LLM cache statistics')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        # Run standard preparation
        prepare_datasets(
            output_dir=args.output_dir,
            sources=args.sources,
            use_augmentation=args.augmentation,
            deduplicate=not args.no_deduplicate
        )
        
    elif args.command == 'generate-descriptions':
        # Generate LLM descriptions
        process_jsonl_with_llm(
            input_file=Path(args.input),
            output_file=Path(args.output),
            llm_backend=args.llm,
            batch_size=args.batch_size
        )
        
    elif args.command == 'cache-stats':
        # Show cache statistics
        llm_generator = get_llm_generator()
        stats = llm_generator.get_stats()
        print(f"\nLLM Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"  Sources: {', '.join(stats['sources'])}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()