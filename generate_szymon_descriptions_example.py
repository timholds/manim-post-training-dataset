#!/usr/bin/env python3
"""
Example script showing how to generate descriptions using YouTube transcripts
for Szymon Ozog's dataset following the TRANSCRIPT_STRATEGY.md approach.
"""

import json
import subprocess
from typing import Dict, Optional

def fetch_youtube_transcript(playlist_url: str, video_title: str) -> Optional[str]:
    """
    Fetch YouTube transcript for a specific video.
    In practice, you'd use youtube-dl or yt-dlp.
    """
    # Example command (requires yt-dlp installed):
    # yt-dlp --skip-download --write-auto-sub --sub-format vtt --sub-lang en VIDEO_URL
    
    # For demonstration, return a placeholder
    return f"[Transcript would be fetched here for: {video_title}]"

def generate_description_with_transcript(sample: Dict) -> str:
    """
    Generate enhanced description using code analysis and YouTube transcript.
    This would typically call an LLM API or use claude -p.
    """
    metadata = sample['metadata']
    youtube_meta = metadata['youtube_metadata']
    features = metadata['features']
    code = sample['conversations'][2]['value']
    
    # Only process if needs update and has video
    if not metadata.get('needs_description_update') or not youtube_meta.get('has_video'):
        return sample['conversations'][1]['value']
    
    # Fetch transcript
    transcript = fetch_youtube_transcript(
        youtube_meta['playlist_url'],
        youtube_meta['video_title']
    )
    
    # Extract key code elements
    code_summary = []
    if features['has_voiceover']:
        code_summary.append("includes voiceover narration")
    if features['has_3d']:
        code_summary.append("uses 3D visualizations")
    if features['has_gpu_concepts']:
        code_summary.append("visualizes GPU concepts")
    if features['has_info_theory']:
        code_summary.append("demonstrates information theory concepts")
    
    # Build LLM prompt
    prompt = f"""
Analyze this Manim animation code and YouTube video context to generate a natural user request.

Video Title: {youtube_meta['video_title']}
Code Features: {', '.join(code_summary)}
Main Visual Elements: {', '.join(features.get('main_elements', []))}

Transcript Excerpt: {transcript[:500]}

Code Analysis:
- Class: {metadata['class']}
- File: {metadata['file']}

Generate a natural user request that someone might ask to create this animation.
The request should capture what the animation demonstrates visually.
Do not mention implementation details, just what the user wants to see animated.
"""
    
    # In practice, this would call an LLM
    # For example using claude:
    # result = subprocess.run(['claude', '-p', prompt], capture_output=True, text=True)
    
    # For demonstration, return enhanced placeholder
    return f"Create an animation that {youtube_meta['video_title'].lower()} with {', '.join(code_summary)}"

def process_dataset_with_transcripts(input_file: str, output_file: str):
    """Process dataset and generate enhanced descriptions."""
    enhanced_samples = []
    
    with open(input_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            
            # Generate enhanced description
            new_description = generate_description_with_transcript(sample)
            
            # Update sample
            sample['conversations'][1]['value'] = new_description
            
            # Mark as processed
            if 'metadata' in sample:
                sample['metadata']['description_generated_by'] = 'transcript_enhanced'
                sample['metadata']['needs_description_update'] = False
            
            enhanced_samples.append(sample)
    
    # Save enhanced dataset
    with open(output_file, 'w') as f:
        for sample in enhanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Enhanced {len(enhanced_samples)} samples with transcript-based descriptions")

def main():
    """Example usage."""
    print("Example: How to use YouTube metadata for description generation")
    print("=" * 60)
    
    # Load a sample
    with open('data_szymon_ozog/szymon_ozog_processed.jsonl', 'r') as f:
        sample = json.loads(f.readline())
    
    print(f"Original description: {sample['conversations'][1]['value']}")
    print(f"YouTube metadata: {sample['metadata']['youtube_metadata']}")
    
    # Show how to generate enhanced description
    enhanced = generate_description_with_transcript(sample)
    print(f"\nEnhanced description: {enhanced}")
    
    print("\nTo process the entire dataset:")
    print("python generate_szymon_descriptions_example.py")
    print("\nOr use claude directly:")
    print('claude -p "$(cat prompt_template.txt)" < data_szymon_ozog/szymon_ozog_processed.jsonl')

if __name__ == "__main__":
    main()