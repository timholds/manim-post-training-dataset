#!/usr/bin/env python3
"""
Extract Manim animations from Szymon Ozog's repositories with YouTube metadata:
- Information Theory Videos
- GPU Programming Videos
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# Repository paths
INFO_THEORY_PATH = "data_szymon_ozog/InformationTheory"
GPU_PROGRAMMING_PATH = "data_szymon_ozog/GPU_Programming/manim_scripts"

# Output directory
OUTPUT_DIR = "data_szymon_ozog"

# YouTube URL mappings
YOUTUBE_MAPPINGS = {
    # Information Theory playlist: https://www.youtube.com/watch?v=-j2Z2heUBYc&list=PL5XwKDZZlwaZi041-mB8zd6APQjb5AkBv
    "InformationTheory.py": {
        "playlist_url": "https://www.youtube.com/watch?v=-j2Z2heUBYc&list=PL5XwKDZZlwaZi041-mB8zd6APQjb5AkBv",
        "video_title": "Information Theory - Communication Systems",
        "episode": 4  # Based on voiceover text mentioning "episode 4"
    },
    "entropy.py": {
        "playlist_url": "https://www.youtube.com/watch?v=-j2Z2heUBYc&list=PL5XwKDZZlwaZi041-mB8zd6APQjb5AkBv",
        "video_title": "Information Theory - Entropy",
        "episode": None  # Helper file, may not have dedicated video
    },
    
    # GPU Programming playlist: https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j
    "0_Introduction.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "GPU Programming Introduction",
        "video_index": 0
    },
    "1_CPU_vs_GPU.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "CPU vs GPU Architecture",
        "video_index": 1
    },
    "2_Grid_Blocks_Threads.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "CUDA Grid, Blocks, and Threads",
        "video_index": 2
    },
    "3_Neural_Network.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Neural Network on GPU",
        "video_index": 3
    },
    "4_Backward_Pass.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Backpropagation on GPU",
        "video_index": 4
    },
    "5_PerformanceCharacteristics.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "GPU Performance Characteristics",
        "video_index": 5
    },
    "6_Memory_Hierarchy.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "GPU Memory Hierarchy",
        "video_index": 6
    },
    "7_Tiling.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Tiling Optimization",
        "video_index": 7
    },
    "8_GPU_Architecture.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "GPU Architecture Deep Dive",
        "video_index": 8
    },
    "9_Constant_Memory.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Constant Memory in CUDA",
        "video_index": 9
    },
    "10_Memory_Coalescing.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Memory Coalescing",
        "video_index": 10
    },
    "11_Occupancy.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "GPU Occupancy",
        "video_index": 11
    },
    # Additional videos that might be special topics or bonus content
    "FastSoftmax.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Fast Softmax Implementation",
        "video_index": None
    },
    "HierarchicalTiling.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Hierarchical Tiling",
        "video_index": None
    },
    "HierarchicalTiling_CE.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Hierarchical Tiling CE",
        "video_index": None
    },
    "TensorCores.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Tensor Cores",
        "video_index": None
    },
    "TensorCores_CE.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Tensor Cores CE",
        "video_index": None
    },
    "Parallelism.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "GPU Parallelism",
        "video_index": None
    },
    "Quantization.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Quantization Techniques",
        "video_index": None
    },
    "MoE.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Mixture of Experts",
        "video_index": None
    },
    "NN.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Neural Network Visualization",
        "video_index": None
    },
    "EndScreen_CE.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "End Screen",
        "video_index": None
    },
    "Presentation.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "Presentation Helper",
        "video_index": None
    },
    "how_to_keep_gpu_happy.py": {
        "playlist_url": "https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j",
        "video_title": "How to Keep GPU Happy",
        "video_index": None
    },
    "voicover_gl.py": {
        "playlist_url": None,  # Likely a helper file
        "video_title": None,
        "video_index": None
    }
}

def extract_scene_classes(code: str) -> List[Dict[str, str]]:
    """Extract Scene classes from Python code."""
    # Pattern to match class definitions that inherit from Scene or VoiceoverScene
    scene_pattern = r'class\s+(\w+)\s*\((?:.*(?:Scene|VoiceoverScene).*)\):\s*\n((?:(?:\s{4,}|\t).*\n)*)'
    
    scenes = []
    matches = re.finditer(scene_pattern, code, re.MULTILINE)
    
    for match in matches:
        class_name = match.group(1)
        class_body = match.group(0)
        
        # Get the full class including its methods
        start_pos = match.start()
        lines = code[start_pos:].split('\n')
        
        # Find the end of the class
        class_lines = [lines[0]]  # Start with class definition
        indent_level = None
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '':
                class_lines.append(line)
                continue
                
            # Determine initial indent level
            if indent_level is None and line.strip():
                indent_level = len(line) - len(line.lstrip())
            
            # Check if we're still in the class
            if line.strip() and indent_level is not None:
                current_indent = len(line) - len(line.lstrip())
                if current_indent < indent_level and line.strip():
                    break
            
            class_lines.append(line)
        
        full_class = '\n'.join(class_lines)
        scenes.append({
            'name': class_name,
            'code': full_class
        })
    
    return scenes

def analyze_code_features(code: str) -> Dict[str, any]:
    """Analyze code to extract features for description generation."""
    features = {
        'has_3d': bool(re.search(r'ThreeDScene|THREE|3D|axes_3d', code, re.IGNORECASE)),
        'has_graph': bool(re.search(r'Graph|plot|axes|NumberPlane|CoordinateSystem', code, re.IGNORECASE)),
        'has_text': bool(re.search(r'Text\(|Tex\(|MathTex\(|Title\(', code)),
        'has_animation': bool(re.search(r'animate\.|Create\(|Write\(|Transform\(|FadeIn\(|FadeOut\(', code)),
        'has_voiceover': bool(re.search(r'VoiceoverScene|voiceover\(', code)),
        'has_gpu_concepts': bool(re.search(r'Thread|Block|Grid|CUDA|GPU|kernel|memory', code, re.IGNORECASE)),
        'has_info_theory': bool(re.search(r'entropy|information|channel|communication|probability', code, re.IGNORECASE)),
        'main_elements': []
    }
    
    # Extract main visual elements
    if re.search(r'Square\(|Rectangle\(|Circle\(', code):
        features['main_elements'].append('geometric shapes')
    if re.search(r'Arrow\(|Line\(', code):
        features['main_elements'].append('arrows and lines')
    if re.search(r'Matrix\(|Table\(', code):
        features['main_elements'].append('matrices or tables')
    if re.search(r'Neural|Network|Layer', code, re.IGNORECASE):
        features['main_elements'].append('neural network visualization')
        
    return features

def create_placeholder_description(file_name: str, class_name: str, features: Dict) -> str:
    """Create a placeholder description that will be enhanced with transcript data."""
    # This is a minimal placeholder that indicates transcript enhancement is needed
    base_name = file_name.replace('.py', '').replace('_CE', '')
    
    # Get video metadata
    video_info = YOUTUBE_MAPPINGS.get(file_name, {})
    if video_info.get('video_title'):
        return f"[PLACEHOLDER - Needs transcript enhancement] Create an animation for: {video_info['video_title']}"
    else:
        return f"[PLACEHOLDER - Needs transcript enhancement] Create an animation for: {base_name}"

def process_file(file_path: str, source_name: str) -> List[Dict]:
    """Process a single Python file and extract animations."""
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Skip if file is too small or doesn't contain Manim imports
        if len(code) < 100 or 'from manim import' not in code:
            return samples
        
        # Extract Scene classes
        scenes = extract_scene_classes(code)
        
        # If no scenes found, check if the whole file is a scene
        if not scenes and ('class' in code and 'Scene' in code):
            # Try to extract any class that might be a scene
            scenes = [{
                'name': Path(file_path).stem,
                'code': code
            }]
        
        for scene in scenes:
            # Analyze code features
            features = analyze_code_features(scene['code'])
            
            # Get YouTube metadata
            file_name = Path(file_path).name
            youtube_info = YOUTUBE_MAPPINGS.get(file_name, {})
            
            # Create placeholder description
            description = create_placeholder_description(
                file_name,
                scene['name'],
                features
            )
            
            # Format the code properly
            formatted_code = f"from manim import *\n\n{scene['code']}"
            
            # Create sample in conversation format
            sample = {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
                    },
                    {
                        "from": "user",
                        "value": description
                    },
                    {
                        "from": "assistant",
                        "value": f"```python\n{formatted_code}\n```"
                    }
                ],
                "source": source_name,
                "metadata": {
                    "file": file_name,
                    "class": scene['name'],
                    "features": features,
                    "needs_description_update": True,
                    "youtube_metadata": {
                        "playlist_url": youtube_info.get("playlist_url"),
                        "video_title": youtube_info.get("video_title"),
                        "video_index": youtube_info.get("video_index"),
                        "episode": youtube_info.get("episode"),
                        "has_video": youtube_info.get("playlist_url") is not None
                    }
                }
            }
            
            samples.append(sample)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return samples

def main():
    """Main extraction function."""
    all_samples = []
    
    # Process Information Theory repository
    print("Processing Information Theory repository...")
    info_theory_files = [
        "InformationTheory.py",
        "entropy.py"
    ]
    
    for file_name in info_theory_files:
        file_path = os.path.join(INFO_THEORY_PATH, file_name)
        if os.path.exists(file_path):
            samples = process_file(file_path, "szymon_ozog_info_theory")
            all_samples.extend(samples)
            print(f"  Extracted {len(samples)} scenes from {file_name}")
            if samples:
                print(f"    YouTube metadata: {samples[0]['metadata']['youtube_metadata']}")
    
    # Process GPU Programming repository
    print("\nProcessing GPU Programming repository...")
    if os.path.exists(GPU_PROGRAMMING_PATH):
        for file_name in sorted(os.listdir(GPU_PROGRAMMING_PATH)):
            if file_name.endswith('.py') and not file_name.startswith('__'):
                file_path = os.path.join(GPU_PROGRAMMING_PATH, file_name)
                samples = process_file(file_path, "szymon_ozog_gpu")
                all_samples.extend(samples)
                if samples:
                    print(f"  Extracted {len(samples)} scenes from {file_name}")
                    if samples[0]['metadata']['youtube_metadata'].get('has_video'):
                        print(f"    Video: {samples[0]['metadata']['youtube_metadata']['video_title']}")
    
    # Save to output file
    output_path = os.path.join(OUTPUT_DIR, "szymon_ozog_processed.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nTotal samples extracted: {len(all_samples)}")
    print(f"Output saved to: {output_path}")
    
    # Print summary of YouTube metadata
    videos_with_metadata = sum(1 for s in all_samples if s['metadata']['youtube_metadata'].get('has_video'))
    print(f"\nSamples with YouTube video metadata: {videos_with_metadata}/{len(all_samples)}")
    print(f"All samples marked for transcript-based description enhancement")

if __name__ == "__main__":
    main()