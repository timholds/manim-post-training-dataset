#!/usr/bin/env python3
"""
Extract Manim animations from Szymon Ozog's repositories:
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
    """Create a placeholder description based on file name and code features."""
    # Map file names to general topics
    topic_map = {
        # Information Theory
        'entropy': 'entropy and information theory concepts',
        'InformationTheory': 'communication systems and information theory',
        
        # GPU Programming
        '0_Introduction': 'introduction to GPU programming concepts',
        '1_CPU_vs_GPU': 'comparison between CPU and GPU architectures',
        '2_Grid_Blocks_Threads': 'CUDA grid, blocks, and threads hierarchy',
        '3_Neural_Network': 'neural network implementation on GPU',
        '4_Backward_Pass': 'backpropagation and gradient computation',
        '5_PerformanceCharacteristics': 'GPU performance characteristics',
        '6_Memory_Hierarchy': 'GPU memory hierarchy and types',
        '7_Tiling': 'tiling optimization techniques',
        '8_GPU_Architecture': 'GPU architecture deep dive',
        '9_Constant_Memory': 'constant memory usage in CUDA',
        '10_Memory_Coalescing': 'memory coalescing for performance',
        '11_Occupancy': 'GPU occupancy and resource utilization',
        'FastSoftmax': 'optimized softmax implementation',
        'HierarchicalTiling': 'hierarchical tiling strategies',
        'TensorCores': 'tensor core operations and optimization',
        'Parallelism': 'parallelism concepts in GPU programming',
        'Quantization': 'quantization techniques for neural networks',
        'MoE': 'mixture of experts architecture',
        'NN': 'neural network visualization'
    }
    
    # Get base topic from file name
    base_name = file_name.replace('.py', '').replace('_CE', '')
    topic = topic_map.get(base_name, 'GPU programming concepts')
    
    # Build description based on features
    desc_parts = [f"Create an animation explaining {topic}"]
    
    if features['has_voiceover']:
        desc_parts.append("with synchronized voiceover narration")
    
    if features['has_3d']:
        desc_parts.append("using 3D visualizations")
    
    if features['has_gpu_concepts']:
        desc_parts.append("showing GPU architecture elements like threads, blocks, and memory")
    
    if features['has_info_theory']:
        desc_parts.append("illustrating information theory principles")
        
    if features['main_elements']:
        desc_parts.append(f"featuring {', '.join(features['main_elements'])}")
    
    return ' '.join(desc_parts) + '.'

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
            
            # Create placeholder description
            description = create_placeholder_description(
                Path(file_path).name,
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
                    "file": Path(file_path).name,
                    "class": scene['name'],
                    "features": features,
                    "needs_description_update": True
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
    
    # Save to output file
    output_path = os.path.join(OUTPUT_DIR, "szymon_ozog_processed.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nTotal samples extracted: {len(all_samples)}")
    print(f"Output saved to: {output_path}")
    
    # Print summary of what needs description updates
    needs_update = sum(1 for s in all_samples if s.get('metadata', {}).get('needs_description_update'))
    print(f"Samples needing description updates: {needs_update}")

if __name__ == "__main__":
    main()