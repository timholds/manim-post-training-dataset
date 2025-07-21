#!/usr/bin/env python3
"""
Integration script to add converted szymon_ozog files to the dataset pipeline

This script:
1. Reads converted ManimCE files
2. Extracts Scene classes 
3. Generates descriptions based on voiceover comments
4. Creates JSONL format for dataset integration
"""

import re
import ast
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SzymonOzogIntegrator:
    def __init__(self, converted_dir: str, output_file: str):
        self.converted_dir = Path(converted_dir)
        self.output_file = Path(output_file)
        self.stats = {
            'files_processed': 0,
            'scenes_extracted': 0,
            'errors': []
        }
    
    def extract_scenes_from_file(self, file_path: Path) -> list:
        """Extract Scene classes from converted file"""
        scenes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Find all Scene-based classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Scene or variants
                    base_names = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_names.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_names.append(base.attr)
                    
                    scene_bases = ['Scene', 'ThreeDScene', 'MovingCameraScene']
                    if any(base in scene_bases for base in base_names):
                        # Extract the class code
                        class_code = ast.get_source_segment(content, node)
                        if class_code:
                            # Extract description from voiceover comments
                            description = self.extract_description(class_code, node.name)
                            
                            # Get relative path for metadata
                            relative_path = file_path.relative_to(self.converted_dir)
                            
                            scenes.append({
                                'name': node.name,
                                'code': class_code,
                                'description': description,
                                'file': str(relative_path),
                                'topic': self.get_topic(relative_path)
                            })
                            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'].append(f"{file_path}: {str(e)}")
        
        return scenes
    
    def extract_description(self, code: str, class_name: str) -> str:
        """Extract description from voiceover comments"""
        voiceover_comments = re.findall(r'# Voiceover: "([^"]+)"', code)
        
        if voiceover_comments:
            # Combine first few voiceover texts
            combined = '. '.join(voiceover_comments[:3])
            if len(combined) > 150:
                combined = combined[:150] + "..."
            return f"Create animation: {combined}"
        
        # Fallback based on class name
        topic_map = {
            'GPU': 'GPU programming visualization',
            'CPU': 'CPU architecture animation',
            'Thread': 'Thread and parallelism demonstration',
            'Memory': 'Memory hierarchy visualization',
            'Communication': 'Communication system diagram',
            'Entropy': 'Information theory entropy visualization',
            'BSC': 'Binary symmetric channel animation'
        }
        
        for keyword, desc in topic_map.items():
            if keyword in class_name:
                return f"{desc} - {class_name}"
        
        return f"Educational animation - {class_name}"
    
    def get_topic(self, file_path: Path) -> str:
        """Get topic from file path"""
        parts = file_path.parts
        
        if 'GPU_Programming' in parts:
            return 'GPU Programming'
        elif 'InformationTheory' in parts:
            return 'Information Theory'
        else:
            return 'General'
    
    def process_all_files(self):
        """Process all converted Python files"""
        py_files = list(self.converted_dir.glob("**/*.py"))
        logger.info(f"Found {len(py_files)} converted files")
        
        all_samples = []
        
        for py_file in py_files:
            self.stats['files_processed'] += 1
            scenes = self.extract_scenes_from_file(py_file)
            
            for scene in scenes:
                self.stats['scenes_extracted'] += 1
                
                # Format as dataset sample
                sample = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "You are a Manim code generator that creates educational animations."
                        },
                        {
                            "from": "user", 
                            "value": scene['description']
                        },
                        {
                            "from": "assistant",
                            "value": f"```python\n{scene['code']}\n```"
                        }
                    ],
                    "source": "szymon_ozog_converted",
                    "metadata": {
                        "file": scene['file'],
                        "class": scene['name'],
                        "topic": scene['topic'],
                        "converted_from_voiceover": True
                    }
                }
                
                all_samples.append(sample)
        
        # Save to JSONL
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Saved {len(all_samples)} samples to {self.output_file}")
        
        return all_samples
    
    def print_report(self):
        """Print integration report"""
        print("\n" + "="*60)
        print("INTEGRATION REPORT")
        print("="*60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Scenes extracted: {self.stats['scenes_extracted']}")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")


def main():
    integrator = SzymonOzogIntegrator(
        converted_dir="data/szymon_ozog_converted",
        output_file="data/szymon_ozog_integrated.jsonl"
    )
    
    samples = integrator.process_all_files()
    integrator.print_report()
    
    # Show a few examples
    print("\nSample entries:")
    for i, sample in enumerate(samples[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Description: {sample['conversations'][1]['value']}")
        print(f"Source file: {sample['metadata']['file']}")
        print(f"Topic: {sample['metadata']['topic']}")


if __name__ == "__main__":
    main()