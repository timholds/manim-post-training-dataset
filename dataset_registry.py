#!/usr/bin/env python3
"""
Unified dataset registry for managing multiple Manim data sources.
Supports extraction, LLM description generation, and caching.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetExtractor(ABC):
    """Base class for dataset extractors."""
    
    @abstractmethod
    def extract(self) -> List[Dict[str, Any]]:
        """Extract samples from the dataset source."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        pass


class LocalCodeExtractor(DatasetExtractor):
    """Extractor for local code repositories."""
    
    def __init__(self, 
                 name: str,
                 repo_path: Path,
                 file_pattern: str = "**/*.py",
                 scene_pattern: str = r'class\s+(\w+)\s*\(Scene\)',
                 metadata_enricher: Optional[callable] = None):
        self.name = name
        self.repo_path = Path(repo_path)
        self.file_pattern = file_pattern
        self.scene_pattern = scene_pattern
        self.metadata_enricher = metadata_enricher or (lambda x: x)
    
    def extract(self) -> List[Dict[str, Any]]:
        """Extract Manim scenes from repository."""
        samples = []
        
        for file_path in self.repo_path.glob(self.file_pattern):
            if 'test' in str(file_path).lower():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all Scene classes
                matches = re.finditer(self.scene_pattern, content)
                
                for match in matches:
                    class_name = match.group(1)
                    
                    # Extract the class definition
                    code = self._extract_class_code(content, class_name)
                    if not code:
                        continue
                    
                    # Create sample with placeholder description
                    sample = self._create_sample(
                        code=code,
                        class_name=class_name,
                        file_path=file_path
                    )
                    
                    # Enrich metadata
                    sample = self.metadata_enricher(sample)
                    
                    samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        return samples
    
    def _extract_class_code(self, content: str, class_name: str) -> Optional[str]:
        """Extract complete class definition."""
        # Simple extraction - can be enhanced
        pattern = rf'class {class_name}\s*\([^)]+\):\s*\n'
        match = re.search(pattern, content)
        
        if not match:
            return None
        
        start = match.start()
        lines = content[start:].split('\n')
        
        # Find the end of the class
        class_lines = [lines[0]]
        indent_level = None
        
        for line in lines[1:]:
            if line.strip() == '':
                class_lines.append(line)
                continue
            
            # Detect indent level
            if indent_level is None and line.strip():
                indent_level = len(line) - len(line.lstrip())
            
            # Check if we're still in the class
            if line.strip() and not line.startswith(' ' * indent_level):
                break
                
            class_lines.append(line)
        
        # Include imports
        imports = self._extract_imports(content)
        
        return imports + '\n\n' + '\n'.join(class_lines)
    
    def _extract_imports(self, content: str) -> str:
        """Extract import statements."""
        lines = content.split('\n')
        imports = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')) and 'manim' in line:
                imports.append(line)
            elif line.strip() and not line.strip().startswith(('#', 'import', 'from')):
                # Stop at first non-import line
                break
        
        return '\n'.join(imports) if imports else "from manim import *"
    
    def _create_sample(self, code: str, class_name: str, file_path: Path) -> Dict[str, Any]:
        """Create a sample with placeholder description."""
        # Analyze code features
        features = self._analyze_code_features(code)
        
        # Create metadata
        metadata = {
            "source": self.name,
            "class_name": class_name,
            "file_path": str(file_path.relative_to(self.repo_path)),
            "code_features": features,
            "needs_description_update": True,
            "description_generated_by": "pending"
        }
        
        # Placeholder description
        placeholder = f"[TO BE GENERATED: Analyze {class_name} animation and create natural user request]"
        
        return {
            "conversations": [
                {
                    "from": "system",
                    "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
                },
                {
                    "from": "user",
                    "value": placeholder
                },
                {
                    "from": "assistant",
                    "value": f"```python\n{code}\n```"
                }
            ],
            "metadata": metadata
        }
    
    def _analyze_code_features(self, code: str) -> Dict[str, Any]:
        """Analyze code to extract features for description generation."""
        features = {
            "has_voiceover": "VoiceoverScene" in code,
            "has_3d": any(x in code for x in ["ThreeDScene", "ThreeDAxes", "Surface"]),
            "has_latex": "MathTex" in code or "Tex(" in code,
            "has_graphs": any(x in code for x in ["Axes", "NumberPlane", "Graph"]),
            "visual_elements": []
        }
        
        # Extract visual elements
        element_patterns = {
            "circles": r'Circle\(',
            "squares": r'Square\(',
            "rectangles": r'Rectangle\(',
            "lines": r'Line\(',
            "arrows": r'Arrow\(',
            "text": r'Text\(',
            "equations": r'MathTex\(',
            "groups": r'VGroup\(',
            "3d_objects": r'(Surface|Sphere|Cube|Cylinder)\('
        }
        
        for element, pattern in element_patterns.items():
            if re.search(pattern, code):
                features["visual_elements"].append(element)
        
        return features
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "name": self.name,
            "type": "local_code",
            "repo_path": str(self.repo_path),
            "file_pattern": self.file_pattern
        }


class Dan4LifeExtractor(LocalCodeExtractor):
    """Specialized extractor for Dan4Life AoC2024 dataset."""
    
    def __init__(self, repo_path: Path = Path("AoC2024_Videos")):
        super().__init__(
            name="dan4life_aoc2024",
            repo_path=repo_path,
            file_pattern="**/scene.py",
            metadata_enricher=self._enrich_aoc_metadata
        )
    
    def _enrich_aoc_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add AoC-specific metadata."""
        file_path = sample["metadata"]["file_path"]
        
        # Extract day number
        day_match = re.search(r'Day_(\d+)', file_path)
        if day_match:
            day = int(day_match.group(1))
            sample["metadata"]["aoc_day"] = day
            sample["metadata"]["original_context"] = f"Advent of Code 2024 Day {day}"
        
        # Check for version
        version_match = re.search(r'Version(\d+)', file_path)
        if version_match:
            sample["metadata"]["version"] = int(version_match.group(1))
        
        return sample


class SzymonOzogExtractor(LocalCodeExtractor):
    """Specialized extractor for Szymon Ozog's educational videos."""
    
    # YouTube URL mappings
    VIDEO_MAPPINGS = {
        "entropy.py": {
            "url": "https://www.youtube.com/watch?v=-j2Z2heUBYc",
            "title": "Why use Entropy? | Information Theory - Part 1"
        },
        "capacity.py": {
            "url": "https://www.youtube.com/watch?v=r0aVt3uQJNU",
            "title": "The most fundamental theorem | Information Theory Part 2"
        },
        # Add more mappings as needed
    }
    
    def __init__(self, repo_paths: List[Path]):
        self.repo_paths = repo_paths
        self.extractors = []
        
        for repo_path in repo_paths:
            topic = "gpu" if "GPU" in str(repo_path) else "info_theory"
            extractor = LocalCodeExtractor(
                name=f"szymon_ozog_{topic}",
                repo_path=repo_path,
                file_pattern="**/*.py",
                metadata_enricher=self._enrich_video_metadata
            )
            self.extractors.append(extractor)
    
    def extract(self) -> List[Dict[str, Any]]:
        """Extract from all repositories."""
        all_samples = []
        for extractor in self.extractors:
            samples = extractor.extract()
            all_samples.extend(samples)
        return all_samples
    
    def _enrich_video_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add YouTube video metadata."""
        file_name = Path(sample["metadata"]["file_path"]).name
        
        if file_name in self.VIDEO_MAPPINGS:
            video_info = self.VIDEO_MAPPINGS[file_name]
            sample["metadata"]["youtube_url"] = video_info["url"]
            sample["metadata"]["video_title"] = video_info["title"]
            sample["metadata"]["has_transcript"] = True
        
        return sample
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "szymon_ozog",
            "type": "local_code_multi",
            "repo_paths": [str(p) for p in self.repo_paths]
        }


class LLMDescriptionGenerator:
    """Handles LLM-based description generation with caching."""
    
    def __init__(self, cache_dir: Path = Path(".cache/llm_descriptions")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index."""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index."""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_cache_key(self, code: str, metadata: Dict[str, Any]) -> str:
        """Generate cache key for code + metadata."""
        cache_data = {
            "code": code,
            "source": metadata.get("source"),
            "class_name": metadata.get("class_name"),
            "context": metadata.get("original_context")
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get_cached_description(self, code: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Get cached description if available."""
        cache_key = self._get_cache_key(code, metadata)
        
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    return cached["description"]
        
        return None
    
    def cache_description(self, code: str, metadata: Dict[str, Any], 
                         description: str, llm_model: str = "unknown"):
        """Cache a generated description."""
        cache_key = self._get_cache_key(code, metadata)
        
        cache_data = {
            "description": description,
            "metadata": metadata,
            "llm_model": llm_model,
            "timestamp": datetime.now().isoformat(),
            "code_hash": hashlib.md5(code.encode()).hexdigest()
        }
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        self.cache_index[cache_key] = {
            "file": f"{cache_key}.json",
            "source": metadata.get("source"),
            "timestamp": cache_data["timestamp"]
        }
        self._save_cache_index()
    
    def generate_batch(self, samples: List[Dict[str, Any]], 
                      llm_function: callable,
                      use_transcripts: bool = True) -> List[Dict[str, Any]]:
        """Generate descriptions for a batch of samples."""
        updated_samples = []
        samples_needing_llm = []
        
        # Check cache first
        for sample in samples:
            code = sample["conversations"][2]["value"]
            # Remove markdown wrapping for cache lookup
            if code.startswith("```python"):
                code = code[9:-3].strip()
            
            metadata = sample.get("metadata", {})
            
            cached_desc = self.get_cached_description(code, metadata)
            if cached_desc:
                sample["conversations"][1]["value"] = cached_desc
                sample["metadata"]["description_generated_by"] = "cache"
                updated_samples.append(sample)
            else:
                samples_needing_llm.append(sample)
        
        logger.info(f"Found {len(updated_samples)} cached descriptions")
        logger.info(f"Need to generate {len(samples_needing_llm)} new descriptions")
        
        # Generate new descriptions
        if samples_needing_llm:
            # Prepare for LLM call
            llm_inputs = []
            
            for sample in samples_needing_llm:
                code = sample["conversations"][2]["value"]
                if code.startswith("```python"):
                    code = code[9:-3].strip()
                
                metadata = sample.get("metadata", {})
                
                llm_input = {
                    "code": code,
                    "metadata": metadata,
                    "features": metadata.get("code_features", {}),
                }
                
                # Add transcript info if available
                if use_transcripts and metadata.get("youtube_url"):
                    llm_input["youtube_url"] = metadata["youtube_url"]
                    llm_input["video_title"] = metadata.get("video_title", "")
                
                llm_inputs.append(llm_input)
            
            # Call LLM (this would be your actual implementation)
            descriptions = llm_function(llm_inputs)
            
            # Update samples and cache
            for sample, desc in zip(samples_needing_llm, descriptions):
                sample["conversations"][1]["value"] = desc
                sample["metadata"]["description_generated_by"] = "llm_gpt4"
                
                # Cache for future use
                code = sample["conversations"][2]["value"]
                if code.startswith("```python"):
                    code = code[9:-3].strip()
                
                self.cache_description(
                    code, 
                    sample["metadata"], 
                    desc, 
                    "gpt4"
                )
                
                updated_samples.append(sample)
        
        return updated_samples


class DatasetRegistry:
    """Central registry for all dataset extractors."""
    
    def __init__(self):
        self.extractors: Dict[str, DatasetExtractor] = {}
        self.llm_generator = LLMDescriptionGenerator()
    
    def register(self, name: str, extractor: DatasetExtractor):
        """Register a dataset extractor."""
        self.extractors[name] = extractor
    
    def extract_all(self, dataset_names: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Extract samples from all registered datasets."""
        if dataset_names:
            extractors_to_use = {k: v for k, v in self.extractors.items() if k in dataset_names}
        else:
            extractors_to_use = self.extractors
        
        all_samples = {}
        
        for name, extractor in extractors_to_use.items():
            logger.info(f"Extracting from {name}...")
            samples = extractor.extract()
            all_samples[name] = samples
            logger.info(f"Extracted {len(samples)} samples from {name}")
        
        return all_samples
    
    def save_for_llm_processing(self, samples: Dict[str, List[Dict[str, Any]]], 
                               output_dir: Path):
        """Save samples for LLM description generation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all samples
        all_samples = []
        for dataset_name, dataset_samples in samples.items():
            all_samples.extend(dataset_samples)
        
        # Save for LLM processing
        llm_file = output_dir / "samples_for_llm.jsonl"
        with open(llm_file, 'w') as f:
            for sample in all_samples:
                llm_input = {
                    "code": sample["conversations"][2]["value"],
                    "metadata": sample["metadata"],
                    "task": "Generate a natural user request for this Manim animation"
                }
                f.write(json.dumps(llm_input) + '\n')
        
        logger.info(f"Saved {len(all_samples)} samples for LLM processing to {llm_file}")
        
        # Save raw samples with placeholders
        raw_file = output_dir / "raw_samples_with_placeholders.jsonl"
        with open(raw_file, 'w') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Saved raw samples to {raw_file}")
        
        # Save metadata
        metadata = {
            "total_samples": len(all_samples),
            "by_dataset": {name: len(samples) for name, samples in samples.items()},
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        metadata_file = output_dir / "extraction_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_with_llm(self, input_file: Path, llm_function: callable, 
                        output_file: Path) -> List[Dict[str, Any]]:
        """Process samples with LLM descriptions."""
        # Load samples
        samples = []
        with open(input_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Generate descriptions
        updated_samples = self.llm_generator.generate_batch(samples, llm_function)
        
        # Save processed samples
        with open(output_file, 'w') as f:
            for sample in updated_samples:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Saved {len(updated_samples)} samples with descriptions to {output_file}")
        
        return updated_samples


# Example usage
def setup_registry():
    """Set up the dataset registry with all extractors."""
    registry = DatasetRegistry()
    
    # Register Dan4Life extractor
    registry.register("dan4life_aoc2024", Dan4LifeExtractor())
    
    # Register Szymon Ozog extractors
    szymon_repos = [
        Path("InformationTheory"),
        Path("GPU_Programming")
    ]
    registry.register("szymon_ozog", SzymonOzogExtractor(szymon_repos))
    
    # Add more extractors as needed
    # registry.register("reducible", ReducibleExtractor())
    # registry.register("kilacoda", KilacodaExtractor())
    
    return registry


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Manim datasets for LLM processing")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to extract")
    parser.add_argument("--output-dir", default="data_extracted", help="Output directory")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set up registry
    registry = setup_registry()
    
    # Extract samples
    samples = registry.extract_all(args.datasets)
    
    # Save for LLM processing
    output_dir = Path(args.output_dir)
    registry.save_for_llm_processing(samples, output_dir)
    
    print(f"\nExtraction complete!")
    print(f"Next steps:")
    print(f"1. Use LLM to generate descriptions from {output_dir}/samples_for_llm.jsonl")
    print(f"2. Run process_with_llm() to update samples with descriptions")
    print(f"3. Run prepare_data_enhanced.py with the processed dataset")