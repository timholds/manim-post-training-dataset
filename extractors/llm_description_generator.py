"""
LLM Description Generator for Manim dataset extractors.
Provides caching and batch processing for description generation.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


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
        # Include relevant metadata that affects description
        cache_data = {
            "code": code,
            "source": metadata.get("source"),
            "original_context": metadata.get("original_context"),
            "youtube_url": metadata.get("youtube_url"),
            "video_title": metadata.get("video_title")
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
    
    def analyze_code_features(self, code: str) -> Dict[str, Any]:
        """Analyze code to extract features for description generation."""
        features = {
            "has_3d": any(x in code for x in ["ThreeDScene", "ThreeDAxes", "Surface"]),
            "has_latex": "MathTex" in code or "Tex(" in code,
            "has_graphs": any(x in code for x in ["Axes", "NumberPlane", "Graph"]),
            "has_voiceover": "VoiceoverScene" in code,
            "has_binary_ops": "binary" in code.lower() or "0b" in code,
            "has_xor": "^" in code or "XOR" in code or "oplus" in code,
            "has_animations": ".animate" in code,
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
            "3d_objects": r'(Surface|Sphere|Cube|Cylinder)\(',
            "triangles": r'Triangle\(',
            "polygons": r'Polygon\(',
            "dots": r'Dot\('
        }
        
        for element, pattern in element_patterns.items():
            if re.search(pattern, code):
                features["visual_elements"].append(element)
        
        return features
    
    def process_batch(self, samples: List[Dict[str, Any]], 
                     llm_function: callable = None,
                     use_cache: bool = True) -> List[Dict[str, Any]]:
        """Process a batch of samples, using cache when available."""
        updated_samples = []
        samples_needing_llm = []
        
        for sample in samples:
            # Check if needs description update
            if not sample.get("metadata", {}).get("needs_description", True):
                updated_samples.append(sample)
                continue
            
            code = sample.get("code", "")
            metadata = sample.get("metadata", {})
            
            # Add code features to metadata
            if "code_features" not in metadata:
                metadata["code_features"] = self.analyze_code_features(code)
            
            # Check cache
            if use_cache:
                cached_desc = self.get_cached_description(code, metadata)
                if cached_desc:
                    sample["description"] = cached_desc
                    metadata["description_generated_by"] = "cache"
                    metadata["needs_description"] = False
                    updated_samples.append(sample)
                    continue
            
            samples_needing_llm.append(sample)
        
        logger.info(f"Found {len(updated_samples) - len(samples_needing_llm)} cached descriptions")
        logger.info(f"Need to generate {len(samples_needing_llm)} new descriptions")
        
        # Generate new descriptions if LLM function provided
        if samples_needing_llm and llm_function:
            descriptions = llm_function(samples_needing_llm)
            
            for sample, desc in zip(samples_needing_llm, descriptions):
                sample["description"] = desc
                sample["metadata"]["description_generated_by"] = "llm"
                sample["metadata"]["needs_description"] = False
                
                # Cache the result
                if use_cache:
                    self.cache_description(
                        sample["code"],
                        sample["metadata"],
                        desc,
                        "llm"
                    )
                
                updated_samples.append(sample)
        else:
            # Return with placeholders intact
            updated_samples.extend(samples_needing_llm)
        
        return updated_samples
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        for entry in self.cache_index.values():
            cache_file = self.cache_dir / entry["file"]
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        return {
            "total_entries": len(self.cache_index),
            "cache_size_mb": total_size / 1024 / 1024,
            "sources": list(set(entry.get("source", "unknown") 
                              for entry in self.cache_index.values()))
        }


# Global instance for convenience
_llm_generator = None

def get_llm_generator() -> LLMDescriptionGenerator:
    """Get or create the global LLM generator instance."""
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMDescriptionGenerator()
    return _llm_generator