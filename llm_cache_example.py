#!/usr/bin/env python3
"""
Example implementation of LLM caching for description generation.
This shows how to avoid redundant LLM calls across pipeline runs.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
import time

class LLMDescriptionCache:
    """Cache for LLM-generated descriptions based on code content."""
    
    def __init__(self, cache_dir: Path = Path(".cache/llm_descriptions")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "index.json"
        self.cache_index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load cache index from disk."""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_code_hash(self, code: str, metadata: Dict = None) -> str:
        """Generate stable hash for code + metadata."""
        # Include relevant metadata that affects description
        cache_key = {
            "code": code,
            "day": metadata.get("day") if metadata else None,
            "version": metadata.get("version") if metadata else None,
            "context": metadata.get("original_context") if metadata else None
        }
        
        # Create stable JSON representation
        cache_str = json.dumps(cache_key, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(self, code: str, metadata: Dict = None) -> Optional[Dict]:
        """Get cached description if available."""
        code_hash = self._get_code_hash(code, metadata)
        
        if code_hash in self.cache_index:
            cache_entry = self.cache_index[code_hash]
            cache_file = self.cache_dir / f"{code_hash}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        return None
    
    def set(self, code: str, description: str, metadata: Dict = None, 
            llm_model: str = "unknown", llm_response: Dict = None):
        """Cache a generated description."""
        code_hash = self._get_code_hash(code, metadata)
        
        # Create cache entry
        cache_data = {
            "description": description,
            "metadata": metadata,
            "llm_model": llm_model,
            "llm_response": llm_response,  # Full response for debugging
            "timestamp": time.time(),
            "code_preview": code[:500] + "..." if len(code) > 500 else code
        }
        
        # Save to file
        cache_file = self.cache_dir / f"{code_hash}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        # Update index
        self.cache_index[code_hash] = {
            "file": f"{code_hash}.json",
            "timestamp": cache_data["timestamp"],
            "day": metadata.get("day") if metadata else None
        }
        self._save_index()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache_index),
            "cache_size_mb": sum(
                (self.cache_dir / entry["file"]).stat().st_size 
                for entry in self.cache_index.values()
                if (self.cache_dir / entry["file"]).exists()
            ) / 1024 / 1024,
            "oldest_entry": min(
                (entry["timestamp"] for entry in self.cache_index.values()),
                default=None
            ),
            "newest_entry": max(
                (entry["timestamp"] for entry in self.cache_index.values()),
                default=None
            )
        }

def process_with_cache(samples, llm_function, cache: LLMDescriptionCache):
    """Process samples with caching to avoid redundant LLM calls."""
    
    samples_needing_llm = []
    cached_count = 0
    
    for sample in samples:
        code = sample["conversations"][2]["value"]
        metadata = sample.get("metadata", {})
        
        # Check cache first
        cached = cache.get(code, metadata)
        if cached:
            # Use cached description
            sample["conversations"][1]["value"] = cached["description"]
            sample["metadata"]["description_generated_by"] = f"cache_{cached['llm_model']}"
            cached_count += 1
        else:
            # Need LLM generation
            samples_needing_llm.append(sample)
    
    print(f"Found {cached_count} cached descriptions")
    print(f"Need to generate {len(samples_needing_llm)} new descriptions")
    
    # Batch process samples needing LLM
    if samples_needing_llm:
        # This would be your actual LLM call
        descriptions = llm_function(samples_needing_llm)
        
        # Cache the results
        for sample, desc in zip(samples_needing_llm, descriptions):
            code = sample["conversations"][2]["value"]
            metadata = sample.get("metadata", {})
            
            sample["conversations"][1]["value"] = desc
            sample["metadata"]["description_generated_by"] = "llm_gpt4"
            
            # Cache for next time
            cache.set(code, desc, metadata, "gpt4", {"raw_response": desc})
    
    return samples

# Example usage
if __name__ == "__main__":
    # Initialize cache
    cache = LLMDescriptionCache()
    
    # Show cache stats
    print("Cache stats:", cache.stats())
    
    # Example: Check if code is cached
    example_code = "from manim import *\nclass Day22(Scene):..."
    example_metadata = {"day": 22, "source": "dan4life"}
    
    cached_result = cache.get(example_code, example_metadata)
    if cached_result:
        print("Found in cache:", cached_result["description"][:100])
    else:
        print("Not in cache, would need LLM call")
        
        # Simulate LLM response
        generated_desc = "Create an animation demonstrating XOR operations..."
        cache.set(example_code, generated_desc, example_metadata, "gpt4")
        print("Cached for future use")