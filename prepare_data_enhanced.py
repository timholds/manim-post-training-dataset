#!/usr/bin/env python3
"""
Enhanced data preparation pipeline for multiple Manim datasets.
Supports automatic downloading from Kaggle and HuggingFace.
"""

import json
import pandas as pd
from pathlib import Path
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
import os
import subprocess
import sys
from datasets import load_dataset
import requests
from tqdm import tqdm
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System prompt for all datasets
SYSTEM_PROMPT = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."

# Prompt variations for augmentation
PROMPT_TEMPLATES = [
    "{description}",  # Original format
    "Create a Manim animation that {description_lower}",
    "Write Manim code to {description_lower}",
    "Generate a Manim scene that {description_lower}",
    "Implement a Manim animation for: {description}",
    "Using Manim, {description_lower}",
]

# Dataset configurations
DATASETS = {
    "manimbench": {
        "type": "kaggle",
        "kaggle_dataset": "ravidussilva/manim-sft",
        "file": "manim_sft_dataset.parquet",
        "description_field": "Reviewed Description",
        "code_field": "Code",
        "split_field": "Split",
        "expected_samples": 400
    },
    "bespoke_manim": {
        "type": "huggingface",
        "dataset_name": "bespokelabs/bespoke-manim",
        "description_field": "question",  # The question/prompt field
        "code_field": "python_code",      # The actual Manim code
        "split": "train",
        "expected_samples": 1000
    },
    "thanks_dataset": {
        "type": "huggingface",
        "dataset_name": "thanhkt/manim_code",
        "description_field": "input",  # Corrected field name
        "code_field": "output",
        "split": "train",
        "expected_samples": 4400
    },
    "dan4life_aoc2024": {
        "type": "local",
        "file": "data_dan4life/dan4life_aoc2024.jsonl",
        "description_field": "conversations[1].value",
        "code_field": "conversations[2].value",
        "expected_samples": 24
    },
    "szymon_ozog": {
        "type": "local",
        "file": "data_szymon_ozog/szymon_ozog_processed.jsonl",
        "description_field": "conversations[1].value",
        "code_field": "conversations[2].value",
        "expected_samples": 29
    }
}

def ensure_proper_code_format(code: str) -> str:
    """Ensure code has proper Scene class structure."""
    code = code.strip()
    
    # Check if code already has proper structure
    if "class" in code and "Scene" in code and "def construct" in code:
        return code
    
    # Add minimal Scene structure if missing
    if not code.startswith("from manim import"):
        # Extract imports if they exist elsewhere in the code
        lines = code.split('\n')
        imports = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')) and 'manim' in line:
                imports.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct with proper structure
        if not imports:
            imports = ["from manim import *"]
        
        code = '\n'.join(imports) + '\n\n'
        code += "class AnimationScene(Scene):\n"
        code += "    def construct(self):\n"
        
        # Indent the rest of the code
        for line in other_lines:
            if line.strip():
                code += "        " + line + "\n"
            else:
                code += "\n"
    
    else:
        # Code has imports but maybe not proper class structure
        if "class" not in code or "Scene" not in code:
            # Split at imports
            parts = code.split('\n\n', 1)
            imports = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            
            return f"""{imports}

class AnimationScene(Scene):
    def construct(self):
        {chr(10).join('        ' + line for line in rest.split(chr(10)) if line.strip())}"""
    
    return code

def create_conversation(description: str, code: str, prompt_variation_idx: int = 0) -> dict:
    """Create a conversation with system prompt and structured output."""
    # Format user prompt based on variation index
    template = PROMPT_TEMPLATES[prompt_variation_idx % len(PROMPT_TEMPLATES)]
    user_prompt = template.format(
        description=description,
        description_lower=description.lower() if not description[0].isupper() else description[0].lower() + description[1:]
    )
    
    # Ensure code is properly formatted
    formatted_code = ensure_proper_code_format(code)
    
    # Wrap code in markdown blocks as expected by the system prompt
    assistant_response = f"```python\n{formatted_code}\n```"
    
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "user", "value": user_prompt},
            {"from": "assistant", "value": assistant_response}
        ]
    }

def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> Optional[Path]:
    """Download dataset from Kaggle using kagglehub."""
    try:
        # Try importing kagglehub
        import kagglehub
        
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        dataset_path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Downloaded to: {dataset_path}")
        return Path(dataset_path)
        
    except ImportError:
        logger.error("kagglehub not installed. Install with: pip install kagglehub")
        logger.info("Make sure you have Kaggle API credentials configured")
        return None
    except Exception as e:
        logger.error(f"Failed to download Kaggle dataset: {e}")
        logger.info("Make sure you have Kaggle API credentials configured:")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Create new API token")
        logger.info("3. Place kaggle.json in ~/.kaggle/")
        return None

def load_kaggle_dataset(config: Dict[str, Any], cache_dir: Path) -> Optional[pd.DataFrame]:
    """Load Kaggle dataset, downloading if necessary."""
    dataset_name = config["kaggle_dataset"]
    file_name = config["file"]
    
    # Check cache first
    cache_file = cache_dir / f"{dataset_name.replace('/', '_')}_{file_name}"
    
    if cache_file.exists():
        logger.info(f"Loading cached dataset from {cache_file}")
        return pd.read_parquet(cache_file)
    
    # Download dataset
    download_path = download_kaggle_dataset(dataset_name, cache_dir)
    if not download_path:
        return None
    
    # Find the parquet file
    parquet_file = download_path / file_name
    if not parquet_file.exists():
        # Search for any parquet file
        parquet_files = list(download_path.glob("*.parquet"))
        if parquet_files:
            parquet_file = parquet_files[0]
            logger.info(f"Using parquet file: {parquet_file}")
        else:
            logger.error(f"No parquet file found in {download_path}")
            return None
    
    # Load and cache
    df = pd.read_parquet(parquet_file)
    df.to_parquet(cache_file)
    logger.info(f"Cached dataset to {cache_file}")
    
    return df

def load_huggingface_dataset(config: Dict[str, Any], cache_dir: Path) -> Optional[pd.DataFrame]:
    """Load HuggingFace dataset."""
    dataset_name = config["dataset_name"]
    split = config.get("split", "train")
    
    try:
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Cache it
        cache_file = cache_dir / f"{dataset_name.replace('/', '_')}_{split}.parquet"
        df.to_parquet(cache_file)
        logger.info(f"Cached dataset to {cache_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
        return None

def load_local_dataset(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load local JSONL dataset."""
    file_path = Path(config["file"])
    
    if not file_path.exists():
        logger.error(f"Local file not found: {file_path}")
        return None
    
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Extract fields based on config paths
                desc_path = config["description_field"].split('.')
                code_path = config["code_field"].split('.')
                
                # Navigate through nested structure
                desc_value = item
                for key in desc_path:
                    if '[' in key and ']' in key:
                        # Handle array access like "conversations[1]"
                        base_key = key[:key.index('[')]
                        index = int(key[key.index('[')+1:key.index(']')])
                        desc_value = desc_value[base_key][index]
                    else:
                        desc_value = desc_value[key]
                
                code_value = item
                for key in code_path:
                    if '[' in key and ']' in key:
                        base_key = key[:key.index('[')]
                        index = int(key[key.index('[')+1:key.index(']')])
                        code_value = code_value[base_key][index]
                    else:
                        code_value = code_value[key]
                
                # Clean code from markdown if needed
                if code_value.startswith("```python"):
                    code_value = code_value.split("```python")[1].split("```")[0].strip()
                
                data.append({
                    config["description_field"]: desc_value,
                    config["code_field"]: code_value,
                    "source": item.get("source", "unknown")
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading local dataset: {e}")
        return None

def process_dataset(dataset_name: str, config: Dict[str, Any], cache_dir: Path) -> List[Dict[str, Any]]:
    """Process a single dataset based on its configuration."""
    logger.info(f"\nProcessing dataset: {dataset_name}")
    
    # Load dataset based on type
    if config["type"] == "kaggle":
        df = load_kaggle_dataset(config, cache_dir)
    elif config["type"] == "huggingface":
        df = load_huggingface_dataset(config, cache_dir)
    elif config["type"] == "local":
        df = load_local_dataset(config)
    else:
        logger.error(f"Unknown dataset type: {config['type']}")
        return []
    
    if df is None:
        logger.error(f"Failed to load dataset {dataset_name}")
        return []
    
    logger.info(f"Loaded {len(df)} samples from {dataset_name}")
    
    # Process samples
    processed_data = []
    desc_field = config["description_field"]
    code_field = config["code_field"]
    
    # Check if fields exist
    if desc_field not in df.columns or code_field not in df.columns:
        logger.error(f"Required fields not found. Available columns: {df.columns.tolist()}")
        # Try to auto-detect fields
        desc_candidates = [col for col in df.columns if any(term in col.lower() for term in ['desc', 'inst', 'prompt', 'question'])]
        code_candidates = [col for col in df.columns if any(term in col.lower() for term in ['code', 'output', 'answer', 'response'])]
        
        if desc_candidates and code_candidates:
            desc_field = desc_candidates[0]
            code_field = code_candidates[0]
            logger.info(f"Auto-detected fields: description='{desc_field}', code='{code_field}'")
        else:
            return []
    
    for idx, row in df.iterrows():
        try:
            description = str(row[desc_field])
            code = str(row[code_field])
            
            # Skip invalid entries
            if not description or not code or description == 'nan' or code == 'nan':
                continue
            
            # Clean up code if needed (handle markdown if present in source data)
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
            # Remove literal \n from the beginning (common in thanks_dataset)
            if code.startswith('\\n'):
                code = code[2:].lstrip()
            
            # Check for minimal validity
            if len(code) < 20 or len(description) < 5:
                continue
            
            # Create base conversation
            processed_data.append({
                "description": description,
                "code": code,
                "source": dataset_name
            })
            
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} samples from {dataset_name}")
    return processed_data

def split_dataset(data: List[Dict[str, Any]], test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and test sets."""
    random.seed(seed)
    random.shuffle(data)
    
    test_size = int(len(data) * test_ratio)
    test_data = data[:test_size]
    train_data = data[test_size:]
    
    return train_data, test_data

# Source priority for deduplication (higher = keep when duplicates found)
SOURCE_PRIORITY = {
    "manimbench": 4,      # Highest quality, reviewed descriptions
    "bespoke_manim": 3,   # Rich context, transcripts
    "thanks_dataset": 2   # Large dataset
}

def normalize_description(desc: str) -> str:
    """Normalize description for comparison."""
    # Lowercase and normalize whitespace
    return ' '.join(desc.lower().split())

def deduplicate_data(all_data: List[Dict[str, Any]]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Remove duplicate descriptions across datasets.
    Returns deduplicated data and statistics.
    """
    from collections import defaultdict
    
    # Group by normalized description
    desc_groups = defaultdict(list)
    
    for item in all_data:
        norm_desc = normalize_description(item['description'])
        desc_groups[norm_desc].append(item)
    
    # Statistics tracking
    stats = {
        'total_raw': len(all_data),
        'unique_descriptions': len(desc_groups),
        'duplicates_removed': 0,
        'duplicates_by_source': defaultdict(int),
        'kept_by_source': defaultdict(int),
        'cross_source_duplicates': 0,
        'within_source_duplicates': 0,
        'examples_removed': []
    }
    
    # Process each group of duplicates
    deduplicated = []
    
    for norm_desc, items in desc_groups.items():
        if len(items) == 1:
            # No duplicates
            deduplicated.append(items[0])
            stats['kept_by_source'][items[0]['source']] += 1
        else:
            # Found duplicates - need to choose best one
            sources = [item['source'] for item in items]
            unique_sources = set(sources)
            
            if len(unique_sources) > 1:
                stats['cross_source_duplicates'] += 1
            else:
                stats['within_source_duplicates'] += 1
            
            # Sort by source priority
            sorted_items = sorted(items, key=lambda x: SOURCE_PRIORITY.get(x['source'], 0), reverse=True)
            
            # Keep the best one
            best_item = sorted_items[0]
            deduplicated.append(best_item)
            stats['kept_by_source'][best_item['source']] += 1
            
            # Track what we removed
            for item in sorted_items[1:]:
                stats['duplicates_removed'] += 1
                stats['duplicates_by_source'][item['source']] += 1
                
                # Save some examples (limit to prevent huge files)
                if len(stats['examples_removed']) < 100:
                    stats['examples_removed'].append({
                        'description': item['description'],
                        'source': item['source'],
                        'kept_source': best_item['source'],
                        'normalized_desc': norm_desc
                    })
    
    # Log summary
    logger.info(f"\nDeduplication Summary:")
    logger.info(f"  Original samples: {stats['total_raw']:,}")
    logger.info(f"  Unique samples: {len(deduplicated):,}")
    logger.info(f"  Duplicates removed: {stats['duplicates_removed']:,}")
    logger.info(f"  Cross-source duplicates: {stats['cross_source_duplicates']:,}")
    logger.info(f"  Within-source duplicates: {stats['within_source_duplicates']:,}")
    
    return deduplicated, stats

def prepare_all_datasets(output_dir: str, dataset_names: Optional[List[str]] = None, use_augmentation: bool = False, deduplicate: bool = True):
    """Prepare all configured datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path.home() / ".cache" / "manim_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Select datasets to process
    if dataset_names:
        datasets_to_process = {name: DATASETS[name] for name in dataset_names if name in DATASETS}
    else:
        datasets_to_process = DATASETS
    
    # Process all datasets
    all_data = []
    dataset_stats = {}
    
    for dataset_name, config in datasets_to_process.items():
        data = process_dataset(dataset_name, config, cache_dir)
        all_data.extend(data)
        dataset_stats[dataset_name] = {
            "samples": len(data),
            "expected": config.get("expected_samples", "unknown")
        }
    
    if not all_data:
        logger.error("No data processed from any dataset!")
        return
    
    logger.info(f"\nTotal samples collected: {len(all_data)}")
    
    # Apply deduplication if requested
    dedup_stats = None
    if deduplicate:
        all_data, dedup_stats = deduplicate_data(all_data)
        logger.info(f"After deduplication: {len(all_data)} samples")
    
    # Split into train/test
    train_data, test_data = split_dataset(all_data, test_ratio=0.1)
    
    # Apply augmentation to training data
    augmented_train = []
    for item in train_data:
        # Always include original
        conv = create_conversation(item["description"], item["code"], 0)
        conv["source"] = item["source"]
        augmented_train.append(conv)
        
        # Add augmented versions
        if use_augmentation:
            # Add 1-2 more variations per sample
            num_augmentations = random.randint(1, 2)
            for i in range(1, num_augmentations + 1):
                conv = create_conversation(item["description"], item["code"], i)
                conv["source"] = item["source"]
                augmented_train.append(conv)
    
    # Process test data (no augmentation)
    augmented_test = []
    for item in test_data:
        conv = create_conversation(item["description"], item["code"], 0)
        conv["source"] = item["source"]
        augmented_test.append(conv)
    
    # Save datasets
    train_file = output_path / "train.json"
    test_file = output_path / "test.json"
    stats_file = output_path / "dataset_stats.json"
    
    # Write in JSON lines format
    with open(train_file, 'w') as f:
        for item in augmented_train:
            # Keep source field for tracking
            f.write(json.dumps(item) + '\n')
    
    with open(test_file, 'w') as f:
        for item in augmented_test:
            # Keep source field for tracking
            f.write(json.dumps(item) + '\n')
    
    # Save statistics
    final_stats = {
        "dataset_stats": dataset_stats,
        "total_samples": {
            "raw": dedup_stats['total_raw'] if dedup_stats else len(all_data),
            "after_deduplication": len(all_data) if deduplicate else None,
            "train_before_augmentation": len(train_data),
            "train_after_augmentation": len(augmented_train),
            "test": len(augmented_test),
            "augmentation_factor": len(augmented_train) / len(train_data) if train_data else 0
        },
        "source_distribution": {
            source: len([item for item in augmented_train if item.get("source") == source])
            for source in dataset_stats.keys()
        },
        "deduplication_applied": deduplicate
    }
    
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Save deduplication report if applicable
    if dedup_stats:
        dedup_report_file = output_path / "deduplication_report.json"
        dedup_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_raw_samples": dedup_stats['total_raw'],
                "unique_samples": dedup_stats['unique_descriptions'],
                "duplicates_removed": dedup_stats['duplicates_removed'],
                "reduction_percentage": (dedup_stats['duplicates_removed'] / dedup_stats['total_raw'] * 100) if dedup_stats['total_raw'] > 0 else 0,
                "cross_source_duplicates": dedup_stats['cross_source_duplicates'],
                "within_source_duplicates": dedup_stats['within_source_duplicates']
            },
            "duplicates_by_source": dict(dedup_stats['duplicates_by_source']),
            "kept_by_source": dict(dedup_stats['kept_by_source']),
            "source_priority": SOURCE_PRIORITY
        }
        
        with open(dedup_report_file, 'w') as f:
            json.dump(dedup_report, f, indent=2)
        
        # Save removed examples
        removed_examples_file = output_path / "removed_duplicates.json"
        with open(removed_examples_file, 'w') as f:
            json.dump(dedup_stats['examples_removed'], f, indent=2)
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("DATASET PREPARATION COMPLETE" + (" WITH DEDUPLICATION" if deduplicate else ""))
    logger.info("="*60)
    if dedup_stats:
        logger.info(f"Original samples: {dedup_stats['total_raw']:,}")
        logger.info(f"After deduplication: {len(all_data):,} ({dedup_stats['duplicates_removed']:,} removed, {dedup_stats['duplicates_removed']/dedup_stats['total_raw']*100:.1f}% reduction)")
    logger.info(f"Train samples: {len(augmented_train)} ({len(augmented_train)/len(train_data):.1f}x augmentation)")
    logger.info(f"Test samples: {len(augmented_test)}")
    logger.info(f"\nDataset statistics saved to: {stats_file}")
    if dedup_stats:
        logger.info(f"Deduplication report saved to: {output_path / 'deduplication_report.json'}")
        logger.info(f"Removed examples saved to: {output_path / 'removed_duplicates.json'}")
    logger.info("\nSource distribution:")
    for source, count in final_stats["source_distribution"].items():
        logger.info(f"  {source}: {count} samples")
    
    # Show sample
    logger.info("\nSample conversation from training set:")
    if augmented_train:
        sample = augmented_train[0]
        logger.info(f"System: {sample['conversations'][0]['value']}")
        logger.info(f"User: {sample['conversations'][1]['value']}")
        logger.info(f"Assistant: {sample['conversations'][2]['value'][:200]}...")

def main():
    """Main function with CLI support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Manim datasets for fine-tuning")
    parser.add_argument("--output-dir", default="data_formatted", help="Output directory for processed datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), 
                        help="Specific datasets to process (default: all)")
    parser.add_argument("--augmentation", action="store_true", help="Enable data augmentation (default: disabled)")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable deduplication (default: enabled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Check for required packages
    try:
        import kagglehub
    except ImportError:
        logger.warning("kagglehub not installed. Install with: pip install kagglehub")
        logger.warning("Kaggle datasets will not be available")
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        sys.exit(1)
    
    prepare_all_datasets(
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        use_augmentation=args.augmentation,
        deduplicate=not args.no_deduplicate
    )

if __name__ == "__main__":
    main()