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
        "description_field": "instruction",  # May need adjustment based on actual schema
        "code_field": "output",
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
    "manim_codegen": {
        "type": "huggingface", 
        "dataset_name": "generaleoley/manim-codegen",
        "description_field": "query",  # Corrected field name
        "code_field": "answer",  # Corrected field name
        "split": "train",
        "expected_samples": 1600
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
    
    # Wrap in code blocks
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

def process_dataset(dataset_name: str, config: Dict[str, Any], cache_dir: Path) -> List[Dict[str, Any]]:
    """Process a single dataset based on its configuration."""
    logger.info(f"\nProcessing dataset: {dataset_name}")
    
    # Load dataset based on type
    if config["type"] == "kaggle":
        df = load_kaggle_dataset(config, cache_dir)
    elif config["type"] == "huggingface":
        df = load_huggingface_dataset(config, cache_dir)
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
            
            # Clean up code if needed
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
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

def prepare_all_datasets(output_dir: str, dataset_names: Optional[List[str]] = None, use_augmentation: bool = True):
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
            "raw": len(all_data),
            "train_before_augmentation": len(train_data),
            "train_after_augmentation": len(augmented_train),
            "test": len(augmented_test),
            "augmentation_factor": len(augmented_train) / len(train_data) if train_data else 0
        },
        "source_distribution": {
            source: len([item for item in augmented_train if item.get("source") == source])
            for source in dataset_stats.keys()
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Train samples: {len(augmented_train)} ({len(augmented_train)/len(train_data):.1f}x augmentation)")
    logger.info(f"Test samples: {len(augmented_test)}")
    logger.info(f"\nDataset statistics saved to: {stats_file}")
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
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
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
        use_augmentation=not args.no_augmentation
    )

if __name__ == "__main__":
    main()