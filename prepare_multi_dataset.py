#!/usr/bin/env python3
"""
Enhanced multi-dataset preparation pipeline for Manim fine-tuning.
Combines multiple datasets while handling different formats and maintaining quality.
"""

import json
import pandas as pd
from pathlib import Path
import logging
import random
from typing import Dict, List, Optional, Tuple
import hashlib
from collections import Counter
import re
import ast
from datasets import load_dataset
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Single, clear system prompt
SYSTEM_PROMPT = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."

# Prompt variations for augmentation
PROMPT_TEMPLATES = [
    "{description}",  # Original format
    "Create a Manim animation that {description_lower}",
    "Write Manim code to {description_lower}",
    "Generate a Manim scene that {description_lower}",
]

class DatasetConfig:
    """Configuration for each dataset source."""
    def __init__(self, name: str, path: str, format: str, 
                 description_col: str, code_col: str, 
                 split_col: Optional[str] = None,
                 filter_func: Optional[callable] = None):
        self.name = name
        self.path = path
        self.format = format
        self.description_col = description_col
        self.code_col = code_col
        self.split_col = split_col
        self.filter_func = filter_func

# Dataset configurations
DATASET_CONFIGS = [
    DatasetConfig(
        name="manimbench",
        path="/home/timholds/.cache/kagglehub/datasets/ravidussilva/manim-sft/versions/2/manim_sft_dataset.parquet",
        format="parquet",
        description_col="Reviewed Description",
        code_col="Code",
        split_col="Split"
    ),
    DatasetConfig(
        name="bespoke-manim",
        path="bespokelabs/bespoke-manim",  # HuggingFace dataset
        format="huggingface",
        description_col="question",  # We'll process multiple fields
        code_col="python_code",
        split_col=None  # Will use ratio-based split
    ),
    DatasetConfig(
        name="manim-codegen",
        path="generaleoley/manim-codegen",  # HuggingFace dataset
        format="huggingface",
        description_col="query",
        code_col="answer",
        split_col=None
    ),
    DatasetConfig(
        name="thanks-manim",
        path="thanhkt/manim_code",  # HuggingFace dataset
        format="huggingface",
        description_col="input",
        code_col="output",
        split_col=None
    ),
]

def ensure_proper_code_format(code: str) -> str:
    """Ensure code has proper imports and Scene class structure."""
    # Add imports if missing
    if "from manim import" not in code and "import manim" not in code:
        code = "from manim import *\n\n" + code
    
    # Check if code has Scene class structure
    lines = code.strip().split('\n')
    has_scene_class = any('class' in line and 'Scene' in line for line in lines)
    
    if not has_scene_class:
        # Find where imports end
        code_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(('from ', 'import ', '#')):
                code_start = i
                break
        
        # Extract imports and code body
        imports = '\n'.join(lines[:code_start])
        code_body = '\n'.join(lines[code_start:])
        
        # Wrap code in Scene class
        if imports:
            return f"""{imports}

class AnimationScene(Scene):
    def construct(self):
        {chr(10).join('        ' + line for line in code_body.split(chr(10)) if line.strip())}"""
        else:
            return f"""from manim import *

class AnimationScene(Scene):
    def construct(self):
        {chr(10).join('        ' + line for line in code.split(chr(10)) if line.strip())}"""
    
    return code

def extract_clean_code(code: str) -> str:
    """Extract clean Python code from various formats."""
    if not code:
        return ""
        
    # Remove literal \n at start
    if code.startswith('\\n'):
        code = code[2:]
    
    # Extract from markdown if present
    if '```' in code:
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', code, re.DOTALL)
        if code_blocks:
            code = code_blocks[0]
    
    return code.strip()

def create_description_from_bespoke(sample: dict) -> str:
    """Create a description from Bespoke Manim sample."""
    if sample.get('question'):
        return f"Create a Manim animation to demonstrate: {sample['question']}"
    elif sample.get('title'):
        return f"Create a Manim animation titled: {sample['title']}"
    elif sample.get('narration'):
        first_sentence = sample['narration'].split('.')[0].strip() + '.'
        return f"Create a Manim animation that shows: {first_sentence}"
    else:
        return "Create a Manim animation"

def load_dataset_from_config(config: DatasetConfig) -> pd.DataFrame:
    """Load a dataset based on its configuration."""
    logger.info(f"Loading dataset '{config.name}' from {config.path}")
    
    if config.format == "parquet":
        df = pd.read_parquet(config.path)
    elif config.format == "json":
        df = pd.read_json(config.path)
    elif config.format == "jsonl":
        df = pd.read_json(config.path, lines=True)
    elif config.format == "csv":
        df = pd.read_csv(config.path)
    elif config.format == "huggingface":
        # Load HuggingFace dataset
        dataset = load_dataset(config.path)
        
        # Convert to pandas DataFrame
        data_list = []
        for split in ['train', 'test']:
            if split in dataset:
                for item in dataset[split]:
                    # Special handling for different datasets
                    if config.name == "bespoke-manim":
                        description = create_description_from_bespoke(item)
                        code = extract_clean_code(item.get(config.code_col, ''))
                    else:
                        description = item.get(config.description_col, '')
                        code = extract_clean_code(item.get(config.code_col, ''))
                    
                    data_list.append({
                        'description': description,
                        'code': code,
                        'split': split
                    })
        
        df = pd.DataFrame(data_list)
    else:
        raise ValueError(f"Unsupported format: {config.format}")
    
    # For non-HuggingFace datasets, standardize column names
    if config.format != "huggingface":
        df = df.rename(columns={
            config.description_col: "description",
            config.code_col: "code"
        })
    
    # Add source tracking
    df["source"] = config.name
    
    # Handle splits for non-HuggingFace datasets
    if config.format != "huggingface":
        if config.split_col and config.split_col in df.columns:
            df["split"] = df[config.split_col]
        else:
            # Default 80/20 split
            df["split"] = "train"
            test_indices = df.sample(frac=0.2, random_state=42).index
            df.loc[test_indices, "split"] = "test"
    
    # Apply custom filter if provided
    if config.filter_func:
        df = config.filter_func(df)
    
    logger.info(f"Loaded {len(df)} samples from '{config.name}'")
    return df

def filter_quality_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter samples based on quality criteria."""
    initial_count = len(df)
    
    # Filter by code length
    df = df[df["code"].str.len() > 50]  # Minimum code length
    df = df[df["code"].str.len() < 5000]  # Maximum code length
    
    # Ensure code has Scene class or can be wrapped
    valid_samples = []
    for idx, row in df.iterrows():
        try:
            code = row["code"]
            # Clean the code first
            code = extract_clean_code(code)
            
            # Check if it has basic Manim structure
            has_manim_import = bool(re.search(r'from\s+manim\s+import|import\s+manim', code))
            has_scene_class = bool(re.search(r'class\s+\w+\s*\([^)]*Scene[^)]*\)', code))
            has_construct = bool(re.search(r'def\s+construct\s*\(self\)', code))
            
            # If missing basic structure but has some animation calls, we can wrap it
            has_animation_calls = bool(re.search(r'self\.(play|wait|add|remove|render)\(', code))
            
            if has_manim_import or has_scene_class or has_animation_calls:
                # Try to compile the code
                formatted_code = ensure_proper_code_format(code)
                ast.parse(formatted_code)  # Use AST parsing instead of compile
                valid_samples.append(idx)
        except:
            continue
    
    df = df.loc[valid_samples]
    
    logger.info(f"Quality filter: {initial_count} -> {len(df)} samples")
    return df

def deduplicate_samples(df: pd.DataFrame, similarity_threshold: float = 0.85) -> pd.DataFrame:
    """Remove duplicate or near-duplicate code samples."""
    initial_count = len(df)
    
    # Create code hashes for exact duplicates
    df["code_hash"] = df["code"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    
    # Remove exact duplicates, keeping first occurrence
    df = df.drop_duplicates(subset=["code_hash"], keep="first")
    
    # Near-duplicate detection with normalized code
    def normalize_code(code):
        # Remove comments and normalize whitespace
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\s+', ' ', code)
        return code.strip()
    
    df["norm_code"] = df["code"].apply(normalize_code)
    df["norm_hash"] = df["norm_code"].apply(lambda x: hashlib.md5(x[:500].encode()).hexdigest())
    
    # Group by normalized hash prefix and check similarity within groups
    unique_indices = []
    seen_codes = {}
    
    for idx, row in df.iterrows():
        norm_hash = row["norm_hash"]
        norm_code = row["norm_code"]
        
        is_duplicate = False
        if norm_hash in seen_codes:
            # Check similarity with codes in the same hash group
            for seen_idx, seen_code in seen_codes[norm_hash]:
                if SequenceMatcher(None, norm_code, seen_code).ratio() > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_indices.append(idx)
            if norm_hash not in seen_codes:
                seen_codes[norm_hash] = []
            seen_codes[norm_hash].append((idx, norm_code))
    
    df = df.loc[unique_indices]
    df = df.drop(columns=["code_hash", "norm_code", "norm_hash"])
    
    logger.info(f"Deduplication: {initial_count} -> {len(df)} samples")
    return df

def balance_datasets(df: pd.DataFrame, max_ratio: float = 3.0) -> pd.DataFrame:
    """Balance dataset sources to prevent one from dominating."""
    source_counts = df["source"].value_counts()
    logger.info(f"Original source distribution:\n{source_counts}")
    
    if len(source_counts) == 1:
        return df
    
    # Find minimum count and maximum allowed count
    min_count = source_counts.min()
    max_count = int(min_count * max_ratio)
    
    balanced_dfs = []
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        if len(source_df) > max_count:
            # Sample down to max_count
            source_df = source_df.sample(n=max_count, random_state=42)
        balanced_dfs.append(source_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    logger.info(f"Balanced source distribution:\n{balanced_df['source'].value_counts()}")
    return balanced_df

def create_conversation(description: str, code: str, source: str, prompt_variation_idx: int = 0) -> dict:
    """Create a conversation with system prompt and structured output."""
    # Format user prompt based on variation index
    if prompt_variation_idx == 0:
        user_prompt = description  # Original format
    else:
        template = PROMPT_TEMPLATES[prompt_variation_idx % len(PROMPT_TEMPLATES)]
        user_prompt = template.format(
            description=description,
            description_lower=description.lower()
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
        ],
        "metadata": {
            "source": source,
            "original_description": description
        }
    }

def prepare_multi_dataset(
    configs: List[DatasetConfig], 
    output_dir: str, 
    use_variations: bool = True,
    balance_sources: bool = True,
    quality_filter: bool = True
):
    """Load and prepare multiple datasets."""
    # Load all datasets
    all_dfs = []
    for config in configs:
        try:
            df = load_dataset_from_config(config)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load dataset '{config.name}': {e}")
            continue
    
    if not all_dfs:
        raise ValueError("No datasets loaded successfully")
    
    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset size: {len(combined_df)}")
    
    # Apply quality filters
    if quality_filter:
        combined_df = filter_quality_samples(combined_df)
        combined_df = deduplicate_samples(combined_df)
    
    # Balance dataset sources
    if balance_sources and len(configs) > 1:
        combined_df = balance_datasets(combined_df)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data
    train_data = []
    test_data = []
    
    for idx, row in combined_df.iterrows():
        description = row['description']
        code = row['code']
        split = row['split']
        source = row['source']
        
        if split == 'train' and use_variations:
            # Create 2 variations for each training sample
            for i in range(2):
                conversation = create_conversation(description, code, source, prompt_variation_idx=i)
                train_data.append(conversation)
        elif split == 'train':
            # No variations, just original
            conversation = create_conversation(description, code, source, prompt_variation_idx=0)
            train_data.append(conversation)
        else:
            # Test data - always use original format
            conversation = create_conversation(description, code, source, prompt_variation_idx=0)
            test_data.append(conversation)
    
    # Save datasets
    for name, data in [("train", train_data), ("test", test_data)]:
        file_path = output_dir / f"{name}.json"
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(data)} {name} samples to {file_path}")
    
    # Save metadata
    metadata = {
        "datasets": [config.name for config in configs],
        "total_samples": len(combined_df),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "augmentation_factor": len(train_data) / len(combined_df[combined_df['split'] == 'train']) if use_variations else 1.0,
        "source_distribution": combined_df['source'].value_counts().to_dict()
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Show statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total samples: {len(combined_df)}")
    logger.info(f"Train samples: {len(train_data)} ({metadata['augmentation_factor']:.1f}x augmentation)")
    logger.info(f"Test samples: {len(test_data)}")
    logger.info(f"Source distribution: {metadata['source_distribution']}")

def main():
    OUTPUT_DIR = "data_formatted_multi"
    USE_VARIATIONS = True
    BALANCE_SOURCES = True
    QUALITY_FILTER = True
    
    random.seed(42)
    prepare_multi_dataset(
        DATASET_CONFIGS, 
        OUTPUT_DIR, 
        USE_VARIATIONS,
        BALANCE_SOURCES,
        QUALITY_FILTER
    )

if __name__ == "__main__":
    main()