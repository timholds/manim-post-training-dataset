#!/usr/bin/env python3
"""
Enhanced data preparation with key improvements:
1. System prompt for role definition
2. Structured code block outputs
3. Ensures proper Scene class structure in all code
4. 2x augmentation through prompt variations for training data
"""

import json
import pandas as pd
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Single, clear system prompt
SYSTEM_PROMPT = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."

# Simple prompt variations (optional - set USE_VARIATIONS to False to disable)
PROMPT_TEMPLATES = [
    "{description}",  # Original format
    "Create a Manim animation that {description_lower}",
    "Write Manim code to {description_lower}",
    "Generate a Manim scene that {description_lower}",
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

def create_conversation(description: str, code: str, prompt_variation_idx: int = 0) -> dict:
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
        ]
    }

def prepare_dataset(parquet_path: str, output_dir: str, use_variations: bool = True):
    """Load and prepare the dataset with essential improvements."""
    # Load data
    logger.info(f"Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data
    train_data = []
    test_data = []
    
    for idx, row in df.iterrows():
        description = row['Reviewed Description']
        code = row['Code']
        split = row['Split']
        
        if split == 'train' and use_variations:
            # Create 2 variations for each training sample
            for i in range(2):
                conversation = create_conversation(description, code, prompt_variation_idx=i)
                train_data.append(conversation)
        elif split == 'train':
            # No variations, just original
            conversation = create_conversation(description, code, prompt_variation_idx=0)
            train_data.append(conversation)
        else:
            # Test data - always use original format
            conversation = create_conversation(description, code, prompt_variation_idx=0)
            test_data.append(conversation)
    
    # Save datasets
    for name, data in [("train", train_data), ("test", test_data)]:
        file_path = output_dir / f"{name}.json"
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(data)} {name} samples to {file_path}")
    
    # Show statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Original samples: {len(df)}")
    orig_train = len(df[df['Split'] == 'train'])
    logger.info(f"Train samples: {len(train_data)} ({len(train_data)/orig_train:.1f}x augmentation)")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Show sample
    logger.info("\nSample conversation:")
    sample = train_data[0]
    logger.info(f"System: {sample['conversations'][0]['value']}")
    logger.info(f"User: {sample['conversations'][1]['value']}")
    logger.info(f"Assistant: {sample['conversations'][2]['value'][:100]}...")

def main():
    DATASET_PATH = "/home/timholds/.cache/kagglehub/datasets/ravidussilva/manim-sft/versions/2/manim_sft_dataset.parquet"
    OUTPUT_DIR = "data_formatted"
    USE_VARIATIONS = True  # Set to False for no augmentation
    
    random.seed(42)
    prepare_dataset(DATASET_PATH, OUTPUT_DIR, USE_VARIATIONS)

if __name__ == "__main__":
    main()