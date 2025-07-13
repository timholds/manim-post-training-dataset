"""Utility functions for extractors."""

import re
from typing import Dict, Any, List
import hashlib


def fix_missing_imports(code: str) -> str:
    """Add 'from manim import *' if no import statements found."""
    code = code.strip()
    
    # Check if any import statement exists
    if "import" not in code:
        # Add the standard manim import
        code = "from manim import *\n\n" + code
    
    return code


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


def create_conversation(description: str, code: str, system_prompt: str = None) -> Dict[str, Any]:
    """Create a conversation with system prompt and structured output."""
    if system_prompt is None:
        system_prompt = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
    
    # Ensure code is properly formatted
    formatted_code = ensure_proper_code_format(code)
    
    # Wrap code in markdown blocks
    assistant_response = f"```python\n{formatted_code}\n```"
    
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "user", "value": description},
            {"from": "assistant", "value": assistant_response}
        ]
    }


def normalize_description(desc: str) -> str:
    """Normalize description for comparison."""
    # Lowercase and normalize whitespace
    return ' '.join(desc.lower().split())


def get_content_hash(content: str) -> str:
    """Get a hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Prompt variations for augmentation
PROMPT_TEMPLATES = [
    "{description}",  # Original format
    "Create a Manim animation that {description_lower}",
    "Write Manim code to {description_lower}",
    "Generate a Manim scene that {description_lower}",
    "Implement a Manim animation for: {description}",
    "Using Manim, {description_lower}",
]


def augment_prompt(description: str, variation_idx: int = 0) -> str:
    """Create prompt variations for data augmentation."""
    template = PROMPT_TEMPLATES[variation_idx % len(PROMPT_TEMPLATES)]
    
    # Handle case for description_lower
    description_lower = description
    if description and description[0].isupper():
        description_lower = description[0].lower() + description[1:]
    
    return template.format(
        description=description,
        description_lower=description_lower
    )


def normalize_code(code: str) -> str:
    """Normalize code for comparison purposes only.
    
    This removes comments, normalizes whitespace, and standardizes
    formatting to detect duplicates. The original code is always
    preserved in the dataset.
    """
    lines = []
    
    for line in code.split('\n'):
        # Remove comments
        line = line.split('#')[0].rstrip()
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Normalize whitespace (convert tabs to spaces, multiple spaces to single)
        line = ' '.join(line.split())
        
        lines.append(line)
    
    # Join and normalize further
    normalized = '\n'.join(lines)
    
    # Remove multiple newlines
    while '\n\n' in normalized:
        normalized = normalized.replace('\n\n', '\n')
    
    return normalized.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using a hybrid approach.
    
    Returns a value between 0 (completely different) and 1 (identical).
    Uses hashing for exact matches and token-based Jaccard similarity for fuzzy matching.
    """
    if not text1 or not text2:
        return 0.0
    
    # Stage 1: Quick hash check for exact matches
    if hash(text1) == hash(text2):
        return 1.0
    
    # Stage 2: Length-based early termination
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    len_ratio = min(len1, len2) / max(len1, len2)
    if len_ratio < 0.5:  # Too different in size
        return len_ratio * 0.5  # Scale down to indicate low similarity
    
    # Stage 3: Token-based Jaccard similarity
    # Split on whitespace and common delimiters
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    
    if not tokens1 or not tokens2:
        return len_ratio * 0.5
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    
    # Combine length ratio and Jaccard similarity
    # Weight Jaccard more heavily as it's more meaningful for code
    return (jaccard * 0.8 + len_ratio * 0.2)