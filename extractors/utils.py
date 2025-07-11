"""Utility functions for extractors."""

import re
from typing import Dict, Any, List
import hashlib


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