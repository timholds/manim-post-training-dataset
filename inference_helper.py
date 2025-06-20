#!/usr/bin/env python3
"""
Helper script for inference with the fine-tuned Manim code generator.
Shows how to properly format prompts for the model.
"""

import torch
from pathlib import Path
from unsloth import FastLanguageModel
from manim_code_extractor import ManimCodeExtractor

# Constants
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."

def load_model(model_path="models/lora_model"):
    """Load the fine-tuned model."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_manim_code(prompt, model, tokenizer, temperature=0.7, max_tokens=512, extract=True):
    """Generate Manim code from a prompt."""
    # Format the prompt
    formatted_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).input_ids.to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Use ManimCodeExtractor for proper extraction
    if extract:
        extractor = ManimCodeExtractor()
        return extractor.extract(full_response)
    else:
        # Return raw response for debugging
        return full_response

def extract_code(response):
    """Extract Python code from the response using ManimCodeExtractor."""
    extractor = ManimCodeExtractor()
    return extractor.extract(response)

def main():
    """Example usage."""
    # Example prompts
    prompts = [
        "Create a simple animation showing a red circle moving from left to right",
        "Generate a Manim scene that displays the text 'Hello Manim!' with a fade-in effect",
        "Write code to animate a square rotating 360 degrees while changing color from blue to green",
    ]
    
    # Load model
    model, tokenizer = load_model()
    
    print("\n=== MANIM CODE GENERATOR ===\n")
    
    for prompt in prompts:
        print(f"PROMPT: {prompt}")
        print("-" * 80)
        
        # Generate code (already extracted by default)
        code = generate_manim_code(prompt, model, tokenizer)
        print(f"GENERATED CODE:\n{code}\n")
        
        # Optionally validate the code
        extractor = ManimCodeExtractor()
        validation = extractor.validate(code)
        if not validation.is_valid:
            print(f"VALIDATION ERRORS: {validation.errors}")
        
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()