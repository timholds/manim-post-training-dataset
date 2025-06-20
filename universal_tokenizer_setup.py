#!/usr/bin/env python3
"""
Universal tokenizer setup for popular code generation models.
Handles padding tokens, chat templates, and special configurations.
"""

from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


def setup_tokenizer(model_name: str, trust_remote_code: bool = False):
    """
    Set up tokenizer with proper padding configuration for any model.
    
    Args:
        model_name: HuggingFace model name or path
        trust_remote_code: Whether to trust remote code (required for some models)
    
    Returns:
        Configured tokenizer
    """
    # Map of known models to their specific configurations
    MODEL_CONFIGS = {
        # Qwen family
        "qwen": {
            "pad_token_preference": ["eos_token", "unk_token", "custom"],
            "chat_template": "chatml",  # <|im_start|>, <|im_end|>
            "needs_trust": False,
        },
        # CodeLlama family
        "codellama": {
            "pad_token_preference": ["eos_token", "unk_token", "custom"],
            "chat_template": None,  # No standard template
            "needs_trust": False,
        },
        # DeepSeek family
        "deepseek": {
            "pad_token_preference": ["existing", "eos_token"],  # Has <pad> token
            "chat_template": "custom",  # Uses thinking tokens
            "needs_trust": True,
        },
        # CodeGemma family
        "gemma": {
            "pad_token_preference": ["existing", "eos_token"],  # Has <pad> token
            "chat_template": "gemma",  # <start_of_turn>, <end_of_turn>
            "needs_trust": False,
        },
        # Stable Code family
        "stable": {
            "pad_token_preference": ["eos_token", "custom"],
            "chat_template": None,
            "needs_trust": False,
        },
    }
    
    # Detect model family
    model_family = None
    model_lower = model_name.lower()
    for family in MODEL_CONFIGS:
        if family in model_lower:
            model_family = family
            break
    
    # Load tokenizer with appropriate settings
    config = MODEL_CONFIGS.get(model_family, {})
    if config.get("needs_trust", False) or trust_remote_code:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up padding token based on preference order
    if not tokenizer.pad_token:
        preferences = config.get("pad_token_preference", ["eos_token", "unk_token", "custom"])
        
        for pref in preferences:
            if pref == "existing":
                # Check if model already has a pad token
                if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                    break
            elif pref == "eos_token":
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
                    break
            elif pref == "unk_token":
                if tokenizer.unk_token and tokenizer.unk_token != tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.unk_token
                    logger.info(f"Set pad_token to unk_token: {tokenizer.unk_token}")
                    break
            elif pref == "custom":
                # Add a custom pad token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added custom [PAD] token")
                break
    
    # Always set padding side for causal LM
    tokenizer.padding_side = "right"
    
    # Log final configuration
    logger.info(f"Tokenizer setup complete for {model_name}:")
    logger.info(f"  - Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"  - Vocab size: {len(tokenizer)}")
    
    # Warn if pad and eos are the same
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        logger.warning(
            "Pad token and EOS token are the same. This may affect the model's ability "
            "to generate EOS tokens properly. Consider using DataCollatorForLanguageModeling "
            "with proper attention masking."
        )
    
    return tokenizer


def format_chat_prompt(tokenizer, messages, model_family=None):
    """
    Format messages into the appropriate chat template for the model.
    
    Args:
        tokenizer: The tokenizer instance
        messages: List of message dicts with 'role' and 'content'
        model_family: Optional model family name for custom formatting
    
    Returns:
        Formatted prompt string
    """
    # Try to use the tokenizer's built-in chat template first
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            logger.warning("Failed to apply chat template, using manual formatting")
    
    # Manual formatting based on model family
    if not model_family:
        # Try to detect from tokenizer
        if "<|im_start|>" in tokenizer.vocab:
            model_family = "qwen"
        elif "<start_of_turn>" in tokenizer.vocab:
            model_family = "gemma"
    
    # Format based on detected or specified family
    if model_family == "qwen":
        # ChatML format
        formatted = ""
        for msg in messages:
            formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    elif model_family == "gemma":
        # Gemma format
        formatted = "<bos>"
        for msg in messages:
            role = "model" if msg['role'] == "assistant" else msg['role']
            formatted += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
        formatted += "<start_of_turn>model\n"
        return formatted
    
    else:
        # Generic format
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        formatted += "assistant: "
        return formatted


# Example usage
if __name__ == "__main__":
    # Test with different models
    test_models = [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "codellama/CodeLlama-7b-Instruct-hf",
        "google/codegemma-2b",
    ]
    
    for model in test_models:
        print(f"\n{'='*50}")
        print(f"Testing: {model}")
        print('='*50)
        
        try:
            tokenizer = setup_tokenizer(model)
            
            # Test chat formatting
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a hello world program."}
            ]
            
            prompt = format_chat_prompt(tokenizer, messages)
            print(f"\nFormatted prompt:\n{prompt}")
            
        except Exception as e:
            print(f"Error: {e}")