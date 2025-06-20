#!/usr/bin/env python3
"""
Universal fine-tuning script for various code generation models on ManimBench dataset.
Supports multiple model families with automatic tokenizer configuration.
"""

import torch
import json
import logging
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import wandb
import os
import sys
from datetime import datetime
import gc
import subprocess
import atexit
import time
from typing import Dict, Any, Tuple, Optional

# Import our universal tokenizer setup
from universal_tokenizer_setup import setup_tokenizer, format_chat_prompt

from logging.handlers import RotatingFileHandler

# Configure logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler with rotation (10MB max, keep 3 backups)
Path("logs").mkdir(exist_ok=True)
file_handler = RotatingFileHandler(
    "logs/training.log", maxBytes=10*1024*1024, backupCount=3
)
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Model configurations mapping
MODEL_CONFIGS = {
    # QWEN family
    "qwen": {
        "hf_names": ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
        "unsloth_names": ["unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": "chatml",
        "trust_remote_code": False,
    },
    # CodeLlama family
    "codellama": {
        "hf_names": ["codellama/CodeLlama-7b-Instruct-hf", "codellama/CodeLlama-13b-Instruct-hf"],
        "unsloth_names": ["unsloth/codellama-7b-instruct-bnb-4bit", "unsloth/codellama-13b-instruct-bnb-4bit"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": None,
        "trust_remote_code": False,
    },
    # DeepSeek family
    "deepseek": {
        "hf_names": ["deepseek-ai/deepseek-coder-1.3b-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"],
        "unsloth_names": [],  # No unsloth versions available
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": "deepseek",
        "trust_remote_code": True,
    },
    # CodeGemma family
    "gemma": {
        "hf_names": ["google/codegemma-2b", "google/codegemma-7b-it"],
        "unsloth_names": ["unsloth/codegemma-2b-bnb-4bit", "unsloth/codegemma-7b-it-bnb-4bit"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": "gemma",
        "trust_remote_code": False,
    },
    # Stable Code family
    "stable": {
        "hf_names": ["stabilityai/stable-code-3b"],
        "unsloth_names": [],  # No unsloth versions available
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": None,
        "trust_remote_code": False,
    },
}

# Default configuration
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Training configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
SAVE_STEPS = 50
LOGGING_STEPS = 10

# Wandb configuration
WANDB_PROJECT = "manim-post-train"
WANDB_ENTITY = None
USE_WANDB = True

# GPU optimization defaults
FP16 = True
GRADIENT_CHECKPOINTING = True
OPTIM = "adamw_8bit"
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def detect_model_family(model_name: str) -> Tuple[str, Dict[str, Any]]:
    """Detect model family from model name and return configuration."""
    model_lower = model_name.lower()
    
    for family, config in MODEL_CONFIGS.items():
        if family in model_lower:
            return family, config
    
    # Default configuration for unknown models
    logger.warning(f"Unknown model family for {model_name}, using default configuration")
    return "unknown", {
        "hf_names": [],
        "unsloth_names": [],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": None,
        "trust_remote_code": False,
    }


def get_unsloth_model_name(model_name: str, model_family: str, config: Dict[str, Any]) -> Optional[str]:
    """Try to find corresponding Unsloth quantized model."""
    # Check if exact unsloth model exists
    if "unsloth/" in model_name:
        return model_name
    
    # Try to map to unsloth version
    for hf_name, unsloth_name in zip(config.get("hf_names", []), config.get("unsloth_names", [])):
        if hf_name in model_name or model_name in hf_name:
            logger.info(f"Found Unsloth version: {unsloth_name}")
            return unsloth_name
    
    # No unsloth version found
    return None


class GPUManager:
    """Manages GPU resources by stopping/starting Ollama service."""
    
    def __init__(self):
        self.ollama_was_running = False
        self.ollama_stopped = False
        
    def check_ollama_running(self):
        """Check if Ollama service is running."""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', 'ollama'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() == 'active'
        except:
            return False
    
    def stop_ollama(self):
        """Stop Ollama service to free GPU memory."""
        if self.check_ollama_running():
            self.ollama_was_running = True
            logger.info("Stopping Ollama service to free GPU memory...")
            try:
                subprocess.run(['sudo', 'systemctl', 'stop', 'ollama'], check=True)
                self.ollama_stopped = True
                time.sleep(2)  # Give it time to release GPU
                logger.info("✓ Ollama service stopped")
            except subprocess.CalledProcessError:
                logger.warning("Failed to stop Ollama service. You may need to run: sudo systemctl stop ollama")
        else:
            logger.info("Ollama service is not running")
    
    def start_ollama(self):
        """Restart Ollama service if it was running before."""
        if self.ollama_was_running and self.ollama_stopped:
            logger.info("Restarting Ollama service...")
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'ollama'], check=True)
                logger.info("✓ Ollama service restarted")
            except subprocess.CalledProcessError:
                logger.warning("Failed to restart Ollama. You may need to run: sudo systemctl start ollama")
    
    def get_gpu_memory_info(self):
        """Get current GPU memory usage."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                return used, total
            return None, None
        except:
            return None, None
    
    def print_gpu_status(self):
        """Print current GPU memory status."""
        used, total = self.get_gpu_memory_info()
        if used and total:
            logger.info(f"GPU Memory: {used}MB / {total}MB ({used/total*100:.1f}% used)")


# Initialize GPU manager
gpu_manager = GPUManager()


def load_dataset(file_path, tokenizer, model_family, chat_template):
    """Load dataset from JSON lines file and format for specific model."""
    logger.info(f"Loading dataset from {file_path}")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to format expected by trainer
    formatted_data = []
    for item in data:
        # Extract components
        system_prompt = item['conversations'][0]['value']  # System
        user_instruction = item['conversations'][1]['value']  # User
        assistant_response = item['conversations'][2]['value']  # Assistant
        
        # Format messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction},
        ]
        
        # Get formatted prompt using model-specific template
        try:
            # Try tokenizer's chat template first
            if hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = format_chat_prompt(tokenizer, messages, model_family)
        except:
            # Fallback to manual formatting
            prompt = format_chat_prompt(tokenizer, messages, model_family)
        
        # Add the assistant response
        text = prompt + assistant_response
        
        # Add appropriate ending token based on model
        if model_family == "qwen":
            text += "<|im_end|>"
        elif model_family == "gemma":
            text += "<end_of_turn>"
        elif tokenizer.eos_token:
            text += tokenizer.eos_token
        
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Loaded {len(dataset)} samples with {model_family} formatting")
    return dataset


def setup_model_and_tokenizer(model_name: str, model_family: str, config: Dict[str, Any]):
    """Initialize model and tokenizer with Unsloth optimizations."""
    logger.info(f"Loading model: {model_name}")
    
    # Try to use Unsloth version if available
    unsloth_model = get_unsloth_model_name(model_name, model_family, config)
    use_unsloth = unsloth_model is not None
    
    if use_unsloth:
        logger.info(f"Using Unsloth optimized model: {unsloth_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=unsloth_model,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    else:
        logger.info(f"Loading standard model (no Unsloth version available)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=config.get("trust_remote_code", False),
        )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth" if use_unsloth else True,
        random_state=42,
    )
    
    # Set up tokenizer using our universal setup
    tokenizer = setup_tokenizer(model_name, trust_remote_code=config.get("trust_remote_code", False))
    
    logger.info("Model and tokenizer initialized successfully")
    return model, tokenizer


def create_training_arguments(output_dir):
    """Create training arguments optimized for 16GB GPU."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,  # Use FP16 for training
        bf16=False,  # A4500 doesn't support BF16
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,  # Evaluate at same frequency as saving
        optim=OPTIM,  # 8-bit optimizer to save memory
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=42,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping
        report_to="wandb" if USE_WANDB else "none",
        push_to_hub=False,
        logging_first_step=True,
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=False,
    )


class MemoryCallback(TrainerCallback):
    """Callback to clear CUDA cache periodically."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:  # Every 50 steps
            gc.collect()
            torch.cuda.empty_cache()


def train_model(model_name: str):
    """Main training function."""
    # Detect model family
    model_family, model_config = detect_model_family(model_name)
    
    # Setup paths
    train_data_path = Path(getattr(sys.modules[__name__], 'TRAIN_PATH', "data_formatted/train.json"))
    test_data_path = Path(getattr(sys.modules[__name__], 'EVAL_PATH', "data_formatted/test.json"))
    
    # Create model-specific output directory
    model_safe_name = model_name.replace("/", "_").replace(":", "_")
    output_dir = Path(getattr(sys.modules[__name__], 'OUTPUT_DIR', f"models/{model_safe_name}_lora"))
    
    # Check if training data exists
    if not train_data_path.exists() or not test_data_path.exists():
        logger.error(f"Dataset not found. Run prepare_data.py first.")
        return
    
    # Initialize wandb
    if USE_WANDB:
        run_name = f"{model_family}-manim-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "model_name": model_name,
                "model_family": model_family,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "max_seq_length": MAX_SEQ_LENGTH,
                "warmup_ratio": WARMUP_RATIO,
                "optimizer": "adamw_8bit",
                "scheduler": "cosine",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            },
            tags=["manim", "code-generation", model_family, "unsloth"]
        )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, model_family, model_config)
    
    # Load datasets with model-specific formatting
    train_dataset = load_dataset(train_data_path, tokenizer, model_family, model_config.get("chat_template"))
    eval_dataset = load_dataset(test_data_path, tokenizer, model_family, model_config.get("chat_template"))
    
    # Print model info
    logger.info(f"Model has {model.num_parameters():,} parameters")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/model.num_parameters()*100:.2f}%)")
    
    # Log model info to wandb
    if USE_WANDB:
        wandb.log({
            "model/total_parameters": model.num_parameters(),
            "model/trainable_parameters": trainable_params,
            "model/trainable_percentage": trainable_params/model.num_parameters()*100,
            "dataset/train_size": len(train_dataset),
            "dataset/eval_size": len(eval_dataset),
        })
    
    # Create training arguments
    training_args = create_training_arguments(str(output_dir))
    
    # Create data collator that properly handles attention masks
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,  # Helps with GPU efficiency
    )
    
    # Create trainer with Unsloth optimizations
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Disable packing for stability
        args=training_args,
        data_collator=data_collator,  # Use our custom data collator
        callbacks=[MemoryCallback()],
    )
    
    # Start training (resume from checkpoint if exists)
    checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
    resume_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None
    
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        logger.info("Starting training from scratch...")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    logger.info("Saving LoRA adapters...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save merged model for easier deployment
    merged_dir = Path(f"models/{model_safe_name}_merged")
    logger.info(f"Saving merged 16-bit model to {merged_dir}...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    
    logger.info("Training completed successfully!")
    
    # Print final statistics
    if trainer.state.log_history:
        final_loss = trainer.state.log_history[-1].get('loss', 'N/A')
        logger.info(f"Final training loss: {final_loss}")
        
        # Log final metrics to wandb
        if USE_WANDB:
            # Calculate training metrics
            losses = [log.get('loss', 0) for log in trainer.state.log_history if 'loss' in log]
            learning_rates = [log.get('learning_rate', 0) for log in trainer.state.log_history if 'learning_rate' in log]
            
            wandb.log({
                "final/loss": final_loss if final_loss != 'N/A' else None,
                "final/perplexity": float(torch.exp(torch.tensor(final_loss))) if final_loss != 'N/A' else None,
                "final/total_steps": trainer.state.global_step,
                "final/epoch": trainer.state.epoch,
                "metrics/avg_loss": sum(losses) / len(losses) if losses else None,
                "metrics/min_loss": min(losses) if losses else None,
                "metrics/final_learning_rate": learning_rates[-1] if learning_rates else None,
            })
    
    # Close wandb run
    if USE_WANDB:
        wandb.finish()
    
    return str(output_dir), model_family


def test_generation(model_path: str, model_family: str):
    """Test the fine-tuned model with a sample prompt."""
    logger.info("\nTesting model generation...")
    
    # Load the saved model
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error("Model not found. Train the model first.")
        return
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Test prompt
    system_prompt = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
    test_prompt = "Create a Manim animation that shows a circle transforming into a square"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": test_prompt},
    ]
    
    # Format prompt based on model family
    formatted_prompt = format_chat_prompt(tokenizer, messages, model_family)
    
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
            max_new_tokens=512,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"\nPrompt: {test_prompt}")
    logger.info(f"\nGenerated response:\n{response}")


def load_models_from_file(file_path="models.txt"):
    """Load model names from models.txt file."""
    models = []
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    models.append(line)
    return models


def get_full_model_name(model_shorthand: str) -> str:
    """Convert model shorthand to full HuggingFace model name."""
    # Mapping of common shorthands to full names
    MODEL_SHORTCUTS = {
        # QWEN
        "qwen-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        
        # CodeLlama
        "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
        "codellama:7b": "codellama/CodeLlama-7b-Instruct-hf",
        "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
        
        # CodeGemma
        "codegemma-2b": "google/codegemma-2b",
        "codegemma-7b": "google/codegemma-7b-it",
        "codegemma:7b": "google/codegemma-7b-it",
        
        # DeepSeek
        "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "deepseek-coder-v2:7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
        
        # Stable Code
        "stable-code-3b": "stabilityai/stable-code-3b",
        "stable-code:3b": "stabilityai/stable-code-3b",
    }
    
    # Return full name if it's a shortcut, otherwise return as-is
    return MODEL_SHORTCUTS.get(model_shorthand.lower(), model_shorthand)


if __name__ == "__main__":
    import argparse
    
    # Load available models from file
    available_models = load_models_from_file()
    
    parser = argparse.ArgumentParser(
        description="Fine-tune various models on Manim dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use full model name
  python fine_tune.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"
  
  # Use shorthand
  python fine_tune.py --model qwen-7b
  
  # Use model from models.txt (if it contains 'qwen2.5-coder:7b')
  python fine_tune.py --model qwen2.5-coder:7b
  
  # List supported models
  python fine_tune.py --list-models
        """
    )
    
    parser.add_argument("--model", type=str, help="Model name or shorthand (e.g., 'qwen-7b', 'codellama-7b')")
    parser.add_argument("--list-models", action="store_true", help="List all supported models and exit")
    parser.add_argument("--train-data", type=str, default="data_formatted/train.json", help="Path to training data")
    parser.add_argument("--eval-data", type=str, default="data_formatted/test.json", help="Path to evaluation data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--skip-test", action="store_true", help="Skip test generation after training")
    parser.add_argument("--keep-ollama", action="store_true", help="Don't stop Ollama service during training")
    args = parser.parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("\nSupported Models:")
        print("================")
        print("\nQWEN Family:")
        print("  - qwen-1.5b, qwen-7b")
        print("  - Full: Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct")
        print("\nCodeLlama Family:")
        print("  - codellama-7b, codellama-13b")
        print("  - Full: codellama/CodeLlama-7b-Instruct-hf")
        print("\nCodeGemma Family:")
        print("  - codegemma-2b, codegemma-7b")
        print("  - Full: google/codegemma-7b-it")
        print("\nDeepSeek Family:")
        print("  - deepseek-1.3b, deepseek-6.7b")
        print("  - Full: deepseek-ai/deepseek-coder-6.7b-instruct")
        print("\nStable Code Family:")
        print("  - stable-code-3b")
        print("  - Full: stabilityai/stable-code-3b")
        
        if available_models:
            print(f"\nModels in models.txt:")
            for model in available_models:
                full_name = get_full_model_name(model)
                if full_name != model:
                    print(f"  - {model} -> {full_name}")
                else:
                    print(f"  - {model}")
        exit(0)
    
    # Require model argument if not listing
    if not args.model:
        parser.error("--model is required unless using --list-models")
    
    # Convert shorthand to full model name
    model_name = get_full_model_name(args.model)
    if model_name != args.model:
        logger.info(f"Using model: {model_name} (from shorthand: {args.model})")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        exit(1)
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check GPU memory status
    gpu_manager.print_gpu_status()
    
    # Stop Ollama if needed (unless --keep-ollama flag is set)
    if not args.keep_ollama:
        gpu_manager.stop_ollama()
        gpu_manager.print_gpu_status()
    
    # Register cleanup function
    def cleanup():
        """Cleanup function to restart Ollama on exit."""
        if not args.keep_ollama:
            gpu_manager.start_ollama()
    
    atexit.register(cleanup)
    
    # Override paths with args
    sys.modules[__name__].TRAIN_PATH = args.train_data
    sys.modules[__name__].EVAL_PATH = args.eval_data
    if args.output_dir:
        sys.modules[__name__].OUTPUT_DIR = args.output_dir
    
    try:
        # Run training with the full model name
        output_path, model_family = train_model(model_name)
        
        # Test generation
        if not args.skip_test:
            test_generation(output_path, model_family)
    finally:
        # Ensure cleanup happens even on error
        cleanup()