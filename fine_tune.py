#!/usr/bin/env python3
"""
Completely isolated version that delays ALL GPU-related imports.
This version should work even when GPU is corrupted.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Only safe imports at module level
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

# Configuration constants (no imports needed)
MODEL_CONFIGS = {
    "qwen": {
        "hf_names": ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
        "unsloth_names": ["unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "chat_template": "chatml",
        "trust_remote_code": False,
    },
    "codellama": {
        "hf_names": ["codellama/CodeLlama-7b-Instruct-hf", "codellama/CodeLlama-13b-Instruct-hf"],
        "unsloth_names": ["unsloth/codellama-7b-instruct-bnb-4bit", "unsloth/codellama-13b-instruct-bnb-4bit"],
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


def get_full_model_name(model_shorthand: str) -> str:
    """Convert model shorthand to full HuggingFace model name."""
    MODEL_SHORTCUTS = {
        "qwen-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
        "codellama:7b": "codellama/CodeLlama-7b-Instruct-hf",
        "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
    }
    return MODEL_SHORTCUTS.get(model_shorthand.lower(), model_shorthand)


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


def check_gpu_health():
    """Check if GPU is in a healthy state without importing torch."""
    import subprocess
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "nvidia-smi failed"
        
        # Check for ERR! in output
        if "ERR!" in result.stdout:
            return False, "GPU in error state (ERR! detected)"
        
        # Check for "Unknown Error"
        if "Unknown Error" in result.stdout:
            return False, "GPU in unknown error state"
        
        return True, "GPU healthy"
    except Exception as e:
        return False, str(e)


def reset_gpu():
    """Attempt to reset GPU."""
    import subprocess
    
    logger.info("Attempting GPU reset...")
    
    # Try the reset script if it exists
    if os.path.exists("./reset_gpu.sh"):
        result = subprocess.run(['sudo', './reset_gpu.sh'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("GPU reset completed")
            time.sleep(3)
            return True
    
    # Fallback: try PCI reset directly
    try:
        pci_path = "/sys/bus/pci/devices/0000:01:00.0/reset"
        if os.path.exists(pci_path):
            subprocess.run(['sudo', 'bash', '-c', f'echo 1 > {pci_path}'], check=True)
            time.sleep(3)
            return True
    except:
        pass
    
    return False


def safe_train_model(model_name: str):
    """Import all GPU libraries and train only after GPU is verified healthy."""
    logger.info("GPU is healthy, importing torch and unsloth...")
    
    # Import torch first
    import torch
    
    # Verify CUDA after torch import
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available to PyTorch after import")
    
    # Import GPU-heavy libraries
    import gc
    import subprocess
    import atexit
    import pandas as pd
    import wandb
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
    from datasets import Dataset
    from universal_tokenizer_setup import setup_tokenizer, format_chat_prompt
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
        return None, None
    
    def cleanup_gpu_resources():
        """Properly clean up GPU resources to prevent corruption."""
        logger.info("Cleaning up GPU resources...")
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(1)
            logger.info("✓ GPU cleanup completed")
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")
    
    def get_unsloth_model_name(model_name: str, model_family: str, config: Dict[str, Any]) -> Optional[str]:
        """Try to find corresponding Unsloth quantized model."""
        if "unsloth/" in model_name:
            return model_name
        
        for hf_name, unsloth_name in zip(config.get("hf_names", []), config.get("unsloth_names", [])):
            if hf_name in model_name or model_name in hf_name:
                logger.info(f"Found Unsloth version: {unsloth_name}")
                return unsloth_name
        
        return None
    
    def load_dataset(file_path, tokenizer, model_family, chat_template):
        """Load dataset from JSON lines file and format for specific model."""
        logger.info(f"Loading dataset from {file_path}")
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        formatted_data = []
        for item in data:
            system_prompt = item['conversations'][0]['value']
            user_instruction = item['conversations'][1]['value']
            assistant_response = item['conversations'][2]['value']
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction},
            ]
            
            try:
                if hasattr(tokenizer, 'apply_chat_template'):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = format_chat_prompt(tokenizer, messages, model_family)
            except:
                prompt = format_chat_prompt(tokenizer, messages, model_family)
            
            text = prompt + assistant_response
            
            if model_family == "qwen":
                text += "<|im_end|>"
            elif tokenizer.eos_token:
                text += tokenizer.eos_token
            
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Loaded {len(dataset)} samples with {model_family} formatting")
        return dataset
    
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
                "optimizer": OPTIM,
                "scheduler": LR_SCHEDULER_TYPE,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            },
            tags=["manim", "code-generation", model_family, "unsloth"]
        )
    
    try:
        # Setup model and tokenizer
        logger.info(f"Loading model: {model_name}")
        
        unsloth_model = get_unsloth_model_name(model_name, model_family, model_config)
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
                trust_remote_code=model_config.get("trust_remote_code", False),
            )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=model_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth" if use_unsloth else True,
            random_state=42,
        )
        
        # Set up tokenizer
        tokenizer = setup_tokenizer(model_name, trust_remote_code=model_config.get("trust_remote_code", False))
        
        logger.info("Model and tokenizer initialized successfully")
        
        # Load datasets
        train_dataset = load_dataset(train_data_path, tokenizer, model_family, model_config.get("chat_template"))
        eval_dataset = load_dataset(test_data_path, tokenizer, model_family, model_config.get("chat_template"))
        
        # Print model info
        logger.info(f"Model has {model.num_parameters():,} parameters")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/model.num_parameters()*100:.2f}%)")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=FP16,
            bf16=False,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=SAVE_STEPS,
            optim=OPTIM,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=42,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=MAX_GRAD_NORM,
            report_to="wandb" if USE_WANDB else "none",
            push_to_hub=False,
            logging_first_step=True,
            save_total_limit=3,
            load_best_model_at_end=False,
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        class MemoryCallback(TrainerCallback):
            """Callback to clear CUDA cache periodically."""
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            data_collator=data_collator,
            callbacks=[MemoryCallback()],
        )
        
        # Start training
        checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
        resume_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None
        
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            logger.info("Starting training from scratch...")
        
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Save final model - ONLY LoRA adapters
        logger.info("Saving LoRA adapters...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        logger.info("Training completed successfully!")
        
        # Clean up
        del model
        del trainer
        del tokenizer
        gc.collect()
        
        return str(output_dir), model_family
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        cleanup_gpu_resources()
        if USE_WANDB:
            wandb.finish()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Isolated fine-tuning script that handles GPU corruption")
    parser.add_argument("--model", type=str, required=True, help="Model name or shorthand")
    parser.add_argument("--train-data", type=str, default="data_formatted/train.json", help="Path to training data")
    parser.add_argument("--eval-data", type=str, default="data_formatted/test.json", help="Path to evaluation data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-test", action="store_true", help="Skip test generation after training")
    parser.add_argument("--force-reset", action="store_true", help="Force GPU reset before training")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum GPU reset retries")
    
    args = parser.parse_args()
    
    # Convert shorthand to full model name
    model_name = get_full_model_name(args.model)
    if model_name != args.model:
        logger.info(f"Using model: {model_name} (from shorthand: {args.model})")
    
    # Force reset if requested
    if args.force_reset:
        reset_gpu()
    
    # Check GPU health with retries
    retry_count = 0
    while retry_count < args.max_retries:
        healthy, message = check_gpu_health()
        
        if healthy:
            logger.info(f"✓ GPU health check passed: {message}")
            break
        else:
            logger.error(f"✗ GPU health check failed: {message}")
            
            if retry_count < args.max_retries - 1:
                logger.info(f"Attempting GPU reset (attempt {retry_count + 1}/{args.max_retries})...")
                if reset_gpu():
                    retry_count += 1
                    continue
                else:
                    logger.error("GPU reset failed")
            
            logger.error("\nGPU is not healthy. Please try:")
            logger.error("1. Run: sudo ./reset_gpu.sh")
            logger.error("2. Reboot the system")
            sys.exit(1)
    
    # Override paths with args
    sys.modules[__name__].TRAIN_PATH = args.train_data
    sys.modules[__name__].EVAL_PATH = args.eval_data
    if args.output_dir:
        sys.modules[__name__].OUTPUT_DIR = args.output_dir
    
    # GPU is healthy, proceed with training
    logger.info("\nStarting training with healthy GPU...")
    
    try:
        output_path, model_family = safe_train_model(model_name)
        logger.info("✓ Training completed successfully!")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        # Check GPU state after failure
        healthy, message = check_gpu_health()
        if not healthy:
            logger.error(f"GPU is now unhealthy: {message}")
            logger.error("Run: sudo ./reset_gpu.sh")
        sys.exit(1)


if __name__ == "__main__":
    main()