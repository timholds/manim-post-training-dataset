#!/usr/bin/env python3
"""
Safe wrapper for fine-tuning that handles GPU issues gracefully.
"""

import subprocess
import sys
import time
import os
import torch

def check_gpu_health():
    """Check if GPU is in a healthy state."""
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "nvidia-smi failed"
        
        # Check for ERR! in output
        if "ERR!" in result.stdout:
            return False, "GPU in error state (ERR! detected)"
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            return False, "CUDA not available to PyTorch"
        
        return True, "GPU healthy"
    except Exception as e:
        return False, str(e)

def reset_gpu():
    """Attempt to reset GPU."""
    print("Attempting GPU reset...")
    
    # Try the reset script if it exists
    if os.path.exists("./reset_gpu.sh"):
        result = subprocess.run(['sudo', './reset_gpu.sh'], capture_output=True, text=True)
        if result.returncode == 0:
            print("GPU reset completed")
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

def safe_import_and_train(model_name, args):
    """Import Unsloth and run training only after GPU is verified."""
    # Delay import until GPU is verified
    from unsloth import FastLanguageModel
    
    # Import the training function
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fine_tune import train_model, test_generation, get_full_model_name
    
    # Convert shorthand to full model name
    full_model_name = get_full_model_name(model_name)
    if full_model_name != model_name:
        print(f"Using model: {full_model_name} (from shorthand: {model_name})")
    
    # Override paths with args
    import fine_tune
    fine_tune.TRAIN_PATH = args.train_data
    fine_tune.EVAL_PATH = args.eval_data
    if args.output_dir:
        fine_tune.OUTPUT_DIR = args.output_dir
    
    # Override LoRA hyperparameters
    fine_tune.LORA_RANK = args.lora_rank
    fine_tune.LORA_ALPHA = args.lora_alpha
    fine_tune.LORA_DROPOUT = args.lora_dropout
    
    # Override training hyperparameters
    fine_tune.BATCH_SIZE = args.batch_size
    fine_tune.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    fine_tune.LEARNING_RATE = args.learning_rate
    fine_tune.NUM_EPOCHS = args.num_epochs
    fine_tune.WARMUP_RATIO = args.warmup_ratio
    fine_tune.MAX_SEQ_LENGTH = args.max_seq_length
    
    # Override optimization parameters
    fine_tune.WEIGHT_DECAY = args.weight_decay
    fine_tune.LR_SCHEDULER_TYPE = args.lr_scheduler_type
    fine_tune.MAX_GRAD_NORM = args.max_grad_norm
    fine_tune.OPTIM = args.optim
    
    # Override monitoring parameters
    fine_tune.SAVE_STEPS = args.save_steps
    fine_tune.LOGGING_STEPS = args.logging_steps
    fine_tune.WANDB_PROJECT = args.wandb_project
    
    # Handle wandb flags
    if args.no_wandb:
        fine_tune.USE_WANDB = False
    elif args.use_wandb:
        fine_tune.USE_WANDB = True
    
    # Override GPU optimization flags
    fine_tune.FP16 = args.fp16
    fine_tune.GRADIENT_CHECKPOINTING = args.gradient_checkpointing
    
    # Disable Ollama management if requested
    if args.keep_ollama:
        fine_tune.gpu_manager.stop_ollama = lambda: None
        fine_tune.gpu_manager.start_ollama = lambda: None
    
    try:
        # Run training
        output_path, model_family = train_model(full_model_name)
        
        # Test generation
        if not args.skip_test:
            test_generation(output_path, model_family)
            
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe fine-tuning wrapper with GPU health checks")
    parser.add_argument("--model", type=str, required=True, help="Model name or shorthand")
    parser.add_argument("--train-data", type=str, default="data_formatted/train.json", help="Path to training data")
    parser.add_argument("--eval-data", type=str, default="data_formatted/test.json", help="Path to evaluation data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-test", action="store_true", help="Skip test generation after training")
    parser.add_argument("--keep-ollama", action="store_true", help="Don't stop Ollama service")
    parser.add_argument("--force-reset", action="store_true", help="Force GPU reset before training")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum GPU reset retries")
    
    # LoRA hyperparameters
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout (default: 0.0)")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length (default: 2048)")
    
    # Optimization parameters
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="LR scheduler type (default: cosine)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer (default: adamw_8bit)")
    
    # Monitoring parameters
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps (default: 10)")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="manim-post-train", help="W&B project name")
    
    # GPU optimization flags
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 training (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 training")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing (default: True)")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", help="Disable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Force reset if requested
    if args.force_reset:
        reset_gpu()
    
    # Check GPU health with retries
    retry_count = 0
    while retry_count < args.max_retries:
        healthy, message = check_gpu_health()
        
        if healthy:
            print(f"✓ GPU health check passed: {message}")
            break
        else:
            print(f"✗ GPU health check failed: {message}")
            
            if retry_count < args.max_retries - 1:
                print(f"Attempting GPU reset (attempt {retry_count + 1}/{args.max_retries})...")
                if reset_gpu():
                    retry_count += 1
                    continue
                else:
                    print("GPU reset failed")
            
            print("\nGPU is not healthy. Please try:")
            print("1. Run: sudo ./reset_gpu.sh")
            print("2. Run: python fix_gpu.py")
            print("3. Reboot the system")
            sys.exit(1)
    
    # GPU is healthy, proceed with training
    print("\nStarting training with healthy GPU...")
    
    # Run training with delayed imports
    success = safe_import_and_train(args.model, args)
    
    if success:
        print("\n✓ Training completed successfully!")
    else:
        print("\n✗ Training failed. Checking GPU state...")
        healthy, message = check_gpu_health()
        if not healthy:
            print(f"GPU is now unhealthy: {message}")
            print("Run: sudo ./reset_gpu.sh")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()