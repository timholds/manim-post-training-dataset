#!/usr/bin/env python3
"""
Run training in Docker to avoid GPU corruption
"""

import subprocess
import os
import sys
from pathlib import Path

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True)
        return result.returncode == 0
    except:
        return False

def build_docker_image():
    """Build a Docker image with all dependencies"""
    
    dockerfile = '''FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git \\
    transformers \\
    trl \\
    peft \\
    accelerate \\
    bitsandbytes \\
    datasets \\
    wandb

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV WANDB_DISABLED=true
'''
    
    # Write Dockerfile
    with open('Dockerfile.training', 'w') as f:
        f.write(dockerfile)
    
    print("Building Docker image...")
    result = subprocess.run(
        ['docker', 'build', '-f', 'Dockerfile.training', '-t', 'manim-training', '.'],
        check=True
    )
    
    # Clean up
    os.remove('Dockerfile.training')
    print("✅ Docker image built")

def run_training_in_docker(model, train_data, eval_data, output_dir):
    """Run training inside Docker container"""
    
    # Get absolute paths
    workspace = Path.cwd()
    
    # Docker run command
    cmd = [
        'docker', 'run',
        '--rm',  # Remove container after exit
        '--gpus', 'all',  # Enable GPU
        '-v', f'{workspace}:/workspace',  # Mount current directory
        '-w', '/workspace',  # Working directory
        'manim-training',
        'python', 'fine_tune.py',
        '--model', model,
        '--train-data', train_data,
        '--eval-data', eval_data,
        '--output-dir', output_dir
    ]
    
    print(f"Running training in Docker container...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd)
    
    return result.returncode == 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training in Docker to avoid GPU corruption")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--train-data", type=str, default="data_formatted/train.json")
    parser.add_argument("--eval-data", type=str, default="data_formatted/test.json")
    parser.add_argument("--output-dir", type=str, default="models/lora_model")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    
    args = parser.parse_args()
    
    # Check Docker
    if not check_docker():
        print("❌ Docker not found. Please install Docker first.")
        print("Visit: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    # Build image if requested or if it doesn't exist
    if args.build or subprocess.run(
        ['docker', 'images', '-q', 'manim-training'],
        capture_output=True
    ).stdout.strip() == b'':
        build_docker_image()
    
    # Run training
    success = run_training_in_docker(
        args.model,
        args.train_data,
        args.eval_data,
        args.output_dir
    )
    
    if success:
        print("\n✅ Training completed successfully!")
        print("GPU should remain healthy since training ran in isolated container")
    else:
        print("\n❌ Training failed")
    
    # Check GPU state
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0 and "ERR!" not in result.stdout:
        print("\n✅ GPU is healthy!")
    else:
        print("\n❌ GPU is corrupted even with Docker isolation!")

if __name__ == "__main__":
    main()