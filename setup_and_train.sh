#!/bin/bash
# Setup and training script for Manim code generation model

set -e  # Exit on error

# Default model
DEFAULT_MODEL="qwen-7b"

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [MODEL]"
    echo ""
    echo "Train a model for Manim code generation."
    echo ""
    echo "Arguments:"
    echo "  MODEL    Model to fine-tune (default: $DEFAULT_MODEL)"
    echo ""
    echo "Available models:"
    echo "  - qwen-1.5b, qwen-7b (Qwen family)"
    echo "  - codellama-7b, codellama-13b (CodeLlama family)"
    echo "  - codegemma-2b, codegemma-7b (CodeGemma family)"
    echo "  - deepseek-1.3b, deepseek-6.7b (DeepSeek family)"
    echo "  - stable-code-3b (Stable Code family)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Use default model (qwen-7b)"
    echo "  $0 codellama-7b     # Train CodeLlama 7B"
    echo "  $0 qwen-1.5b        # Train smaller QWEN model"
    echo ""
    echo "For full list of models, run: python fine_tune.py --list-models"
    exit 0
fi

MODEL="${1:-$DEFAULT_MODEL}"

echo "=================================================="
echo "Manim Code Generation Model - Setup and Training"
echo "=================================================="
echo "Model: $MODEL"
echo ""

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if manim-env exists
if [ ! -d "manim-env" ]; then
    echo "Error: manim-env not found. Please create the virtual environment first."
    echo "Run: python -m venv manim-env"
    exit 1
fi

# Activate the virtual environment
echo "Activating manim-env..."
source manim-env/bin/activate

# Check for wandb login
echo -e "\nChecking Weights & Biases configuration..."
if ! wandb login --verify > /dev/null 2>&1; then
    echo "WARNING: Not logged into Weights & Biases."
    echo "To enable metrics logging, run: wandb login"
    echo "Or set WANDB_API_KEY environment variable"
    echo ""
    read -p "Continue without wandb logging? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please run 'wandb login' and try again."
        exit 1
    fi
    export WANDB_MODE=offline
fi

# Install dependencies
echo -e "\n1. Installing dependencies..."
uv pip install -r requirements.txt

# Check CUDA availability
echo -e "\n2. Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Prepare the dataset
echo -e "\n3. Preparing ManimBench dataset..."
python prepare_data.py

# Check if data was created successfully
if [ ! -f "data/train.json" ] || [ ! -f "data/test.json" ]; then
    echo "Error: Dataset preparation failed"
    exit 1
fi

# Fine-tune the model
echo -e "\n4. Starting model fine-tuning..."
echo "This will take approximately 30-45 minutes on a 16GB GPU"
echo "Note: Ollama service will be temporarily stopped to free GPU memory"
echo "      (use --keep-ollama flag if you have enough GPU memory)"
python fine_tune.py --model "$MODEL"

# Check if model was created successfully
if [ ! -d "models/lora_model" ] || [ ! -d "models/merged_model" ]; then
    echo "Error: Model training failed"
    exit 1
fi

# Convert to Ollama format
echo -e "\n5. Converting model to Ollama format..."
python convert_to_ollama.py

# Run inference tests
echo -e "\n6. Running inference tests..."
python test_inference.py

# Run comprehensive evaluation
echo -e "\n7. Running comprehensive model evaluation..."
python evaluate_model.py

echo -e "\n=================================================="
echo "Training completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Test the Ollama model: ollama run manim-coder 'Create a circle animation'"
echo "2. Run interactive mode: python test_inference.py --interactive"
echo "3. Check test results: cat test_results.txt"
echo "4. View evaluation report: cat evaluation_report.txt"
echo "5. Check wandb dashboard: https://wandb.ai/${WANDB_ENTITY:-$USER}/manim-post-train"
echo ""
echo "Model files:"
echo "- LoRA adapters: models/lora_model/"
echo "- Merged model: models/merged_model/"
echo "- Ollama model: models/manim-coder.gguf"
echo ""
echo "Metrics logged to Weights & Biases project: manim-post-train"