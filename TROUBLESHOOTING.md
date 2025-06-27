# Manim Fine-Tuning Troubleshooting Guide

## Quick Diagnostics

### Check Everything is Working
```bash
# 1. Test GPU
python diagnose_corruption.py

# 2. Test environment
python -c "import torch, manim, unsloth; print('All imports OK')"

# 3. Test data pipeline
python prepare_data.py
ls -la data_formatted/

# 4. Test training (small)
python fine_tune.py --model qwen-1.5b --max-steps 10
```

## Common Issues & Solutions

### Installation Issues

#### "Module not found: unsloth"
```bash
# Activate environment first
source manim-env/bin/activate

# Install with uv
uv pip install unsloth

# Or reinstall everything
uv pip install -r requirements.txt
```

#### "CUDA not available"
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### "Manim import error"
```bash
# Ensure you're in the right environment
which python  # Should show manim-env/bin/python

# Reinstall Manim
uv pip install manim

# Test Manim
manim --version
```

### Data Preparation Issues

#### "Dataset file not found"
```bash
# Check dataset location
ls -la /home/timholds/.cache/kagglehub/datasets/ravidussilva/manim-sft/versions/2/

# If missing, update path in prepare_data.py:
# DATASET_PATH = "/your/actual/path/to/dataset.parquet"
```

#### "KeyError: 'Reviewed Description'"
Dataset has different column names. Check columns:
```python
import pandas as pd
df = pd.read_parquet("path/to/dataset.parquet")
print(df.columns)
# Update column names in prepare_data.py
```

#### "Empty training data"
```bash
# Check data files
wc -l data_formatted/*.json

# Regenerate if needed
rm -rf data_formatted/
python prepare_data.py
```

### Training Issues

#### "CUDA out of memory"
```python
# Option 1: Reduce batch size
python fine_tune.py --model qwen-1.5b --batch-size 2

# Option 2: Use smaller model
python fine_tune.py --model qwen-0.5b

# Option 3: Clear cache and retry
python -c "import torch; torch.cuda.empty_cache()"
python fine_tune.py --model qwen-1.5b

# Option 4: Use 8-bit instead of 4-bit
# Edit fine_tune.py: load_in_4bit=False, load_in_8bit=True
```

#### "Training loss not decreasing"
```bash
# Check learning rate
# Edit fine_tune.py: learning_rate=5e-5 (try 1e-4 or 1e-5)

# Check data quality
python -c "
import json
with open('data_formatted/train.json') as f:
    data = [json.loads(line) for line in f]
    print(data[0])  # Inspect samples
"

# Try different model
python fine_tune.py --model deepseek-1.3b
```

#### "GPU corruption during training"
See GPU_TROUBLESHOOTING.md. Quick fix:
```bash
# Use Docker isolation
python train_docker.py --model qwen-1.5b

# Or use daemon mode
python train_daemon.py --model qwen-1.5b --daemon-mode
```

### Model Output Issues

#### "Model generates garbage"
```python
# 1. Check prompt format matches training
# In test_inference.py, ensure prompt uses same format as training

# 2. Use correct model path
model_path = "./outputs/your-actual-model"

# 3. Try different generation parameters
# temperature=0.3, top_p=0.9, repetition_penalty=1.1
```

#### "Model won't load in Ollama"
```bash
# Check GGUF conversion
ls -la *.gguf

# Reconvert if needed
python convert_to_ollama.py ./outputs/qwen-1.5b/

# Create Ollama model manually
ollama create manim-coder -f Modelfile

# Test
ollama run manim-coder "Create a circle"
```

#### "Generated code has syntax errors"
```python
# Add post-processing to fix common issues
def fix_generated_code(code):
    # Ensure imports
    if "from manim import" not in code:
        code = "from manim import *\n\n" + code
    
    # Ensure Scene class
    if "class" not in code:
        code = f"""class AnimationScene(Scene):
    def construct(self):
        {code}"""
    
    return code
```

### Performance Issues

#### "Training too slow"
```bash
# Enable mixed precision training
# In fine_tune.py: fp16=True or bf16=True

# Increase batch size if memory allows
python fine_tune.py --model qwen-1.5b --batch-size 8

# Use gradient accumulation
# gradient_accumulation_steps=4 with batch_size=2
```

#### "Inference too slow"
```bash
# Use quantized model
ollama run manim-coder  # Already quantized

# Or use vLLM for faster inference
uv pip install vllm
# See vllm_inference.py example
```

### Environment Issues

#### "Multiple Python environments conflicting"
```bash
# Always use the project environment
source manim-env/bin/activate

# Check which Python
which python  # Should be in manim-env

# Never use conda in this project
# If conda is active: conda deactivate
```

#### "Permission denied errors"
```bash
# For model outputs
chmod -R 755 outputs/

# For GPU reset
sudo chmod +x reset_gpu.sh

# For Docker
sudo usermod -aG docker $USER
# Log out and back in
```

## Debugging Techniques

### 1. Enable Verbose Logging
```python
# In scripts, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Test Individual Components
```python
# Test data loading
from prepare_data import prepare_dataset
prepare_dataset("path", "output", use_variations=False)

# Test model loading
from fine_tune import load_model_and_tokenizer
model, tokenizer = load_model_and_tokenizer("qwen-0.5b")

# Test generation
from test_inference import generate_code
code = generate_code(model, tokenizer, "Draw a circle")
```

### 3. Monitor Resources
```bash
# Terminal 1: GPU
watch -n 1 nvidia-smi

# Terminal 2: CPU/Memory
htop

# Terminal 3: Disk I/O
iotop
```

### 4. Check Logs
```bash
# Training logs
tail -f outputs/qwen-*/training.log

# System logs
sudo journalctl -f

# Python crashes
dmesg | grep -i python
```

## Emergency Recovery

### Complete Reset
```bash
# 1. Stop all processes
pkill -f python
sudo fuser -k /dev/nvidia*

# 2. Clear outputs
rm -rf outputs/* data_formatted/*

# 3. Reset GPU
sudo ./reset_gpu.sh

# 4. Rebuild environment
deactivate
rm -rf manim-env/
python -m venv manim-env
source manim-env/bin/activate
pip install uv
uv pip install -r requirements.txt

# 5. Start fresh
python prepare_data.py
python fine_tune.py --model qwen-0.5b --max-steps 10
```

## Getting Help

### Information to Provide
1. Full error message and stack trace
2. Output of `python diagnose_corruption.py`
3. Your command and parameters
4. Dataset information (size, format)
5. Environment details: `pip list | grep -E "torch|transformers|unsloth"`

### Where to Get Help
1. Check existing issues: GitHub repo issues
2. Unsloth Discord for Unsloth-specific issues
3. Manim community for Manim-specific issues
4. This documentation for common problems

## Validation Checklist

Before training:
- [ ] GPU healthy: `nvidia-smi` shows no errors
- [ ] Environment active: `which python` shows manim-env
- [ ] Data prepared: `ls data_formatted/*.json`
- [ ] Disk space available: `df -h` shows >50GB free

During training:
- [ ] Loss decreasing
- [ ] GPU memory stable
- [ ] No error messages
- [ ] Checkpoints saving

After training:
- [ ] Model files exist: `ls outputs/*/adapter_model.safetensors`
- [ ] Test generation works
- [ ] Ollama conversion successful
- [ ] Generated code runs in Manim