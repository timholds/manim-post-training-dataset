# CUDA/NVIDIA Driver Fix Instructions

## Current Issue
Your system has NVIDIA driver 575 installed, but PyTorch cannot detect your GPU. The nvidia-smi output shows errors (ERR!) which indicates a driver problem.

## Root Cause
The upgrade to NVIDIA driver 575 has caused compatibility issues. There's likely a mismatch between:
- NVIDIA kernel modules (575)
- NVIDIA userspace libraries (mixed 550/575)
- CUDA runtime expectations

## Recommended Solutions (in order of preference)

### Option 1: Clean Driver Reinstall (Recommended)
```bash
# 1. Remove all NVIDIA packages
sudo apt-get purge nvidia-* cuda-* libnvidia-*

# 2. Add the official NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

# 3. Install a stable driver version (550 or 560)
sudo apt-get install nvidia-driver-550

# 4. Reboot
sudo reboot

# 5. After reboot, verify
nvidia-smi
```

### Option 2: Downgrade PyTorch to Match Driver
```bash
source manim-env/bin/activate
uv pip uninstall torch torchvision
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

### Option 3: Use Docker with NVIDIA Container Toolkit
```bash
# Install Docker and NVIDIA Container Toolkit
# Then run:
docker run --gpus all -it pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
```

### Option 4: Temporary CPU-Only Training
Use the `fine_tune_qwen_cpu.py` script I created as a temporary workaround.

## Quick Diagnostic Commands
```bash
# Check driver status
nvidia-smi

# Check loaded modules
lsmod | grep nvidia

# Check CUDA in Python
source manim-env/bin/activate
python -c "import torch; print(torch.cuda.is_available())"

# Check driver files
ls -la /usr/lib/x86_64-linux-gnu/libnvidia*
dpkg -l | grep nvidia
```

## After Fixing
Once CUDA is working again, you can run:
```bash
source manim-env/bin/activate
python fine_tune_qwen.py
```

## Emergency Workaround
If you need to proceed immediately without fixing the driver:
1. Use the CPU-only script: `python fine_tune_qwen_cpu.py`
2. Or use Google Colab with their free GPU
3. Or use a cloud GPU service (Vast.ai, Lambda Labs, etc.)