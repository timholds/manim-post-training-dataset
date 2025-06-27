# GPU Troubleshooting Guide

## Critical Issue: GPU Corruption with 4-bit Quantization

### Problem Description
When using Unsloth with 4-bit quantized models on NVIDIA driver 575, the GPU becomes corrupted during Python process exit. This happens AFTER successful training completion.

### Symptoms
- Training runs successfully to completion
- Model saves correctly
- GPU corruption occurs during Python process termination
- Error: "GPU is lost" or kernel crashes
- System may require reboot

### Root Cause
Bug in CUDA context cleanup when Python exits after using 4-bit quantized models with Unsloth. The issue is specific to:
- NVIDIA driver 575.x
- Unsloth library
- 4-bit quantization (not 8-bit or 16-bit)

## Solutions (In Order of Preference)

### 1. Docker Isolation (Recommended)
```bash
python train_docker.py --model qwen-1.5b
```
**Pros**: Complete isolation, no host GPU corruption
**Cons**: Slightly slower startup, requires Docker

### 2. Training Daemon Mode
```bash
python train_daemon.py --model qwen-1.5b --daemon-mode
```
**Pros**: Keeps Python process alive, avoiding exit corruption
**Cons**: Uses more system resources between runs

### 3. Manual GPU Reset
```bash
python fine_tune.py --model qwen-1.5b
sudo ./reset_gpu.sh  # Run IMMEDIATELY after training
```
**Pros**: Simple, works with existing scripts
**Cons**: Requires sudo, manual intervention

### 4. Alternative Approaches
- Use 8-bit quantization: `--quantization 8bit`
- Use HuggingFace without Unsloth: `--no-unsloth`
- Downgrade NVIDIA driver to 550.x

## GPU Reset Procedures

### Automatic Reset Script
```bash
sudo ./reset_gpu.sh
```
This script:
1. Unloads NVIDIA kernel modules
2. Resets GPU state
3. Reloads modules
4. Verifies GPU health

### Manual Reset Steps
```bash
# 1. Kill all GPU processes
sudo fuser -k /dev/nvidia*

# 2. Reset GPU state
sudo nvidia-smi --gpu-reset

# 3. If that fails, unload/reload modules
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
```

### Nuclear Option (Last Resort)
```bash
sudo reboot
```

## Diagnostic Tools

### 1. GPU Health Check
```bash
python diagnose_corruption.py
```
Provides comprehensive GPU diagnostics:
- Memory usage and availability
- Process list
- Driver and CUDA versions
- Corruption indicators

### 2. Quick GPU Status
```bash
nvidia-smi
```
Check for:
- "GPU is lost" errors
- Unusual memory usage
- Stuck processes

### 3. Test GPU Without Training
```bash
python test_gpu_minimal.py
```
Tests basic GPU operations without Unsloth

## Prevention Strategies

### Before Training
1. Check GPU health: `python diagnose_corruption.py`
2. Clear any stuck processes: `sudo fuser -k /dev/nvidia*`
3. Ensure adequate cooling (monitor temps during training)

### During Training
1. Monitor GPU: `watch -n 1 nvidia-smi`
2. Check system logs: `sudo dmesg -w | grep -i nvidia`
3. Watch for memory leaks or temperature issues

### After Training
1. Use provided workarounds (Docker/daemon/reset)
2. Verify model outputs before extensive testing
3. Document any issues for future reference

## Common Issues & Solutions

### "CUDA out of memory"
```bash
# Reduce batch size
python fine_tune.py --model qwen-1.5b --batch-size 2

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Use gradient checkpointing
# (enabled by default in our scripts)
```

### "GPU is lost"
```bash
# Immediate action required
sudo ./reset_gpu.sh

# If persists, full module reload
sudo ./fix_gpu.py --force
```

### "Process stuck on GPU"
```bash
# Find stuck processes
nvidia-smi | grep python

# Kill specific process
sudo kill -9 <PID>

# Kill all GPU processes
sudo fuser -k /dev/nvidia*
```

### "Module nvidia not found"
```bash
# Reinstall NVIDIA drivers
sudo apt update
sudo apt install --reinstall nvidia-driver-575

# Or downgrade to stable version
sudo apt install nvidia-driver-550
```

## Monitoring During Training

### Real-time GPU Monitor
```bash
# Terminal 1: GPU stats
watch -n 1 nvidia-smi

# Terminal 2: System logs
sudo journalctl -f | grep -i nvidia

# Terminal 3: Training logs
tail -f training.log
```

### Key Metrics to Watch
- **GPU Memory**: Should stay under 15GB for 16GB GPU
- **Temperature**: Keep under 83°C
- **Power Draw**: Normal is 60-90W for A4500
- **GPU Utilization**: Should be 80-100% during training

## Emergency Recovery

### If System Becomes Unresponsive
1. Try switching to TTY: `Ctrl+Alt+F2`
2. Login and run: `sudo systemctl restart gdm`
3. If that fails: `sudo reboot`

### If GPU Permanently Corrupted
1. Boot in recovery mode
2. Purge NVIDIA drivers: `sudo apt purge nvidia-*`
3. Reinstall drivers
4. Test with simple CUDA operations first

## Best Practices

1. **Always use workarounds** for 4-bit training
2. **Test small first**: Run 10 steps before full training
3. **Monitor actively**: Don't leave training unattended
4. **Document issues**: Note any warnings or errors
5. **Regular backups**: Save model checkpoints frequently

## Future Fixes

This issue should be resolved when:
- Unsloth releases a patch
- NVIDIA updates driver to 580+
- PyTorch updates CUDA handling

Until then, use the provided workarounds to ensure stable training.