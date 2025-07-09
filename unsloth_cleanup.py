#!/usr/bin/env python3
"""
Comprehensive cleanup script for Unsloth/CUDA to prevent GPU corruption
"""

import os
import gc
import sys
import time
import subprocess
import logging

logger = logging.getLogger(__name__)

def aggressive_cleanup():
    """Aggressive cleanup to prevent GPU corruption between runs"""
    
    # Import torch only if available
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        
    logger.info("Starting aggressive GPU cleanup...")
    
    # Step 1: Clear all Python references
    # Get all modules that might hold GPU references
    modules_to_clear = [
        'unsloth', 'unsloth.models', 'unsloth.tokenizers',
        'transformers', 'transformers.models',
        'trl', 'peft', 'accelerate',
        'bitsandbytes', 'datasets',
    ]
    
    for module in list(sys.modules.keys()):
        if any(module.startswith(m) for m in modules_to_clear):
            logger.info(f"Removing module: {module}")
            del sys.modules[module]
    
    # Step 2: Force garbage collection multiple times
    for i in range(3):
        gc.collect()
        time.sleep(0.1)
    
    # Step 3: Clear CUDA if available
    if has_torch and torch.cuda.is_available():
        try:
            # Clear all CUDA caches
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset memory stats
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
            
            # Try to reset CUDA context (experimental)
            if hasattr(torch.cuda, 'reset'):
                torch.cuda.reset()
                
            logger.info("CUDA cleanup completed")
        except Exception as e:
            logger.warning(f"CUDA cleanup error: {e}")
    
    # Step 4: Clear environment variables that might affect CUDA
    cuda_env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_LAUNCH_BLOCKING',
        'PYTORCH_CUDA_ALLOC_CONF',
        'TORCH_CUDA_ARCH_LIST',
    ]
    
    for var in cuda_env_vars:
        if var in os.environ:
            logger.info(f"Clearing {var}")
            del os.environ[var]
    
    # Step 5: Final garbage collection
    gc.collect()
    
    # Step 6: Small delay to ensure cleanup
    time.sleep(1)
    
    logger.info("Aggressive cleanup completed")

def kill_gpu_processes():
    """Kill any lingering GPU processes"""
    try:
        # Find processes using GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), 9)
                    logger.info(f"Killed GPU process: {pid}")
                except:
                    pass
    except:
        pass

def reset_cuda_device():
    """Attempt to reset CUDA device (requires root)"""
    try:
        # Try nvidia-ml-py if available
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # This might help reset the device state
            pynvml.nvmlDeviceResetApplicationsClocks(handle)
            
        pynvml.nvmlShutdown()
        logger.info("NVML reset completed")
    except:
        # If pynvml not available, try nvidia-smi
        try:
            subprocess.run(['nvidia-smi', '-r'], capture_output=True)
            logger.info("nvidia-smi reset attempted")
        except:
            pass

def full_cleanup():
    """Run full cleanup sequence"""
    aggressive_cleanup()
    kill_gpu_processes()
    # Only attempt device reset if we have permissions
    if os.geteuid() == 0:
        reset_cuda_device()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    full_cleanup()
    print("Cleanup completed. GPU should be ready for next run.")