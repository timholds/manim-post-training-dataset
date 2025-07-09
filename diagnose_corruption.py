#!/usr/bin/env python3
"""
Diagnose the exact conditions that cause GPU corruption
"""

import subprocess
import os
import sys
import time

def check_gpu():
    """Detailed GPU check"""
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    
    print("\n--- GPU Status ---")
    print(f"Return code: {result.returncode}")
    
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:200]}")
        return False
        
    # Check for various error patterns
    errors = ["ERR!", "Unknown Error", "Unable to determine", "No devices were found"]
    for error in errors:
        if error in result.stdout or error in result.stderr:
            print(f"Found error pattern: '{error}'")
            return False
    
    # Print first few lines of nvidia-smi
    lines = result.stdout.split('\n')[:10]
    for line in lines:
        print(line)
    
    return True

def check_processes():
    """Check for lingering Python/CUDA processes"""
    print("\n--- Process Check ---")
    
    # Check Python processes
    result = subprocess.run(['pgrep', '-f', 'python'], capture_output=True, text=True)
    if result.stdout:
        pids = result.stdout.strip().split('\n')
        print(f"Python processes running: {pids}")
        
        # Get more info about each process
        for pid in pids:
            try:
                cmd_result = subprocess.run(['ps', '-p', pid, '-o', 'cmd='], capture_output=True, text=True)
                print(f"  PID {pid}: {cmd_result.stdout.strip()[:60]}...")
            except:
                pass
    else:
        print("No Python processes found")
    
    # Check GPU processes
    try:
        result = subprocess.run(['nvidia-smi', 'pmon', '-c', '1'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nGPU process monitor:")
            print(result.stdout)
    except:
        pass

def check_environment():
    """Check environment variables"""
    print("\n--- Environment Check ---")
    
    cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k or 'NVIDIA' in k}
    if cuda_vars:
        print("CUDA-related environment variables:")
        for k, v in cuda_vars.items():
            print(f"  {k}={v}")
    else:
        print("No CUDA environment variables set")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        print(f"\nLD_LIBRARY_PATH: {ld_path}")

def check_driver_info():
    """Get detailed driver information"""
    print("\n--- Driver Information ---")
    
    try:
        # Driver version
        result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(line.strip())
    except:
        pass
    
    # Kernel modules
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nNVIDIA kernel modules:")
            for line in result.stdout.split('\n'):
                if 'nvidia' in line.lower():
                    print(f"  {line}")
    except:
        pass

def test_import_sequence():
    """Test if specific import sequences cause issues"""
    print("\n--- Testing Import Sequences ---")
    
    sequences = [
        ["torch", "unsloth"],
        ["unsloth", "torch"],
        ["transformers", "unsloth", "torch"],
        ["torch", "transformers", "unsloth"],
    ]
    
    for seq in sequences:
        print(f"\nTesting sequence: {' -> '.join(seq)}")
        
        # Create test script
        imports = '\n'.join(f"import {mod}" for mod in seq)
        script = f"""
import sys
{imports}
print("Imports successful")
"""
        
        with open('test_import.py', 'w') as f:
            f.write(script)
        
        # Run in subprocess
        result = subprocess.run([sys.executable, 'test_import.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Success")
        else:
            print(f"  ❌ Failed: {result.stderr[:100]}")
        
        os.remove('test_import.py')
        
        # Check GPU after each
        if not check_gpu():
            print(f"  ❌ GPU corrupted by sequence: {' -> '.join(seq)}")
            return False
    
    return True

def check_file_descriptors():
    """Check for file descriptor leaks"""
    print("\n--- File Descriptor Check ---")
    
    try:
        # Current process
        pid = os.getpid()
        fd_path = f"/proc/{pid}/fd"
        if os.path.exists(fd_path):
            fds = os.listdir(fd_path)
            print(f"Current process has {len(fds)} file descriptors")
            
            # Check for CUDA-related
            for fd in fds[:10]:  # Check first 10
                try:
                    link = os.readlink(f"{fd_path}/{fd}")
                    if 'nvidia' in link or 'cuda' in link:
                        print(f"  FD {fd}: {link}")
                except:
                    pass
    except:
        pass

def main():
    print("GPU Corruption Diagnostic Tool")
    print("==============================")
    
    # Full system check
    check_gpu()
    check_processes()
    check_environment()
    check_driver_info()
    check_file_descriptors()
    
    # Import sequence test
    print("\n" + "="*50)
    if test_import_sequence():
        print("\n✅ All import sequences OK")
    else:
        print("\n❌ Found problematic import sequence")
    
    # Final GPU check
    print("\n" + "="*50)
    print("Final GPU check:")
    if check_gpu():
        print("\n✅ GPU is healthy")
    else:
        print("\n❌ GPU is corrupted")

if __name__ == "__main__":
    main()