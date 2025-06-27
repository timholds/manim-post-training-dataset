#!/bin/bash
# Safe training wrapper that ensures GPU cleanup between runs

echo "Safe Training Wrapper"
echo "===================="

# Function to check GPU health
check_gpu() {
    nvidia-smi > /dev/null 2>&1
    return $?
}

# Function to cleanup
cleanup() {
    echo "Running cleanup..."
    python unsloth_cleanup.py
    
    # Extra cleanup steps
    pkill -f "python.*train" 2>/dev/null
    sleep 2
}

# Check initial GPU state
if ! check_gpu; then
    echo "❌ GPU is corrupted before training!"
    echo "Run: sudo ./reset_gpu.sh"
    exit 1
fi

echo "✅ GPU healthy, starting training..."

# Run training
python fine_tune.py "$@"
TRAIN_EXIT_CODE=$?

# Always run cleanup, regardless of success/failure
cleanup

# Check GPU state after training
if ! check_gpu; then
    echo "⚠️  GPU corrupted after training!"
    echo "Attempting automatic recovery..."
    
    # Try to reset without sudo first
    if [ -f "/sys/bus/pci/devices/0000:01:00.0/reset" ]; then
        echo "Attempting PCI reset..."
        echo 1 > /sys/bus/pci/devices/0000:01:00.0/reset 2>/dev/null || {
            echo "PCI reset requires sudo. Run: sudo ./reset_gpu.sh"
        }
    fi
fi

exit $TRAIN_EXIT_CODE