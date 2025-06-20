#!/usr/bin/env python3
"""Shared utilities for managing Ollama service and models."""

import subprocess
import time
import requests
import sys
import ollama
from pathlib import Path

class OllamaManager:
    """Manages Ollama service lifecycle and model availability."""
    
    def __init__(self, required_models=None, timeout=30):
        self.required_models = required_models or []
        self.timeout = timeout
        self.ollama_pid_file = Path("/tmp/ollama_manager.pid")
        self.started_service = False
        
    def is_ollama_running(self):
        """Check if Ollama service is accessible."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama(self):
        """Start Ollama service in background if not running."""
        if self.is_ollama_running():
            print("✓ Ollama is already running")
            return True
            
        print("Starting Ollama service...")
        try:
            # Start ollama in background, redirect output to prevent terminal pollution
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=lambda: None  # Detach from parent
            )
            
            # Save PID for potential cleanup
            self.ollama_pid_file.write_text(str(process.pid))
            self.started_service = True  # Track that we started it
            
            # Wait for service to be ready
            for i in range(self.timeout):
                if self.is_ollama_running():
                    print("✓ Ollama service started successfully")
                    return True
                time.sleep(1)
                if i % 5 == 0:
                    print(f"  Waiting for Ollama to start... ({i}s)")
            
            print("✗ Ollama failed to start within timeout")
            return False
            
        except FileNotFoundError:
            print("✗ Ollama not found. Please install it first:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            return False
        except Exception as e:
            print(f"✗ Failed to start Ollama: {str(e)}")
            return False
    
    def list_available_models(self):
        """Get list of locally available models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except:
            return []
    
    def pull_model(self, model_name):
        """Pull a model if not already available."""
        available_models = self.list_available_models()
        
        # Check if model already exists (handle version tags)
        model_base = model_name.split(':')[0]
        if any(model_base in avail for avail in available_models):
            print(f"✓ Model {model_name} already available")
            return True
            
        print(f"Pulling model {model_name}...")
        try:
            # Use subprocess to show progress
            result = subprocess.run(
                ["ollama", "pull", model_name],
                check=True,
                text=True
            )
            print(f"✓ Successfully pulled {model_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to pull {model_name}: {e}")
            return False
        except Exception as e:
            print(f"✗ Error pulling {model_name}: {str(e)}")
            return False
    
    def ensure_ready(self):
        """Ensure Ollama is running and required models are available."""
        print("🔧 Checking Ollama setup...")
        
        # Start Ollama if needed
        if not self.start_ollama():
            return False
        
        # Pull required models
        if self.required_models:
            print(f"\nChecking required models: {', '.join(self.required_models)}")
            for model in self.required_models:
                if not self.pull_model(model):
                    print(f"\n⚠️  Warning: Could not pull {model}, but continuing...")
        
        print("\n✅ Ollama is ready!")
        return True
    
    def cleanup(self):
        """Optional cleanup - stops Ollama if we started it."""
        if self.started_service and self.ollama_pid_file.exists():
            try:
                pid = int(self.ollama_pid_file.read_text())
                subprocess.run(["kill", str(pid)], capture_output=True)
                self.ollama_pid_file.unlink()
                print("✓ Stopped Ollama service")
                self.started_service = False
            except:
                pass
    
    def __enter__(self):
        """Context manager entry - start Ollama service."""
        if not self.ensure_ready():
            raise RuntimeError("Failed to start Ollama service")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup Ollama service."""
        self.cleanup()


def ensure_ollama_ready(required_models=None):
    """Convenience function to ensure Ollama is ready with minimal code."""
    manager = OllamaManager(required_models=required_models)
    if not manager.ensure_ready():
        print("\n❌ Failed to setup Ollama")
        sys.exit(1)
    return manager


if __name__ == "__main__":
    # Test the utility
    print("Testing Ollama utilities...")
    manager = ensure_ollama_ready(["qwen2.5-coder:1.5b", "llama3.2:3b"])
    print("\nAvailable models:", manager.list_available_models())