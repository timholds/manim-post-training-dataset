#!/usr/bin/env python3
"""
Training daemon that keeps Python process alive between training runs
to avoid GPU corruption on process exit
"""

import subprocess
import sys
import time
import os
import json
from pathlib import Path

class TrainingDaemon:
    def __init__(self):
        self.process = None
        self.loaded = False
        
    def start(self):
        """Start the daemon process"""
        daemon_script = '''
import sys
import json
import torch
import gc
from unsloth import FastLanguageModel
from pathlib import Path

# Keep these loaded in memory
model = None
tokenizer = None
base_model_name = None

print("DAEMON:READY", flush=True)

while True:
    try:
        line = sys.stdin.readline().strip()
        if not line:
            continue
            
        if line == "EXIT":
            break
            
        if line == "STATUS":
            print(f"DAEMON:STATUS:{'LOADED' if model is not None else 'EMPTY'}", flush=True)
            continue
            
        if line.startswith("TRAIN:"):
            # Parse training command
            cmd_data = json.loads(line[6:])
            model_name = cmd_data['model']
            train_data = cmd_data['train_data']
            eval_data = cmd_data['eval_data']
            output_dir = cmd_data['output_dir']
            
            print(f"DAEMON:TRAINING:START", flush=True)
            
            # Import training function
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from fine_tune import safe_train_model
            
            try:
                # Run training
                output_path, model_family = safe_train_model(model_name)
                print(f"DAEMON:TRAINING:SUCCESS:{output_path}", flush=True)
            except Exception as e:
                print(f"DAEMON:TRAINING:ERROR:{str(e)}", flush=True)
            
            # Cleanup but keep process alive
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        elif line == "CLEANUP":
            # Force cleanup
            if model is not None:
                del model
                model = None
            if tokenizer is not None:
                del tokenizer
                tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("DAEMON:CLEANUP:DONE", flush=True)
            
    except Exception as e:
        print(f"DAEMON:ERROR:{str(e)}", flush=True)

print("DAEMON:EXIT", flush=True)
'''
        
        # Write daemon script
        with open('.training_daemon.py', 'w') as f:
            f.write(daemon_script)
        
        # Start daemon process
        self.process = subprocess.Popen(
            [sys.executable, '.training_daemon.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for ready
        while True:
            line = self.process.stdout.readline().strip()
            if line == "DAEMON:READY":
                print("✅ Training daemon started")
                break
    
    def train(self, model, train_data, eval_data, output_dir):
        """Send training command to daemon"""
        if not self.process:
            self.start()
        
        # Send training command
        cmd = {
            'model': model,
            'train_data': train_data,
            'eval_data': eval_data,
            'output_dir': output_dir
        }
        
        self.process.stdin.write(f"TRAIN:{json.dumps(cmd)}\n")
        self.process.stdin.flush()
        
        # Read output
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                continue
                
            if line.startswith("DAEMON:TRAINING:"):
                status = line.split(":", 2)[2]
                if status == "START":
                    print("Training started...")
                elif status.startswith("SUCCESS"):
                    output_path = status.split(":", 1)[1] if ":" in status else ""
                    print(f"✅ Training completed: {output_path}")
                    return True
                elif status.startswith("ERROR"):
                    error = status.split(":", 1)[1] if ":" in status else "Unknown error"
                    print(f"❌ Training failed: {error}")
                    return False
            else:
                # Pass through other output
                print(line)
    
    def cleanup(self):
        """Request cleanup"""
        if self.process:
            self.process.stdin.write("CLEANUP\n")
            self.process.stdin.flush()
            
            # Wait for confirmation
            while True:
                line = self.process.stdout.readline().strip()
                if line == "DAEMON:CLEANUP:DONE":
                    print("✅ Cleanup completed")
                    break
    
    def stop(self):
        """Stop the daemon"""
        if self.process:
            self.process.stdin.write("EXIT\n")
            self.process.stdin.flush()
            self.process.wait()
            print("✅ Training daemon stopped")
            
            # Remove daemon script
            if os.path.exists('.training_daemon.py'):
                os.remove('.training_daemon.py')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Training daemon to avoid GPU corruption")
    parser.add_argument("--model", type=str, help="Model name or shorthand")
    parser.add_argument("--train-data", type=str, default="data_formatted/train.json")
    parser.add_argument("--eval-data", type=str, default="data_formatted/test.json")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--daemon-mode", action="store_true", help="Keep daemon running")
    
    args = parser.parse_args()
    
    # Create daemon
    daemon = TrainingDaemon()
    
    try:
        if args.model:
            # Run training
            success = daemon.train(
                args.model,
                args.train_data,
                args.eval_data,
                args.output_dir
            )
            
            if not args.daemon_mode:
                # Stop daemon after training
                daemon.stop()
            else:
                print("\nDaemon mode: Process kept alive to avoid GPU corruption")
                print("Run more training commands or Ctrl+C to exit")
                
                # Interactive mode
                while True:
                    try:
                        cmd = input("\nEnter command (train/cleanup/exit): ").strip()
                        if cmd == "exit":
                            break
                        elif cmd == "cleanup":
                            daemon.cleanup()
                        elif cmd.startswith("train "):
                            parts = cmd.split()
                            if len(parts) > 1:
                                daemon.train(parts[1], args.train_data, args.eval_data, args.output_dir)
                    except KeyboardInterrupt:
                        break
        
    finally:
        daemon.stop()

if __name__ == "__main__":
    main()