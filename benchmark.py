#!/usr/bin/env python3
"""Simple Manim code generation benchmark tool."""

import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from tabulate import tabulate
import tempfile
import os
import ollama
from ollama_utils import ensure_ollama_ready, OllamaManager
import argparse
from manim_code_extractor import ManimCodeExtractor

def generate_manim_code(model_name, prompt):
    """Generate Manim code using ollama."""
    # Prepare the prompt to encourage Manim code generation
    full_prompt = f"""Generate Python code using Manim to create an animation for: {prompt}

The code should be a complete, runnable Manim script with proper imports and a Scene class.
Only output the Python code, no explanations."""
    
    try:
        response = ollama.generate(
            model=model_name, 
            prompt=full_prompt,
            options={
                'temperature': 0.4,
                'top_p': 0.9
            }
        )
        return response['response']
    except Exception as e:
        raise Exception(f"Failed to connect to Ollama. Please ensure Ollama is running (ollama serve). Error: {str(e)}")

def extract_python_code(text):
    """Extract Python code from the response using ManimCodeExtractor."""
    extractor = ManimCodeExtractor()
    return extractor.extract(text)

def test_manim_compilation(code, generate_video=False, video_counter=None):
    """Test if the generated code compiles with Manim."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    video_size = None
    
    try:
        if generate_video:
            # Use persistent benchmark_videos directory
            video_dir = Path("benchmark_videos")
            video_dir.mkdir(exist_ok=True)
            
            # Generate a unique filename for this test
            video_filename = f"test_{video_counter:03d}"
            
            # Run without dry_run to actually generate the video
            cmd = f"source manim-env/bin/activate && manim -ql {temp_file} -o {video_filename} --media_dir {video_dir}"
            compile_result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, 
                text=True,
                timeout=60  # Increased timeout for video generation
            )
            
            # Check for generated video files
            if compile_result.returncode == 0:
                # Find any .mp4 files in the video directory
                for root, dirs, files in os.walk(video_dir):
                    for file in files:
                        if file.endswith('.mp4') and video_filename in file:
                            video_path = os.path.join(root, file)
                            video_size = os.path.getsize(video_path)
                            break
                    if video_size:
                        break
        else:
            # Dry run for quick compilation check
            compile_result = subprocess.run(
                ["bash", "-c", f"source manim-env/bin/activate && manim -ql {temp_file} --dry_run"],
                capture_output=True, 
                text=True,
                timeout=30
            )
        
        success = compile_result.returncode == 0
        error = compile_result.stderr if not success else ""
        
        return success, error, video_size
    finally:
        os.unlink(temp_file)

def run_benchmark(generate_videos=False):
    """Run the benchmark on all models and prompts."""
    # Read inputs
    if not Path("benchmark_queries.txt").exists():
        print("Creating sample benchmark_queries.txt...")
        Path("benchmark_queries.txt").write_text("""Create a sine wave animation
Animate a bouncing ball
Show the derivative of x^2
Create a 3D rotating cube
Animate the Pythagorean theorem""")
    
    if not Path("models.txt").exists():
        print("Creating sample models.txt...")
        Path("models.txt").write_text("""qwen2.5-coder:7b
llama3.2:3b""")
    
    prompts = [p.strip() for p in Path("benchmark_queries.txt").read_text().strip().split('\n') if p.strip()]
    models = [m.strip() for m in Path("models.txt").read_text().strip().split('\n') if m.strip()]
    
    print(f"Testing {len(models)} models on {len(prompts)} prompts...\n")
    
    results = {}
    detailed_results = []
    
    # Clear benchmark_videos directory if it exists
    if generate_videos:
        video_dir = Path("benchmark_videos")
        if video_dir.exists():
            subprocess.run(["rm", "-rf", str(video_dir)], capture_output=True)
        print(f"📹 Videos will be saved to: {video_dir.absolute()}")
    
    video_counter = 0
    
    for model in models:
        print(f"\nTesting model: {model}")
        results[model] = {"total": 0, "passed": 0, "details": {}}
        
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...", end=' ', flush=True)
            
            try:
                # Generate code
                generated_code = generate_manim_code(model, prompt)
                code = extract_python_code(generated_code)
                
                # Validate code structure first
                extractor = ManimCodeExtractor()
                validation = extractor.validate(code)
                
                if not validation.is_valid:
                    # Skip compilation test if validation fails
                    results[model]["total"] += 1
                    print("✗ (Invalid structure)")
                    results[model]["details"][prompt] = {
                        "success": False,
                        "error": f"Validation failed: {'; '.join(validation.errors)}",
                        "generated_code": code,
                        "video_size": None
                    }
                    continue
                
                # Test compilation
                video_counter += 1
                success, error, video_size = test_manim_compilation(code, generate_video=generate_videos, video_counter=video_counter)
                
                results[model]["total"] += 1
                if success:
                    results[model]["passed"] += 1
                    if video_size:
                        if video_size > 1024 * 1024:
                            size_str = f" ({video_size / (1024 * 1024):.1f}MB)"
                        else:
                            size_str = f" ({video_size / 1024:.0f}KB)"
                        print(f"✓{size_str}")
                    else:
                        print("✓")
                else:
                    print("✗")
                    if error and len(error) < 200:
                        print(f"      Error: {error.strip()}")
                
                results[model]["details"][prompt] = {
                    "success": success,
                    "error": error if not success else None,
                    "generated_code": code,
                    "video_size": video_size
                }
                
            except Exception as e:
                print(f"✗ (Generation failed: {str(e)[:50]})")
                results[model]["total"] += 1
                results[model]["details"][prompt] = {
                    "success": False,
                    "error": str(e),
                    "generated_code": None,
                    "video_size": None
                }
    
    # Calculate success rates
    for model in models:
        if results[model]["total"] > 0:
            results[model]["success_rate"] = results[model]["passed"] / results[model]["total"] * 100
        else:
            results[model]["success_rate"] = 0
    
    # Save raw results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results, prompts, models

def create_visualizations(results, prompts, models, show_video_sizes=False):
    """Create bar chart and table visualizations."""
    # Create bar chart
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    success_rates = [results[m]["success_rate"] for m in model_names]
    
    bars = plt.bar(model_names, success_rates, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'][:len(model_names)])
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.title('Manim Code Generation Benchmark Results', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    
    # Add counts below model names
    ax = plt.gca()
    for i, model in enumerate(model_names):
        passed = results[model]["passed"]
        total = results[model]["total"]
        ax.text(i, -5, f'({passed}/{total})', ha='center', va='top', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved bar chart to benchmark_results.png")
    
    # Create detailed table
    table_data = []
    if show_video_sizes:
        headers = ["Prompt"] + [f"{model}\n(Status | Size)" for model in models]
    else:
        headers = ["Prompt"] + models
    
    for prompt in prompts:
        row = [prompt[:40] + "..." if len(prompt) > 40 else prompt]
        for model in models:
            if prompt in results[model]["details"]:
                success = results[model]["details"][prompt]["success"]
                video_size = results[model]["details"][prompt].get("video_size")
                
                if show_video_sizes and video_size is not None:
                    # Format file size in KB or MB
                    if video_size > 1024 * 1024:
                        size_str = f"{video_size / (1024 * 1024):.1f}MB"
                    else:
                        size_str = f"{video_size / 1024:.0f}KB"
                    row.append(f"✓ | {size_str}" if success else "✗")
                else:
                    row.append("✓" if success else "✗")
            else:
                row.append("-")
        table_data.append(row)
    
    # Add summary row
    table_data.append(["---"] * len(headers))
    summary_row = ["TOTAL"]
    for model in models:
        rate = results[model]["success_rate"]
        passed = results[model]["passed"]
        total = results[model]["total"]
        summary_row.append(f"{rate:.1f}% ({passed}/{total})")
    table_data.append(summary_row)
    
    # Print and save table
    table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    print("\n" + table_str)
    
    with open("benchmark_table.txt", "w") as f:
        f.write(f"Manim Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_str)
        f.write("\n\n")
        
        # Add model ranking
        f.write("Model Ranking by Success Rate:\n")
        f.write("-" * 40 + "\n")
        ranked_models = sorted(models, key=lambda m: results[m]["success_rate"], reverse=True)
        for i, model in enumerate(ranked_models, 1):
            f.write(f"{i}. {model}: {results[model]['success_rate']:.1f}%\n")
    
    print("\n✓ Saved detailed table to benchmark_table.txt")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manim Code Generation Benchmark Tool")
    parser.add_argument("--generate-videos", action="store_true", 
                        help="Actually generate video files (slower but shows file sizes)")
    args = parser.parse_args()
    
    print("🚀 Manim Code Generation Benchmark Tool")
    print("=" * 40)
    if args.generate_videos:
        print("📹 Video generation mode enabled (this will be slower)")
    
    # Read models from file to know what to pull
    if Path("models.txt").exists():
        models_to_pull = [m.strip() for m in Path("models.txt").read_text().strip().split('\n') if m.strip()]
    else:
        models_to_pull = ["qwen2.5-coder:7b", "llama3.2:3b"]
    
    # Use context manager for automatic Ollama lifecycle management
    try:
        with OllamaManager(required_models=models_to_pull) as ollama_mgr:
            results, prompts, models = run_benchmark(generate_videos=args.generate_videos)
            create_visualizations(results, prompts, models, show_video_sizes=args.generate_videos)
            
            print("\n📊 Summary:")
            print("-" * 40)
            for model in models:
                rate = results[model]["success_rate"]
                print(f"{model}: {rate:.1f}% success rate")
            
            print("\n✅ Benchmark complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()