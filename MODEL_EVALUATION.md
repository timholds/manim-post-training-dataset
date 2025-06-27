# Model Evaluation Guide for Manim Fine-Tuning

## Overview
This guide covers comprehensive evaluation methods for Manim-generating models, from basic functionality tests to advanced quality metrics.

## Quick Evaluation

### 1. Basic Functionality Test
```bash
# Test with simple prompts
python test_inference.py --model ./outputs/qwen-1.5b \
  --prompt "Create a blue circle that fades in"

# Test with Ollama
ollama run manim-coder "Draw a red square that rotates"
```

### 2. Automated Test Suite
```bash
python evaluate_model.py --model ./outputs/qwen-1.5b \
  --test-set data_formatted/test.json \
  --output-dir evaluation_results/
```

## Evaluation Metrics

### 1. Code Validity (Most Important)
**Syntax Validity**: Can the code be parsed?
```python
def check_syntax(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
```

**Manim Execution**: Does the code run without errors?
```python
def check_manim_execution(code):
    try:
        # Save to temp file
        with open('temp_scene.py', 'w') as f:
            f.write(code)
        
        # Try to render
        result = subprocess.run(
            ['manim', 'temp_scene.py', '-ql', '--dry_run'],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False
```

### 2. Code Quality Metrics

**Structure Compliance**:
- Has proper imports
- Contains Scene class
- Has construct method
- Follows Manim conventions

**Complexity Metrics**:
- Number of Manim objects used
- Animation variety
- Code length vs instruction complexity

### 3. Semantic Accuracy
Does the generated code match the instruction?

```python
def evaluate_semantic_match(instruction, generated_code):
    # Extract key terms from instruction
    instruction_lower = instruction.lower()
    
    # Check for mentioned objects
    if "circle" in instruction_lower:
        if "Circle" not in generated_code:
            return 0.0
    
    # Check for mentioned colors
    if "red" in instruction_lower:
        if "RED" not in generated_code and "#ff0000" not in generated_code:
            return 0.5
    
    # Check for animations
    if "rotate" in instruction_lower:
        if "Rotate" not in generated_code and "rotate" not in generated_code:
            return 0.5
    
    return 1.0
```

## Comprehensive Evaluation Pipeline

### Create Evaluation Script
```python
#!/usr/bin/env python3
"""evaluate_model.py - Comprehensive model evaluation"""

import json
import subprocess
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import tempfile
import os

class ManimModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = []
        
    def evaluate_dataset(self, test_file: str) -> Dict:
        """Evaluate model on entire test dataset."""
        with open(test_file) as f:
            test_data = [json.loads(line) for line in f]
        
        for item in test_data:
            instruction = item['conversations'][1]['value']
            expected = item['conversations'][2]['value']
            
            # Generate code
            generated = self.generate_code(instruction)
            
            # Evaluate
            scores = self.evaluate_single(instruction, generated, expected)
            self.results.append({
                'instruction': instruction,
                'generated': generated,
                'expected': expected,
                **scores
            })
        
        return self.compute_summary()
    
    def evaluate_single(self, instruction: str, generated: str, expected: str = None) -> Dict:
        """Evaluate a single generation."""
        scores = {}
        
        # Extract code from markdown if needed
        code = self.extract_code(generated)
        
        # 1. Syntax validity
        scores['syntax_valid'] = self.check_syntax(code)
        
        # 2. Manim executability
        scores['manim_executable'] = self.check_manim_execution(code)
        
        # 3. Structure compliance
        scores['has_imports'] = 'from manim import' in code or 'import manim' in code
        scores['has_scene_class'] = 'class' in code and 'Scene' in code
        scores['has_construct'] = 'def construct' in code
        
        # 4. Semantic accuracy
        scores['semantic_score'] = self.evaluate_semantic_match(instruction, code)
        
        # 5. Code quality
        scores['code_length'] = len(code)
        scores['num_animations'] = self.count_animations(code)
        scores['num_mobjects'] = self.count_mobjects(code)
        
        # 6. Similarity to expected (if provided)
        if expected:
            scores['similarity_score'] = self.calculate_similarity(code, expected)
        
        return scores
    
    def check_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except:
            return False
    
    def check_manim_execution(self, code: str) -> bool:
        """Check if code executes in Manim without errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['manim', temp_file, '-ql', '--dry_run'],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False
        finally:
            os.unlink(temp_file)
    
    def count_animations(self, code: str) -> int:
        """Count number of animations in code."""
        animations = ['play', 'wait', 'add', 'remove', 'become', 'transform']
        count = sum(1 for anim in animations if f'.{anim}(' in code.lower())
        return count
    
    def count_mobjects(self, code: str) -> int:
        """Count number of Manim objects created."""
        mobjects = ['Circle', 'Square', 'Rectangle', 'Line', 'Arrow', 'Text', 
                   'MathTex', 'Dot', 'VGroup', 'Triangle', 'Polygon']
        count = sum(1 for obj in mobjects if obj in code)
        return count
    
    def evaluate_semantic_match(self, instruction: str, code: str) -> float:
        """Evaluate how well code matches instruction semantically."""
        score = 1.0
        instruction_lower = instruction.lower()
        
        # Define mappings
        object_mappings = {
            'circle': 'Circle',
            'square': 'Square',
            'line': 'Line',
            'text': 'Text',
            'arrow': 'Arrow',
            'triangle': 'Triangle'
        }
        
        color_mappings = {
            'red': ['RED', '#ff0000', '#FF0000'],
            'blue': ['BLUE', '#0000ff', '#0000FF'],
            'green': ['GREEN', '#00ff00', '#00FF00'],
            'yellow': ['YELLOW', '#ffff00', '#FFFF00']
        }
        
        animation_mappings = {
            'fade': ['FadeIn', 'FadeOut'],
            'rotate': ['Rotate', 'rotate'],
            'scale': ['Scale', 'scale'],
            'move': ['shift', 'move_to', 'animate.shift'],
            'grow': ['GrowFromCenter', 'grow']
        }
        
        # Check objects
        for term, manim_class in object_mappings.items():
            if term in instruction_lower and manim_class not in code:
                score *= 0.8
        
        # Check colors
        for color, variations in color_mappings.items():
            if color in instruction_lower:
                if not any(var in code for var in variations):
                    score *= 0.9
        
        # Check animations
        for action, implementations in animation_mappings.items():
            if action in instruction_lower:
                if not any(impl in code for impl in implementations):
                    score *= 0.85
        
        return score
    
    def compute_summary(self) -> Dict:
        """Compute summary statistics from all results."""
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_samples': len(df),
            'syntax_valid_rate': df['syntax_valid'].mean(),
            'executable_rate': df['manim_executable'].mean(),
            'has_imports_rate': df['has_imports'].mean(),
            'has_scene_class_rate': df['has_scene_class'].mean(),
            'has_construct_rate': df['has_construct'].mean(),
            'avg_semantic_score': df['semantic_score'].mean(),
            'avg_code_length': df['code_length'].mean(),
            'avg_animations': df['num_animations'].mean(),
            'avg_mobjects': df['num_mobjects'].mean()
        }
        
        # Success tiers
        df['fully_valid'] = df['syntax_valid'] & df['manim_executable']
        df['structurally_correct'] = df['has_imports'] & df['has_scene_class'] & df['has_construct']
        df['high_quality'] = df['fully_valid'] & df['structurally_correct'] & (df['semantic_score'] > 0.8)
        
        summary['fully_valid_rate'] = df['fully_valid'].mean()
        summary['structurally_correct_rate'] = df['structurally_correct'].mean()
        summary['high_quality_rate'] = df['high_quality'].mean()
        
        return summary
    
    def generate_report(self, output_dir: str):
        """Generate detailed evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save raw results
        df = pd.DataFrame(self.results)
        df.to_csv(output_dir / 'evaluation_results.csv', index=False)
        
        # Save summary
        summary = self.compute_summary()
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate report
        report = self.format_report(summary)
        with open(output_dir / 'report.md', 'w') as f:
            f.write(report)
        
        print(f"Evaluation complete. Results saved to {output_dir}")
        print("\nSummary:")
        print(report)
    
    def format_report(self, summary: Dict) -> str:
        """Format summary into readable report."""
        report = f"""# Model Evaluation Report

## Overview
- Total samples evaluated: {summary['total_samples']}
- Model: {self.model_path}

## Success Metrics
- **Syntax Valid**: {summary['syntax_valid_rate']:.1%}
- **Manim Executable**: {summary['executable_rate']:.1%}
- **Fully Valid**: {summary['fully_valid_rate']:.1%}
- **High Quality**: {summary['high_quality_rate']:.1%}

## Structure Compliance
- Has imports: {summary['has_imports_rate']:.1%}
- Has Scene class: {summary['has_scene_class_rate']:.1%}
- Has construct method: {summary['has_construct_rate']:.1%}
- Structurally correct: {summary['structurally_correct_rate']:.1%}

## Quality Metrics
- Average semantic score: {summary['avg_semantic_score']:.2f}
- Average code length: {summary['avg_code_length']:.0f} chars
- Average animations per scene: {summary['avg_animations']:.1f}
- Average objects per scene: {summary['avg_mobjects']:.1f}

## Performance Grade
"""
        
        # Assign grade
        score = (summary['fully_valid_rate'] * 0.4 + 
                summary['structurally_correct_rate'] * 0.3 +
                summary['avg_semantic_score'] * 0.3)
        
        if score >= 0.9:
            grade = "A - Excellent"
        elif score >= 0.8:
            grade = "B - Good"
        elif score >= 0.7:
            grade = "C - Satisfactory"
        elif score >= 0.6:
            grade = "D - Needs Improvement"
        else:
            grade = "F - Poor"
        
        report += f"**Overall Grade: {grade}** (Score: {score:.1%})\n"
        
        return report
```

## Testing Strategies

### 1. Unit Tests for Specific Capabilities
```python
# Test basic shapes
shape_tests = [
    "Create a red circle",
    "Draw a blue square", 
    "Make a green triangle",
    "Show a yellow rectangle"
]

# Test animations
animation_tests = [
    "Create a circle that fades in",
    "Make a square that rotates 90 degrees",
    "Draw a line that grows from left to right",
    "Create text that scales up"
]

# Test complex scenes
complex_tests = [
    "Create three circles in a row and make them bounce",
    "Draw a coordinate system with labeled axes",
    "Animate a ball rolling down a ramp",
    "Show the Pythagorean theorem with visual proof"
]
```

### 2. Regression Testing
Keep a set of prompts that should always work:
```python
REGRESSION_TESTS = [
    {
        "prompt": "Create a circle",
        "must_contain": ["Circle()", "self.play", "self.add"]
    },
    {
        "prompt": "Write Hello World text",
        "must_contain": ["Text", "Hello World", "self.play"]
    }
]
```

### 3. Edge Case Testing
```python
EDGE_CASES = [
    "",  # Empty prompt
    "Create a circle" * 100,  # Very long prompt
    "Create a 🔵 circle",  # Unicode
    "Make something beautiful",  # Vague instruction
    "Debug this: Circle().shift(UP)",  # Code in prompt
]
```

## Continuous Evaluation

### Automated Pipeline
```bash
#!/bin/bash
# evaluate_pipeline.sh

# Train model
python fine_tune.py --model $1

# Convert to Ollama
python convert_to_ollama.py outputs/$1

# Run evaluation
python evaluate_model.py --model outputs/$1 \
  --test-set data_formatted/test.json \
  --output-dir evaluation_results/$1

# Compare with baseline
python compare_models.py \
  --baseline outputs/baseline \
  --new-model outputs/$1 \
  --output comparison_report.md
```

### Evaluation Metrics to Track
1. **Validity Rate**: % of syntactically valid outputs
2. **Execution Rate**: % that run in Manim without errors
3. **Semantic Accuracy**: How well output matches instruction
4. **Code Quality**: Length, complexity, style
5. **Generation Speed**: Tokens per second
6. **Model Size**: Parameters and disk usage

## Human Evaluation

### Visual Quality Assessment
```bash
# Generate videos for human review
python generate_test_videos.py --model outputs/qwen-1.5b \
  --prompts test_prompts.txt \
  --output-dir test_videos/

# Create side-by-side comparisons
python create_comparison_grid.py \
  --videos test_videos/*.mp4 \
  --output comparison.mp4
```

### Evaluation Rubric
Rate each generation on 1-5 scale:
1. **Correctness**: Does it do what was asked?
2. **Visual Quality**: Is the animation smooth and appealing?
3. **Code Quality**: Is the code clean and efficient?
4. **Creativity**: Does it show appropriate creativity?
5. **Completeness**: Is it a complete, working solution?

## Model Comparison

### A/B Testing Framework
```python
def compare_models(model_a, model_b, test_prompts):
    results = []
    
    for prompt in test_prompts:
        code_a = generate_with_model(model_a, prompt)
        code_b = generate_with_model(model_b, prompt)
        
        # Compare execution success
        exec_a = check_manim_execution(code_a)
        exec_b = check_manim_execution(code_b)
        
        # Compare code quality
        quality_a = evaluate_code_quality(code_a)
        quality_b = evaluate_code_quality(code_b)
        
        results.append({
            'prompt': prompt,
            'model_a_success': exec_a,
            'model_b_success': exec_b,
            'model_a_quality': quality_a,
            'model_b_quality': quality_b
        })
    
    return results
```

## Best Practices

1. **Evaluate Early and Often**: Run quick tests during training
2. **Use Diverse Test Sets**: Include easy, medium, and hard prompts
3. **Track Metrics Over Time**: Monitor improvement across versions
4. **Balance Metrics**: Don't optimize for one metric at expense of others
5. **Include Real User Prompts**: Test with actual use cases
6. **Visual Inspection**: Always manually review some outputs
7. **Version Everything**: Keep evaluation results for each model version

## Red Flags to Watch For

1. **Syntax validity < 80%**: Model hasn't learned Python properly
2. **Execution rate < 60%**: Model doesn't understand Manim
3. **Semantic score < 0.7**: Model ignores instructions
4. **Very short outputs**: Model might be collapsing
5. **Repetitive code**: Overfitting or degradation
6. **Import errors**: Formatting issues in training data