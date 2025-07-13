#!/usr/bin/env python3
"""
Show exactly how quality scores are calculated with real examples.
"""

import json
import ast
from pathlib import Path

def calculate_quality_score(desc, code):
    """
    Calculate quality score exactly as done in the analysis.
    This is from compare_thanks_dataset.py
    """
    quality_score = 100
    issues = []
    
    # 1. Syntax error check (-30 points)
    try:
        ast.parse(code)
    except SyntaxError:
        quality_score -= 30
        issues.append("syntax_error (-30)")
    
    # 2. Short description (-10 points)
    if len(desc) < 50:
        quality_score -= 10
        issues.append(f"short_desc ({len(desc)} chars) (-10)")
    
    # 3. Short code (-10 points)
    if len(code) < 200:
        quality_score -= 10
        issues.append(f"short_code ({len(code)} chars) (-10)")
    
    # 4. Missing imports (-15 points)
    if "import" not in code:
        quality_score -= 15
        issues.append("no_imports (-15)")
    
    # 5. No animation methods (-5 points)
    animation_keywords = ["play", "wait", "add", "animate", "transform", "move_to"]
    if not any(kw in code for kw in animation_keywords):
        quality_score -= 5
        issues.append("no_animation (-5)")
    
    # 6. No math objects (-5 points)
    math_objects = ["Text", "Tex", "MathTex", "Circle", "Square", "Dot", "Arrow", "Axes"]
    if not any(obj in code for obj in math_objects):
        quality_score -= 5
        issues.append("no_objects (-5)")
    
    # 7. Placeholders (-10 points)
    if any(marker in code for marker in ["TODO", "FIXME", "..."]):
        quality_score -= 10
        issues.append("placeholder (-10)")
    
    return quality_score, issues

def main():
    # Load a few samples
    train_file = Path("data_formatted_v2/train.json")
    thanks_samples = []
    
    with open(train_file) as f:
        for line in f:
            sample = json.loads(line)
            if sample.get('source') == 'thanks_dataset':
                thanks_samples.append(sample)
                if len(thanks_samples) >= 20:
                    break
    
    print("="*100)
    print("HOW QUALITY SCORES ARE CALCULATED")
    print("="*100)
    
    print("\nScoring system (starts at 100):")
    print("  - Syntax error: -30 points")
    print("  - Short description (<50 chars): -10 points")
    print("  - Short code (<200 chars): -10 points")  
    print("  - Missing imports: -15 points")
    print("  - No animation methods (play, wait, etc): -5 points")
    print("  - No math objects (Text, Circle, etc): -5 points")
    print("  - Has placeholders (TODO, FIXME, ...): -10 points")
    
    print("\n" + "="*100)
    print("REAL EXAMPLES WITH SCORE BREAKDOWN")
    print("="*100)
    
    # Find samples with different scores
    score_examples = {
        "100": None,
        "90-99": None,
        "80-89": None,
        "70-79": None,
        "<70": None
    }
    
    for sample in thanks_samples:
        desc = sample['conversations'][1]['value']
        code = sample['conversations'][2]['value']
        
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        score, issues = calculate_quality_score(desc, code)
        
        # Categorize
        if score == 100 and score_examples["100"] is None:
            score_examples["100"] = (desc, code, score, issues)
        elif 90 <= score < 100 and score_examples["90-99"] is None:
            score_examples["90-99"] = (desc, code, score, issues)
        elif 80 <= score < 90 and score_examples["80-89"] is None:
            score_examples["80-89"] = (desc, code, score, issues)
        elif 70 <= score < 80 and score_examples["70-79"] is None:
            score_examples["70-79"] = (desc, code, score, issues)
        elif score < 70 and score_examples["<70"] is None:
            score_examples["<70"] = (desc, code, score, issues)
    
    # Show examples
    for category, example in score_examples.items():
        if example:
            desc, code, score, issues = example
            print(f"\n{'='*80}")
            print(f"SCORE: {score}/100 (Category: {category})")
            print(f"{'='*80}")
            
            print(f"\nDescription ({len(desc)} chars):")
            print(desc[:150] + "..." if len(desc) > 150 else desc)
            
            print(f"\nCode ({len(code)} chars):")
            print("```python")
            print(code[:300] + "..." if len(code) > 300 else code)
            print("```")
            
            print(f"\nScore calculation:")
            print(f"  Starting score: 100")
            for issue in issues:
                print(f"  {issue}")
            print(f"  Final score: {score}")
            
            print(f"\nThis sample would:")
            if score >= 85:
                print("  ✓ Be kept with a quality threshold of 85")
            else:
                print("  ✗ Be filtered with a quality threshold of 85")
    
    print("\n" + "="*100)
    print("WHAT THIS MEANS FOR FILTERING")
    print("="*100)
    
    print("\nIf you set quality threshold to 85:")
    print("  - Keeps samples with score 85-100")
    print("  - Main reasons for scoring below 85:")
    print("    • Missing imports alone = 85 (100-15)")
    print("    • Short code + no animation = 85 (100-10-5)")
    print("    • Any 2 major issues typically drops below 85")
    
    print("\nIf you set quality threshold to 90:")
    print("  - Only minor issues allowed")
    print("  - Filters out samples missing imports")
    print("  - Very strict - removes ~20-30% of data")
    
    print("\nIf you set quality threshold to 80:")
    print("  - Allows samples with 2 issues")
    print("  - Good balance of quality and quantity")
    print("  - Filters out seriously flawed samples")

if __name__ == "__main__":
    main()