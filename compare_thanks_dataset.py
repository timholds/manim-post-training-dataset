#!/usr/bin/env python3
"""
Detailed comparison of thanks_dataset vs other sources.
"""

import json
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
import statistics

def analyze_source_quality(file_path: Path):
    """Analyze quality metrics for each source."""
    
    source_metrics = defaultdict(lambda: {
        "samples": [],
        "desc_lengths": [],
        "code_lengths": [],
        "issues": Counter(),
        "examples": {"good": [], "bad": []},
        "unique_patterns": set(),
        "quality_scores": []
    })
    
    with open(file_path) as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                source = sample.get('source', 'unknown')
                metrics = source_metrics[source]
                
                if 'conversations' in sample and len(sample['conversations']) >= 3:
                    desc = sample['conversations'][1].get('value', '')
                    code = sample['conversations'][2].get('value', '')
                    
                    # Clean code
                    if '```python' in code:
                        code = code.split('```python')[1].split('```')[0].strip()
                    elif '```' in code:
                        code = code.split('```')[1].split('```')[0].strip()
                    
                    metrics["desc_lengths"].append(len(desc))
                    metrics["code_lengths"].append(len(code))
                    
                    # Quality score (0-100)
                    quality_score = 100
                    issues = []
                    
                    # Check various quality aspects
                    try:
                        ast.parse(code)
                    except SyntaxError:
                        quality_score -= 30
                        issues.append("syntax_error")
                        metrics["issues"]["syntax_errors"] += 1
                    
                    if len(desc) < 50:
                        quality_score -= 10
                        issues.append("short_desc")
                        metrics["issues"]["short_descriptions"] += 1
                    
                    if len(code) < 200:
                        quality_score -= 10
                        issues.append("short_code")
                        metrics["issues"]["short_code"] += 1
                    
                    if "import" not in code:
                        quality_score -= 15
                        issues.append("no_imports")
                        metrics["issues"]["missing_imports"] += 1
                    
                    # Check for animation methods
                    animation_keywords = ["play", "wait", "add", "animate", "transform", "move_to"]
                    if not any(kw in code for kw in animation_keywords):
                        quality_score -= 5
                        issues.append("no_animation")
                        metrics["issues"]["no_animation_methods"] += 1
                    
                    # Check for math objects
                    math_objects = ["Text", "Tex", "MathTex", "Circle", "Square", "Dot", "Arrow", "Axes"]
                    if not any(obj in code for obj in math_objects):
                        quality_score -= 5
                        issues.append("no_objects")
                        metrics["issues"]["no_math_objects"] += 1
                    
                    # Check for placeholders
                    if any(marker in code for marker in ["TODO", "FIXME", "..."]):
                        quality_score -= 10
                        issues.append("placeholder")
                        metrics["issues"]["has_placeholders"] += 1
                    
                    # Extract unique patterns
                    classes = re.findall(r'class\s+(\w+)', code)
                    for cls in classes:
                        metrics["unique_patterns"].add(cls)
                    
                    metrics["quality_scores"].append(quality_score)
                    
                    # Store examples
                    sample_data = {
                        "desc": desc[:100] + "..." if len(desc) > 100 else desc,
                        "code_preview": code[:200] + "..." if len(code) > 200 else code,
                        "issues": issues,
                        "score": quality_score
                    }
                    
                    if quality_score >= 90 and len(metrics["examples"]["good"]) < 3:
                        metrics["examples"]["good"].append(sample_data)
                    elif quality_score < 60 and len(metrics["examples"]["bad"]) < 3:
                        metrics["examples"]["bad"].append(sample_data)
                    
                    metrics["samples"].append(sample_data)
                    
            except Exception as e:
                pass
    
    return source_metrics

def print_comparison_report(source_metrics):
    """Print detailed comparison report."""
    
    print("="*100)
    print("DETAILED SOURCE COMPARISON REPORT")
    print("="*100)
    
    # Calculate summary statistics for each source
    summary = {}
    for source, metrics in source_metrics.items():
        if not metrics["samples"]:
            continue
            
        summary[source] = {
            "total_samples": len(metrics["samples"]),
            "avg_quality": statistics.mean(metrics["quality_scores"]) if metrics["quality_scores"] else 0,
            "avg_desc_len": statistics.mean(metrics["desc_lengths"]) if metrics["desc_lengths"] else 0,
            "avg_code_len": statistics.mean(metrics["code_lengths"]) if metrics["code_lengths"] else 0,
            "unique_classes": len(metrics["unique_patterns"]),
            "main_issues": metrics["issues"].most_common(3)
        }
    
    # Print comparison table
    print("\nQUALITY COMPARISON TABLE")
    print("-"*100)
    print(f"{'Source':<20} {'Samples':<10} {'Avg Quality':<12} {'Avg Desc Len':<15} {'Avg Code Len':<15} {'Top Issues'}")
    print("-"*100)
    
    for source in sorted(summary.keys(), key=lambda x: summary[x]["avg_quality"], reverse=True):
        s = summary[source]
        issues_str = ", ".join([f"{issue[0]}({issue[1]})" for issue in s["main_issues"][:2]])
        
        print(f"{source:<20} {s['total_samples']:<10} {s['avg_quality']:<12.1f} "
              f"{s['avg_desc_len']:<15.0f} {s['avg_code_len']:<15.0f} {issues_str}")
    
    # Detailed analysis of thanks_dataset
    print("\n" + "="*100)
    print("THANKS_DATASET DETAILED ANALYSIS")
    print("="*100)
    
    if "thanks_dataset" in source_metrics:
        thanks = source_metrics["thanks_dataset"]
        
        print(f"\nTotal samples: {len(thanks['samples'])}")
        print(f"Average quality score: {statistics.mean(thanks['quality_scores']):.1f}")
        print(f"Quality score range: {min(thanks['quality_scores'])}-{max(thanks['quality_scores'])}")
        
        print("\nQuality distribution:")
        score_ranges = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "<60": 0}
        for score in thanks["quality_scores"]:
            if score >= 90:
                score_ranges["90-100"] += 1
            elif score >= 80:
                score_ranges["80-89"] += 1
            elif score >= 70:
                score_ranges["70-79"] += 1
            elif score >= 60:
                score_ranges["60-69"] += 1
            else:
                score_ranges["<60"] += 1
        
        for range_name, count in score_ranges.items():
            pct = count / len(thanks['samples']) * 100
            print(f"  {range_name}: {count} samples ({pct:.1f}%)")
        
        print("\nIssue breakdown:")
        for issue, count in thanks["issues"].most_common():
            pct = count / len(thanks['samples']) * 100
            print(f"  {issue}: {count} samples ({pct:.1f}%)")
        
        print("\nGood examples from thanks_dataset:")
        for i, example in enumerate(thanks["examples"]["good"][:2], 1):
            print(f"\n  Example {i} (score: {example['score']}):")
            print(f"    Description: {example['desc']}")
            print(f"    Code preview: {example['code_preview'][:100]}...")
        
        print("\nProblematic examples from thanks_dataset:")
        for i, example in enumerate(thanks["examples"]["bad"][:2], 1):
            print(f"\n  Example {i} (score: {example['score']}, issues: {', '.join(example['issues'])}):")
            print(f"    Description: {example['desc']}")
            print(f"    Code preview: {example['code_preview'][:100]}...")
    
    # Compare with best source
    best_source = max(summary.keys(), key=lambda x: summary[x]["avg_quality"])
    if best_source != "thanks_dataset" and "thanks_dataset" in summary:
        print("\n" + "="*100)
        print(f"COMPARISON: thanks_dataset vs {best_source}")
        print("="*100)
        
        thanks_summary = summary["thanks_dataset"]
        best_summary = summary[best_source]
        
        print(f"\nQuality difference: {best_summary['avg_quality'] - thanks_summary['avg_quality']:.1f} points")
        print(f"Description length: thanks={thanks_summary['avg_desc_len']:.0f}, {best_source}={best_summary['avg_desc_len']:.0f}")
        print(f"Code length: thanks={thanks_summary['avg_code_len']:.0f}, {best_source}={best_summary['avg_code_len']:.0f}")
        
        # Calculate what we'd lose by removing thanks_dataset
        total_samples = sum(s["total_samples"] for s in summary.values())
        thanks_pct = thanks_summary["total_samples"] / total_samples * 100
        
        print(f"\nImpact of removing thanks_dataset:")
        print(f"  Would lose: {thanks_summary['total_samples']} samples ({thanks_pct:.1f}% of total)")
        print(f"  Remaining samples: {total_samples - thanks_summary['total_samples']}")
        
        # Calculate quality improvement
        current_avg_quality = sum(s["total_samples"] * s["avg_quality"] for s in summary.values()) / total_samples
        without_thanks_total = total_samples - thanks_summary["total_samples"]
        without_thanks_quality = (sum(s["total_samples"] * s["avg_quality"] for k, s in summary.items() if k != "thanks_dataset") / without_thanks_total)
        
        print(f"\nDataset quality impact:")
        print(f"  Current average quality: {current_avg_quality:.1f}")
        print(f"  Quality without thanks_dataset: {without_thanks_quality:.1f}")
        print(f"  Quality improvement: +{without_thanks_quality - current_avg_quality:.1f} points")

def main():
    """Run the analysis."""
    train_file = Path("data_formatted_v2/train.json")
    
    if not train_file.exists():
        print("No training data found")
        return
    
    print("Analyzing sources...")
    source_metrics = analyze_source_quality(train_file)
    print_comparison_report(source_metrics)
    
    # Save detailed metrics
    output = {}
    for source, metrics in source_metrics.items():
        output[source] = {
            "total_samples": len(metrics["samples"]),
            "avg_quality": statistics.mean(metrics["quality_scores"]) if metrics["quality_scores"] else 0,
            "issues": dict(metrics["issues"]),
            "avg_desc_length": statistics.mean(metrics["desc_lengths"]) if metrics["desc_lengths"] else 0,
            "avg_code_length": statistics.mean(metrics["code_lengths"]) if metrics["code_lengths"] else 0,
        }
    
    with open("thanks_dataset_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nDetailed metrics saved to thanks_dataset_comparison.json")

if __name__ == "__main__":
    main()