#!/usr/bin/env python3
"""
Comprehensive quality analysis of all data sources.
Provides detailed metrics to help decide which sources to keep/remove.
"""

import json
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceAnalyzer:
    """Analyze individual data sources for quality metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "total_samples": 0,
            "syntax_errors": 0,
            "missing_imports": 0,
            "empty_constructs": 0,
            "placeholder_code": 0,
            "short_descriptions": 0,
            "short_code": 0,
            "no_animation_methods": 0,
            "no_math_objects": 0,
            "avg_description_length": 0,
            "avg_code_length": 0,
            "unique_descriptions": set(),
            "code_patterns": Counter(),
            "error_examples": [],
            "good_examples": [],
            "description_lengths": [],
            "code_lengths": []
        })
    
    def analyze_sample(self, sample: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Analyze a single sample and update metrics."""
        metrics = self.metrics[source]
        metrics["total_samples"] += 1
        
        # Extract description and code
        desc = ""
        code = ""
        
        if 'conversations' in sample and len(sample['conversations']) >= 3:
            desc = sample['conversations'][1].get('value', '')
            code = sample['conversations'][2].get('value', '')
            
            # Clean markdown from code
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
        
        # Track lengths
        metrics["description_lengths"].append(len(desc))
        metrics["code_lengths"].append(len(code))
        
        # Check for various issues
        issues = []
        
        # 1. Syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            metrics["syntax_errors"] += 1
            issues.append(f"Syntax error: {str(e)[:50]}")
            if len(metrics["error_examples"]) < 3:
                metrics["error_examples"].append({
                    "description": desc[:100],
                    "code": code[:200],
                    "error": str(e)
                })
        
        # 2. Missing imports
        if "import" not in code:
            metrics["missing_imports"] += 1
            issues.append("Missing imports")
        
        # 3. Empty constructs
        if "def construct(self):" in code and ("pass" in code or "..." in code):
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if "def construct(self):" in line:
                    next_lines = lines[i+1:i+5]
                    if all(l.strip() in ["pass", "...", ""] for l in next_lines):
                        metrics["empty_constructs"] += 1
                        issues.append("Empty construct method")
                        break
        
        # 4. Placeholder code
        placeholder_patterns = [
            r"TODO", r"FIXME", r"XXX", r"HACK",
            r"# Your code here", r"# Implementation goes here",
            r"\.\.\.[ ]*$"
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                metrics["placeholder_code"] += 1
                issues.append(f"Contains placeholder: {pattern}")
                break
        
        # 5. Short descriptions
        if len(desc) < 30:
            metrics["short_descriptions"] += 1
            issues.append(f"Short description ({len(desc)} chars)")
        
        # 6. Short code
        if len(code) < 100:
            metrics["short_code"] += 1
            issues.append(f"Short code ({len(code)} chars)")
        
        # 7. Animation methods
        animation_methods = ["play", "wait", "add", "remove", "move_to", "shift", 
                           "scale", "rotate", "animate", "transform"]
        if not any(method in code for method in animation_methods):
            metrics["no_animation_methods"] += 1
            issues.append("No animation methods")
        
        # 8. Math objects
        math_objects = ["Text", "Tex", "MathTex", "Circle", "Square", "Line",
                       "Arrow", "Dot", "Graph", "Axes", "Vector", "Matrix",
                       "Mobject", "VMobject", "Group"]
        if not any(obj in code for obj in math_objects):
            metrics["no_math_objects"] += 1
            issues.append("No math objects")
        
        # Track unique descriptions
        metrics["unique_descriptions"].add(desc.lower().strip())
        
        # Track code patterns
        if "class" in code:
            class_matches = re.findall(r'class\s+(\w+)', code)
            for class_name in class_matches:
                metrics["code_patterns"][class_name] += 1
        
        # Save good examples
        if not issues and len(metrics["good_examples"]) < 3:
            metrics["good_examples"].append({
                "description": desc[:100],
                "code": code[:200]
            })
        
        return {"issues": issues}
    
    def generate_source_report(self, source: str) -> Dict[str, Any]:
        """Generate comprehensive report for a source."""
        m = self.metrics[source]
        
        if m["total_samples"] == 0:
            return {"error": "No samples found"}
        
        # Calculate averages
        avg_desc_len = sum(m["description_lengths"]) / len(m["description_lengths"]) if m["description_lengths"] else 0
        avg_code_len = sum(m["code_lengths"]) / len(m["code_lengths"]) if m["code_lengths"] else 0
        
        # Calculate percentages
        report = {
            "total_samples": m["total_samples"],
            "unique_descriptions": len(m["unique_descriptions"]),
            "duplicate_rate": (m["total_samples"] - len(m["unique_descriptions"])) / m["total_samples"] * 100,
            
            "quality_issues": {
                "syntax_errors": {
                    "count": m["syntax_errors"],
                    "percentage": m["syntax_errors"] / m["total_samples"] * 100
                },
                "missing_imports": {
                    "count": m["missing_imports"],
                    "percentage": m["missing_imports"] / m["total_samples"] * 100
                },
                "empty_constructs": {
                    "count": m["empty_constructs"],
                    "percentage": m["empty_constructs"] / m["total_samples"] * 100
                },
                "placeholder_code": {
                    "count": m["placeholder_code"],
                    "percentage": m["placeholder_code"] / m["total_samples"] * 100
                },
                "short_descriptions": {
                    "count": m["short_descriptions"],
                    "percentage": m["short_descriptions"] / m["total_samples"] * 100
                },
                "short_code": {
                    "count": m["short_code"],
                    "percentage": m["short_code"] / m["total_samples"] * 100
                },
                "no_animation_methods": {
                    "count": m["no_animation_methods"],
                    "percentage": m["no_animation_methods"] / m["total_samples"] * 100
                },
                "no_math_objects": {
                    "count": m["no_math_objects"],
                    "percentage": m["no_math_objects"] / m["total_samples"] * 100
                }
            },
            
            "statistics": {
                "avg_description_length": avg_desc_len,
                "avg_code_length": avg_code_len,
                "min_description_length": min(m["description_lengths"]) if m["description_lengths"] else 0,
                "max_description_length": max(m["description_lengths"]) if m["description_lengths"] else 0,
                "min_code_length": min(m["code_lengths"]) if m["code_lengths"] else 0,
                "max_code_length": max(m["code_lengths"]) if m["code_lengths"] else 0
            },
            
            "examples": {
                "errors": m["error_examples"][:3],
                "good": m["good_examples"][:3]
            },
            
            "common_classes": m["code_patterns"].most_common(10)
        }
        
        # Calculate overall quality score (0-100)
        penalties = {
            "syntax_errors": 30,
            "missing_imports": 15,
            "empty_constructs": 20,
            "placeholder_code": 15,
            "short_descriptions": 5,
            "short_code": 5,
            "no_animation_methods": 5,
            "no_math_objects": 5
        }
        
        quality_score = 100
        for issue, penalty in penalties.items():
            issue_rate = m[issue] / m["total_samples"]
            quality_score -= issue_rate * penalty
        
        report["quality_score"] = max(0, quality_score)
        
        return report


def main():
    """Analyze all sources and generate comprehensive report."""
    data_dir = Path("data_formatted_v2")
    train_file = data_dir / "train.json"
    
    if not train_file.exists():
        logger.error("No training data found")
        return
    
    analyzer = SourceAnalyzer()
    
    logger.info("Analyzing all data sources...")
    
    # Process all samples
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                logger.info(f"Processed {i} samples...")
            
            try:
                sample = json.loads(line.strip())
                source = sample.get('source', 'unknown')
                analyzer.analyze_sample(sample, source)
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
    
    # Generate reports for all sources
    all_reports = {}
    sources = sorted(analyzer.metrics.keys())
    
    for source in sources:
        all_reports[source] = analyzer.generate_source_report(source)
    
    # Print comprehensive report
    print("\n" + "="*100)
    print("COMPREHENSIVE DATA SOURCE QUALITY ANALYSIS")
    print("="*100)
    
    # Summary table
    print("\nQUALITY SUMMARY")
    print("-"*100)
    print(f"{'Source':<20} {'Samples':<10} {'Quality':<10} {'Syntax Err':<12} {'No Imports':<12} {'Empty':<10} {'Recommend'}")
    print("-"*100)
    
    for source in sources:
        r = all_reports[source]
        if "error" in r:
            continue
        
        q = r["quality_issues"]
        score = r["quality_score"]
        
        # Recommendation
        if score >= 90:
            recommend = "KEEP (High)"
        elif score >= 70:
            recommend = "KEEP (Good)"
        elif score >= 50:
            recommend = "REVIEW"
        else:
            recommend = "REMOVE/FIX"
        
        print(f"{source:<20} {r['total_samples']:<10} {score:<10.1f} "
              f"{q['syntax_errors']['percentage']:<11.1f}% "
              f"{q['missing_imports']['percentage']:<11.1f}% "
              f"{q['empty_constructs']['percentage']:<9.1f}% "
              f"{recommend}")
    
    # Detailed source analysis
    print("\n\nDETAILED SOURCE ANALYSIS")
    print("="*100)
    
    for source in sources:
        r = all_reports[source]
        if "error" in r:
            continue
        
        print(f"\n## {source.upper()}")
        print(f"Quality Score: {r['quality_score']:.1f}/100")
        print(f"Total Samples: {r['total_samples']}")
        print(f"Unique Descriptions: {r['unique_descriptions']} ({100-r['duplicate_rate']:.1f}% unique)")
        
        print("\nIssue Breakdown:")
        for issue, data in r["quality_issues"].items():
            if data["count"] > 0:
                print(f"  - {issue}: {data['count']} samples ({data['percentage']:.1f}%)")
        
        print("\nCode Statistics:")
        stats = r["statistics"]
        print(f"  - Avg description length: {stats['avg_description_length']:.0f} chars (range: {stats['min_description_length']}-{stats['max_description_length']})")
        print(f"  - Avg code length: {stats['avg_code_length']:.0f} chars (range: {stats['min_code_length']}-{stats['max_code_length']})")
        
        if r["examples"]["errors"]:
            print("\nExample Errors:")
            for ex in r["examples"]["errors"][:2]:
                print(f"  - Description: {ex['description'][:50]}...")
                print(f"    Error: {ex['error']}")
        
        if r["common_classes"]:
            print("\nMost Common Classes:")
            for class_name, count in r["common_classes"][:5]:
                print(f"  - {class_name}: {count} times")
    
    # Final recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    for source in sources:
        r = all_reports[source]
        if "error" in r:
            continue
        
        score = r["quality_score"]
        q = r["quality_issues"]
        
        print(f"\n{source}:")
        
        if score >= 90:
            print("  ✓ KEEP - High quality source")
        elif score >= 70:
            print("  ✓ KEEP - Good quality, minor issues")
        elif score >= 50:
            print("  ? REVIEW - Significant issues, consider cleaning or filtering")
            if q["syntax_errors"]["percentage"] > 20:
                print("    - Major issue: High syntax error rate")
            if q["missing_imports"]["percentage"] > 30:
                print("    - Major issue: Many samples missing imports")
        else:
            print("  ✗ REMOVE or MAJOR CLEANUP REQUIRED")
            print("    - Quality too low for effective training")
            print(f"    - Would lose {r['total_samples']} samples")
    
    # Save detailed report
    with open("source_quality_report.json", 'w') as f:
        json.dump(all_reports, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to source_quality_report.json")


if __name__ == "__main__":
    main()