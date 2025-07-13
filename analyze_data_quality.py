#!/usr/bin/env python3
"""
Comprehensive data quality analysis tool for the Manim dataset.
Identifies various quality issues across all data sources.
"""

import json
import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyzes data quality issues in Manim datasets."""
    
    def __init__(self, data_dir: str = "data_formatted_v2"):
        self.data_dir = Path(data_dir)
        self.issues = defaultdict(list)
        self.stats = defaultdict(int)
        self.samples_by_source = defaultdict(int)
        
    def analyze_code_quality(self, code: str, sample_id: int, source: str) -> List[str]:
        """Analyze code for various quality issues."""
        issues = []
        
        # Basic checks
        if len(code) < 50:
            issues.append("code_too_short")
            self.issues["code_too_short"].append({
                "id": sample_id, 
                "source": source, 
                "length": len(code)
            })
        
        # Check for incomplete code markers
        if any(marker in code for marker in ["TODO", "FIXME", "XXX", "HACK"]):
            issues.append("incomplete_code")
            self.issues["incomplete_code"].append({
                "id": sample_id,
                "source": source
            })
        
        # Check for valid Python structure
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append("syntax_error")
            self.issues["syntax_error"].append({
                "id": sample_id,
                "source": source,
                "error": str(e)
            })
        
        # Check for imports
        if "import" not in code:
            issues.append("missing_imports")
            self.issues["missing_imports"].append({
                "id": sample_id,
                "source": source
            })
        
        # Check for Scene class
        if not any(pattern in code for pattern in ["class", "Scene", "construct"]):
            issues.append("missing_scene_structure")
            self.issues["missing_scene_structure"].append({
                "id": sample_id,
                "source": source
            })
        
        # Check for empty construct method
        if "def construct(self):" in code and "pass" in code:
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if "def construct(self):" in line:
                    # Check next few lines for only pass/whitespace
                    next_lines = lines[i+1:i+5]
                    if all(l.strip() in ["pass", ""] for l in next_lines):
                        issues.append("empty_construct")
                        self.issues["empty_construct"].append({
                            "id": sample_id,
                            "source": source
                        })
        
        # Check for placeholder content
        placeholder_patterns = [
            r"# Your code here",
            r"# Implementation goes here",
            r"# Add your",
            r"\.\.\.[ ]*$",  # Ellipsis at end of line
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append("placeholder_content")
                self.issues["placeholder_content"].append({
                    "id": sample_id,
                    "source": source,
                    "pattern": pattern
                })
                break
        
        return issues
    
    def analyze_description_quality(self, desc: str, sample_id: int, source: str) -> List[str]:
        """Analyze description for quality issues."""
        issues = []
        
        # Length check
        if len(desc) < 20:
            issues.append("description_too_short")
            self.issues["description_too_short"].append({
                "id": sample_id,
                "source": source,
                "length": len(desc)
            })
        
        # Check for balanced parentheses
        if desc.count('(') != desc.count(')'):
            issues.append("unbalanced_parentheses")
            self.issues["unbalanced_parentheses"].append({
                "id": sample_id,
                "source": source
            })
        
        # Check for generic descriptions
        generic_patterns = [
            r"^Create a Manim animation$",
            r"^Animation demonstrating$",
            r"^Create an animation$",
            r"^Manim scene$",
        ]
        for pattern in generic_patterns:
            if re.match(pattern, desc, re.IGNORECASE):
                issues.append("generic_description")
                self.issues["generic_description"].append({
                    "id": sample_id,
                    "source": source
                })
                break
        
        # Check for placeholder markers
        if any(marker in desc for marker in ["TODO", "FIXME", "[", "]", "INSERT", "PLACEHOLDER"]):
            issues.append("description_has_placeholders")
            self.issues["description_has_placeholders"].append({
                "id": sample_id,
                "source": source
            })
        
        return issues
    
    def check_code_description_mismatch(self, desc: str, code: str, sample_id: int, source: str) -> List[str]:
        """Check if code and description match."""
        issues = []
        
        # Extract class names from code
        class_names = re.findall(r'class\s+(\w+)', code)
        
        # Check if description mentions any concept from the code
        desc_lower = desc.lower()
        code_lower = code.lower()
        
        # Common mathematical concepts
        math_concepts = [
            "derivative", "integral", "matrix", "vector", "graph",
            "function", "equation", "theorem", "proof", "transform",
            "series", "sequence", "limit", "differential", "gradient"
        ]
        
        desc_has_math = any(concept in desc_lower for concept in math_concepts)
        code_has_math = any(concept in code_lower for concept in math_concepts)
        
        if code_has_math and not desc_has_math:
            issues.append("description_missing_math_context")
            self.issues["description_missing_math_context"].append({
                "id": sample_id,
                "source": source
            })
        
        return issues
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single JSONL file."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {}
        
        total_samples = 0
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                total_samples += 1
                try:
                    sample = json.loads(line.strip())
                    source = sample.get('source', 'unknown')
                    self.samples_by_source[source] += 1
                    
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
                    
                    # Analyze quality
                    code_issues = self.analyze_code_quality(code, i, source)
                    desc_issues = self.analyze_description_quality(desc, i, source)
                    mismatch_issues = self.check_code_description_mismatch(desc, code, i, source)
                    
                    # Track overall sample quality
                    all_issues = code_issues + desc_issues + mismatch_issues
                    if len(all_issues) == 0:
                        self.stats["perfect_samples"] += 1
                    elif len(all_issues) == 1:
                        self.stats["minor_issues"] += 1
                    elif len(all_issues) <= 3:
                        self.stats["moderate_issues"] += 1
                    else:
                        self.stats["severe_issues"] += 1
                    
                except Exception as e:
                    self.issues["parse_error"].append({
                        "id": i,
                        "error": str(e)
                    })
        
        self.stats["total_samples"] = total_samples
        return self.stats
    
    def generate_report(self) -> str:
        """Generate a comprehensive quality report."""
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall statistics
        report.append(f"\nTotal samples analyzed: {self.stats['total_samples']}")
        report.append(f"Perfect samples: {self.stats['perfect_samples']} ({self.stats['perfect_samples']/self.stats['total_samples']*100:.1f}%)")
        report.append(f"Minor issues (1 issue): {self.stats['minor_issues']} ({self.stats['minor_issues']/self.stats['total_samples']*100:.1f}%)")
        report.append(f"Moderate issues (2-3 issues): {self.stats['moderate_issues']} ({self.stats['moderate_issues']/self.stats['total_samples']*100:.1f}%)")
        report.append(f"Severe issues (4+ issues): {self.stats['severe_issues']} ({self.stats['severe_issues']/self.stats['total_samples']*100:.1f}%)")
        
        # Samples by source
        report.append("\nSamples by source:")
        for source, count in sorted(self.samples_by_source.items()):
            report.append(f"  {source}: {count}")
        
        # Issues breakdown
        report.append("\nIssues breakdown:")
        
        # Sort issues by severity
        issue_severity = {
            "syntax_error": 5,
            "parse_error": 5,
            "missing_scene_structure": 4,
            "empty_construct": 4,
            "code_too_short": 3,
            "missing_imports": 3,
            "placeholder_content": 3,
            "incomplete_code": 2,
            "description_too_short": 2,
            "generic_description": 2,
            "description_missing_math_context": 1,
            "unbalanced_parentheses": 1,
            "description_has_placeholders": 1,
        }
        
        sorted_issues = sorted(self.issues.items(), 
                             key=lambda x: issue_severity.get(x[0], 0), 
                             reverse=True)
        
        for issue_type, instances in sorted_issues:
            severity = issue_severity.get(issue_type, 0)
            severity_label = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"][min(severity, 4)]
            
            report.append(f"\n  [{severity_label}] {issue_type}: {len(instances)} samples")
            
            # Show examples for critical issues
            if severity >= 3 and len(instances) > 0:
                report.append("    Examples:")
                for inst in instances[:3]:
                    report.append(f"      - Sample {inst['id']} from {inst.get('source', 'unknown')}")
                    if 'error' in inst:
                        report.append(f"        Error: {inst['error'][:100]}...")
        
        # Source-specific analysis
        report.append("\n\nSource-specific quality issues:")
        source_issues = defaultdict(lambda: defaultdict(int))
        
        for issue_type, instances in self.issues.items():
            for inst in instances:
                source = inst.get('source', 'unknown')
                source_issues[source][issue_type] += 1
        
        for source in sorted(self.samples_by_source.keys()):
            if source in source_issues:
                total_samples = self.samples_by_source[source]
                report.append(f"\n  {source} ({total_samples} samples):")
                
                sorted_source_issues = sorted(source_issues[source].items(),
                                            key=lambda x: x[1],
                                            reverse=True)
                
                for issue_type, count in sorted_source_issues[:5]:
                    percentage = count / total_samples * 100
                    report.append(f"    - {issue_type}: {count} ({percentage:.1f}%)")
        
        return "\n".join(report)
    
    def save_detailed_issues(self, output_file: str = "quality_issues_detailed.json"):
        """Save detailed issue information for further analysis."""
        detailed_data = {
            "summary": self.stats,
            "samples_by_source": dict(self.samples_by_source),
            "issues": dict(self.issues),
            "issue_counts": {
                issue_type: len(instances) 
                for issue_type, instances in self.issues.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        logger.info(f"Detailed issues saved to {output_file}")


def main():
    """Run quality analysis on the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Manim dataset quality")
    parser.add_argument("--data-dir", default="data_formatted_v2", help="Data directory")
    parser.add_argument("--output", default="quality_report.txt", help="Output report file")
    parser.add_argument("--detailed", action="store_true", help="Save detailed issues JSON")
    
    args = parser.parse_args()
    
    analyzer = DataQualityAnalyzer(args.data_dir)
    
    # Analyze train and test files
    train_file = analyzer.data_dir / "train.json"
    test_file = analyzer.data_dir / "test.json"
    
    if train_file.exists():
        logger.info("Analyzing training data...")
        analyzer.analyze_file(train_file)
    
    if test_file.exists():
        logger.info("Analyzing test data...")
        analyzer.analyze_file(test_file)
    
    # Generate report
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {args.output}")
    
    # Save detailed issues if requested
    if args.detailed:
        analyzer.save_detailed_issues()


if __name__ == "__main__":
    main()