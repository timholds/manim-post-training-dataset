#!/usr/bin/env python3
"""
Analyze rendering failures and suggest/apply bulk fixes.
"""

import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RenderingFailureAnalyzer:
    """Analyze patterns in rendering failures and suggest fixes."""
    
    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.fix_suggestions = {}
        
    def analyze_failures(self, failed_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failed samples and categorize issues."""
        
        analysis = {
            "total_failures": len(failed_samples),
            "failure_categories": defaultdict(int),
            "fixable_categories": defaultdict(int),
            "patterns": defaultdict(list),
            "suggested_fixes": {}
        }
        
        for sample in failed_samples:
            error = sample.get("render_error", "")
            stderr = sample.get("stderr", "")
            code = sample.get("code", "")
            
            # Categorize the failure
            category, is_fixable, fix_suggestion = self._categorize_failure(error, stderr, code)
            
            analysis["failure_categories"][category] += 1
            if is_fixable:
                analysis["fixable_categories"][category] += 1
                if category not in analysis["suggested_fixes"]:
                    analysis["suggested_fixes"][category] = fix_suggestion
            
            # Store example for pattern analysis
            if len(analysis["patterns"][category]) < 5:
                analysis["patterns"][category].append({
                    "source": sample.get("source"),
                    "description": sample.get("description", "")[:100],
                    "error_snippet": error[:200]
                })
        
        return analysis
    
    def _categorize_failure(self, error: str, stderr: str, code: str) -> Tuple[str, bool, Dict[str, Any]]:
        """Categorize a failure and determine if it's fixable."""
        
        error_lower = (error + stderr).lower()
        
        # Missing imports
        if "no module named 'manim'" in error_lower or "importerror" in error_lower:
            return "missing_imports", True, {
                "type": "add_imports",
                "fix": "from manim import *",
                "position": "start"
            }
        
        # Scene class issues
        if "error: the following arguments are required: scene_names" in error_lower:
            return "no_scene_specified", True, {
                "type": "extract_scene_name",
                "fix": "auto_detect"
            }
        
        # Inheritance issues
        if "scene" not in code and re.search(r'class\s+\w+\s*\(\s*\)', code):
            return "missing_scene_inheritance", True, {
                "type": "fix_inheritance",
                "pattern": r'class\s+(\w+)\s*\(\s*\)',
                "replacement": r'class \1(Scene)'
            }
        
        # Method issues
        if "takes 0 positional arguments but 1 was given" in error_lower and "construct" in error_lower:
            return "construct_missing_self", True, {
                "type": "fix_method_signature",
                "pattern": r'def\s+construct\s*\(\s*\):',
                "replacement": r'def construct(self):'
            }
        
        # Animation method issues
        if "nameerror" in error_lower and any(method in error_lower for method in ["play", "wait", "add"]):
            return "missing_self_in_methods", True, {
                "type": "add_self_to_methods",
                "methods": ["play", "wait", "add", "remove"]
            }
        
        # Syntax errors
        if "syntaxerror" in error_lower:
            # Check for common syntax issues
            if "invalid syntax" in error_lower:
                return "syntax_error", False, {}
            elif "indentationerror" in error_lower:
                return "indentation_error", True, {
                    "type": "fix_indentation",
                    "fix": "convert_tabs_to_spaces"
                }
        
        # Timeout
        if "timeout" in error_lower:
            return "render_timeout", False, {}
        
        # No output
        if "no video file generated" in error_lower:
            return "no_video_output", False, {}
        
        # Other
        return "other_error", False, {}
    
    def generate_bulk_fixes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate bulk fix recommendations based on analysis."""
        
        bulk_fixes = []
        
        for category, count in analysis["fixable_categories"].items():
            if count == 0:
                continue
                
            fix_info = analysis["suggested_fixes"].get(category, {})
            if not fix_info:
                continue
            
            bulk_fix = {
                "category": category,
                "affected_count": count,
                "fix_type": fix_info.get("type"),
                "fix_details": fix_info,
                "estimated_success_rate": self._estimate_success_rate(category, count, analysis["failure_categories"][category])
            }
            
            bulk_fixes.append(bulk_fix)
        
        # Sort by affected count
        bulk_fixes.sort(key=lambda x: x["affected_count"], reverse=True)
        
        return bulk_fixes
    
    def _estimate_success_rate(self, category: str, fixable: int, total: int) -> float:
        """Estimate success rate for a fix category."""
        
        # Based on empirical observations
        success_rates = {
            "missing_imports": 0.95,
            "no_scene_specified": 0.90,
            "missing_scene_inheritance": 0.95,
            "construct_missing_self": 0.98,
            "missing_self_in_methods": 0.90,
            "indentation_error": 0.85,
        }
        
        base_rate = success_rates.get(category, 0.5)
        
        # Adjust based on proportion of fixable
        if total > 0:
            return base_rate * (fixable / total)
        return base_rate
    
    def apply_bulk_fix(self, code: str, fix_type: str, fix_details: Dict[str, Any]) -> str:
        """Apply a specific fix to code."""
        
        if fix_type == "add_imports":
            if "from manim import" not in code and "import manim" not in code:
                return fix_details["fix"] + "\n" + code
                
        elif fix_type == "fix_inheritance":
            pattern = fix_details["pattern"]
            replacement = fix_details["replacement"]
            return re.sub(pattern, replacement, code)
            
        elif fix_type == "fix_method_signature":
            pattern = fix_details["pattern"]
            replacement = fix_details["replacement"]
            return re.sub(pattern, replacement, code)
            
        elif fix_type == "add_self_to_methods":
            methods = fix_details["methods"]
            fixed_code = code
            for method in methods:
                # Add self. to method calls that don't have it
                pattern = rf'(?<!self\.)(?<!\.)\b{method}\s*\('
                replacement = f'self.{method}('
                fixed_code = re.sub(pattern, replacement, fixed_code)
            return fixed_code
            
        elif fix_type == "fix_indentation":
            if fix_details["fix"] == "convert_tabs_to_spaces":
                return code.replace('\t', '    ')
        
        return code


def main():
    parser = argparse.ArgumentParser(description="Analyze rendering failures from validation")
    parser.add_argument("report_file", help="Path to rendering_validation_report.json")
    parser.add_argument("--output", default="rendering_analysis.json", help="Output analysis file")
    parser.add_argument("--suggest-fixes", action="store_true", help="Generate fix suggestions")
    parser.add_argument("--source-filter", help="Only analyze failures from specific source")
    
    args = parser.parse_args()
    
    # Load report
    with open(args.report_file, 'r') as f:
        report = json.load(f)
    
    failed_samples = report.get("failed_examples", [])
    
    if args.source_filter:
        failed_samples = [s for s in failed_samples if s.get("source") == args.source_filter]
        logger.info(f"Filtering to {len(failed_samples)} failures from {args.source_filter}")
    
    if not failed_samples:
        logger.error("No failed samples found in report")
        return
    
    # Analyze failures
    analyzer = RenderingFailureAnalyzer()
    analysis = analyzer.analyze_failures(failed_samples)
    
    # Generate fix suggestions if requested
    if args.suggest_fixes:
        bulk_fixes = analyzer.generate_bulk_fixes(analysis)
        analysis["bulk_fix_suggestions"] = bulk_fixes
    
    # Save analysis
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RENDERING FAILURE ANALYSIS")
    logger.info("="*60)
    logger.info(f"Total failures analyzed: {analysis['total_failures']}")
    logger.info("\nFailure categories:")
    for category, count in sorted(analysis["failure_categories"].items(), key=lambda x: x[1], reverse=True):
        fixable = analysis["fixable_categories"].get(category, 0)
        logger.info(f"  {category}: {count} ({fixable} fixable)")
    
    if args.suggest_fixes and analysis.get("bulk_fix_suggestions"):
        logger.info("\nRecommended bulk fixes:")
        for fix in analysis["bulk_fix_suggestions"][:5]:
            logger.info(f"\n  Fix: {fix['category']}")
            logger.info(f"    Affected samples: {fix['affected_count']}")
            logger.info(f"    Estimated success rate: {fix['estimated_success_rate']:.1%}")
            logger.info(f"    Fix type: {fix['fix_type']}")
    
    logger.info(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()