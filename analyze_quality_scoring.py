#!/usr/bin/env python3
"""
Show exactly how quality scoring works with real examples.
"""

import json
import ast
from pathlib import Path
from extractors.quality_validator import QualityValidator

def analyze_sample_quality(sample_data, validator):
    """Analyze a single sample and show detailed scoring."""
    
    # Extract description and code
    desc = sample_data['conversations'][1]['value']
    code = sample_data['conversations'][2]['value']
    
    # Clean code from markdown
    if '```python' in code:
        code = code.split('```python')[1].split('```')[0].strip()
    elif '```' in code:
        code = code.split('```')[1].split('```')[0].strip()
    
    # Create sample dict for validator
    sample = {
        "description": desc,
        "code": code,
        "source": sample_data.get('source', 'unknown')
    }
    
    # Validate
    is_valid, issues = validator.validate_sample(sample)
    
    return {
        "description": desc[:150] + "..." if len(desc) > 150 else desc,
        "code_preview": code[:300] + "..." if len(code) > 300 else code,
        "desc_length": len(desc),
        "code_length": len(code),
        "is_valid": is_valid,
        "issues": issues,
        "full_code": code  # Keep full code for detailed analysis
    }

def main():
    """Show quality scoring with real examples."""
    
    # Initialize validators
    strict_validator = QualityValidator(strict_mode=True)
    lenient_validator = QualityValidator(strict_mode=False)
    
    # Load some samples from thanks_dataset
    train_file = Path("data_formatted_v2/train.json")
    
    thanks_samples = []
    all_samples = []
    
    with open(train_file) as f:
        for line in f:
            sample = json.loads(line)
            if sample.get('source') == 'thanks_dataset':
                thanks_samples.append(sample)
            all_samples.append(sample)
    
    print("="*100)
    print("QUALITY SCORING EXPLAINED WITH REAL EXAMPLES")
    print("="*100)
    
    print("\nHOW THE QUALITY VALIDATOR WORKS:")
    print("-" * 50)
    print("The validator checks for these issues:\n")
    
    print("CRITICAL (reject in all modes):")
    print("  - Syntax errors: Code must parse as valid Python")
    print("  - No Scene class: Must have a class inheriting from Scene")
    print("  - Empty construct: The construct() method can't be empty")
    print("  - Code too short: Less than 50 characters\n")
    
    print("HIGH (reject in strict mode only):")
    print("  - Missing imports: No import statements found")
    print("  - Placeholder content: Contains TODO, FIXME, ellipsis")
    print("  - Description too short: Less than 20 characters\n")
    
    print("MEDIUM/LOW (warnings only):")
    print("  - No animation methods (play, wait, add, etc.)")
    print("  - No math objects (Text, Circle, etc.)")
    print("  - Generic descriptions")
    print("  - Formatting issues\n")
    
    # Find examples of different quality levels
    print("="*100)
    print("REAL EXAMPLES FROM thanks_dataset")
    print("="*100)
    
    # Categories to find
    categories = {
        "perfect": {"found": [], "criteria": lambda r: r["is_valid"] and len(r["issues"]) == 0},
        "good_but_warnings": {"found": [], "criteria": lambda r: r["is_valid"] and len(r["issues"]) > 0},
        "fail_strict": {"found": [], "criteria": lambda r: not r["is_valid"]},
        "missing_imports": {"found": [], "criteria": lambda r: any("Missing import" in i for i in r["issues"])},
        "short_code": {"found": [], "criteria": lambda r: any("Code too short" in i for i in r["issues"])},
        "placeholder": {"found": [], "criteria": lambda r: any("placeholder" in i for i in r["issues"])}
    }
    
    # Analyze samples
    for i, sample in enumerate(thanks_samples[:500]):  # Check first 500
        result = analyze_sample_quality(sample, strict_validator)
        
        for cat_name, cat_data in categories.items():
            if len(cat_data["found"]) < 2 and cat_data["criteria"](result):
                cat_data["found"].append((i, result))
    
    # Show examples from each category
    print("\n1. PERFECT SAMPLES (pass all checks):")
    print("-" * 80)
    for idx, (i, result) in enumerate(categories["perfect"]["found"][:1]):
        print(f"\nExample {idx+1}:")
        print(f"Description: {result['description']}")
        print(f"Code length: {result['code_length']} chars")
        print(f"Issues: None")
        print(f"Would pass strict mode: YES")
        print(f"Would pass lenient mode: YES")
        print(f"\nCode preview:")
        print(result['code_preview'])
    
    print("\n\n2. GOOD SAMPLES WITH WARNINGS (pass but have minor issues):")
    print("-" * 80)
    for idx, (i, result) in enumerate(categories["good_but_warnings"]["found"][:1]):
        print(f"\nExample {idx+1}:")
        print(f"Description: {result['description']}")
        print(f"Code length: {result['code_length']} chars")
        print(f"Issues: {result['issues']}")
        print(f"Would pass strict mode: YES (warnings don't fail)")
        print(f"Would pass lenient mode: YES")
        print(f"\nCode preview:")
        print(result['code_preview'])
    
    print("\n\n3. SAMPLES THAT FAIL STRICT MODE:")
    print("-" * 80)
    
    # Re-analyze with lenient mode to show difference
    for idx, (i, sample_data) in enumerate(categories["fail_strict"]["found"][:2]):
        # Get the original sample
        original_sample = thanks_samples[i]
        
        # Analyze with both validators
        strict_result = analyze_sample_quality(original_sample, strict_validator)
        lenient_result = analyze_sample_quality(original_sample, lenient_validator)
        
        print(f"\nExample {idx+1}:")
        print(f"Description: {strict_result['description']}")
        print(f"Code length: {strict_result['code_length']} chars")
        print(f"Issues: {strict_result['issues']}")
        print(f"Would pass STRICT mode: {strict_result['is_valid']}")
        print(f"Would pass LENIENT mode: {lenient_result['is_valid']}")
        print(f"\nCode preview:")
        print(strict_result['code_preview'])
        
        # Show why it failed
        critical_issues = [i for i in strict_result['issues'] if i.startswith("[CRITICAL]")]
        high_issues = [i for i in strict_result['issues'] if i.startswith("[HIGH]")]
        if critical_issues:
            print(f"\nFailed due to CRITICAL issues: {critical_issues}")
        elif high_issues:
            print(f"\nFailed due to HIGH issues (strict mode): {high_issues}")
    
    print("\n\n4. SPECIFIC ISSUE: MISSING IMPORTS")
    print("-" * 80)
    for idx, (i, result) in enumerate(categories["missing_imports"]["found"][:1]):
        print(f"\nExample {idx+1}:")
        print(f"Description: {result['description']}")
        print(f"Issues: {[i for i in result['issues'] if 'import' in i]}")
        print(f"Would pass strict mode: NO (HIGH severity issue)")
        print(f"Would pass lenient mode: YES (only CRITICAL issues fail)")
        print(f"\nFull code (to show the issue):")
        print(result['full_code'][:500] + "..." if len(result['full_code']) > 500 else result['full_code'])
    
    # Summary statistics
    print("\n\n" + "="*100)
    print("SUMMARY STATISTICS FOR thanks_dataset")
    print("="*100)
    
    # Test a larger sample
    test_count = min(500, len(thanks_samples))
    strict_pass = 0
    lenient_pass = 0
    issue_counts = {}
    
    for sample in thanks_samples[:test_count]:
        strict_result = analyze_sample_quality(sample, strict_validator)
        lenient_result = analyze_sample_quality(sample, lenient_validator)
        
        if strict_result["is_valid"]:
            strict_pass += 1
        if lenient_result["is_valid"]:
            lenient_pass += 1
        
        for issue in strict_result["issues"]:
            severity = issue.split("]")[0] + "]"
            issue_type = issue.split("] ")[1] if "] " in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    print(f"\nOut of {test_count} samples from thanks_dataset:")
    print(f"  - Pass STRICT mode: {strict_pass} ({strict_pass/test_count*100:.1f}%)")
    print(f"  - Pass LENIENT mode: {lenient_pass} ({lenient_pass/test_count*100:.1f}%)")
    print(f"  - Would be filtered in STRICT: {test_count - strict_pass} ({(test_count - strict_pass)/test_count*100:.1f}%)")
    
    print("\nMost common issues:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {issue}: {count} samples ({count/test_count*100:.1f}%)")

if __name__ == "__main__":
    main()