"""
Quality validation framework for Manim dataset extraction.
Ensures high-quality data at the extraction stage.
"""

import ast
import re
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QualityValidator:
    """Validates quality of Manim code samples during extraction."""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, reject samples with any critical issues
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "issues_by_type": {}
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single sample.
        
        Returns:
            (is_valid, list_of_issues)
        """
        self.validation_stats["total_checked"] += 1
        
        issues = []
        description = sample.get("description", "")
        code = sample.get("code", "")
        
        # Run all validation checks
        issues.extend(self._validate_description(description))
        issues.extend(self._validate_code_structure(code))
        issues.extend(self._validate_code_quality(code))
        issues.extend(self._validate_code_description_alignment(description, code))
        
        # Determine if sample passes
        critical_issues = [i for i in issues if i.startswith("[CRITICAL]")]
        high_issues = [i for i in issues if i.startswith("[HIGH]")]
        
        is_valid = True
        if self.strict_mode:
            is_valid = len(critical_issues) == 0 and len(high_issues) == 0
        else:
            is_valid = len(critical_issues) == 0
        
        # Update stats
        if is_valid:
            self.validation_stats["passed"] += 1
        else:
            self.validation_stats["failed"] += 1
        
        for issue in issues:
            issue_type = issue.split("]")[0] + "]"
            self.validation_stats["issues_by_type"][issue_type] = \
                self.validation_stats["issues_by_type"].get(issue_type, 0) + 1
        
        return is_valid, issues
    
    def _validate_description(self, description: str) -> List[str]:
        """Validate description quality."""
        issues = []
        
        # Length check
        if len(description) < 20:
            issues.append(f"[HIGH] Description too short ({len(description)} chars)")
        
        # Check for placeholders
        placeholder_patterns = [
            r"\[.*?\]",  # Square brackets
            r"TODO", r"FIXME", r"XXX",
            r"INSERT", r"PLACEHOLDER",
            r"<.*?>",  # Angle brackets
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                issues.append(f"[HIGH] Description contains placeholder: {pattern}")
                break
        
        # Check for generic descriptions
        generic_starts = [
            "Create a Manim animation",
            "Create an animation",
            "Animation demonstrating",
            "Manim scene",
        ]
        if any(description.strip().startswith(start) and len(description) < 50 
               for start in generic_starts):
            issues.append("[MEDIUM] Description is too generic")
        
        # Check for balanced parentheses
        if description.count('(') != description.count(')'):
            issues.append("[LOW] Unbalanced parentheses in description")
        
        # Check for proper sentence structure
        if description and not description[0].isupper():
            issues.append("[LOW] Description should start with capital letter")
        
        if description and not description.rstrip().endswith(('.', '!', '?')):
            issues.append("[LOW] Description should end with punctuation")
        
        return issues
    
    def _validate_code_structure(self, code: str) -> List[str]:
        """Validate code has proper structure."""
        issues = []
        
        # Basic length check
        if len(code) < 50:
            issues.append(f"[CRITICAL] Code too short ({len(code)} chars)")
            return issues  # No point checking further
        
        # Check for syntax errors
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(f"[CRITICAL] Syntax error: {str(e)}")
            return issues  # Can't analyze further with syntax errors
        
        # Check for imports
        has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) 
                         for node in ast.walk(tree))
        if not has_imports:
            issues.append("[HIGH] Missing import statements")
        
        # Check for Scene class
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        scene_classes = []
        
        for cls in classes:
            # Check if inherits from Scene-like class
            for base in cls.bases:
                if isinstance(base, ast.Name):
                    if 'Scene' in base.id:
                        scene_classes.append(cls)
                        break
        
        if not scene_classes:
            issues.append("[CRITICAL] No Scene class found")
            return issues
        
        # Check for construct method
        for scene_class in scene_classes:
            has_construct = False
            construct_is_empty = False
            
            for node in scene_class.body:
                if isinstance(node, ast.FunctionDef) and node.name == "construct":
                    has_construct = True
                    
                    # Check if construct is empty or just has pass
                    if len(node.body) == 0:
                        construct_is_empty = True
                    elif len(node.body) == 1:
                        if isinstance(node.body[0], ast.Pass):
                            construct_is_empty = True
                        elif isinstance(node.body[0], ast.Expr) and \
                             isinstance(node.body[0].value, ast.Constant) and \
                             node.body[0].value.value == ...:
                            construct_is_empty = True
            
            if not has_construct:
                issues.append(f"[HIGH] Scene class '{scene_class.name}' missing construct method")
            elif construct_is_empty:
                issues.append(f"[CRITICAL] Empty construct method in '{scene_class.name}'")
        
        return issues
    
    def _validate_code_quality(self, code: str) -> List[str]:
        """Validate code quality and completeness."""
        issues = []
        
        # Check for incomplete code markers
        incomplete_markers = ["TODO", "FIXME", "XXX", "HACK", "BUG", "REFACTOR"]
        for marker in incomplete_markers:
            if marker in code:
                issues.append(f"[HIGH] Code contains incomplete marker: {marker}")
                break
        
        # Check for placeholder patterns
        placeholder_patterns = [
            r"#\s*Your code here",
            r"#\s*Implementation goes here",
            r"#\s*Add your.*here",
            r"#\s*Fill in.*",
            r"#\s*Complete this.*",
            r"\.\.\.\s*$",  # Ellipsis at end of line (not in string)
            r"pass\s*#\s*TODO",
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                issues.append(f"[HIGH] Code contains placeholder pattern")
                break
        
        # Check for common animation methods
        animation_methods = [
            "play", "wait", "add", "remove", "move_to", "shift", "scale",
            "rotate", "set_color", "fade", "transform", "animate"
        ]
        
        has_animation = any(method in code for method in animation_methods)
        if not has_animation:
            issues.append("[MEDIUM] Code lacks animation methods - might be incomplete")
        
        # Check for mathematical objects (good sign of actual content)
        math_objects = [
            "Text", "Tex", "MathTex", "Circle", "Square", "Line",
            "Arrow", "Dot", "Graph", "Axes", "Vector", "Matrix",
            "Polygon", "Arc", "Angle", "BraceBetweenPoints"
        ]
        
        has_math_objects = any(obj in code for obj in math_objects)
        if not has_math_objects:
            issues.append("[MEDIUM] Code lacks mathematical objects")
        
        return issues
    
    def _validate_code_description_alignment(self, description: str, code: str) -> List[str]:
        """Check if code and description are aligned."""
        issues = []
        
        # Extract key concepts from description
        desc_lower = description.lower()
        
        # Common mathematical concepts
        math_concepts = {
            "derivative": ["derivative", "differentiate", "d/dx", "prime"],
            "integral": ["integral", "integrate", "area under", "antiderivative"],
            "matrix": ["matrix", "matrices", "determinant", "eigenvalue"],
            "vector": ["vector", "cross product", "dot product", "magnitude"],
            "graph": ["graph", "plot", "axes", "coordinate"],
            "equation": ["equation", "solve", "formula", "expression"],
            "theorem": ["theorem", "proof", "lemma", "corollary"],
            "transform": ["transform", "transformation", "map", "morph"],
            "probability": ["probability", "random", "chance", "likelihood"],
            "geometry": ["circle", "triangle", "polygon", "angle", "perpendicular"],
        }
        
        # Check which concepts are mentioned
        mentioned_concepts = []
        for concept, keywords in math_concepts.items():
            if any(keyword in desc_lower for keyword in keywords):
                mentioned_concepts.append(concept)
        
        # If description mentions math concepts, code should have related objects
        if mentioned_concepts:
            code_lower = code.lower()
            found_implementation = False
            
            for concept in mentioned_concepts:
                # Check if concept appears to be implemented
                if concept in code_lower or any(kw in code_lower for kw in math_concepts[concept]):
                    found_implementation = True
                    break
            
            if not found_implementation:
                issues.append("[MEDIUM] Description mentions concepts not clearly implemented in code")
        
        # Check if class names align with description
        class_names = re.findall(r'class\s+(\w+)', code)
        if class_names:
            # Check if any class name relates to description
            desc_words = set(re.findall(r'\w+', desc_lower))
            class_words = set(word.lower() for name in class_names 
                            for word in re.findall(r'[A-Z][a-z]+|[a-z]+', name))
            
            if not desc_words.intersection(class_words):
                issues.append("[LOW] Class names don't reflect description content")
        
        return issues
    
    def get_validation_report(self) -> str:
        """Get a summary of validation statistics."""
        report = []
        report.append("=== Quality Validation Report ===")
        report.append(f"Total samples checked: {self.validation_stats['total_checked']}")
        report.append(f"Passed: {self.validation_stats['passed']} ({self.validation_stats['passed']/max(1, self.validation_stats['total_checked'])*100:.1f}%)")
        report.append(f"Failed: {self.validation_stats['failed']} ({self.validation_stats['failed']/max(1, self.validation_stats['total_checked'])*100:.1f}%)")
        
        if self.validation_stats["issues_by_type"]:
            report.append("\nIssues by severity:")
            for issue_type in ["[CRITICAL]", "[HIGH]", "[MEDIUM]", "[LOW]"]:
                count = self.validation_stats["issues_by_type"].get(issue_type, 0)
                if count > 0:
                    report.append(f"  {issue_type}: {count}")
        
        return "\n".join(report)


class QualityFilter:
    """Filter samples based on quality criteria."""
    
    def __init__(self, validator: Optional[QualityValidator] = None):
        """Initialize filter with optional custom validator."""
        self.validator = validator or QualityValidator(strict_mode=True)
        self.filtered_stats = {
            "total_input": 0,
            "total_output": 0,
            "filtered_by_source": {}
        }
    
    def filter_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of samples, keeping only high-quality ones."""
        filtered = []
        
        for sample in samples:
            self.filtered_stats["total_input"] += 1
            source = sample.get("source", "unknown")
            
            is_valid, issues = self.validator.validate_sample(sample)
            
            if is_valid:
                filtered.append(sample)
                self.filtered_stats["total_output"] += 1
            else:
                # Track what we filtered
                if source not in self.filtered_stats["filtered_by_source"]:
                    self.filtered_stats["filtered_by_source"][source] = {
                        "count": 0,
                        "issues": []
                    }
                
                self.filtered_stats["filtered_by_source"][source]["count"] += 1
                if len(self.filtered_stats["filtered_by_source"][source]["issues"]) < 10:
                    self.filtered_stats["filtered_by_source"][source]["issues"].append({
                        "sample_desc": sample.get("description", "")[:100],
                        "issues": issues[:3]  # First 3 issues
                    })
        
        return filtered
    
    def get_filter_report(self) -> str:
        """Get filtering statistics."""
        report = []
        report.append("=== Quality Filtering Report ===")
        report.append(f"Total input samples: {self.filtered_stats['total_input']}")
        report.append(f"Total output samples: {self.filtered_stats['total_output']}")
        report.append(f"Filtered out: {self.filtered_stats['total_input'] - self.filtered_stats['total_output']} ({(self.filtered_stats['total_input'] - self.filtered_stats['total_output'])/max(1, self.filtered_stats['total_input'])*100:.1f}%)")
        
        if self.filtered_stats["filtered_by_source"]:
            report.append("\nFiltered by source:")
            for source, data in sorted(self.filtered_stats["filtered_by_source"].items()):
                report.append(f"  {source}: {data['count']} samples filtered")
                if data["issues"]:
                    report.append("    Example issues:")
                    for ex in data["issues"][:3]:
                        report.append(f"      - {ex['sample_desc'][:50]}...")
                        for issue in ex["issues"]:
                            report.append(f"        {issue}")
        
        return "\n".join(report)