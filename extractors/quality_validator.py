"""
Quality validation framework for Manim dataset extraction.
Ensures high-quality data at the extraction stage.

IMPORTANT: Placeholder descriptions are EXPECTED and ACCEPTED at this stage!
- Descriptions starting with PLACEHOLDER_DESCRIPTION are intentionally allowed
- These will be filled later by an LLM
- The validator specifically skips validation for known placeholder patterns
- See _validate_description() method for the list of accepted placeholders

The quality issues we check for are in the CODE itself, not placeholder descriptions.
"""

import ast
import re
from typing import Dict, Any, List, Tuple, Optional
import logging

from .constants import PLACEHOLDER_DESCRIPTION

logger = logging.getLogger(__name__)


class QualityValidator:
    """Validates quality of Manim code samples during extraction."""
    
    def __init__(self, strict_mode: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, reject samples with any critical issues
            config: Quality configuration dict from quality_config.json
        """
        self.strict_mode = strict_mode
        self.config = config or {}
        self.validation_stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "issues_by_type": {}
        }
    
    def validate_sample(self, sample: Dict[str, Any], source_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate a single sample.
        
        Args:
            sample: The sample to validate
            source_id: Optional source identifier for source-specific config
        
        Returns:
            (is_valid, list_of_issues)
        """
        self.validation_stats["total_checked"] += 1
        
        # Get merged config for this source
        merged_config = self._get_merged_config(source_id)
        
        issues = []
        description = sample.get("description", "")
        code = sample.get("code", "")
        
        # Run all validation checks with config
        issues.extend(self._validate_description(description, merged_config))
        issues.extend(self._validate_code_structure(code, merged_config))
        issues.extend(self._validate_code_quality(code, merged_config))
        issues.extend(self._validate_code_description_alignment(description, code, merged_config))
        issues.extend(self._validate_code_executability(code, merged_config))
        
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
    
    def _get_merged_config(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Get merged configuration for a specific source."""
        # Start with global settings
        merged = self.config.get("global_settings", {}).copy()
        
        # Apply source-specific overrides if available
        if source_id and source_id in self.config.get("source_overrides", {}):
            source_config = self.config["source_overrides"][source_id]
            # Deep merge the configurations
            for key, value in source_config.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        
        # Include validation_actions settings
        if "validation_actions" in self.config:
            validation_actions = self.config["validation_actions"]
            if "allow_through" in validation_actions:
                merged["allow_through"] = validation_actions["allow_through"]
            if "must_reject" in validation_actions:
                merged["must_reject"] = validation_actions["must_reject"]
            
        return merged
    
    def _validate_description(self, description: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate description quality.
        
        Simple rules:
        1. Empty descriptions are CRITICAL errors
        2. If description starts with a known placeholder pattern, it's valid (no other checks)
        3. Otherwise, apply basic quality checks
        """
        issues = []
        
        # Rule 1: Empty descriptions are always bad
        if not description or description.strip() == "":
            return ["[CRITICAL] Description is empty"]
        
        # Rule 2: Known placeholders are always OK (they'll be filled by LLM later)
        # This includes our standard placeholder and any description starting with [PLACEHOLDER
        if description.startswith(PLACEHOLDER_DESCRIPTION) or description.startswith("[PLACEHOLDER"):
            return []
        
        # Rule 3: For non-placeholder descriptions, apply quality checks
        # Check if it looks like a placeholder that we missed
        if re.search(r"\[(TODO|FIXME|INSERT|TBD|PLACEHOLDER)", description, re.IGNORECASE):
            issues.append("[HIGH] Description contains placeholder pattern")
        
        # Check for overly generic descriptions
        generic_starts = ["Create a Manim animation", "Create an animation", "Manim scene"]
        if any(description.startswith(start) for start in generic_starts) and len(description) < 50:
            issues.append("[MEDIUM] Description is too generic")
        
        # Basic grammar checks (LOW priority, won't block in strict mode)
        if not description[0].isupper():
            issues.append("[LOW] Description should start with capital letter")
        
        if not description.rstrip().endswith(('.', '!', '?', ':')):
            issues.append("[LOW] Description should end with punctuation")
        
        return issues
    
    def _validate_code_structure(self, code: str, config: Dict[str, Any]) -> List[str]:
        """Validate code has proper structure."""
        issues = []
        
        # Check must_reject rules first
        must_reject = config.get("must_reject", {})
        
        # Basic length check - use config if available
        if must_reject.get("code_below_minimum", {}).get("enabled", True):
            min_length = must_reject.get("code_below_minimum", {}).get("min_length", 30)
            if len(code) < min_length:
                issues.append(f"[CRITICAL] Code too short ({len(code)} chars, min: {min_length})")
                return issues  # No point checking further
        elif len(code) < 50:
            issues.append(f"[CRITICAL] Code too short ({len(code)} chars)")
            return issues  # No point checking further
        
        # Check for syntax errors
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(f"[CRITICAL] Syntax error: {str(e)}")
            return issues  # Can't analyze further with syntax errors
        
        # Check for imports and build alias mapping
        has_imports = False
        alias_map = {}  # Maps aliases to their original names
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                has_imports = True
                
                # Build alias mapping for Scene classes
                if isinstance(node, ast.ImportFrom) and node.module and 'manim' in node.module:
                    for alias in node.names:
                        # alias.name is the original name, alias.asname is the alias
                        if alias.asname:
                            alias_map[alias.asname] = alias.name
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            alias_map[alias.asname] = alias.name
        
        if not has_imports:
            issues.append("[HIGH] Missing import statements")
        
        # Check for Scene class
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        scene_classes = []
        
        # Valid Scene classes in Manim
        # This includes all common Scene subclasses to avoid false positives
        # Previously, we only checked if class name contained "Scene", which missed
        # valid subclasses like ThreeDScene (would fail because base is "ThreeDScene" not "Scene")
        valid_scene_classes = {
            'Scene', 'ThreeDScene', 'VoiceoverScene', 'MovingCameraScene',
            'ZoomedScene', 'InteractiveScene', 'SampleSpaceScene', 'LiveStreamingScene',
            'GraphScene', 'LinearTransformationScene', 'VectorScene', 'SpecialThreeDScene',
            'SpaceScene'  # From manim-physics
        }
        
        # Build a map of all classes for multi-level inheritance checking
        class_map = {cls.name: cls for cls in classes}
        
        def is_scene_class(cls, visited=None):
            """Recursively check if a class inherits from a Scene class."""
            if visited is None:
                visited = set()
            if cls.name in visited:
                return False
            visited.add(cls.name)
            
            for base in cls.bases:
                # Case 1: Direct inheritance (e.g., class MyScene(Scene))
                if isinstance(base, ast.Name):
                    base_name = base.id
                    
                    # Check if this is an alias
                    if base_name in alias_map:
                        original_name = alias_map[base_name]
                        if original_name in valid_scene_classes:
                            return True
                    
                    # Check direct name
                    if base_name in valid_scene_classes:
                        return True
                    
                    # Check if base is another class in this file
                    if base_name in class_map:
                        if is_scene_class(class_map[base_name], visited):
                            return True
                            
                # Case 2: Module-qualified (e.g., class MyScene(manim.Scene))
                elif isinstance(base, ast.Attribute):
                    if base.attr in valid_scene_classes:
                        return True
            return False
        
        # Check each class
        for cls in classes:
            if is_scene_class(cls):
                scene_classes.append(cls)
        
        if not scene_classes:
            issues.append("[CRITICAL] No Scene class found")
            return issues
        
        # First pass: Build a map of which classes have construct methods
        classes_with_construct = {}  # Maps class name to (has_construct, is_empty)
        
        for cls in classes:
            has_construct = False
            construct_is_empty = False
            
            for node in cls.body:
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
                    break
            
            classes_with_construct[cls.name] = (has_construct, construct_is_empty)
        
        # Helper function to check if a class has construct method through inheritance
        def has_construct_in_chain(cls_name, visited=None):
            """Check if a class or any of its parents has a construct method."""
            if visited is None:
                visited = set()
            if cls_name in visited:
                return False, False
            visited.add(cls_name)
            
            # Check if this class directly has construct
            if cls_name in classes_with_construct:
                has_it, is_empty = classes_with_construct[cls_name]
                if has_it:
                    return True, is_empty
            
            # Check parent classes
            if cls_name in class_map:
                cls = class_map[cls_name]
                for base in cls.bases:
                    parent_name = None
                    
                    if isinstance(base, ast.Name):
                        parent_name = base.id
                    elif isinstance(base, ast.Attribute):
                        # For cases like manim.Scene, we can't check inheritance
                        # but these are base classes that should have construct
                        continue
                    
                    if parent_name and parent_name in class_map:
                        has_parent_construct, parent_is_empty = has_construct_in_chain(parent_name, visited)
                        if has_parent_construct:
                            return True, parent_is_empty
            
            return False, False
        
        # Second pass: Check each Scene class for construct method (including inherited)
        for scene_class in scene_classes:
            has_construct, construct_is_empty = has_construct_in_chain(scene_class.name)
            
            if not has_construct:
                issues.append(f"[HIGH] Scene class '{scene_class.name}' missing construct method")
            elif construct_is_empty:
                # Only report empty construct if it's directly in this class
                # (not inherited empty construct)
                if scene_class.name in classes_with_construct and classes_with_construct[scene_class.name][0]:
                    issues.append(f"[CRITICAL] Empty construct method in '{scene_class.name}'")
        
        return issues
    
    def _validate_code_quality(self, code: str, config: Dict[str, Any]) -> List[str]:
        """Validate code quality and completeness."""
        issues = []
        
        # Get allow_through settings
        allow_through = config.get("allow_through", {})
        
        # Check for obvious placeholder patterns that indicate incomplete code
        # Note: We do NOT check for TODO/FIXME/XXX as these often appear in valid
        # function names (e.g., opacity_updater) and cause false positives
        placeholder_patterns = [
            r"#\s*Your code here",
            r"#\s*Implementation goes here",
            r"#\s*Add your.*here",
            r"#\s*Fill in.*",
            r"#\s*Complete this.*",
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                issues.append(f"[HIGH] Code contains placeholder pattern")
                break
        
        # Check for code ellipsis (not in strings or comments)
        # This is more complex and needs special handling
        try:
            tree = ast.parse(code)
            # Check for ellipsis in the AST (actual code ellipsis)
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value == ...:
                    issues.append(f"[HIGH] Code contains placeholder ellipsis (...)")
                    break
        except:
            # If AST parsing fails, we already catch that in _validate_code_structure
            pass
        
        # Check for actual animation execution (not in comments)
        # First, remove comments to avoid false positives
        code_without_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Check for animation commands - but respect config
        if not allow_through.get("no_animation_methods", {}).get("enabled", False):
            animation_commands = [
                r"self\.play\(", r"self\.add\(", r"self\.wait\(",
                r"self\.remove\(", r"self\.bring_to_front\(", r"self\.bring_to_back\("
            ]
            
            has_animation_command = any(
                re.search(pattern, code_without_comments) for pattern in animation_commands
            )
            
            if not has_animation_command:
                issues.append("[CRITICAL] No animation commands found (play/add/wait)")
        
        # Check for common animation methods - but respect config
        if not allow_through.get("no_animation_methods", {}).get("enabled", False):
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
            "Polygon", "Arc", "Angle", "BraceBetweenPoints",
            # Geometric objects
            "Triangle", "Rectangle", "Ellipse", "Sector", "Annulus",
            "RegularPolygon", "RoundedRectangle", "Cutout",
            # Coordinate systems and grids
            "NumberPlane", "ComplexPlane", "PolarPlane", "CoordinateSystem",
            "Axes", "ThreeDAxes", "NumberLine",
            # 3D objects
            "Cube", "Sphere", "Cylinder", "Cone", "Prism", "Pyramid",
            "Polyhedron", "Dodecahedron", "Icosahedron", "Octahedron",
            # Specialized mathematical objects
            "YoungTableau", "ParametricFunction", "ImplicitFunction",
            "VectorField", "StreamLines", "FunctionGraph",
            # Graphs and trees
            "Graph", "DiGraph", "Tree", "GenericGraph",
            # Tables and arrays
            "Table", "MobjectTable", "DecimalTable", "IntegerTable"
        ]
        
        has_math_objects = any(obj in code for obj in math_objects)
        if not has_math_objects:
            issues.append("[MEDIUM] Code lacks mathematical objects")
        
        return issues
    
    def _validate_code_description_alignment(self, description: str, code: str, config: Dict[str, Any]) -> List[str]:
        """
        Check if code and description are aligned.
        Only check for non-placeholder descriptions.
        Keep it simple - just check for obvious mismatches.
        """
        # Skip all checks for placeholder descriptions
        if description.startswith("[PLACEHOLDER") or description.startswith(PLACEHOLDER_DESCRIPTION):
            return []
        
        issues = []
        desc_lower = description.lower()
        code_lower = code.lower()
        
        # Simple check: If description mentions specific shapes/objects, code should have them
        shape_mappings = {
            "circle": ["Circle(", "circle"],
            "square": ["Square(", "square"],
            "triangle": ["Triangle(", "Polygon", "triangle"],
            "line": ["Line(", "line"],
            "arrow": ["Arrow(", "arrow"],
            "text": ["Text(", "Tex(", "MathTex("],
            "graph": ["Graph(", "Axes(", "NumberPlane(", "plot"],
            "matrix": ["Matrix(", "matrix"],
            "vector": ["Vector(", "Arrow(", "vector"],
        }
        
        for shape, code_patterns in shape_mappings.items():
            if shape in desc_lower:
                if not any(pattern in code for pattern in code_patterns):
                    issues.append(f"[MEDIUM] Description mentions '{shape}' but code doesn't implement it")
                    break  # Only report first mismatch to avoid noise
        
        return issues
    
    def _validate_code_executability(self, code: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate that the code can be parsed and potentially executed.
        Tests if the transformed code is syntactically valid.
        """
        issues = []
        
        # Skip if executability check is disabled
        if not config.get("check_executability", True):
            return issues
        
        # Test 1: Basic syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"[CRITICAL] Syntax error in code: {e}")
            return issues  # No point in further checks if syntax is broken
        except Exception as e:
            issues.append(f"[CRITICAL] Failed to parse code: {e}")
            return issues
        
        # Test 2: Check for unescaped string literals that would cause warnings
        try:
            # Look for potential escape sequence issues
            if re.search(r'(MathTex|Text|Tex)\s*\(\s*["\'][^"\']*\\[^\\r][^"\']*["\']', code):
                issues.append("[MEDIUM] Potential invalid escape sequences in TeX strings")
        except Exception:
            pass
        
        # Test 3: Check for common runtime issues that can be statically detected
        runtime_issues = []
        
        # Check for undefined variables (simplified check)
        try:
            tree = ast.parse(code)
            
            # Find all variable names used
            used_names = set()
            defined_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)
                    elif isinstance(node.ctx, ast.Store):
                        defined_names.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
            
            # Common built-ins and manim imports that are usually available
            common_builtins = {
                'len', 'range', 'enumerate', 'zip', 'sum', 'min', 'max',
                'print', 'str', 'int', 'float', 'bool', 'list', 'dict',
                'True', 'False', 'None', 'self'
            }
            
            common_manim = {
                'Scene', 'Text', 'MathTex', 'Tex', 'Circle', 'Square', 'Line',
                'Arrow', 'Dot', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WHITE', 'BLACK',
                'RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'PURPLE', 'PI', 'TAU',
                'Create', 'Write', 'FadeIn', 'FadeOut', 'Transform', 'ReplacementTransform',
                'Axes', 'NumberPlane', 'Graph', 'Vector', 'Matrix', 'Polygon'
            }
            
            # Check for potentially undefined variables
            undefined = used_names - defined_names - common_builtins - common_manim
            if undefined:
                # Filter out likely false positives
                likely_undefined = [name for name in undefined 
                                  if not name.startswith('_') and 
                                  not name.isupper() and  # Likely constants
                                  len(name) > 1]
                if likely_undefined:
                    issues.append(f"[MEDIUM] Potentially undefined variables: {', '.join(list(likely_undefined)[:5])}")
        
        except Exception:
            # If static analysis fails, don't add issues
            pass
        
        # Test 4: Check for missing required methods in Scene classes
        try:
            if 'class' in code and 'Scene' in code:
                tree = ast.parse(code)
                has_construct = False
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == 'construct':
                        has_construct = True
                        break
                
                if not has_construct:
                    issues.append("[HIGH] Scene class missing construct method")
        except Exception:
            pass
        
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
    
    def __init__(self, validator: Optional[QualityValidator] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize filter with optional custom validator."""
        self.validator = validator or QualityValidator(strict_mode=True, config=config)
        self.config = config or {}
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
            
            is_valid, issues = self.validator.validate_sample(sample, source_id=source)
            
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