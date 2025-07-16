"""
Conservative code fixing for common Manim API issues.
Only applies high-confidence, low-risk fixes that don't change code semantics.
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FixResult:
    """Result of applying a fix to code."""
    success: bool
    original_code: str
    fixed_code: str
    fixes_applied: List[str]
    fix_count: int

class ManimCodeFixer:
    """Conservative code fixer for common Manim API issues."""
    
    def __init__(self, aggressive_mode: bool = False):
        self.aggressive_mode = aggressive_mode
        self.stats = {
            'samples_processed': 0,
            'samples_fixed': 0,
            'fixes_by_type': {},
            'fixes_by_source': {}
        }
    
    def fix_api_calls(self, code: str) -> Tuple[str, List[str]]:
        """Fix deprecated API calls with high confidence."""
        fixes_applied = []
        original_code = code
        
        # Pattern 1: obj.set_width(val) -> obj.width = val
        # Very safe: only matches exact pattern with clear intent
        width_pattern = r'(\w+)\.set_width\(\s*([^)]+)\s*\)'
        if re.search(width_pattern, code):
            code = re.sub(width_pattern, r'\1.width = \2', code)
            fixes_applied.append('set_width_to_property')
        
        # Pattern 2: obj.set_height(val) -> obj.height = val
        height_pattern = r'(\w+)\.set_height\(\s*([^)]+)\s*\)'
        if re.search(height_pattern, code):
            code = re.sub(height_pattern, r'\1.height = \2', code)
            fixes_applied.append('set_height_to_property')
        
        # Pattern 3: obj.get_width() -> obj.width
        # Only if it's clearly a getter (no assignment context)
        get_width_pattern = r'(\w+)\.get_width\(\s*\)'
        if re.search(get_width_pattern, code):
            code = re.sub(get_width_pattern, r'\1.width', code)
            fixes_applied.append('get_width_to_property')
        
        # Pattern 4: obj.get_height() -> obj.height
        get_height_pattern = r'(\w+)\.get_height\(\s*\)'
        if re.search(get_height_pattern, code):
            code = re.sub(get_height_pattern, r'\1.height', code)
            fixes_applied.append('get_height_to_property')
        
        return code, fixes_applied
    
    def fix_imports(self, code: str) -> Tuple[str, List[str]]:
        """Add missing common imports that are safe to add."""
        fixes_applied = []
        
        # Common constants that are safe to import
        SAFE_CONSTANTS = {
            'WHITE', 'BLACK', 'RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'PURPLE',
            'UP', 'DOWN', 'LEFT', 'RIGHT', 'IN', 'OUT',
            'PI', 'TAU', 'DEGREES'
        }
        
        # Check if we already have a manim import
        has_manim_import = bool(re.search(r'from\s+manim\s+import', code) or 
                               re.search(r'import\s+manim', code))
        
        if has_manim_import:
            return code, fixes_applied
        
        # Find constants used in code that we can safely import
        used_constants = set()
        for constant in SAFE_CONSTANTS:
            # Look for the constant used as a standalone identifier
            if re.search(rf'\b{constant}\b', code):
                used_constants.add(constant)
        
        if used_constants:
            # Add import at the top
            import_line = f"from manim import {', '.join(sorted(used_constants))}\n"
            code = import_line + code
            fixes_applied.append(f'added_imports_{len(used_constants)}')
        
        return code, fixes_applied
    
    def fix_manimgl_to_manimce(self, code: str) -> Tuple[str, List[str]]:
        """Convert old ManimGL code to ManimCE syntax."""
        fixes_applied = []
        
        # Fix 1: Replace manimlib imports with manim imports
        if 'from manimlib.imports import *' in code:
            code = code.replace('from manimlib.imports import *', 'from manim import *')
            fixes_applied.append('manimlib_to_manim_import')
        elif 'from manimlib' in code:
            code = re.sub(r'from manimlib(\.[a-zA-Z0-9_.]+)? import', r'from manim import', code)
            fixes_applied.append('manimlib_to_manim_import')
        
        # Fix 2: TexMobject -> MathTex
        if 'TexMobject' in code:
            code = re.sub(r'\bTexMobject\b', 'MathTex', code)
            fixes_applied.append('texmobject_to_mathtex')
        
        # Fix 3: TextMobject -> Text  
        if 'TextMobject' in code:
            code = re.sub(r'\bTextMobject\b', 'Text', code)
            fixes_applied.append('textmobject_to_text')
        
        # Fix 4: Remove tex_to_color_map parameter
        # Pattern: tex_to_color_map={...} or tex_to_color_map=dict
        tex_color_pattern = r',?\s*tex_to_color_map\s*=\s*(\{[^}]*\}|\w+)'
        if re.search(tex_color_pattern, code):
            code = re.sub(tex_color_pattern, '', code)
            fixes_applied.append('removed_tex_to_color_map')
        
        # Fix 5: Fix Axes parameters (x_min/x_max -> x_range)
        # More robust Axes conversion
        if 'Axes' in code and ('x_min' in code or 'x_max' in code):
            # Simple approach: just replace the parameter names
            code = re.sub(r'\bx_min\s*=\s*([-\d.]+)\s*,\s*x_max\s*=\s*([-\d.]+)', 
                         r'x_range=[\1, \2]', code)
            code = re.sub(r'\by_min\s*=\s*([-\d.]+)\s*,\s*y_max\s*=\s*([-\d.]+)', 
                         r'y_range=[\1, \2]', code)
            fixes_applied.append('axes_params_updated')
        
        # Fix 6: NumberLine parameters
        if 'number_line_config' in code and 'exclude_zero_from_default_numbers' in code:
            code = re.sub(r'exclude_zero_from_default_numbers', 'include_numbers', code)
            fixes_applied.append('numberline_config_updated')
        
        # Fix 7: VMobject import (common in old manim)
        if 'VMobject' in code and 'from manim import *' in code:
            # VMobject is already in manim import *, no need to fix
            pass
        
        # Fix 8: .center -> .get_center() (deprecated property to method)
        if '.center' in code and '.get_center()' not in code:
            # Replace .center with .get_center() but avoid .get_center() cases
            # Pattern: obj.center (word boundary) or obj.center[index]
            if '.center[' in code:
                # Handle array access: obj.center[0] -> obj.get_center()[0]
                code = re.sub(r'\.center\[', '.get_center()[', code)
                fixes_applied.append('center_to_get_center')
            elif re.search(r'\.center\b', code):
                # Handle simple property access: obj.center -> obj.get_center()
                code = re.sub(r'\.center\b', '.get_center()', code)
                fixes_applied.append('center_to_get_center')
        
        # Fix 9: ShowCreation -> Create
        if 'ShowCreation' in code:
            code = re.sub(r'\bShowCreation\b', 'Create', code)
            fixes_applied.append('showcreation_to_create')
        
        # Fix 10: CONFIG dict (remove or convert to init)
        if 'CONFIG' in code and 'CONFIG = {' in code:
            # Remove CONFIG dict - it's deprecated in ManimCE
            # This regex handles multi-line CONFIG dicts with nested braces
            lines = code.split('\n')
            new_lines = []
            in_config = False
            brace_count = 0
            
            for line in lines:
                if 'CONFIG' in line and '=' in line and '{' in line:
                    in_config = True
                    brace_count = line.count('{') - line.count('}')
                    continue
                elif in_config:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        in_config = False
                    continue
                new_lines.append(line)
            
            new_code = '\n'.join(new_lines)
            if new_code != code:
                code = new_code
                fixes_applied.append('removed_config_dict')
        
        # Fix 11: Handle directional fade animations FIRST
        # Important: Do these BEFORE the general FadeInFrom fix
        directional_fades = {
            'FadeInFromDown': 'DOWN',
            'FadeInFromUp': 'UP', 
            'FadeInFromLeft': 'LEFT',
            'FadeInFromRight': 'RIGHT',
            'FadeInFromPoint': None,  # Requires special handling
            'FadeInFromLarge': None,  # scale=2 instead of shift
        }
        
        for old_anim, direction in directional_fades.items():
            if old_anim in code:
                if old_anim == 'FadeInFromLarge':
                    # FadeInFromLarge(obj) -> FadeIn(obj, scale=2)
                    code = re.sub(rf'\b{old_anim}\s*\(\s*([^)]+)\s*\)', r'FadeIn(\1, scale=2)', code)
                    fixes_applied.append('fadeinfromlarge_to_fadein')
                elif old_anim == 'FadeInFromPoint':
                    # Skip - needs point parameter
                    continue
                elif direction:
                    # FadeInFromDown(obj) -> FadeIn(obj, shift=DOWN)
                    code = re.sub(rf'\b{old_anim}\s*\(\s*([^)]+)\s*\)', rf'FadeIn(\1, shift={direction})', code)
                    fixes_applied.append(f'{old_anim.lower()}_to_fadein')
        
        # Fix 12: DrawBorderThenFill -> DrawBorderThenFill (ensure it's imported correctly)
        # Note: DrawBorderThenFill exists in ManimCE but sometimes needs proper import
        if 'DrawBorderThenFill' in code and 'from manim import *' not in code:
            # Add to imports if using specific imports
            import_pattern = r'(from manim import[^)\n]+)'
            if re.search(import_pattern, code):
                code = re.sub(import_pattern, r'\1, DrawBorderThenFill', code)
                fixes_applied.append('added_drawborderthenfill_import')
        
        # Fix 13: FadeInFrom -> FadeIn with shift parameter (general case)
        # Pattern: FadeInFrom(obj, direction) -> FadeIn(obj, shift=direction)
        fadein_pattern = r'FadeInFrom\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
        if re.search(fadein_pattern, code):
            code = re.sub(fadein_pattern, r'FadeIn(\1, shift=\2)', code)
            fixes_applied.append('fadeinfrom_to_fadein')
        
        # Fix 14: GrowFromCenter -> GrowFromCenter (ensure proper import)
        # GrowFromCenter exists in both but might need import adjustment
        if 'GrowFromCenter' in code and 'from manim import *' not in code:
            import_pattern = r'(from manim import[^)\n]+)'
            if re.search(import_pattern, code) and 'GrowFromCenter' not in code:
                code = re.sub(import_pattern, r'\1, GrowFromCenter', code)
                fixes_applied.append('added_growfromcenter_import')
        
        
        # FadeOutAndShift -> FadeOut(shift=...)
        fadeout_pattern = r'FadeOutAndShift\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
        if re.search(fadeout_pattern, code):
            code = re.sub(fadeout_pattern, r'FadeOut(\1, shift=\2)', code)
            fixes_applied.append('fadeoutandshift_to_fadeout')
        
        # Fix 15: VoiceoverScene -> Scene (remove manim-voiceover dependency)
        if 'VoiceoverScene' in code:
            # Replace VoiceoverScene with Scene in class definitions
            code = re.sub(r'\bVoiceoverScene\b', 'Scene', code)
            fixes_applied.append('voiceoverscene_to_scene')
            
            # Remove manim_voiceover imports
            code = re.sub(r'from manim_voiceover[^\n]*\n', '', code)
            code = re.sub(r'import manim_voiceover[^\n]*\n', '', code)
            
            # Remove voiceover-specific code patterns
            # Remove with self.voiceover(...) blocks but keep the code inside
            voiceover_pattern = r'with self\.voiceover\([^)]*\):\s*\n((?:[ \t]+.*\n)*)'
            matches = re.finditer(voiceover_pattern, code)
            for match in reversed(list(matches)):
                # Extract the indented code block
                indented_block = match.group(1)
                # Remove one level of indentation
                dedented_block = re.sub(r'^    ', '', indented_block, flags=re.MULTILINE)
                # Replace the with statement with the dedented code
                code = code[:match.start()] + dedented_block + code[match.end():]
            
            # Remove self.set_speech_service(...) calls
            code = re.sub(r'self\.set_speech_service\([^)]*\)\s*\n', '', code)
            
            # Comment out tracker variable updates instead of removing
            code = re.sub(r'(tracker\.set_value\([^)]*\))', r'# \1', code)
        
        return code, fixes_applied
    
    def fix_tex_string_escapes(self, code: str) -> Tuple[str, List[str]]:
        """Fix TeX string escape sequences by converting to raw strings."""
        fixes_applied = []
        
        # Simple, targeted fix for the most common case: Tex/MathTex/Text with backslashes
        # Only fix strings that contain problematic escape sequences
        tex_functions = ['MathTex', 'Text', 'Tex', 'TexMobject', 'TextMobject']
        
        for func_name in tex_functions:
            # Find all occurrences of Tex("...") or Tex('...')
            pattern = rf'{func_name}\s*\(\s*(["\'])([^"\']*)\1'
            
            for match in re.finditer(pattern, code):
                quote_char = match.group(1)
                string_content = match.group(2)
                
                # Check if this string has problematic escape sequences
                # Common LaTeX commands that cause issues: \L, \p, \c, \b, \f, \n, \r, \t
                if '\\' in string_content:
                    # Check if it's already a raw string
                    start = match.start()
                    if start > 0 and code[start-1] == 'r':
                        continue
                    
                    # Only fix if it contains invalid Python escape sequences
                    try:
                        # Try to parse as a regular string literal
                        compile(f'"{string_content}"', '<string>', 'eval')
                    except SyntaxError:
                        # This will fail for invalid escape sequences like \L
                        # Convert to raw string
                        old_text = match.group(0)
                        new_text = f'{func_name}(r{quote_char}{string_content}{quote_char})'
                        code = code.replace(old_text, new_text, 1)
                        if 'tex_string_to_raw' not in fixes_applied:
                            fixes_applied.append('tex_string_to_raw')
        
        return code, fixes_applied
    
    def apply_fixes(self, sample: Dict[str, Any]) -> FixResult:
        """Apply all conservative fixes to a sample."""
        original_code = sample['code']
        current_code = original_code
        all_fixes = []
        
        # Apply ManimGL to ManimCE conversion first (highest priority)
        current_code, manimgl_fixes = self.fix_manimgl_to_manimce(current_code)
        all_fixes.extend(manimgl_fixes)
        
        # Apply API fixes
        current_code, api_fixes = self.fix_api_calls(current_code)
        all_fixes.extend(api_fixes)
        
        # Apply TeX string escape fixes
        current_code, tex_fixes = self.fix_tex_string_escapes(current_code)
        all_fixes.extend(tex_fixes)
        
        # Apply import fixes (only if not already converted from manimlib)
        if 'manimlib_to_manim_import' not in all_fixes:
            current_code, import_fixes = self.fix_imports(current_code)
            all_fixes.extend(import_fixes)
        
        # Update stats
        self.stats['samples_processed'] += 1
        source = sample.get('source', 'unknown')
        if source not in self.stats['fixes_by_source']:
            self.stats['fixes_by_source'][source] = 0
        
        if all_fixes:
            self.stats['samples_fixed'] += 1
            self.stats['fixes_by_source'][source] += 1
            
            for fix_type in all_fixes:
                if fix_type not in self.stats['fixes_by_type']:
                    self.stats['fixes_by_type'][fix_type] = 0
                self.stats['fixes_by_type'][fix_type] += 1
        
        return FixResult(
            success=len(all_fixes) > 0,
            original_code=original_code,
            fixed_code=current_code,
            fixes_applied=all_fixes,
            fix_count=len(all_fixes)
        )
    
    def get_stats_report(self) -> str:
        """Generate a summary report of fixes applied."""
        if self.stats['samples_processed'] == 0:
            return "No samples processed."
        
        report = []
        report.append(f"Code Fixing Summary:")
        report.append(f"  Samples processed: {self.stats['samples_processed']:,}")
        report.append(f"  Samples fixed: {self.stats['samples_fixed']:,}")
        report.append(f"  Fix rate: {self.stats['samples_fixed']/self.stats['samples_processed']*100:.1f}%")
        
        if self.stats['fixes_by_type']:
            report.append(f"  Fix types:")
            for fix_type, count in sorted(self.stats['fixes_by_type'].items()):
                report.append(f"    {fix_type}: {count:,}")
        
        if self.stats['fixes_by_source']:
            report.append(f"  Fixes by source:")
            sorted_sources = sorted(self.stats['fixes_by_source'].items(), 
                                  key=lambda x: x[1], reverse=True)
            for source, count in sorted_sources[:10]:  # Top 10
                report.append(f"    {source}: {count:,}")
        
        return '\n'.join(report)


def fix_dataset_codes(samples: List[Dict[str, Any]], aggressive_mode: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply conservative code fixes to a dataset.
    
    Args:
        samples: List of samples to fix
        aggressive_mode: Enable more aggressive fixes (unused for now)
    
    Returns:
        Tuple of (fixed_samples, fix_statistics)
    """
    fixer = ManimCodeFixer(aggressive_mode=aggressive_mode)
    fixed_samples = []
    
    logger.info(f"🔧 Applying conservative code fixes to {len(samples):,} samples...")
    
    for sample in samples:
        fix_result = fixer.apply_fixes(sample)
        
        # Create new sample with fixed code
        fixed_sample = sample.copy()
        fixed_sample['code'] = fix_result.fixed_code
        
        # Track what was fixed
        if fix_result.success:
            fixed_sample['auto_fixed'] = True
            fixed_sample['fixes_applied'] = fix_result.fixes_applied
            fixed_sample['original_code'] = fix_result.original_code
        
        fixed_samples.append(fixed_sample)
    
    logger.info(fixer.get_stats_report())
    
    return fixed_samples, fixer.stats