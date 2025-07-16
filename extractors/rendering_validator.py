"""
Rendering validation for Manim code samples.
Tests if code actually produces video output.
"""

import subprocess
import tempfile
import os
import ast
import re
import time
import traceback
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from .code_fixer import ManimCodeFixer

logger = logging.getLogger(__name__)


def _worker_validate_sample(idx: int, sample: Dict[str, Any], timeout: int, quality: str, fix_common_issues: bool, dry_run: bool = False, save_videos_dir: Optional[str] = None, fast_mode: bool = False, use_cache: bool = True) -> Tuple[bool, Dict, Dict]:
    """Worker function for parallel validation."""
    # Suppress logging in worker processes to avoid interfering with progress bar
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    validator = RenderingValidator(timeout=timeout, quality=quality, cleanup=True, fix_common_issues=fix_common_issues, dry_run=dry_run, save_videos_dir=save_videos_dir, fast_mode=fast_mode, use_cache=use_cache)
    code = sample.get("code", "")
    sample_id = f"{sample.get('source', 'unknown')}_{idx}"
    success, details = validator.validate_render(code, sample_id)
    # Return success, details, and stats separately
    return success, details, validator.stats


class RenderingValidator:
    """Validates Manim code by attempting to render it."""
    
    def __init__(self, 
                 timeout: int = 30,
                 quality: str = "low_quality",
                 cleanup: bool = True,
                 fix_common_issues: bool = True,
                 dry_run: bool = False,
                 save_videos_dir: Optional[str] = None,
                 fast_mode: bool = False,
                 use_cache: bool = True):
        """
        Initialize rendering validator.
        
        Args:
            timeout: Max seconds to wait for render
            quality: Manim quality setting (low_quality for speed)
            cleanup: Whether to clean up temp files
            fix_common_issues: Whether to attempt auto-fixes
            dry_run: If True, only validate syntax without rendering
            save_videos_dir: Directory to save rendered videos (None to not save)
            fast_mode: If True, only render last frame as PNG instead of full video
            use_cache: If True, skip rendering if video already exists in save_videos_dir
        """
        self.timeout = timeout
        self.quality = quality
        self.cleanup = cleanup
        self.fix_common_issues = fix_common_issues
        self.dry_run = dry_run
        self.save_videos_dir = save_videos_dir
        self.fast_mode = fast_mode
        self.use_cache = use_cache
        
        # Create save directory if specified
        if self.save_videos_dir:
            Path(self.save_videos_dir).mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_validated": 0,
            "successful_renders": 0,
            "failed_renders": 0,
            "fixed_and_rendered": 0,
            "cached_videos": 0,
            "error_types": {}
        }
    
    def validate_render(self, code: str, sample_id: str = "test") -> Tuple[bool, Dict[str, Any]]:
        """
        Attempt to render code and return validation result.
        
        Returns:
            (success, details_dict)
        """
        self.stats["total_validated"] += 1
        
        # Try original code first
        success, details = self._try_render(code, sample_id)
        
        if not success and self.fix_common_issues:
            # Attempt to fix and retry
            fixed_code = self._attempt_fixes(code, details.get("error", ""), details.get("stderr", ""))
            if fixed_code != code:
                success, details = self._try_render(fixed_code, sample_id + "_fixed")
                if success:
                    details["was_fixed"] = True
                    details["fixes_applied"] = self._get_applied_fixes(code, fixed_code)
                    self.stats["fixed_and_rendered"] += 1
        
        # Update stats
        if success:
            self.stats["successful_renders"] += 1
            if details.get("cached", False):
                self.stats["cached_videos"] += 1
        else:
            self.stats["failed_renders"] += 1
            error_type = self._categorize_error(details.get("error", ""), details.get("stderr", ""))
            self.stats["error_types"][error_type] = self.stats["error_types"].get(error_type, 0) + 1
        
        return success, details
    
    def _try_render(self, code: str, sample_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to render code once."""
        # Always do basic validation first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, {
                "error": f"Syntax error at line {e.lineno}: {e.msg}",
                "line": e.lineno
            }
        
        # Check for Scene class
        scene_name = self._extract_scene_name(code)
        if not scene_name:
            return False, {"error": "No Scene class found"}
        
        # Check for construct method
        if "def construct" not in code:
            return False, {"error": "No construct method found"}
        
        # If dry run, stop here
        if self.dry_run:
            return True, {"dry_run": True, "scene_name": scene_name}
        
        # Check cache if enabled and save directory exists
        if self.use_cache and self.save_videos_dir and not self.fast_mode:
            cached_path = Path(self.save_videos_dir) / f"{sample_id}.mp4"
            if cached_path.exists() and cached_path.stat().st_size > 0:
                return True, {
                    "render_time": 0,
                    "file_size": cached_path.stat().st_size,
                    "output_path": str(cached_path),
                    "saved_path": str(cached_path),
                    "fast_mode": False,
                    "cached": True,
                    "stdout": "Video loaded from cache"
                }
        
        # Otherwise do actual render
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to temp file
            script_path = Path(tmpdir) / f"{sample_id}.py"
            script_path.write_text(code)
            
            # Prepare manim command
            output_dir = Path(tmpdir) / "media"
            cmd = [
                "manim", "-ql",  # Quick low quality
                "--disable_caching",
                "--media_dir", str(output_dir)
            ]
            
            # Add fast mode flag if enabled
            if self.fast_mode:
                cmd.append("--save_last_frame")
            
            cmd.append(str(script_path))
            
            # Extract scene name if possible
            scene_name = self._extract_scene_name(code)
            if scene_name:
                cmd.append(scene_name)
            
            start_time = time.time()
            
            try:
                # Run manim
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )
                
                render_time = time.time() - start_time
                
                if result.returncode == 0:
                    # Check if output was actually created
                    if self.fast_mode:
                        # In fast mode, look for PNG files
                        output_files = list(output_dir.rglob("*.png"))
                    else:
                        # In normal mode, look for any video files (mp4, mov, avi, etc.)
                        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
                        output_files = []
                        for ext in video_extensions:
                            output_files.extend(list(output_dir.rglob(ext)))
                        
                        # If no video files found, also check for any files in media directory
                        if not output_files:
                            all_files = list(output_dir.rglob("*"))
                            # Filter out directories and hidden files
                            output_files = [f for f in all_files if f.is_file() and not f.name.startswith('.') and f.stat().st_size > 100]
                    
                    if output_files:
                        file_size = output_files[0].stat().st_size
                        temp_output_path = str(output_files[0])
                        
                        # Save output if directory specified and not in fast mode
                        saved_output_path = None
                        if self.save_videos_dir and not self.fast_mode:
                            saved_output_path = self._save_video(output_files[0], sample_id)
                        
                        return True, {
                            "render_time": render_time,
                            "file_size": file_size,
                            "output_path": temp_output_path,
                            "saved_path": saved_output_path,
                            "fast_mode": self.fast_mode,
                            "stdout": result.stdout[-500:] if result.stdout else ""
                        }
                    else:
                        # Debug: show what files were actually created
                        all_files = list(output_dir.rglob("*"))
                        file_list = [str(f) for f in all_files if f.is_file()]
                        return False, {
                            "error": f"No {'PNG' if self.fast_mode else 'video'} file generated",
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "debug_files_found": file_list[:10]  # Show first 10 files for debugging
                        }
                else:
                    return False, {
                        "error": "Manim command failed",
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                    
            except subprocess.TimeoutExpired:
                return False, {
                    "error": "Render timeout",
                    "timeout": self.timeout
                }
            except Exception as e:
                return False, {
                    "error": f"Unexpected error: {str(e)}",
                    "traceback": traceback.format_exc()
                }
    
    def _save_video(self, video_path: Path, sample_id: str) -> Optional[str]:
        """Save video to persistent directory."""
        try:
            import shutil
            # Create a safe filename from sample_id
            safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', sample_id)
            saved_path = Path(self.save_videos_dir) / f"{safe_filename}.mp4"
            
            # Handle filename conflicts
            counter = 1
            while saved_path.exists():
                saved_path = Path(self.save_videos_dir) / f"{safe_filename}_{counter}.mp4"
                counter += 1
            
            shutil.copy2(video_path, saved_path)
            logger.info(f"Saved video: {saved_path}")
            return str(saved_path)
        except Exception as e:
            logger.warning(f"Failed to save video for {sample_id}: {e}")
            return None
    
    def _extract_scene_name(self, code: str) -> Optional[str]:
        """Extract the Scene class name from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Scene
                    for base in node.bases:
                        if isinstance(base, ast.Name) and 'Scene' in base.id:
                            return node.name
        except:
            pass
        return None
    
    def _categorize_error(self, error: str, stderr: str = "") -> str:
        """Categorize error type for statistics."""
        # Combine error and stderr for better categorization
        full_error = error.lower() + " " + stderr.lower()
        
        if "no module named" in full_error or "import" in full_error:
            return "import_error"
        elif "syntax" in full_error:
            return "syntax_error"
        elif "timeout" in full_error:
            return "timeout"
        elif "no video file" in full_error or "no png file" in full_error:
            return "no_output"
        elif "attributeerror" in full_error:
            return "attribute_error"
        elif "typeerror" in full_error:
            return "type_error"
        elif "nameerror" in full_error:
            return "name_error"
        else:
            return "other"
    
    def _attempt_fixes(self, code: str, error: str, stderr: str = "") -> str:
        """Attempt to fix common issues in code."""
        fixed_code = code
        
        # First, try applying general fixes with ManimCodeFixer
        try:
            fixer = ManimCodeFixer()
            sample = {'code': fixed_code}
            fix_result = fixer.apply_fixes(sample)
            if fix_result.success:
                fixed_code = fix_result.fixed_code
        except Exception as e:
            logger.debug(f"ManimCodeFixer failed: {e}")
        
        # Combine error and stderr for better error detection
        full_error = error.lower() + " " + stderr.lower()
        
        # Fix 1: Add missing imports
        if "no module named 'manim'" in full_error or not re.search(r'from manim import|import manim', code):
            fixed_code = "from manim import *\n" + fixed_code
        
        # Fix 2: Ensure proper indentation (convert tabs to spaces)
        fixed_code = fixed_code.replace('\t', '    ')
        
        # Fix 3: Fix common Scene inheritance issues
        fixed_code = re.sub(
            r'class\s+(\w+)\s*\(\s*\)',
            r'class \1(Scene)',
            fixed_code
        )
        
        # Fix 4: Add self parameter to construct if missing
        fixed_code = re.sub(
            r'def\s+construct\s*\(\s*\):',
            r'def construct(self):',
            fixed_code
        )
        
        # Fix 5: Fix common animation method issues
        # Change play(Create(obj)) to self.play(Create(obj))
        fixed_code = re.sub(
            r'(?<!self\.)(?<!\.)\b(play|wait|add|remove)\s*\(',
            r'self.\1(',
            fixed_code
        )
        
        # Fix 6: Handle missing Scene name in command
        if "error: the following arguments are required: scene_names" in error:
            # This is handled by extracting scene name
            pass
        
        # Fix 7: Fix deprecated animations based on NameError in stderr
        if "nameerror" in full_error:
            # ShowCreation -> Create
            if "showcreation" in full_error:
                fixed_code = re.sub(r'\bShowCreation\b', 'Create', fixed_code)
            
            # Uncreate -> Unwrite
            if "uncreate" in full_error:
                fixed_code = re.sub(r'\bUncreate\b', 'Unwrite', fixed_code)
            
            # DrawBorderThenFill -> Create
            if "drawborderthenfill" in full_error:
                fixed_code = re.sub(r'\bDrawBorderThenFill\b', 'Create', fixed_code)
            
            # FadeInFromDown -> FadeIn with shift parameter
            if "fadeinfromdown" in full_error:
                fixed_code = re.sub(r'\bFadeInFromDown\b', 'FadeIn', fixed_code)
            
            # FadeOutAndShiftDown -> FadeOut with shift parameter
            if "fadeoutandshiftdown" in full_error:
                fixed_code = re.sub(r'\bFadeOutAndShiftDown\b', 'FadeOut', fixed_code)
            
            # GrowFromCenter -> GrowFromPoint
            if "growfromcenter" in full_error:
                fixed_code = re.sub(r'\bGrowFromCenter\b', 'GrowFromPoint', fixed_code)
            
            # ShowSubmobjectsOneByOne -> AnimationGroup with lag_ratio
            if "showsubmobjectsonebyone" in full_error:
                fixed_code = re.sub(r'\bShowSubmobjectsOneByOne\b', 'Create', fixed_code)
        
        # Fix 8: Fix deprecated classes based on NameError
        if "nameerror" in full_error:
            # TexMobject -> MathTex
            if "texmobject" in full_error:
                fixed_code = re.sub(r'\bTexMobject\b', 'MathTex', fixed_code)
            
            # TextMobject -> Text
            if "textmobject" in full_error:
                fixed_code = re.sub(r'\bTextMobject\b', 'Text', fixed_code)
        
        # Fix 9: Fix attribute errors
        if "attributeerror" in full_error:
            # set_width -> .width property
            if "set_width" in full_error:
                fixed_code = re.sub(r'(\w+)\.set_width\(\s*([^)]+)\s*\)', r'\1.width = \2', fixed_code)
            
            # set_height -> .height property
            if "set_height" in full_error:
                fixed_code = re.sub(r'(\w+)\.set_height\(\s*([^)]+)\s*\)', r'\1.height = \2', fixed_code)
            
            # get_width -> .width property
            if "get_width" in full_error:
                fixed_code = re.sub(r'(\w+)\.get_width\(\s*\)', r'\1.width', fixed_code)
            
            # get_height -> .height property
            if "get_height" in full_error:
                fixed_code = re.sub(r'(\w+)\.get_height\(\s*\)', r'\1.height', fixed_code)
        
        return fixed_code
    
    def _get_applied_fixes(self, original: str, fixed: str) -> List[str]:
        """Identify which fixes were applied."""
        fixes = []
        
        if "from manim import" in fixed and "from manim import" not in original:
            fixes.append("added_imports")
        
        if original.count('\t') > fixed.count('\t'):
            fixes.append("fixed_indentation")
        
        if re.search(r'class\s+\w+\s*\(Scene\)', fixed) and not re.search(r'class\s+\w+\s*\(Scene\)', original):
            fixes.append("fixed_scene_inheritance")
        
        if "self.play" in fixed and "self.play" not in original and "play" in original:
            fixes.append("added_self_to_methods")
        
        # Check for deprecated animation fixes
        deprecated_animations = [
            ("ShowCreation", "Create"),
            ("Uncreate", "Unwrite"),
            ("DrawBorderThenFill", "Create"),
            ("FadeInFromDown", "FadeIn"),
            ("FadeOutAndShiftDown", "FadeOut"),
            ("GrowFromCenter", "GrowFromPoint"),
            ("ShowSubmobjectsOneByOne", "Create")
        ]
        
        for old_name, new_name in deprecated_animations:
            if old_name in original and new_name in fixed and old_name not in fixed:
                fixes.append(f"replaced_{old_name.lower()}")
        
        # Check for deprecated class fixes
        if "TexMobject" in original and "MathTex" in fixed:
            fixes.append("replaced_texmobject")
        if "TextMobject" in original and "Text" in fixed:
            fixes.append("replaced_textmobject")
        
        # Check for property fixes
        if "set_width(" in original and ".width =" in fixed:
            fixes.append("converted_set_width")
        if "set_height(" in original and ".height =" in fixed:
            fixes.append("converted_set_height")
        if "get_width()" in original and ".width" in fixed and "get_width()" not in fixed:
            fixes.append("converted_get_width")
        if "get_height()" in original and ".height" in fixed and "get_height()" not in fixed:
            fixes.append("converted_get_height")
        
        return fixes
    
    def get_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("=== Rendering Validation Report ===")
        if self.dry_run:
            report.append("Mode: DRY RUN (syntax validation only)")
        report.append(f"Total validated: {self.stats['total_validated']}")
        report.append(f"Successful {'validations' if self.dry_run else 'renders'}: {self.stats['successful_renders']} ({self.stats['successful_renders']/max(1, self.stats['total_validated'])*100:.1f}%)")
        report.append(f"Failed {'validations' if self.dry_run else 'renders'}: {self.stats['failed_renders']} ({self.stats['failed_renders']/max(1, self.stats['total_validated'])*100:.1f}%)")
        
        if self.fix_common_issues:
            report.append(f"Fixed and rendered: {self.stats['fixed_and_rendered']} ({self.stats['fixed_and_rendered']/max(1, self.stats['total_validated'])*100:.1f}%)")
        
        if self.use_cache and self.stats.get('cached_videos', 0) > 0:
            report.append(f"Loaded from cache: {self.stats['cached_videos']} ({self.stats['cached_videos']/max(1, self.stats['total_validated'])*100:.1f}%)")
        
        if self.stats["error_types"]:
            report.append("\nError distribution:")
            for error_type, count in sorted(self.stats["error_types"].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {error_type}: {count}")
        
        return "\n".join(report)


class BatchRenderValidator:
    """Efficiently validate multiple samples with progress tracking."""
    
    def __init__(self, 
                 validator: Optional[RenderingValidator] = None,
                 max_workers: Optional[int] = None,
                 save_failed_samples: bool = True,
                 dry_run: bool = False,
                 max_failed_samples: int = 1000):
        """
        Initialize batch validator.
        
        Args:
            validator: RenderingValidator instance
            max_workers: Max parallel validation processes (defaults to CPU count // 2)
            save_failed_samples: Whether to save failed samples for analysis
            dry_run: If True, only validate syntax without rendering
        """
        import multiprocessing
        self.validator = validator or RenderingValidator(dry_run=dry_run)
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() // 2)
        self.save_failed_samples = save_failed_samples
        self.failed_samples = []
        self.dry_run = dry_run
        self.max_failed_samples = max_failed_samples
    
    def validate_dataset(self, samples: List[Dict[str, Any]], 
                        progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a dataset of samples using parallel processing.
        
        Returns:
            (valid_samples, invalid_samples)
        """
        if self.max_workers > 1 and len(samples) > 10:
            # Use parallel processing
            return self._validate_parallel(samples, progress_callback)
        else:
            # Fall back to sequential for small datasets
            return self._validate_sequential(samples, progress_callback)
    
    def _validate_sequential(self, samples: List[Dict[str, Any]], 
                           progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
        """Sequential validation."""
        valid_samples = []
        invalid_samples = []
        
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, len(samples))
            
            code = sample.get("code", "")
            sample_id = f"{sample.get('source', 'unknown')}_{i}"
            
            success, details = self.validator.validate_render(code, sample_id)
            
            if success:
                sample["rendering_validated"] = True
                if details.get("was_fixed"):
                    sample["auto_fixed"] = True
                    sample["fixes_applied"] = details["fixes_applied"]
                valid_samples.append(sample)
            else:
                sample["rendering_failed"] = True
                sample["render_error"] = details.get("error", "Unknown error")
                invalid_samples.append(sample)
                
                if self.save_failed_samples and len(self.failed_samples) < self.max_failed_samples:
                    self.failed_samples.append({
                        "source": sample.get("source"),
                        "description": sample.get("description", "")[:200],
                        "code": sample.get("code", ""),
                        "error": details.get("error", ""),
                        "stderr": details.get("stderr", "")[:500],
                        "sample_id": f"{sample.get('source', 'unknown')}_{i}"
                    })
        
        return valid_samples, invalid_samples
    
    def _validate_parallel(self, samples: List[Dict[str, Any]], 
                         progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
        """Parallel validation using ProcessPoolExecutor."""
        from functools import partial
        
        # Create worker function with fixed parameters
        worker = partial(_worker_validate_sample,
                        timeout=self.validator.timeout,
                        quality=self.validator.quality,
                        fix_common_issues=self.validator.fix_common_issues,
                        dry_run=self.dry_run,
                        save_videos_dir=self.validator.save_videos_dir,
                        fast_mode=self.validator.fast_mode,
                        use_cache=self.validator.use_cache)
        
        # Process samples in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = [(executor.submit(worker, i, sample), i) 
                      for i, sample in enumerate(samples)]
            
            # Collect results
            results = [None] * len(samples)
            completed = 0
            
            for future, idx in futures:
                try:
                    results[idx] = future.result()
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(samples))
                except Exception as e:
                    logger.error(f"Failed to process sample {idx}: {e}")
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(samples))
        
        # Sort results
        valid_samples = []
        invalid_samples = []
        
        # Aggregate stats from all workers
        aggregated_stats = {
            "total_validated": 0,
            "successful_renders": 0,
            "failed_renders": 0,
            "fixed_and_rendered": 0,
            "cached_videos": 0,
            "error_types": {}
        }
        
        for i, result in enumerate(results):
            if result is None:
                continue
                
            sample = samples[i]
            success, details, worker_stats = result
            
            # Aggregate stats from worker
            for key in ["total_validated", "successful_renders", "failed_renders", "fixed_and_rendered", "cached_videos"]:
                aggregated_stats[key] += worker_stats.get(key, 0)
            
            # Aggregate error types
            for error_type, count in worker_stats.get("error_types", {}).items():
                aggregated_stats["error_types"][error_type] = aggregated_stats["error_types"].get(error_type, 0) + count
            
            if success:
                sample["rendering_validated"] = True
                if details.get("was_fixed"):
                    sample["auto_fixed"] = True
                    sample["fixes_applied"] = details["fixes_applied"]
                valid_samples.append(sample)
            else:
                sample["rendering_failed"] = True
                sample["render_error"] = details.get("error", "Unknown error")
                invalid_samples.append(sample)
                
                if self.save_failed_samples and len(self.failed_samples) < self.max_failed_samples:
                    self.failed_samples.append({
                        "source": sample.get("source"),
                        "description": sample.get("description", "")[:200],
                        "code": sample.get("code", ""),
                        "error": details.get("error", ""),
                        "stderr": details.get("stderr", "")[:500],
                        "sample_id": f"{sample.get('source', 'unknown')}_{i}"
                    })
        
        # Update validator stats with aggregated stats
        self.validator.stats = aggregated_stats
        
        return valid_samples, invalid_samples
    
    def get_failed_samples_report(self) -> List[Dict]:
        """Get examples of failed samples for debugging."""
        return self.failed_samples