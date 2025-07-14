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
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _worker_validate_sample(idx: int, sample: Dict[str, Any], timeout: int, quality: str, fix_common_issues: bool, dry_run: bool = False, save_videos_dir: Optional[str] = None) -> Tuple[bool, Dict]:
    """Worker function for parallel validation."""
    validator = RenderingValidator(timeout=timeout, quality=quality, cleanup=True, fix_common_issues=fix_common_issues, dry_run=dry_run, save_videos_dir=save_videos_dir)
    code = sample.get("code", "")
    sample_id = f"{sample.get('source', 'unknown')}_{idx}"
    return validator.validate_render(code, sample_id)


class RenderingValidator:
    """Validates Manim code by attempting to render it."""
    
    def __init__(self, 
                 timeout: int = 30,
                 quality: str = "low_quality",
                 cleanup: bool = True,
                 fix_common_issues: bool = True,
                 dry_run: bool = False,
                 save_videos_dir: Optional[str] = None):
        """
        Initialize rendering validator.
        
        Args:
            timeout: Max seconds to wait for render
            quality: Manim quality setting (low_quality for speed)
            cleanup: Whether to clean up temp files
            fix_common_issues: Whether to attempt auto-fixes
            dry_run: If True, only validate syntax without rendering
            save_videos_dir: Directory to save rendered videos (None to not save)
        """
        self.timeout = timeout
        self.quality = quality
        self.cleanup = cleanup
        self.fix_common_issues = fix_common_issues
        self.dry_run = dry_run
        self.save_videos_dir = save_videos_dir
        
        # Create save directory if specified
        if self.save_videos_dir:
            Path(self.save_videos_dir).mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_validated": 0,
            "successful_renders": 0,
            "failed_renders": 0,
            "fixed_and_rendered": 0,
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
            fixed_code = self._attempt_fixes(code, details.get("error", ""))
            if fixed_code != code:
                success, details = self._try_render(fixed_code, sample_id + "_fixed")
                if success:
                    details["was_fixed"] = True
                    details["fixes_applied"] = self._get_applied_fixes(code, fixed_code)
                    self.stats["fixed_and_rendered"] += 1
        
        # Update stats
        if success:
            self.stats["successful_renders"] += 1
        else:
            self.stats["failed_renders"] += 1
            error_type = self._categorize_error(details.get("error", ""))
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
                "--media_dir", str(output_dir),
                str(script_path)
            ]
            
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
                    # Check if video was actually created
                    video_files = list(output_dir.rglob("*.mp4"))
                    if video_files:
                        video_size = video_files[0].stat().st_size
                        temp_video_path = str(video_files[0])
                        
                        # Save video if directory specified
                        saved_video_path = None
                        if self.save_videos_dir:
                            saved_video_path = self._save_video(video_files[0], sample_id)
                        
                        return True, {
                            "render_time": render_time,
                            "video_size": video_size,
                            "video_path": temp_video_path,
                            "saved_video_path": saved_video_path,
                            "stdout": result.stdout[-500:] if result.stdout else ""
                        }
                    else:
                        return False, {
                            "error": "No video file generated",
                            "stdout": result.stdout,
                            "stderr": result.stderr
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
    
    def _categorize_error(self, error: str) -> str:
        """Categorize error type for statistics."""
        error_lower = error.lower()
        
        if "no module named" in error_lower or "import" in error_lower:
            return "import_error"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "timeout" in error_lower:
            return "timeout"
        elif "no video file" in error_lower:
            return "no_output"
        elif "attributeerror" in error_lower:
            return "attribute_error"
        elif "typeerror" in error_lower:
            return "type_error"
        elif "nameerror" in error_lower:
            return "name_error"
        else:
            return "other"
    
    def _attempt_fixes(self, code: str, error: str) -> str:
        """Attempt to fix common issues in code."""
        fixed_code = code
        
        # Fix 1: Add missing imports
        if "no module named 'manim'" in error.lower() or not re.search(r'from manim import|import manim', code):
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
                 dry_run: bool = False):
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
                
                if self.save_failed_samples and len(self.failed_samples) < 100:
                    self.failed_samples.append({
                        "source": sample.get("source"),
                        "description": sample.get("description", "")[:200],
                        "error": details.get("error", ""),
                        "stderr": details.get("stderr", "")[:500]
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
                        save_videos_dir=self.validator.save_videos_dir)
        
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
        
        for i, result in enumerate(results):
            if result is None:
                continue
                
            sample = samples[i]
            success, details = result
            
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
                
                if self.save_failed_samples and len(self.failed_samples) < 100:
                    self.failed_samples.append({
                        "source": sample.get("source"),
                        "description": sample.get("description", "")[:200],
                        "error": details.get("error", ""),
                        "stderr": details.get("stderr", "")[:500]
                    })
        
        return valid_samples, invalid_samples
    
    def get_failed_samples_report(self) -> List[Dict]:
        """Get examples of failed samples for debugging."""
        return self.failed_samples