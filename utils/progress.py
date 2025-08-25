from tqdm import tqdm
from typing import Optional, Any, Callable, Union, Dict
import time
import sys
from contextlib import contextmanager


class ProgressBar:
    """Wrapper for tqdm progress bars with consistent styling."""
    
    def __init__(self, total: Optional[int] = None, description: str = "", unit: str = "it", 
                 disable: bool = False, leave: bool = True):
        self.pbar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            disable=disable,
            leave=leave,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update(self, n: int = 1):
        """Update progress bar."""
        self.pbar.update(n)
    
    def set_description(self, description: str):
        """Update description."""
        self.pbar.set_description(description)
    
    def set_postfix(self, **kwargs):
        """Set postfix information."""
        self.pbar.set_postfix(**kwargs)
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SpinnerProgress:
    """Simple spinner for indeterminate progress."""
    
    def __init__(self, description: str = "", disable: bool = False):
        self.description = description
        self.disable = disable
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_char = 0
        self.start_time = None
        
    def start(self):
        """Start the spinner."""
        if not self.disable:
            self.start_time = time.time()
            print(f"{self.description}", end="", flush=True)
    
    def update(self, message: str = None):
        """Update spinner."""
        if not self.disable:
            if message:
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(f"\r{self.description} {self.spinner_chars[self.current_char]} {message} [{elapsed:.1f}s]", 
                      end="", flush=True)
            else:
                print(f"\r{self.description} {self.spinner_chars[self.current_char]}", end="", flush=True)
            self.current_char = (self.current_char + 1) % len(self.spinner_chars)
    
    def finish(self, final_message: str = "Complete"):
        """Finish the spinner."""
        if not self.disable:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"\r{self.description} ✓ {final_message} [{elapsed:.1f}s]")
        else:
            print(f"{self.description} ✓ {final_message}")


class MultiStageProgress:
    """Progress tracker for multi-stage operations."""
    
    def __init__(self, stages: Dict[str, int], description: str = "", disable: bool = False):
        """
        Initialize multi-stage progress tracker.
        
        Args:
            stages: Dictionary mapping stage names to their total steps
            description: Overall description of the operation
            disable: Whether to disable progress display
        """
        self.stages = stages
        self.description = description
        self.disable = disable
        self.current_stage = None
        self.stage_progress = {}
        self.overall_progress = None
        self.start_time = time.time()
        
        # Calculate total steps across all stages
        total_steps = sum(stages.values())
        if not disable:
            self.overall_progress = ProgressBar(
                total=total_steps, 
                description=description, 
                unit="steps",
                disable=disable
            )
    
    def start_stage(self, stage_name: str):
        """Start a new stage."""
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        self.current_stage = stage_name
        if not self.disable:
            self.overall_progress.set_description(f"{self.description} - {stage_name}")
    
    def update_stage(self, steps: int = 1, message: str = None):
        """Update current stage progress."""
        if self.current_stage is None:
            raise RuntimeError("No stage is currently active")
        
        if not self.disable:
            self.overall_progress.update(steps)
            if message:
                self.overall_progress.set_postfix_str(message)
    
    def finish_stage(self, message: str = None):
        """Finish current stage."""
        if self.current_stage and not self.disable:
            if message:
                self.overall_progress.set_postfix_str(f"{self.current_stage} - {message}")
        self.current_stage = None
    
    def finish(self, final_message: str = "Complete"):
        """Finish all stages."""
        if not self.disable:
            elapsed = time.time() - self.start_time
            self.overall_progress.set_description(f"{self.description} - {final_message} ({elapsed:.1f}s)")
            self.overall_progress.close()


@contextmanager
def progress_context(description: str, disable: bool = False):
    """Context manager for simple progress indication."""
    spinner = SpinnerProgress(description, disable=disable)
    try:
        spinner.start()
        yield spinner
    finally:
        spinner.finish()


def create_download_progress_callback(description: str = "Downloading", disable: bool = False):
    """Create a progress callback for model downloads."""
    pbar = None
    
    def callback(current: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = ProgressBar(
                total=total, 
                description=description, 
                unit="B", 
                disable=disable,
                leave=False
            )
            pbar.pbar.unit_scale = True
        else:
            pbar.update(current - pbar.pbar.n)
            
        if current >= total and pbar:
            pbar.close()
    
    return callback


def create_transcription_progress(total_duration: float, description: str = "Transcribing", 
                                disable: bool = False) -> ProgressBar:
    """Create progress bar for transcription operations."""
    return ProgressBar(
        total=int(total_duration),
        description=description,
        unit="sec",
        disable=disable
    )


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"


def estimate_processing_time(file_size_mb: float, device: str = "cpu") -> float:
    """Estimate processing time based on file size and device."""
    # Rough estimates based on typical performance
    if device == "cuda":
        # GPU processing: ~70x realtime, assuming 1MB ≈ 1 minute of audio
        return file_size_mb * 60 / 70  # seconds
    else:
        # CPU processing: ~5x realtime
        return file_size_mb * 60 / 5  # seconds


class OperationProgress:
    """High-level progress tracker for entire transcription operations."""
    
    def __init__(self, stages: list, file_path: str, disable: bool = False):
        self.stages = stages
        self.file_path = file_path
        self.disable = disable
        self.current_stage_idx = 0
        self.start_time = time.time()
        
        if not disable:
            print(f"🎤 Starting transcription of: {file_path}")
    
    def start_stage(self, stage_name: str):
        """Start a specific stage."""
        if not self.disable:
            elapsed = time.time() - self.start_time
            print(f"⏳ {stage_name}... [{elapsed:.1f}s elapsed]")
    
    def finish_stage(self, stage_name: str):
        """Finish a specific stage."""
        if not self.disable:
            elapsed = time.time() - self.start_time
            print(f"✓ {stage_name} complete [{elapsed:.1f}s elapsed]")
    
    def finish(self):
        """Finish entire operation."""
        if not self.disable:
            total_elapsed = time.time() - self.start_time
            print(f"🎉 Transcription complete in {format_duration(total_elapsed)}")
    
    def error(self, stage_name: str, error_msg: str):
        """Report error in a stage."""
        if not self.disable:
            elapsed = time.time() - self.start_time
            print(f"❌ {stage_name} failed: {error_msg} [{elapsed:.1f}s elapsed]")


# Global progress settings
_global_quiet = False

def set_global_quiet(quiet: bool):
    """Set global quiet mode for all progress indicators."""
    global _global_quiet
    _global_quiet = quiet

def should_show_progress() -> bool:
    """Check if progress should be shown."""
    return not _global_quiet