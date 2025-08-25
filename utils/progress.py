from tqdm import tqdm
from typing import Optional, Any


class ProgressBar:
    """Wrapper for tqdm progress bars with consistent styling."""
    
    def __init__(self, total: Optional[int] = None, description: str = "", unit: str = "it"):
        self.pbar = tqdm(
            total=total,
            desc=description,
            unit=unit,
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


def create_download_progress_callback(description: str = "Downloading"):
    """Create a progress callback for model downloads."""
    pbar = None
    
    def callback(current: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = ProgressBar(total=total, description=description, unit="B")
        else:
            pbar.update(current - pbar.pbar.n)
    
    return callback