from pathlib import Path
from typing import Optional


class AudioProcessor:
    """Utility class for audio processing operations."""
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in megabytes."""
        path = Path(file_path)
        return path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported audio formats."""
        return ['.mp3', '.m4a', '.wav', '.webm', '.mp4', '.mpga', '.mpeg', '.flac', '.ogg']
    
    @staticmethod
    def estimate_duration(file_path: str) -> Optional[float]:
        """Estimate audio duration (placeholder for future implementation)."""
        # This would require additional audio libraries like librosa or pydub
        return None