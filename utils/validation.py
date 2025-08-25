from pathlib import Path
from typing import List, Tuple


def validate_audio_file(file_path: str, supported_formats: List[str]) -> Tuple[bool, str]:
    """Validate audio file format and existence."""
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File does not exist: {file_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    file_extension = path.suffix.lower()
    if file_extension not in supported_formats:
        return False, f"Unsupported format '{file_extension}'. Supported: {', '.join(supported_formats)}"
    
    return True, "Valid audio file"