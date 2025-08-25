from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Word:
    """Represents a single word with timing information."""
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None


@dataclass
class Segment:
    """Represents a segment of transcribed audio."""
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0
    words: Optional[List[Word]] = None


@dataclass
class Speaker:
    """Represents a detected speaker in the audio."""
    id: str
    label: str
    segments: List[int]
    total_duration: float
    percentage: float


@dataclass
class TranscriptionResult:
    """Complete transcription result with all metadata."""
    text: str
    language: str
    duration: float
    segments: List[Segment]
    speakers: Optional[List[Speaker]] = None
    words: Optional[List[Word]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    error_message: Optional[str] = None
    file_size_mb: Optional[float] = None


@dataclass
class EngineCapabilities:
    """Describes the capabilities of a transcription engine."""
    supports_diarization: bool = False
    supports_word_timestamps: bool = False
    supports_local_processing: bool = False
    max_file_size_mb: Optional[float] = None
    supported_languages: List[str] = None


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file and return structured result."""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate that the file can be processed by this engine."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """Return the capabilities of this engine."""
        pass
    
    def _validate_file_exists(self, file_path: str) -> ValidationResult:
        """Common file existence validation."""
        path = Path(file_path)
        
        if not path.exists():
            return ValidationResult(False, f"File does not exist: {file_path}")
        
        if not path.is_file():
            return ValidationResult(False, f"Path is not a file: {file_path}")
        
        # Calculate file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        return ValidationResult(True, file_size_mb=file_size_mb)
    
    def _validate_file_format(self, file_path: str, supported_formats: List[str]) -> ValidationResult:
        """Common file format validation."""
        path = Path(file_path)
        file_extension = path.suffix.lower()
        
        if file_extension not in supported_formats:
            return ValidationResult(
                False,
                f"Unsupported file format '{file_extension}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        return ValidationResult(True)