import openai
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .base import (
    TranscriptionEngine, TranscriptionResult, Segment, Word, Speaker,
    ValidationResult, EngineCapabilities
)


class OpenAIEngine(TranscriptionEngine):
    """OpenAI API-based transcription engine (existing functionality)."""
    
    SUPPORTED_FORMATS = ['.mp3', '.m4a', '.wav', '.webm', '.mp4', '.mpga', '.mpeg']
    MAX_FILE_SIZE_MB = 25.0
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = openai.OpenAI()
        self.language = config.get('language', 'he')
        self.model = config.get('model', 'whisper-1')
        self.temperature = config.get('temperature', 0.0)
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API."""
        # Validate file first
        validation = self.validate_file(audio_path)
        if not validation.is_valid:
            raise ValueError(validation.error_message)
        
        print(f"Transcribing {audio_path} using OpenAI API...")
        
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="verbose_json",
                    temperature=self.temperature
                )
            
            # Extract basic information
            language_detected = getattr(transcript, 'language', self.language)
            duration = getattr(transcript, 'duration', 0.0)
            
            # Create segments from the response (OpenAI API provides segments)
            segments = []
            if hasattr(transcript, 'segments'):
                for i, seg in enumerate(transcript.segments):
                    segment = Segment(
                        id=i,
                        start=seg.get('start', 0.0),
                        end=seg.get('end', 0.0),
                        text=seg.get('text', ''),
                        confidence=1.0  # OpenAI doesn't provide confidence scores
                    )
                    segments.append(segment)
            else:
                # If no segments, create one for the entire transcript
                segments = [Segment(
                    id=0,
                    start=0.0,
                    end=duration,
                    text=transcript.text,
                    confidence=1.0
                )]
            
            # Create metadata
            metadata = {
                'engine': 'openai',
                'model': self.model,
                'language_requested': self.language,
                'timestamp': datetime.now().isoformat(),
                'file_name': Path(audio_path).name
            }
            
            return TranscriptionResult(
                text=transcript.text,
                language=language_detected,
                duration=duration,
                segments=segments,
                metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI transcription failed: {str(e)}")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate file for OpenAI API processing."""
        # Check file existence
        existence_result = self._validate_file_exists(file_path)
        if not existence_result.is_valid:
            return existence_result
        
        # Check file format
        format_result = self._validate_file_format(file_path, self.SUPPORTED_FORMATS)
        if not format_result.is_valid:
            return format_result
        
        # Check file size
        if existence_result.file_size_mb > self.MAX_FILE_SIZE_MB:
            return ValidationResult(
                False,
                f"File size ({existence_result.file_size_mb:.1f}MB) exceeds "
                f"{self.MAX_FILE_SIZE_MB}MB limit for OpenAI API"
            )
        
        return ValidationResult(True, file_size_mb=existence_result.file_size_mb)
    
    def get_capabilities(self) -> EngineCapabilities:
        """Return OpenAI engine capabilities."""
        return EngineCapabilities(
            supports_diarization=False,
            supports_word_timestamps=False,
            supports_local_processing=False,
            max_file_size_mb=self.MAX_FILE_SIZE_MB,
            supported_languages=['en', 'he', 'ar', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        )