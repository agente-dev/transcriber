from .base import TranscriptionEngine, TranscriptionResult, Segment, Word, Speaker
from .openai_engine import OpenAIEngine

# Try to import WhisperX engine, but don't fail if dependencies are missing
try:
    from .whisperx_engine import WhisperXEngine
    whisperx_available = True
except ImportError:
    WhisperXEngine = None
    whisperx_available = False

__all__ = [
    'TranscriptionEngine',
    'TranscriptionResult',
    'Segment',
    'Word',
    'Speaker',
    'OpenAIEngine'
]

if whisperx_available:
    __all__.append('WhisperXEngine')