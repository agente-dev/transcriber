from typing import Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from engines import TranscriptionEngine, TranscriptionResult, OpenAIEngine
from config import Config


class TranscriptionRouter:
    """Routes transcription requests to the appropriate engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = self._select_engine()
    
    def _select_engine(self) -> TranscriptionEngine:
        """Select the appropriate transcription engine based on configuration."""
        if self.config.engine == "whisperx" or self.config.local:
            # Import here to avoid loading WhisperX if not needed
            try:
                from engines.whisperx_engine import WhisperXEngine
                return WhisperXEngine(self.config)
            except ImportError as e:
                print(f"Warning: WhisperX not available ({e}), falling back to OpenAI API")
                return OpenAIEngine(self._config_to_dict())
        
        return OpenAIEngine(self._config_to_dict())
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config object to dictionary for engine initialization."""
        return {
            'language': self.config.language,
            'model': self.config.openai_model,
            'temperature': self.config.temperature
        }
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file using the selected engine."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        return self.engine.transcribe(audio_path)
    
    def validate_file(self, audio_path: str):
        """Validate audio file using the selected engine."""
        return self.engine.validate_file(audio_path)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current engine."""
        capabilities = self.engine.get_capabilities()
        return {
            'engine_type': self.config.engine,
            'capabilities': capabilities,
            'config': self.config
        }