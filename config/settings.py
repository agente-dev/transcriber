import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class WhisperXConfig:
    """WhisperX-specific configuration."""
    model: str = 'large-v2'
    device: str = 'auto'
    compute_type: str = 'float16'
    batch_size: int = 16


@dataclass
class DiarizationConfig:
    """Speaker diarization configuration."""
    enabled: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    clustering: str = 'AgglomerativeClustering'


@dataclass
class AlignmentConfig:
    """Word alignment configuration."""
    enabled: bool = True
    language_models: Dict[str, str] = field(default_factory=lambda: {
        'he': 'ivrit-ai/wav2vec2-large-xlsr-53-hebrew',
        'en': 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
    })


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration: float = 0.25


@dataclass
class OutputConfig:
    """Output formatting configuration."""
    default_format: str = 'markdown'
    include_word_timestamps: bool = False
    include_confidence_scores: bool = False


@dataclass
class Config:
    """Main configuration class."""
    # Engine selection
    engine: str = 'openai'  # 'openai' or 'whisperx'
    
    # OpenAI settings
    openai_model: str = 'whisper-1'
    language: str = 'he'
    temperature: float = 0.0
    
    # WhisperX settings
    whisperx: WhisperXConfig = field(default_factory=WhisperXConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # API keys
    huggingface_token: Optional[str] = None
    
    # Local processing flags
    local: bool = False
    word_timestamps: bool = False
    diarize: bool = False
    
    # CLI overrides
    output_format: str = 'markdown'
    output_path: Optional[str] = None
    verbose: bool = False


def load_config(config_path: Optional[str] = None, **kwargs) -> Config:
    """Load configuration from file and environment."""
    config_dict = {}
    
    # Load from YAML file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config_dict.update(file_config)
    
    # Load from default config location
    default_config_path = Path.home() / '.whisperx' / 'config' / 'default.yaml'
    if default_config_path.exists():
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
            if default_config:
                # Merge with file config, giving precedence to explicit config
                for key, value in default_config.items():
                    if key not in config_dict:
                        config_dict[key] = value
    
    # Load from environment variables
    env_config = {
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN'),
        'language': os.getenv('WHISPER_LANGUAGE', 'he'),
        'engine': os.getenv('WHISPER_ENGINE', 'openai')
    }
    
    # Remove None values
    env_config = {k: v for k, v in env_config.items() if v is not None}
    config_dict.update(env_config)
    
    # Apply CLI overrides
    config_dict.update(kwargs)
    
    # Handle nested configurations
    whisperx_config = WhisperXConfig(**config_dict.get('whisperx', {}))
    diarization_config = DiarizationConfig(**config_dict.get('diarization', {}))
    alignment_config = AlignmentConfig(**config_dict.get('alignment', {}))
    vad_config = VADConfig(**config_dict.get('vad', {}))
    output_config = OutputConfig(**config_dict.get('output', {}))
    
    # Remove nested configs from main dict to avoid duplication
    for key in ['whisperx', 'diarization', 'alignment', 'vad', 'output']:
        config_dict.pop(key, None)
    
    return Config(
        whisperx=whisperx_config,
        diarization=diarization_config,
        alignment=alignment_config,
        vad=vad_config,
        output=output_config,
        **config_dict
    )


def create_default_config_file():
    """Create default configuration file."""
    config_dir = Path.home() / '.whisperx' / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        'whisperx': {
            'model': 'large-v2',
            'device': 'auto',
            'compute_type': 'float16',
            'batch_size': 16
        },
        'diarization': {
            'enabled': False,
            'min_speakers': None,
            'max_speakers': None,
            'clustering': 'AgglomerativeClustering'
        },
        'alignment': {
            'enabled': True,
            'language_models': {
                'he': 'ivrit-ai/wav2vec2-large-xlsr-53-hebrew',
                'en': 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
            }
        },
        'vad': {
            'enabled': True,
            'threshold': 0.5,
            'min_speech_duration': 0.25
        },
        'output': {
            'default_format': 'markdown',
            'include_word_timestamps': False,
            'include_confidence_scores': False
        }
    }
    
    config_file = config_dir / 'default.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Created default configuration at: {config_file}")