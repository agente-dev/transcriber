"""
Language-specific model management for word-level alignment.
Handles loading and caching of wav2vec2 models for different languages.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

try:
    import torch
    torch_available = True
except ImportError:
    torch = None
    torch_available = False

try:
    import whisperx
    whisperx_available = True
except ImportError:
    whisperx = None
    whisperx_available = False

logger = logging.getLogger(__name__)


class LanguageModelManager:
    """Manages language-specific alignment models."""
    
    # Language model configurations
    LANGUAGE_MODELS = {
        'he': {
            'model_name': 'ivrit-ai/wav2vec2-large-xlsr-53-hebrew',
            'display_name': 'Hebrew (wav2vec2)',
            'rtl': True
        },
        'en': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-english', 
            'display_name': 'English (wav2vec2)',
            'rtl': False
        },
        'ar': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
            'display_name': 'Arabic (wav2vec2)',
            'rtl': True
        },
        'es': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
            'display_name': 'Spanish (wav2vec2)',
            'rtl': False
        },
        'fr': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
            'display_name': 'French (wav2vec2)',
            'rtl': False
        },
        'de': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
            'display_name': 'German (wav2vec2)',
            'rtl': False
        },
        'it': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-italian',
            'display_name': 'Italian (wav2vec2)',
            'rtl': False
        },
        'pt': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
            'display_name': 'Portuguese (wav2vec2)',
            'rtl': False
        },
        'ru': {
            'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
            'display_name': 'Russian (wav2vec2)',
            'rtl': False
        }
    }
    
    # Fallback to multilingual model
    DEFAULT_MODEL = {
        'model_name': 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
        'display_name': 'Multilingual (wav2vec2)',
        'rtl': False
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, device: str = 'auto'):
        """
        Initialize language model manager.
        
        Args:
            cache_dir: Directory to cache models (default: ~/.whisperx/models/alignment)
            device: Device for model loading ('cuda', 'cpu', 'auto')
        """
        if not whisperx_available:
            raise ImportError("WhisperX is not available. Please install: pip install whisperx")
        
        if not torch_available:
            raise ImportError("PyTorch is not available. Please install: pip install torch")
        
        self.cache_dir = cache_dir or Path.home() / ".whisperx" / "models" / "alignment"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._setup_device(device)
        self.loaded_models: Dict[str, Tuple[Any, Any]] = {}  # language -> (model, metadata)
        
        logger.info(f"Language model manager initialized (device: {self.device})")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device selection."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU detected for alignment: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                device = 'cpu'
                logger.info("Using CPU for word alignment")
        
        elif device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available for alignment, falling back to CPU")
                device = 'cpu'
        
        return device
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages with display names."""
        return {lang: config['display_name'] for lang, config in self.LANGUAGE_MODELS.items()}
    
    def is_rtl_language(self, language: str) -> bool:
        """Check if language uses right-to-left text direction."""
        return self.LANGUAGE_MODELS.get(language, self.DEFAULT_MODEL).get('rtl', False)
    
    def load_model(self, language: str) -> Optional[Tuple[Any, Any]]:
        """
        Load alignment model for specified language.
        
        Args:
            language: Language code (e.g., 'he', 'en')
            
        Returns:
            Tuple of (model, metadata) or None if loading failed
        """
        # Return cached model if available
        if language in self.loaded_models:
            logger.debug(f"Using cached alignment model for {language}")
            return self.loaded_models[language]
        
        # Get model configuration
        model_config = self.LANGUAGE_MODELS.get(language)
        if not model_config:
            logger.warning(f"No alignment model available for language '{language}', using default")
            model_config = self.DEFAULT_MODEL
            language = 'default'
        
        try:
            logger.info(f"Loading alignment model for {language}: {model_config['display_name']}")
            
            # Use WhisperX to load alignment model
            model, metadata = whisperx.load_align_model(
                language_code=language if language != 'default' else 'en',
                device=self.device
            )
            
            # Cache the loaded model
            self.loaded_models[language] = (model, metadata)
            
            logger.info(f"✓ Alignment model loaded for {language}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load alignment model for {language}: {e}")
            
            # Try fallback to English if not already trying English
            if language != 'en' and language != 'default':
                logger.info("Attempting fallback to English alignment model...")
                try:
                    model, metadata = whisperx.load_align_model(
                        language_code='en',
                        device=self.device
                    )
                    
                    # Cache as fallback
                    self.loaded_models[language] = (model, metadata)
                    logger.info(f"✓ Using English alignment model as fallback for {language}")
                    return model, metadata
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback alignment model also failed: {fallback_error}")
            
            return None
    
    def get_model_info(self, language: str) -> Dict[str, Any]:
        """Get information about alignment model for language."""
        model_config = self.LANGUAGE_MODELS.get(language, self.DEFAULT_MODEL)
        
        info = {
            'language': language,
            'model_name': model_config['model_name'],
            'display_name': model_config['display_name'],
            'rtl': model_config['rtl'],
            'device': self.device,
            'loaded': language in self.loaded_models,
            'cache_dir': str(self.cache_dir)
        }
        
        return info
    
    def clear_cache(self):
        """Clear all loaded models from memory."""
        logger.info("Clearing alignment model cache")
        self.loaded_models.clear()
    
    def preload_models(self, languages: list):
        """Preload models for specified languages."""
        logger.info(f"Preloading alignment models for: {languages}")
        
        for language in languages:
            try:
                self.load_model(language)
            except Exception as e:
                logger.warning(f"Failed to preload model for {language}: {e}")
    
    def validate_language_support(self, language: str) -> Dict[str, Any]:
        """Validate if language is supported for alignment."""
        result = {
            'supported': False,
            'model_available': False,
            'fallback_required': False,
            'rtl_support': False,
            'warnings': []
        }
        
        if language in self.LANGUAGE_MODELS:
            result['supported'] = True
            result['rtl_support'] = self.LANGUAGE_MODELS[language]['rtl']
            
            # Try to load model to check availability
            try:
                model_tuple = self.load_model(language)
                result['model_available'] = model_tuple is not None
            except Exception:
                result['model_available'] = False
                result['warnings'].append(f"Model loading failed for {language}")
        
        else:
            result['fallback_required'] = True
            result['warnings'].append(f"No native support for {language}, will use multilingual model")
        
        return result
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for loaded models."""
        memory_info = {
            'loaded_models': len(self.loaded_models),
            'languages': list(self.loaded_models.keys()),
            'device': self.device
        }
        
        if torch.cuda.is_available() and self.device == 'cuda':
            memory_info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
            memory_info['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**2)  # MB
        
        return memory_info