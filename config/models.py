import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from huggingface_hub import hf_hub_download, login
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from utils.progress import create_download_progress_callback, progress_context, should_show_progress


class ModelManager:
    """Manages WhisperX and diarization models."""
    
    def __init__(self, cache_dir: Optional[str] = None, hf_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.whisperx'
        self.models_dir = self.cache_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Hugging Face authentication
        if hf_token:
            self._setup_huggingface_auth(hf_token)
        
        # Model configurations
        self.whisper_models = {
            'tiny': {'size': '39MB', 'params': '39M'},
            'base': {'size': '74MB', 'params': '74M'},
            'small': {'size': '244MB', 'params': '244M'},
            'medium': {'size': '769MB', 'params': '769M'},
            'large': {'size': '1550MB', 'params': '1550M'},
            'large-v2': {'size': '1550MB', 'params': '1550M'},
            'large-v3': {'size': '1550MB', 'params': '1550M'},
        }
        
        self.alignment_models = {
            'he': 'ivrit-ai/wav2vec2-large-xlsr-53-hebrew',
            'en': 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
            'ar': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
            'fr': 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
            'de': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
            'es': 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
            'it': 'jonatasgrosman/wav2vec2-large-xlsr-53-italian',
            'pt': 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
            'ru': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
        }
    
    def _setup_huggingface_auth(self, token: str):
        """Setup Hugging Face authentication."""
        try:
            login(token=token)
            print("✓ Hugging Face authentication successful")
        except Exception as e:
            print(f"Warning: Hugging Face authentication failed: {e}")
    
    def get_cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self.cache_dir
    
    def is_model_cached(self, model_name: str, model_type: str = 'whisper') -> bool:
        """Check if a model is already cached locally."""
        if model_type == 'whisper':
            model_path = self.models_dir / 'whisper' / f'{model_name}.pt'
        elif model_type == 'alignment':
            model_path = self.models_dir / 'alignment' / f'{model_name}'
        elif model_type == 'diarization':
            model_path = self.models_dir / 'diarization' / f'{model_name}'
        else:
            return False
        
        return model_path.exists()
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get information about a Whisper model."""
        return self.whisper_models.get(model_name, {})
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by type."""
        return {
            'whisper': list(self.whisper_models.keys()),
            'alignment': list(self.alignment_models.keys()),
            'diarization': ['pyannote-segmentation', 'pyannote-embedding']
        }
    
    def get_alignment_model_name(self, language: str) -> str:
        """Get the alignment model for a specific language."""
        return self.alignment_models.get(language, self.alignment_models['en'])
    
    def validate_model_selection(self, model_name: str, device: str = 'auto') -> Dict[str, str]:
        """Validate model selection based on available hardware."""
        validation = {'status': 'valid', 'warnings': []}
        
        if model_name not in self.whisper_models:
            validation['status'] = 'error'
            validation['error'] = f"Unknown model: {model_name}"
            return validation
        
        # Check GPU memory requirements
        if device in ['cuda', 'auto'] and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            model_info = self.whisper_models[model_name]
            
            # Rough memory requirements (can vary based on batch size and compute type)
            memory_requirements = {
                'tiny': 1, 'base': 1, 'small': 2, 'medium': 4,
                'large': 6, 'large-v2': 6, 'large-v3': 6
            }
            
            required_memory = memory_requirements.get(model_name, 6)
            
            if gpu_memory_gb < required_memory:
                validation['warnings'].append(
                    f"Model {model_name} may not fit in GPU memory "
                    f"({gpu_memory_gb:.1f}GB available, ~{required_memory}GB required). "
                    f"Consider using a smaller model or CPU processing."
                )
        
        return validation
    
    def download_model_if_needed(self, model_name: str, model_type: str = 'whisper') -> bool:
        """Download model if not already cached."""
        if self.is_model_cached(model_name, model_type):
            return True
        
        disable_progress = not should_show_progress()
        
        try:
            if model_type == 'alignment':
                # Download alignment model from Hugging Face
                model_id = self.get_alignment_model_name(model_name)
                cache_dir = self.models_dir / 'alignment' / model_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                with progress_context(f"Downloading alignment model ({model_name})", disable=disable_progress) as spinner:
                    # Download model files
                    hf_hub_download(
                        repo_id=model_id,
                        filename="pytorch_model.bin",
                        cache_dir=str(cache_dir)
                    )
                    spinner.finish(f"Alignment model {model_name} downloaded")
                
            elif model_type == 'diarization':
                # Diarization models require pyannote.audio and special handling
                if not disable_progress:
                    print("Diarization models will be downloaded automatically by pyannote.audio")
                
            else:
                # WhisperX models are handled by the library itself
                if not disable_progress:
                    print(f"WhisperX will download model {model_name} automatically")
                
            return True
            
        except Exception as e:
            if not disable_progress:
                print(f"Error downloading {model_type} model {model_name}: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, str]:
        """Get information about available devices."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def cleanup_cache(self, keep_recent: int = 2):
        """Clean up old cached models, keeping only the most recent ones."""
        if not self.models_dir.exists():
            return
        
        print(f"Cleaning up model cache, keeping {keep_recent} most recent models...")
        
        # This is a simple implementation - in a real scenario, you'd track usage
        # and clean up based on last access time
        total_size = sum(
            f.stat().st_size for f in self.models_dir.rglob('*') if f.is_file()
        )
        
        print(f"Current cache size: {total_size / (1024**2):.1f}MB")


def get_model_manager(config) -> ModelManager:
    """Get a configured model manager instance."""
    return ModelManager(
        cache_dir=getattr(config, 'cache_dir', None),
        hf_token=getattr(config, 'huggingface_token', None)
    )