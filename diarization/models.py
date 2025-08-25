"""
Diarization model management and configuration.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List
from huggingface_hub import hf_hub_download, login


class DiarizationModelManager:
    """Manages pyannote.audio models for speaker diarization."""
    
    def __init__(self, cache_dir: Optional[str] = None, hf_token: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.whisperx'
        self.diarization_dir = self.cache_dir / 'diarization'
        self.diarization_dir.mkdir(parents=True, exist_ok=True)
        
        self.hf_token = hf_token
        if hf_token:
            self._setup_huggingface_auth(hf_token)
        
        # Pyannote.audio model configurations
        self.diarization_models = {
            'segmentation': {
                'repo_id': 'pyannote/segmentation-3.0',
                'filename': 'pytorch_model.bin',
                'config_filename': 'config.yaml',
                'description': 'Voice activity detection and speaker segmentation'
            },
            'embedding': {
                'repo_id': 'pyannote/wespeaker-voxceleb-resnet34-LM',
                'filename': 'pytorch_model.bin', 
                'config_filename': 'config.yaml',
                'description': 'Speaker embedding extraction'
            },
            'speaker_diarization_3.1': {
                'repo_id': 'pyannote/speaker-diarization-3.1',
                'filename': 'config.yaml',
                'description': 'Complete speaker diarization pipeline'
            }
        }
    
    def _setup_huggingface_auth(self, token: str):
        """Setup Hugging Face authentication for pyannote models."""
        try:
            login(token=token)
            print("✓ Hugging Face authentication successful for diarization models")
            
            # Set environment variable for pyannote.audio
            os.environ['HUGGINGFACE_ACCESS_TOKEN'] = token
            os.environ['HF_TOKEN'] = token
            
        except Exception as e:
            print(f"Warning: Hugging Face authentication failed: {e}")
            print("Note: Pyannote.audio models require Hugging Face authentication")
    
    def is_authenticated(self) -> bool:
        """Check if Hugging Face authentication is available."""
        return bool(self.hf_token) or bool(os.getenv('HUGGINGFACE_ACCESS_TOKEN') or os.getenv('HF_TOKEN'))
    
    def get_required_models(self) -> List[str]:
        """Get list of required diarization models."""
        return ['speaker_diarization_3.1']  # The main pipeline model
    
    def validate_model_access(self) -> Dict[str, any]:
        """Validate access to required diarization models."""
        validation = {
            'status': 'valid',
            'errors': [],
            'warnings': []
        }
        
        if not self.is_authenticated():
            validation['status'] = 'error'
            validation['errors'].append(
                "Hugging Face authentication required for pyannote.audio models. "
                "Please set HUGGINGFACE_ACCESS_TOKEN environment variable or provide --hf-token."
            )
            return validation
        
        # Check model access (basic validation)
        try:
            # This will be handled by pyannote.audio at runtime
            validation['warnings'].append(
                "Diarization models will be downloaded automatically on first use. "
                "This may take some time depending on your internet connection."
            )
        except Exception as e:
            validation['warnings'].append(f"Could not verify model access: {e}")
        
        return validation
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get information about a diarization model."""
        return self.diarization_models.get(model_name, {})
    
    def list_available_models(self) -> List[str]:
        """List available diarization models."""
        return list(self.diarization_models.keys())
    
    def get_pipeline_config(self) -> Dict[str, any]:
        """Get configuration for diarization pipeline."""
        return {
            'segmentation_model': self.diarization_models['segmentation']['repo_id'],
            'embedding_model': self.diarization_models['embedding']['repo_id'],
            'pipeline_model': self.diarization_models['speaker_diarization_3.1']['repo_id'],
            'use_auth_token': self.hf_token,
            'cache_dir': str(self.cache_dir)
        }
    
    def prepare_environment(self) -> bool:
        """Prepare environment for diarization processing."""
        if not self.is_authenticated():
            print("Error: Diarization requires Hugging Face authentication")
            print("Please provide your Hugging Face token using:")
            print("  --hf-token YOUR_TOKEN")
            print("  or set HUGGINGFACE_ACCESS_TOKEN environment variable")
            print("\nTo get a token:")
            print("  1. Visit https://huggingface.co/settings/tokens")
            print("  2. Create a new token with 'Read' permissions") 
            print("  3. Accept the license for pyannote/speaker-diarization-3.1")
            return False
        
        # Create cache directories
        self.diarization_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    def get_memory_requirements(self) -> Dict[str, float]:
        """Get approximate memory requirements for diarization."""
        return {
            'gpu_memory_gb': 2.0,  # Approximate GPU memory needed
            'system_memory_gb': 4.0,  # Approximate system memory needed
            'disk_space_gb': 1.0  # Approximate disk space for models
        }
    
    def check_system_requirements(self) -> Dict[str, any]:
        """Check if system meets diarization requirements."""
        requirements = {
            'status': 'valid',
            'warnings': [],
            'recommendations': []
        }
        
        try:
            import torch
            
            # Check GPU availability
            if not torch.cuda.is_available():
                requirements['warnings'].append(
                    "CUDA not available. Diarization will use CPU and may be slower."
                )
                requirements['recommendations'].append(
                    "Consider installing CUDA for better performance."
                )
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                required_memory = self.get_memory_requirements()['gpu_memory_gb']
                
                if gpu_memory < required_memory:
                    requirements['warnings'].append(
                        f"GPU has {gpu_memory:.1f}GB memory, diarization requires "
                        f"~{required_memory}GB. May fallback to CPU."
                    )
        
        except ImportError:
            requirements['status'] = 'error'
            requirements['warnings'].append("PyTorch not installed")
        
        return requirements