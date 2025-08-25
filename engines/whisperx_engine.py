try:
    import torch
    torch_available = True
except ImportError:
    torch = None
    torch_available = False

try:
    import numpy as np
    numpy_available = True
except ImportError:
    np = None
    numpy_available = False

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import whisperx
    whisperx_available = True
except ImportError:
    whisperx = None
    whisperx_available = False

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from engines.base import (
    TranscriptionEngine, TranscriptionResult, Segment, Word, Speaker,
    ValidationResult, EngineCapabilities
)
from config.models import ModelManager
from diarization.pipeline import DiarizationPipeline


class WhisperXEngine(TranscriptionEngine):
    """WhisperX-based transcription engine with diarization support."""
    
    SUPPORTED_FORMATS = ['.mp3', '.m4a', '.wav', '.webm', '.mp4', '.mpga', '.mpeg', '.flac', '.ogg']
    
    def __init__(self, config):
        if not whisperx_available:
            raise ImportError("WhisperX is not installed. Please run: pip install whisperx")
        
        if not torch_available:
            raise ImportError("PyTorch is not installed. Please run: pip install torch")
        
        super().__init__(config)
        
        # Extract configuration
        self.whisperx_config = getattr(config, 'whisperx', type('obj', (object,), {
            'model': 'large-v2',
            'device': 'auto',
            'compute_type': 'float16',
            'batch_size': 16
        })())
        
        self.diarization_config = getattr(config, 'diarization', type('obj', (object,), {
            'enabled': False,
            'min_speakers': None,
            'max_speakers': None
        })())
        
        self.alignment_config = getattr(config, 'alignment', type('obj', (object,), {
            'enabled': True
        })())
        
        self.vad_config = getattr(config, 'vad', type('obj', (object,), {
            'enabled': True
        })())
        
        # Setup device
        self.device = self._setup_device()
        self.compute_type = self.whisperx_config.compute_type
        
        # Model manager
        self.model_manager = ModelManager(
            hf_token=getattr(config, 'huggingface_token', None)
        )
        
        # Initialize models (lazy loading)
        self.whisper_model = None
        self.alignment_model = None
        self.diarization_pipeline = None
        self.enhanced_diarization_pipeline = None
        
        print(f"✓ WhisperX engine initialized (device: {self.device})")
    
    def _setup_device(self) -> str:
        """Setup and validate device selection."""
        device = self.whisperx_config.device
        
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                device = 'cpu'
                print("! GPU not available, using CPU")
        
        elif device == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                device = 'cpu'
        
        return device
    
    def _load_whisper_model(self):
        """Load WhisperX model (lazy loading)."""
        if self.whisper_model is not None:
            return
        
        model_name = self.whisperx_config.model
        print(f"Loading WhisperX model: {model_name}")
        
        # Validate model selection
        validation = self.model_manager.validate_model_selection(model_name, self.device)
        if validation['status'] == 'error':
            raise ValueError(validation['error'])
        
        for warning in validation.get('warnings', []):
            print(f"Warning: {warning}")
        
        try:
            self.whisper_model = whisperx.load_model(
                model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            print(f"✓ Model loaded successfully")
            
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("GPU out of memory, trying with reduced batch size...")
                # Try with smaller batch size or CPU fallback
                if self.device == 'cuda':
                    print("Falling back to CPU processing")
                    self.device = 'cpu'
                    self.whisper_model = whisperx.load_model(
                        model_name,
                        device=self.device,
                        compute_type='float32'
                    )
                else:
                    raise
            else:
                raise RuntimeError(f"Failed to load WhisperX model: {e}")
    
    def _load_alignment_model(self, language: str):
        """Load alignment model for word-level timestamps."""
        if not self.alignment_config.enabled:
            return None
        
        if self.alignment_model is not None:
            return self.alignment_model
        
        try:
            print(f"Loading alignment model for language: {language}")
            self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device
            )
            print("✓ Alignment model loaded")
            return self.alignment_model
            
        except Exception as e:
            print(f"Warning: Could not load alignment model for {language}: {e}")
            print("Word-level alignment will not be available")
            return None
    
    def _load_diarization_pipeline(self):
        """Load enhanced speaker diarization pipeline."""
        if not self.diarization_config.enabled:
            return None
        
        if self.enhanced_diarization_pipeline is not None:
            return self.enhanced_diarization_pipeline
        
        try:
            print("Loading enhanced speaker diarization pipeline...")
            
            # Create configuration for diarization pipeline
            diarization_config = {
                'cache_dir': str(self.model_manager.cache_dir),
                'device': self.device
            }
            
            self.enhanced_diarization_pipeline = DiarizationPipeline(
                config=diarization_config,
                hf_token=self.model_manager.hf_token,
                device=self.device
            )
            
            print("✓ Enhanced diarization pipeline loaded")
            return self.enhanced_diarization_pipeline
            
        except Exception as e:
            print(f"Warning: Could not load enhanced diarization pipeline: {e}")
            print("Attempting fallback to basic WhisperX diarization...")
            
            # Fallback to basic WhisperX diarization
            try:
                self.diarization_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=self.model_manager.hf_token,
                    device=self.device
                )
                print("✓ Basic diarization pipeline loaded")
                return self.diarization_pipeline
            except Exception as fallback_error:
                print(f"Warning: Fallback diarization also failed: {fallback_error}")
                print("Speaker diarization will not be available")
                return None
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file with WhisperX."""
        # Validate file
        validation = self.validate_file(audio_path)
        if not validation.is_valid:
            raise ValueError(validation.error_message)
        
        print(f"Transcribing {audio_path} using WhisperX...")
        
        try:
            # Load models
            self._load_whisper_model()
            
            # Load and preprocess audio
            print("Loading audio...")
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            print("Transcribing...")
            result = self.whisper_model.transcribe(
                audio, 
                batch_size=self.whisperx_config.batch_size,
                language=getattr(self.config, 'language', None)
            )
            
            # Get detected language
            language = result.get('language', getattr(self.config, 'language', 'he'))
            
            # Word-level alignment
            if self.alignment_config.enabled:
                alignment_model = self._load_alignment_model(language)
                if alignment_model is not None:
                    print("Aligning words...")
                    result = whisperx.align(
                        result["segments"], 
                        self.alignment_model, 
                        self.alignment_metadata, 
                        audio, 
                        self.device, 
                        return_char_alignments=False
                    )
            
            # Speaker diarization
            speakers_list = None
            if self.diarization_config.enabled:
                diarization_pipeline = self._load_diarization_pipeline()
                if diarization_pipeline is not None:
                    
                    # Use enhanced diarization pipeline if available
                    if isinstance(diarization_pipeline, DiarizationPipeline):
                        print("Performing enhanced speaker diarization...")
                        
                        # Run enhanced diarization
                        diarization_result = diarization_pipeline.diarize_audio(
                            audio_path,
                            min_speakers=self.diarization_config.min_speakers,
                            max_speakers=self.diarization_config.max_speakers
                        )
                        
                        # Apply to segments
                        result["segments"] = diarization_pipeline.apply_diarization_to_segments(
                            diarization_result, result["segments"]
                        )
                        
                        # Apply to words if available
                        if 'words' in result and result['words']:
                            all_words = []
                            for segment in result["segments"]:
                                if 'words' in segment:
                                    segment_words = diarization_pipeline.apply_word_level_diarization(
                                        diarization_result, segment['words']
                                    )
                                    segment['words'] = segment_words
                                    all_words.extend(segment_words)
                            result['words'] = all_words
                        
                        # Get enhanced speaker statistics
                        total_duration = max([seg['end'] for seg in result["segments"]]) if result["segments"] else 0.0
                        speaker_stats = diarization_pipeline.get_enhanced_speaker_statistics(
                            diarization_result, result["segments"], total_duration
                        )
                        speakers_list = self._convert_speaker_stats(speaker_stats)
                        
                        # Validate diarization quality
                        validation = diarization_pipeline.validate_diarization_quality(
                            diarization_result,
                            min_speakers=self.diarization_config.min_speakers,
                            max_speakers=self.diarization_config.max_speakers
                        )
                        
                        for warning in validation.get('warnings', []):
                            print(f"Diarization warning: {warning}")
                    
                    else:
                        # Fallback to basic WhisperX diarization
                        print("Using basic WhisperX diarization...")
                        
                        diarize_segments = diarization_pipeline(
                            audio,
                            min_speakers=self.diarization_config.min_speakers,
                            max_speakers=self.diarization_config.max_speakers
                        )
                        
                        # Apply speaker labels to segments
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                        
                        # Extract speaker information
                        speakers_list = self._extract_speaker_info(result["segments"])
            
            # Convert to our format
            return self._convert_result(result, language, Path(audio_path).name, speakers_list)
            
        except Exception as e:
            raise RuntimeError(f"WhisperX transcription failed: {e}")
    
    def _extract_speaker_info(self, segments: List[Dict]) -> List[Speaker]:
        """Extract speaker information from segments."""
        speaker_stats = {}
        
        for segment in segments:
            speaker_id = segment.get('speaker')
            if speaker_id:
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {
                        'segments': [],
                        'total_duration': 0.0
                    }
                
                duration = segment['end'] - segment['start']
                speaker_stats[speaker_id]['segments'].append(segment.get('id', 0))
                speaker_stats[speaker_id]['total_duration'] += duration
        
        # Calculate total duration for percentages
        total_duration = sum(stats['total_duration'] for stats in speaker_stats.values())
        
        speakers = []
        for speaker_id, stats in speaker_stats.items():
            percentage = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            
            speakers.append(Speaker(
                id=speaker_id,
                label=speaker_id,
                segments=stats['segments'],
                total_duration=stats['total_duration'],
                percentage=percentage
            ))
        
        return sorted(speakers, key=lambda x: x.total_duration, reverse=True)
    
    def _convert_speaker_stats(self, speaker_stats: List[Dict]) -> List[Speaker]:
        """Convert enhanced diarization statistics to Speaker objects."""
        speakers = []
        
        for stats in speaker_stats:
            speaker = Speaker(
                id=stats['id'],
                label=stats['label'],
                segments=[],  # Segment indices not available in this format
                total_duration=stats['total_duration'],
                percentage=stats['percentage']
            )
            speakers.append(speaker)
        
        return speakers
    
    def _convert_result(self, result: Dict, language: str, filename: str, speakers: Optional[List[Speaker]]) -> TranscriptionResult:
        """Convert WhisperX result to our standard format."""
        segments = []
        all_words = []
        
        for i, seg in enumerate(result.get("segments", [])):
            # Extract words if available
            words = []
            if 'words' in seg:
                for word_data in seg['words']:
                    word = Word(
                        word=word_data.get('word', ''),
                        start=word_data.get('start', 0.0),
                        end=word_data.get('end', 0.0),
                        confidence=word_data.get('score', 1.0),
                        speaker=word_data.get('speaker')
                    )
                    words.append(word)
                    all_words.append(word)
            
            segment = Segment(
                id=i,
                start=seg.get('start', 0.0),
                end=seg.get('end', 0.0),
                text=seg.get('text', ''),
                speaker=seg.get('speaker'),
                confidence=1.0,  # WhisperX doesn't provide segment confidence
                words=words if words else None
            )
            segments.append(segment)
        
        # Calculate total duration
        duration = max([seg.end for seg in segments]) if segments else 0.0
        
        # Full text
        full_text = " ".join([seg.text.strip() for seg in segments])
        
        # Metadata
        metadata = {
            'engine': 'whisperx',
            'model': self.whisperx_config.model,
            'device': self.device,
            'compute_type': self.compute_type,
            'language_requested': getattr(self.config, 'language', None),
            'diarization_enabled': self.diarization_config.enabled,
            'alignment_enabled': self.alignment_config.enabled,
            'timestamp': datetime.now().isoformat(),
            'file_name': filename
        }
        
        return TranscriptionResult(
            text=full_text,
            language=language,
            duration=duration,
            segments=segments,
            speakers=speakers,
            words=all_words if all_words else None,
            metadata=metadata
        )
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate file for WhisperX processing."""
        # Check file existence
        existence_result = self._validate_file_exists(file_path)
        if not existence_result.is_valid:
            return existence_result
        
        # Check file format
        format_result = self._validate_file_format(file_path, self.SUPPORTED_FORMATS)
        if not format_result.is_valid:
            return format_result
        
        # WhisperX can handle larger files than OpenAI API
        # But we'll still warn for very large files
        if existence_result.file_size_mb > 500:  # 500MB warning threshold
            return ValidationResult(
                True,
                error_message=f"Warning: Large file ({existence_result.file_size_mb:.1f}MB) "
                             f"may take a long time to process",
                file_size_mb=existence_result.file_size_mb
            )
        
        return ValidationResult(True, file_size_mb=existence_result.file_size_mb)
    
    def get_capabilities(self) -> EngineCapabilities:
        """Return WhisperX engine capabilities."""
        return EngineCapabilities(
            supports_diarization=True,
            supports_word_timestamps=True,
            supports_local_processing=True,
            max_file_size_mb=None,  # No hard limit for local processing
            supported_languages=['he', 'en', 'ar', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        )