"""
Enhanced diarization pipeline with advanced speaker identification features.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    pyannote_available = True
except ImportError:
    Pipeline = None
    Annotation = None
    Segment = None
    pyannote_available = False

from .models import DiarizationModelManager
from .speaker_mapping import SpeakerMapper


class DiarizationPipeline:
    """Enhanced speaker diarization pipeline with WhisperX integration."""
    
    def __init__(self, config: Dict[str, any], hf_token: Optional[str] = None, device: str = 'auto'):
        if not pyannote_available:
            raise ImportError(
                "pyannote.audio is not installed. Please run: pip install pyannote.audio"
            )
        
        self.config = config
        self.device = self._setup_device(device)
        self.hf_token = hf_token
        
        # Model manager
        self.model_manager = DiarizationModelManager(
            cache_dir=config.get('cache_dir'),
            hf_token=hf_token
        )
        
        # Pipeline will be loaded lazily
        self.pipeline = None
        self.is_initialized = False
        
        # Speaker mapping system
        self.speaker_mapper = SpeakerMapper(config.get('speaker_mapping', {}))
        
        print(f"✓ Diarization pipeline configured (device: {self.device})")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for diarization processing."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"✓ Using GPU for diarization")
            else:
                device = 'cpu'
                print("! Using CPU for diarization (slower)")
        
        elif device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
            device = 'cpu'
        
        return device
    
    def _load_pipeline(self):
        """Load the pyannote.audio diarization pipeline."""
        if self.pipeline is not None:
            return
        
        # Validate authentication
        validation = self.model_manager.validate_model_access()
        if validation['status'] == 'error':
            for error in validation['errors']:
                print(f"Error: {error}")
            raise RuntimeError("Cannot load diarization pipeline: authentication required")
        
        for warning in validation.get('warnings', []):
            print(f"Warning: {warning}")
        
        try:
            print("Loading pyannote.audio diarization pipeline...")
            
            # Load the speaker diarization pipeline
            pipeline_model = self.model_manager.get_model_info('speaker_diarization_3.1')['repo_id']
            
            self.pipeline = Pipeline.from_pretrained(
                pipeline_model,
                use_auth_token=self.hf_token
            )
            
            # Move to specified device
            if hasattr(self.pipeline, 'to') and self.device == 'cuda':
                self.pipeline = self.pipeline.to(torch.device('cuda'))
            
            self.is_initialized = True
            print("✓ Diarization pipeline loaded successfully")
            
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "token" in error_msg.lower():
                print("Error: Failed to authenticate with Hugging Face")
                print("Please ensure you have:")
                print("1. A valid Hugging Face token")
                print("2. Accepted the license for pyannote/speaker-diarization-3.1")
                print("3. Set the token using --hf-token or HUGGINGFACE_ACCESS_TOKEN")
            else:
                print(f"Error loading diarization pipeline: {e}")
            raise RuntimeError(f"Failed to load diarization pipeline: {e}")
    
    def diarize_audio(
        self, 
        audio_path: str, 
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs
    ) -> Annotation:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            **kwargs: Additional parameters for the pipeline
            
        Returns:
            pyannote.core.Annotation object with speaker segments
        """
        # Load pipeline if not already loaded
        self._load_pipeline()
        
        print(f"Performing speaker diarization on: {Path(audio_path).name}")
        
        try:
            # Prepare pipeline parameters
            params = {}
            if min_speakers is not None:
                params['min_speakers'] = min_speakers
            if max_speakers is not None:
                params['max_speakers'] = max_speakers
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Run diarization
            diarization = self.pipeline(audio_path, **params)
            
            # Log results
            speakers = list(diarization.labels())
            total_speech = sum(segment.end - segment.start for segment in diarization.itersegments())
            
            print(f"✓ Diarization complete: {len(speakers)} speakers detected")
            print(f"  Total speech time: {total_speech:.1f}s")
            
            return diarization
            
        except Exception as e:
            print(f"Error during diarization: {e}")
            raise RuntimeError(f"Diarization failed: {e}")
    
    def apply_diarization_to_segments(
        self,
        diarization: Annotation,
        transcription_segments: List[Dict[str, any]],
        overlap_strategy: str = 'predominant'
    ) -> List[Dict[str, any]]:
        """
        Apply diarization results to transcription segments using enhanced mapping.
        
        Args:
            diarization: Pyannote diarization results
            transcription_segments: List of transcription segments from WhisperX
            overlap_strategy: How to handle overlapping speech
            
        Returns:
            Updated segments with enhanced speaker information
        """
        print("Mapping speakers to transcription segments with enhanced mapping...")
        
        # Use enhanced speaker mapping
        updated_segments = self.speaker_mapper.map_segments_to_speakers(
            diarization, transcription_segments, overlap_strategy
        )
        
        # Validate mapping quality
        validation = self.speaker_mapper.validate_speaker_mapping(updated_segments)
        
        print(f"✓ Speaker mapping completed:")
        print(f"  - Mapped {validation['mapped_segments']}/{len(transcription_segments)} segments")
        print(f"  - Coverage: {validation['mapping_coverage']:.1f}%")
        
        for warning in validation.get('warnings', []):
            print(f"  Warning: {warning}")
        
        return updated_segments
    
    def _get_predominant_speaker(
        self,
        diarization: Annotation,
        start_time: float,
        end_time: float
    ) -> Optional[str]:
        """
        Get the predominant speaker for a given time segment.
        
        Args:
            diarization: Pyannote diarization results
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            Speaker label or None if no speaker found
        """
        segment_range = Segment(start_time, end_time)
        
        # Find overlapping speaker segments
        speaker_durations = {}
        
        for timeline, _, speaker in diarization.itertracks():
            overlap = segment_range.intersect(timeline)
            if overlap:
                overlap_duration = overlap.duration
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                speaker_durations[speaker] += overlap_duration
        
        if not speaker_durations:
            return None
        
        # Return speaker with longest overlap
        predominant_speaker = max(speaker_durations.items(), key=lambda x: x[1])
        return predominant_speaker[0]
    
    def apply_word_level_diarization(
        self,
        diarization: Annotation,
        words: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Apply speaker labels to individual words.
        
        Args:
            diarization: Pyannote diarization results  
            words: List of word-level data from alignment
            
        Returns:
            Updated words with speaker labels
        """
        print("Applying speaker labels to words...")
        
        updated_words = []
        
        for word_data in words:
            if 'start' not in word_data or 'end' not in word_data:
                # Skip words without timestamps
                updated_words.append(word_data)
                continue
                
            start_time = word_data['start']
            end_time = word_data['end']
            
            # Get speaker for this word
            speaker = self._get_predominant_speaker(
                diarization, start_time, end_time
            )
            
            # Update word with speaker
            updated_word = word_data.copy()
            updated_word['speaker'] = speaker
            updated_words.append(updated_word)
        
        return updated_words
    
    def get_speaker_statistics(
        self,
        diarization: Annotation,
        total_duration: float
    ) -> List[Dict[str, any]]:
        """
        Calculate speaker statistics from diarization results.
        
        Args:
            diarization: Pyannote diarization results
            total_duration: Total audio duration
            
        Returns:
            List of speaker statistics
        """
        speaker_stats = {}
        
        # Calculate speaking time for each speaker
        for segment, _, speaker in diarization.itertracks():
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'total_time': 0.0,
                    'segment_count': 0,
                    'segments': []
                }
            
            duration = segment.duration
            speaker_stats[speaker]['total_time'] += duration
            speaker_stats[speaker]['segment_count'] += 1
            speaker_stats[speaker]['segments'].append({
                'start': segment.start,
                'end': segment.end,
                'duration': duration
            })
        
        # Convert to statistics format
        statistics = []
        for speaker, stats in speaker_stats.items():
            percentage = (stats['total_time'] / total_duration * 100) if total_duration > 0 else 0
            
            statistics.append({
                'id': speaker,
                'label': speaker,
                'total_duration': stats['total_time'],
                'percentage': percentage,
                'segment_count': stats['segment_count'],
                'average_segment_duration': stats['total_time'] / stats['segment_count'],
                'segments': stats['segments']
            })
        
        # Sort by speaking time (descending)
        statistics.sort(key=lambda x: x['total_duration'], reverse=True)
        
        return statistics
    
    def get_enhanced_speaker_statistics(
        self,
        diarization: Annotation,
        segments_with_speakers: List[Dict[str, any]],
        total_duration: float
    ) -> List[Dict[str, any]]:
        """Get enhanced speaker statistics using the speaker mapper."""
        return self.speaker_mapper.calculate_enhanced_speaker_statistics(
            diarization, segments_with_speakers, total_duration
        )
    
    def validate_diarization_quality(
        self,
        diarization: Annotation,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Validate diarization quality and provide feedback.
        
        Args:
            diarization: Pyannote diarization results
            min_speakers: Expected minimum speakers
            max_speakers: Expected maximum speakers
            
        Returns:
            Quality assessment
        """
        speakers = list(diarization.labels())
        num_speakers = len(speakers)
        
        validation = {
            'status': 'valid',
            'num_speakers': num_speakers,
            'warnings': [],
            'recommendations': []
        }
        
        # Check speaker count expectations
        if min_speakers and num_speakers < min_speakers:
            validation['warnings'].append(
                f"Detected {num_speakers} speakers, expected at least {min_speakers}"
            )
        
        if max_speakers and num_speakers > max_speakers:
            validation['warnings'].append(
                f"Detected {num_speakers} speakers, expected at most {max_speakers}"
            )
        
        # Check for very short speaker segments (potential oversegmentation)
        total_segments = sum(1 for _ in diarization.itersegments())
        if total_segments > num_speakers * 20:  # Heuristic threshold
            validation['warnings'].append(
                "Many short segments detected, possibly oversegmented"
            )
            validation['recommendations'].append(
                "Consider adjusting clustering parameters"
            )
        
        # Check speaker balance
        speaker_durations = {}
        total_duration = 0
        
        for segment, _, speaker in diarization.itertracks():
            duration = segment.duration
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += duration
            total_duration += duration
        
        # Warn if one speaker dominates (>80%)
        if speaker_durations:
            max_speaker_pct = max(speaker_durations.values()) / total_duration * 100
            if max_speaker_pct > 80:
                validation['warnings'].append(
                    f"One speaker dominates ({max_speaker_pct:.1f}% of speech)"
                )
        
        return validation
    
    def get_capabilities(self) -> Dict[str, any]:
        """Return diarization pipeline capabilities."""
        return {
            'supports_min_max_speakers': True,
            'supports_word_level_diarization': True,
            'supports_speaker_statistics': True,
            'supports_quality_validation': True,
            'required_authentication': True,
            'device_support': ['cpu', 'cuda'],
            'model_info': self.model_manager.list_available_models()
        }