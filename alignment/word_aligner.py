"""
Word-level alignment implementation.
Provides precise word-level timestamps and speaker mapping using wav2vec2 models.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import torch
    import numpy as np
    torch_available = True
except ImportError:
    torch = None
    np = None
    torch_available = False

try:
    import whisperx
    whisperx_available = True
except ImportError:
    whisperx = None
    whisperx_available = False

from .language_models import LanguageModelManager

logger = logging.getLogger(__name__)


@dataclass
class AlignmentConfig:
    """Configuration for word alignment."""
    enabled: bool = True
    return_char_alignments: bool = False
    interpolate_method: str = 'nearest'  # 'nearest', 'linear'
    device: str = 'auto'
    extend_duration: float = 2.0  # seconds to extend alignment context
    rtl_support: bool = True
    min_word_confidence: float = 0.1
    alignment_window: float = 0.02  # alignment precision window (20ms)


class WordAligner:
    """
    Enhanced word-level alignment with speaker mapping support.
    Uses wav2vec2 models for precise word-level timestamp extraction.
    """
    
    def __init__(self, config: AlignmentConfig, cache_dir: Optional[Path] = None):
        """
        Initialize word aligner.
        
        Args:
            config: Alignment configuration
            cache_dir: Directory to cache alignment models
        """
        if not whisperx_available:
            raise ImportError("WhisperX is not available. Please install: pip install whisperx")
        
        if not torch_available:
            raise ImportError("PyTorch is not available. Please install: pip install torch")
        
        self.config = config
        self.cache_dir = cache_dir or Path.home() / ".whisperx" / "models" / "alignment"
        
        # Initialize language model manager
        self.language_manager = LanguageModelManager(
            cache_dir=self.cache_dir, 
            device=config.device
        )
        
        # Current loaded model state
        self.current_language = None
        self.current_model = None
        self.current_metadata = None
        
        logger.info("Word aligner initialized")
    
    def align_segments(
        self, 
        segments: List[Dict], 
        audio_data: Any, 
        language: str = 'he'
    ) -> List[Dict]:
        """
        Apply word-level alignment to transcription segments.
        
        Args:
            segments: List of transcription segments from WhisperX
            audio_data: Audio data loaded by whisperx.load_audio()
            language: Language code for alignment model
            
        Returns:
            Enhanced segments with word-level timestamps
        """
        if not self.config.enabled:
            logger.info("Word alignment disabled, returning original segments")
            return segments
        
        if not segments:
            logger.warning("No segments to align")
            return segments
        
        try:
            # Load appropriate language model
            model, metadata = self._load_model_for_language(language)
            if model is None:
                logger.warning("Could not load alignment model, returning original segments")
                return segments
            
            logger.info(f"Aligning words for {len(segments)} segments ({language})")
            
            # Use WhisperX align function
            aligned_result = whisperx.align(
                segments,
                model,
                metadata,
                audio_data,
                self.language_manager.device,
                return_char_alignments=self.config.return_char_alignments,
                interpolate_method=self.config.interpolate_method,
                extend_duration=self.config.extend_duration
            )
            
            # Post-process alignment results
            enhanced_segments = self._post_process_alignment(
                aligned_result.get('segments', segments),
                language
            )
            
            logger.info(f"✓ Word alignment completed for {len(enhanced_segments)} segments")
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Word alignment failed: {e}")
            logger.info("Returning original segments without word-level timestamps")
            return segments
    
    def align_words_to_speakers(
        self, 
        segments: List[Dict], 
        diarization_result: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Map word-level timestamps to speaker segments.
        
        Args:
            segments: Segments with word-level alignment
            diarization_result: Speaker diarization results
            
        Returns:
            Segments with words mapped to speakers
        """
        if not diarization_result or not segments:
            return segments
        
        try:
            logger.info("Mapping words to speakers")
            
            enhanced_segments = []
            for segment in segments:
                enhanced_segment = segment.copy()
                
                if 'words' in segment and segment['words']:
                    enhanced_words = []
                    
                    for word_data in segment['words']:
                        enhanced_word = word_data.copy()
                        
                        # Find best matching speaker for this word
                        word_start = word_data.get('start', 0.0)
                        word_end = word_data.get('end', 0.0)
                        word_center = (word_start + word_end) / 2
                        
                        best_speaker = self._find_speaker_for_timestamp(
                            word_center, diarization_result
                        )
                        
                        if best_speaker:
                            enhanced_word['speaker'] = best_speaker
                        
                        # Add confidence metrics
                        enhanced_word['alignment_confidence'] = self._calculate_alignment_confidence(
                            word_data, segment
                        )
                        
                        enhanced_words.append(enhanced_word)
                    
                    enhanced_segment['words'] = enhanced_words
                    
                    # Update segment speaker if needed
                    if not enhanced_segment.get('speaker') and enhanced_words:
                        # Use most frequent speaker in words
                        word_speakers = [w.get('speaker') for w in enhanced_words if w.get('speaker')]
                        if word_speakers:
                            most_common_speaker = max(set(word_speakers), key=word_speakers.count)
                            enhanced_segment['speaker'] = most_common_speaker
                
                enhanced_segments.append(enhanced_segment)
            
            logger.info("✓ Words mapped to speakers successfully")
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Failed to map words to speakers: {e}")
            return segments
    
    def validate_alignment_quality(
        self, 
        segments: List[Dict], 
        audio_duration: float
    ) -> Dict[str, Any]:
        """
        Validate quality of word-level alignment.
        
        Args:
            segments: Aligned segments
            audio_duration: Total audio duration
            
        Returns:
            Validation results with quality metrics
        """
        results = {
            'status': 'success',
            'total_segments': len(segments),
            'segments_with_words': 0,
            'total_words': 0,
            'words_with_timestamps': 0,
            'coverage_percentage': 0.0,
            'avg_word_confidence': 0.0,
            'timing_gaps': [],
            'warnings': []
        }
        
        try:
            total_word_duration = 0.0
            confidence_scores = []
            previous_end = 0.0
            
            for segment in segments:
                if 'words' in segment and segment['words']:
                    results['segments_with_words'] += 1
                    
                    for word_data in segment['words']:
                        results['total_words'] += 1
                        
                        # Check for valid timestamps
                        if ('start' in word_data and 'end' in word_data and 
                            word_data['start'] is not None and word_data['end'] is not None):
                            results['words_with_timestamps'] += 1
                            
                            word_duration = word_data['end'] - word_data['start']
                            if word_duration > 0:
                                total_word_duration += word_duration
                            
                            # Check for timing gaps
                            if word_data['start'] > previous_end + 0.1:  # 100ms gap
                                gap_duration = word_data['start'] - previous_end
                                results['timing_gaps'].append({
                                    'start': previous_end,
                                    'end': word_data['start'],
                                    'duration': gap_duration
                                })
                            
                            previous_end = word_data['end']
                        
                        # Collect confidence scores
                        if 'score' in word_data and word_data['score'] is not None:
                            confidence_scores.append(word_data['score'])
                        elif 'alignment_confidence' in word_data:
                            confidence_scores.append(word_data['alignment_confidence'])
            
            # Calculate metrics
            if results['total_words'] > 0:
                results['coverage_percentage'] = (
                    results['words_with_timestamps'] / results['total_words'] * 100
                )
            
            if confidence_scores:
                results['avg_word_confidence'] = sum(confidence_scores) / len(confidence_scores)
            
            # Add warnings for quality issues
            if results['coverage_percentage'] < 80:
                results['warnings'].append(
                    f"Low timestamp coverage: {results['coverage_percentage']:.1f}%"
                )
            
            if results['avg_word_confidence'] < 0.7:
                results['warnings'].append(
                    f"Low average confidence: {results['avg_word_confidence']:.2f}"
                )
            
            if len(results['timing_gaps']) > results['total_segments'] * 0.5:
                results['warnings'].append(
                    f"Many timing gaps detected: {len(results['timing_gaps'])}"
                )
            
            logger.info(f"Alignment validation: {results['coverage_percentage']:.1f}% coverage, "
                       f"{results['avg_word_confidence']:.2f} avg confidence")
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logger.error(f"Alignment validation failed: {e}")
        
        return results
    
    def _load_model_for_language(self, language: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load alignment model for specified language."""
        if (self.current_language == language and 
            self.current_model is not None and 
            self.current_metadata is not None):
            return self.current_model, self.current_metadata
        
        # Load new model
        result = self.language_manager.load_model(language)
        if result is None:
            return None, None
        
        model, metadata = result
        
        # Cache current model
        self.current_language = language
        self.current_model = model
        self.current_metadata = metadata
        
        return model, metadata
    
    def _post_process_alignment(self, segments: List[Dict], language: str) -> List[Dict]:
        """Post-process alignment results for enhanced accuracy."""
        enhanced_segments = []
        
        is_rtl = self.language_manager.is_rtl_language(language)
        
        for segment in segments:
            enhanced_segment = segment.copy()
            
            if 'words' in segment and segment['words']:
                enhanced_words = []
                
                for word_data in segment['words']:
                    enhanced_word = word_data.copy()
                    
                    # Filter low-confidence words
                    confidence = enhanced_word.get('score', 1.0)
                    if confidence < self.config.min_word_confidence:
                        logger.debug(f"Filtering low-confidence word: {enhanced_word.get('word')} "
                                   f"(confidence: {confidence})")
                        continue
                    
                    # Handle RTL text if needed
                    if is_rtl and 'word' in enhanced_word:
                        # Note: RTL handling might need more sophisticated processing
                        # depending on the specific requirements
                        pass
                    
                    # Ensure timestamp validity
                    if 'start' in enhanced_word and 'end' in enhanced_word:
                        if enhanced_word['end'] <= enhanced_word['start']:
                            # Fix invalid timestamps
                            duration = max(0.1, len(enhanced_word.get('word', '')) * 0.05)
                            enhanced_word['end'] = enhanced_word['start'] + duration
                    
                    enhanced_words.append(enhanced_word)
                
                enhanced_segment['words'] = enhanced_words
                
                # Update segment timestamps based on words if available
                if enhanced_words:
                    word_starts = [w.get('start') for w in enhanced_words if w.get('start') is not None]
                    word_ends = [w.get('end') for w in enhanced_words if w.get('end') is not None]
                    
                    if word_starts and word_ends:
                        enhanced_segment['start'] = min(word_starts)
                        enhanced_segment['end'] = max(word_ends)
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _find_speaker_for_timestamp(
        self, 
        timestamp: float, 
        diarization_result: Dict
    ) -> Optional[str]:
        """Find the most likely speaker for a given timestamp."""
        if 'speakers' not in diarization_result:
            return None
        
        best_speaker = None
        best_overlap = 0.0
        
        for speaker_data in diarization_result['speakers']:
            speaker_id = speaker_data.get('id')
            segments = speaker_data.get('segments', [])
            
            for segment in segments:
                start = segment.get('start', 0.0)
                end = segment.get('end', 0.0)
                
                # Check if timestamp falls within speaker segment
                if start <= timestamp <= end:
                    overlap = min(end, timestamp + self.config.alignment_window) - max(start, timestamp - self.config.alignment_window)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker_id
        
        return best_speaker
    
    def _calculate_alignment_confidence(
        self, 
        word_data: Dict, 
        segment: Dict
    ) -> float:
        """Calculate alignment confidence score for a word."""
        confidence = 1.0
        
        # Use existing score if available
        if 'score' in word_data and word_data['score'] is not None:
            confidence *= word_data['score']
        
        # Penalize very short or very long words based on duration
        if 'start' in word_data and 'end' in word_data:
            duration = word_data['end'] - word_data['start']
            word_length = len(word_data.get('word', ''))
            
            if word_length > 0:
                # Expected duration per character (rough estimate)
                expected_duration = word_length * 0.08  # 80ms per character
                
                # Penalize if duration is too different from expected
                duration_ratio = duration / max(expected_duration, 0.1)
                if duration_ratio < 0.3 or duration_ratio > 3.0:
                    confidence *= 0.7
        
        return min(1.0, max(0.0, confidence))
    
    def get_alignment_stats(self) -> Dict[str, Any]:
        """Get statistics about alignment performance."""
        stats = {
            'language_manager': self.language_manager.get_memory_usage(),
            'current_language': self.current_language,
            'config': {
                'enabled': self.config.enabled,
                'device': self.config.device,
                'rtl_support': self.config.rtl_support,
                'min_word_confidence': self.config.min_word_confidence
            }
        }
        
        return stats
    
    def clear_cache(self):
        """Clear alignment model cache."""
        logger.info("Clearing word alignment cache")
        self.language_manager.clear_cache()
        self.current_language = None
        self.current_model = None
        self.current_metadata = None