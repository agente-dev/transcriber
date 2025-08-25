"""
Enhanced speaker segment mapping and labeling utilities.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

try:
    from pyannote.core import Annotation, Segment
    pyannote_available = True
except ImportError:
    Annotation = None
    Segment = None
    pyannote_available = False


class SpeakerMapper:
    """Enhanced speaker mapping and labeling system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Speaker labeling preferences
        self.speaker_label_format = self.config.get('speaker_label_format', 'SPEAKER_{id}')
        self.custom_speaker_names = self.config.get('custom_speaker_names', {})
        
        # Overlap handling settings
        self.overlap_threshold = self.config.get('overlap_threshold', 0.1)  # 10% minimum overlap
        self.overlap_resolution = self.config.get('overlap_resolution', 'predominant')  # or 'merge', 'split'
    
    def map_segments_to_speakers(
        self,
        diarization: Annotation,
        transcription_segments: List[Dict[str, Any]],
        overlap_strategy: str = 'predominant'
    ) -> List[Dict[str, Any]]:
        """
        Enhanced segment-to-speaker mapping with overlap handling.
        
        Args:
            diarization: Pyannote diarization results
            transcription_segments: List of transcription segments
            overlap_strategy: How to handle overlapping speech ('predominant', 'merge', 'split')
            
        Returns:
            Updated segments with speaker information and overlap metadata
        """
        if not pyannote_available:
            raise ImportError("pyannote.audio is required for speaker mapping")
        
        updated_segments = []
        
        for i, segment in enumerate(transcription_segments):
            start_time = segment['start']
            end_time = segment['end']
            segment_range = Segment(start_time, end_time)
            
            # Find all overlapping speakers and their durations
            speaker_overlaps = self._get_speaker_overlaps(diarization, segment_range)
            
            # Apply overlap resolution strategy
            speaker_info = self._resolve_speaker_overlaps(
                speaker_overlaps, overlap_strategy
            )
            
            # Update segment with speaker information
            updated_segment = segment.copy()
            updated_segment.update(speaker_info)
            
            # Add segment metadata
            updated_segment['segment_id'] = i
            updated_segment['overlap_count'] = len(speaker_overlaps)
            
            updated_segments.append(updated_segment)
        
        return updated_segments
    
    def _get_speaker_overlaps(
        self,
        diarization: Annotation,
        segment_range: Segment
    ) -> List[Dict[str, Any]]:
        """Get all speaker overlaps for a segment with detailed information."""
        overlaps = []
        
        for timeline, _, speaker in diarization.itertracks():
            overlap = segment_range.intersect(timeline)
            
            if overlap and overlap.duration > 0:
                overlap_ratio = overlap.duration / segment_range.duration
                
                # Only consider meaningful overlaps
                if overlap_ratio >= self.overlap_threshold:
                    overlaps.append({
                        'speaker': speaker,
                        'timeline': timeline,
                        'overlap': overlap,
                        'overlap_duration': overlap.duration,
                        'overlap_ratio': overlap_ratio,
                        'speaker_timeline_duration': timeline.duration
                    })
        
        # Sort by overlap duration (descending)
        overlaps.sort(key=lambda x: x['overlap_duration'], reverse=True)
        return overlaps
    
    def _resolve_speaker_overlaps(
        self,
        speaker_overlaps: List[Dict[str, Any]],
        strategy: str
    ) -> Dict[str, Any]:
        """Resolve speaker overlaps using specified strategy."""
        if not speaker_overlaps:
            return {'speaker': None, 'speaker_confidence': 0.0}
        
        if strategy == 'predominant':
            return self._predominant_speaker_strategy(speaker_overlaps)
        elif strategy == 'merge':
            return self._merge_speakers_strategy(speaker_overlaps)
        elif strategy == 'split':
            return self._split_segment_strategy(speaker_overlaps)
        else:
            # Default to predominant
            return self._predominant_speaker_strategy(speaker_overlaps)
    
    def _predominant_speaker_strategy(self, overlaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the speaker with the longest overlap."""
        primary_overlap = overlaps[0]
        speaker_id = primary_overlap['speaker']
        
        # Calculate confidence based on overlap ratio and competition
        confidence = primary_overlap['overlap_ratio']
        
        # Reduce confidence if there are competing speakers
        if len(overlaps) > 1:
            second_best_ratio = overlaps[1]['overlap_ratio']
            competition_factor = 1 - (second_best_ratio / primary_overlap['overlap_ratio'])
            confidence *= competition_factor
        
        return {
            'speaker': self._format_speaker_label(speaker_id),
            'speaker_confidence': confidence,
            'speaker_overlap_ratio': primary_overlap['overlap_ratio'],
            'competing_speakers': len(overlaps) - 1 if len(overlaps) > 1 else 0
        }
    
    def _merge_speakers_strategy(self, overlaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple speakers into a combined label."""
        if len(overlaps) == 1:
            return self._predominant_speaker_strategy(overlaps)
        
        # Get speakers with significant overlap (>30% of segment)
        significant_speakers = [
            overlap for overlap in overlaps 
            if overlap['overlap_ratio'] >= 0.3
        ]
        
        if len(significant_speakers) <= 1:
            return self._predominant_speaker_strategy(overlaps)
        
        # Create merged speaker label
        speaker_ids = [self._format_speaker_label(o['speaker']) for o in significant_speakers]
        merged_label = ' + '.join(speaker_ids)
        
        # Average confidence
        avg_confidence = sum(o['overlap_ratio'] for o in significant_speakers) / len(significant_speakers)
        
        return {
            'speaker': merged_label,
            'speaker_confidence': avg_confidence,
            'speaker_overlap_ratio': avg_confidence,
            'merged_speakers': speaker_ids,
            'is_overlapping_speech': True
        }
    
    def _split_segment_strategy(self, overlaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mark segment for potential splitting based on speaker changes."""
        # This strategy flags segments that might need splitting
        # Actual splitting would be implemented at a higher level
        
        primary_overlap = overlaps[0]
        
        result = self._predominant_speaker_strategy(overlaps)
        result.update({
            'needs_splitting': len(overlaps) > 1,
            'split_candidates': [
                {
                    'speaker': self._format_speaker_label(o['speaker']),
                    'start_offset': o['overlap'].start - primary_overlap['overlap'].start,
                    'duration': o['overlap_duration'],
                    'ratio': o['overlap_ratio']
                }
                for o in overlaps[1:] if o['overlap_ratio'] >= 0.2
            ]
        })
        
        return result
    
    def _format_speaker_label(self, speaker_id: str) -> str:
        """Format speaker label according to configuration."""
        # Check for custom names first
        if speaker_id in self.custom_speaker_names:
            return self.custom_speaker_names[speaker_id]
        
        # Use configured format
        return self.speaker_label_format.format(id=speaker_id)
    
    def calculate_enhanced_speaker_statistics(
        self,
        diarization: Annotation,
        segments_with_speakers: List[Dict[str, Any]],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """Calculate comprehensive speaker statistics."""
        speaker_stats = defaultdict(lambda: {
            'total_time': 0.0,
            'segment_count': 0,
            'segments': [],
            'overlapping_segments': 0,
            'average_confidence': 0.0,
            'confidence_scores': [],
            'speaking_turns': 0,
            'longest_turn': 0.0,
            'shortest_turn': float('inf')
        })
        
        # Collect statistics from segments
        previous_speaker = None
        
        for segment in segments_with_speakers:
            speaker = segment.get('speaker')
            if not speaker or speaker == 'None':
                continue
            
            duration = segment['end'] - segment['start']
            confidence = segment.get('speaker_confidence', 1.0)
            is_overlapping = segment.get('is_overlapping_speech', False)
            
            stats = speaker_stats[speaker]
            stats['total_time'] += duration
            stats['segment_count'] += 1
            stats['confidence_scores'].append(confidence)
            
            if is_overlapping:
                stats['overlapping_segments'] += 1
            
            # Track speaking turns
            if speaker != previous_speaker:
                stats['speaking_turns'] += 1
            
            # Track turn durations (simplified - actual implementation would need turn detection)
            stats['longest_turn'] = max(stats['longest_turn'], duration)
            stats['shortest_turn'] = min(stats['shortest_turn'], duration)
            
            previous_speaker = speaker
        
        # Calculate final statistics
        statistics = []
        for speaker, stats in speaker_stats.items():
            if stats['shortest_turn'] == float('inf'):
                stats['shortest_turn'] = 0.0
            
            avg_confidence = sum(stats['confidence_scores']) / len(stats['confidence_scores']) if stats['confidence_scores'] else 0.0
            percentage = (stats['total_time'] / total_duration * 100) if total_duration > 0 else 0
            
            statistics.append({
                'id': speaker,
                'label': speaker,
                'total_duration': stats['total_time'],
                'percentage': percentage,
                'segment_count': stats['segment_count'],
                'average_segment_duration': stats['total_time'] / stats['segment_count'],
                'average_confidence': avg_confidence,
                'speaking_turns': stats['speaking_turns'],
                'overlapping_segments': stats['overlapping_segments'],
                'overlap_percentage': (stats['overlapping_segments'] / stats['segment_count'] * 100) if stats['segment_count'] > 0 else 0,
                'longest_turn': stats['longest_turn'],
                'shortest_turn': stats['shortest_turn'],
                'confidence_scores': stats['confidence_scores']
            })
        
        # Sort by speaking time (descending)
        statistics.sort(key=lambda x: x['total_duration'], reverse=True)
        
        return statistics
    
    def validate_speaker_mapping(
        self,
        segments_with_speakers: List[Dict[str, Any]],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate the quality of speaker mapping."""
        speakers = set()
        unmapped_segments = 0
        low_confidence_segments = 0
        overlapping_segments = 0
        
        for segment in segments_with_speakers:
            speaker = segment.get('speaker')
            confidence = segment.get('speaker_confidence', 0.0)
            is_overlapping = segment.get('is_overlapping_speech', False)
            
            if speaker and speaker != 'None':
                speakers.add(speaker)
            else:
                unmapped_segments += 1
            
            if confidence < 0.5:  # Low confidence threshold
                low_confidence_segments += 1
            
            if is_overlapping:
                overlapping_segments += 1
        
        num_speakers = len(speakers)
        total_segments = len(segments_with_speakers)
        
        validation = {
            'status': 'valid',
            'num_speakers': num_speakers,
            'mapped_segments': total_segments - unmapped_segments,
            'unmapped_segments': unmapped_segments,
            'low_confidence_segments': low_confidence_segments,
            'overlapping_segments': overlapping_segments,
            'mapping_coverage': (total_segments - unmapped_segments) / total_segments * 100 if total_segments > 0 else 0,
            'warnings': [],
            'recommendations': []
        }
        
        # Validate speaker count expectations
        if min_speakers and num_speakers < min_speakers:
            validation['warnings'].append(
                f"Only {num_speakers} speakers detected, expected at least {min_speakers}"
            )
        
        if max_speakers and num_speakers > max_speakers:
            validation['warnings'].append(
                f"Detected {num_speakers} speakers, expected at most {max_speakers}"
            )
        
        # Quality warnings
        if validation['mapping_coverage'] < 80:
            validation['warnings'].append(
                f"Low mapping coverage: {validation['mapping_coverage']:.1f}% of segments mapped to speakers"
            )
        
        if low_confidence_segments > total_segments * 0.2:
            validation['warnings'].append(
                f"High number of low-confidence speaker assignments: {low_confidence_segments}/{total_segments}"
            )
        
        if overlapping_segments > total_segments * 0.3:
            validation['recommendations'].append(
                "High amount of overlapping speech detected. Consider using merge or split strategy."
            )
        
        return validation