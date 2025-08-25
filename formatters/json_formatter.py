import json
from typing import Dict, Any, List
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from engines.base import TranscriptionResult, Segment, Word, Speaker


class JSONFormatter:
    """Format transcription results as JSON."""
    
    def __init__(self, pretty: bool = True, include_words: bool = True, 
                 include_analysis: bool = True, include_statistics: bool = True):
        self.pretty = pretty
        self.include_words = include_words
        self.include_analysis = include_analysis
        self.include_statistics = include_statistics
    
    def format(self, result: TranscriptionResult, filename: str) -> str:
        """Format transcription result as JSON."""
        data = self._build_json_structure(result, filename)
        
        if self.pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False)
    
    def _build_json_structure(self, result: TranscriptionResult, filename: str) -> Dict[str, Any]:
        """Build the JSON data structure."""
        data = {
            "metadata": {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "language": result.language,
                "duration": result.duration,
                "engine_info": result.metadata or {}
            },
            "text": result.text,
            "segments": [self._segment_to_dict(seg) for seg in result.segments]
        }
        
        # Add speakers if available
        if result.speakers:
            data["speakers"] = [self._speaker_to_dict(speaker) for speaker in result.speakers]
            data["metadata"]["speaker_count"] = len(result.speakers)
            
            # Add speaker statistics if requested
            if self.include_statistics:
                data["speaker_statistics"] = self._generate_speaker_statistics(result.speakers, result.duration)
        
        # Add words if available and requested
        if self.include_words and result.words:
            data["words"] = [self._word_to_dict(word) for word in result.words]
            data["metadata"]["word_count"] = len(result.words)
            
            # Add word statistics if requested
            if self.include_statistics:
                data["word_statistics"] = self._generate_word_statistics(result.words)
        
        # Add conversation analysis if requested
        if self.include_analysis and result.speakers and len(result.speakers) > 1:
            data["conversation_analysis"] = self._generate_conversation_analysis(result)
        
        # Add overall statistics
        if self.include_statistics:
            data["overall_statistics"] = self._generate_overall_statistics(result)
        
        return data
    
    def _segment_to_dict(self, segment: Segment) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        seg_dict = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "confidence": segment.confidence
        }
        
        if segment.speaker:
            seg_dict["speaker"] = segment.speaker
        
        if self.include_words and segment.words:
            seg_dict["words"] = [self._word_to_dict(word) for word in segment.words]
        
        return seg_dict
    
    def _word_to_dict(self, word: Word) -> Dict[str, Any]:
        """Convert word to dictionary."""
        word_dict = {
            "word": word.word,
            "start": word.start,
            "end": word.end,
            "confidence": word.confidence
        }
        
        if word.speaker:
            word_dict["speaker"] = word.speaker
        
        return word_dict
    
    def _speaker_to_dict(self, speaker: Speaker) -> Dict[str, Any]:
        """Convert speaker to dictionary."""
        return {
            "id": speaker.id,
            "label": speaker.label,
            "segments": speaker.segments,
            "total_duration": speaker.total_duration,
            "percentage": speaker.percentage
        }
    
    def _generate_speaker_statistics(self, speakers: List[Speaker], total_duration: float) -> Dict[str, Any]:
        """Generate detailed speaker statistics."""
        stats = {
            "total_speakers": len(speakers),
            "dominant_speaker": max(speakers, key=lambda s: s.percentage).label if speakers else None,
            "speaker_balance": {
                "most_active": max(speakers, key=lambda s: s.percentage).label if speakers else None,
                "least_active": min(speakers, key=lambda s: s.percentage).label if speakers else None,
                "balance_ratio": max([s.percentage for s in speakers]) / min([s.percentage for s in speakers]) if speakers and min([s.percentage for s in speakers]) > 0 else None
            },
            "speaking_patterns": []
        }
        
        for speaker in speakers:
            avg_segment_length = (speaker.total_duration / len(speaker.segments) 
                                if hasattr(speaker, 'segments') and speaker.segments else 0)
            
            # Determine speaking style
            if speaker.percentage > 60:
                style = "dominant"
            elif speaker.percentage > 40:
                style = "active"
            elif speaker.percentage > 20:
                style = "regular"
            else:
                style = "occasional"
            
            stats["speaking_patterns"].append({
                "speaker": speaker.label,
                "style": style,
                "average_segment_duration": avg_segment_length,
                "segments_count": len(speaker.segments) if hasattr(speaker, 'segments') and speaker.segments else 0
            })
        
        return stats
    
    def _generate_word_statistics(self, words: List[Word]) -> Dict[str, Any]:
        """Generate word-level statistics."""
        if not words:
            return {}
        
        # Calculate speaking rate (words per minute)
        total_time = max(word.end for word in words) - min(word.start for word in words)
        words_per_minute = len(words) / (total_time / 60) if total_time > 0 else 0
        
        # Confidence statistics
        confidences = [word.confidence for word in words if word.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        low_confidence_words = len([c for c in confidences if c < 0.5])
        
        # Speaker word distribution
        speaker_word_count = {}
        for word in words:
            if word.speaker:
                speaker_word_count[word.speaker] = speaker_word_count.get(word.speaker, 0) + 1
        
        return {
            "total_words": len(words),
            "words_per_minute": round(words_per_minute, 1),
            "average_confidence": round(avg_confidence, 3),
            "low_confidence_count": low_confidence_words,
            "low_confidence_percentage": round((low_confidence_words / len(confidences) * 100), 1) if confidences else 0,
            "speaker_word_distribution": speaker_word_count
        }
    
    def _generate_conversation_analysis(self, result: TranscriptionResult) -> Dict[str, Any]:
        """Generate conversation flow analysis."""
        analysis = {
            "speaker_transitions": [],
            "conversation_flow": [],
            "interaction_patterns": {}
        }
        
        if not result.segments:
            return analysis
        
        # Analyze speaker transitions
        transitions = []
        prev_speaker = None
        
        for segment in result.segments:
            if segment.speaker and segment.speaker != prev_speaker:
                if prev_speaker is not None:
                    transitions.append({
                        "from": prev_speaker,
                        "to": segment.speaker,
                        "timestamp": segment.start
                    })
                prev_speaker = segment.speaker
        
        # Count transition patterns
        transition_counts = {}
        for transition in transitions:
            key = f"{transition['from']} → {transition['to']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        analysis["speaker_transitions"] = sorted(
            [{"pattern": k, "count": v} for k, v in transition_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )
        
        # Conversation timeline analysis
        timeline_segments = []
        current_speaker = None
        segment_start = 0
        
        for segment in result.segments:
            if segment.speaker != current_speaker:
                if current_speaker is not None:
                    timeline_segments.append({
                        "speaker": current_speaker,
                        "start": segment_start,
                        "end": segment.start,
                        "duration": segment.start - segment_start
                    })
                current_speaker = segment.speaker
                segment_start = segment.start
        
        # Add final segment
        if current_speaker and result.segments:
            final_end = result.segments[-1].end
            timeline_segments.append({
                "speaker": current_speaker,
                "start": segment_start,
                "end": final_end,
                "duration": final_end - segment_start
            })
        
        analysis["conversation_flow"] = timeline_segments
        
        # Interaction patterns
        if len(result.speakers) == 2:
            analysis["interaction_patterns"]["conversation_type"] = "dialog"
        elif len(result.speakers) > 2:
            analysis["interaction_patterns"]["conversation_type"] = "group_discussion"
        else:
            analysis["interaction_patterns"]["conversation_type"] = "monologue"
        
        analysis["interaction_patterns"]["total_transitions"] = len(transitions)
        analysis["interaction_patterns"]["average_segment_duration"] = (
            sum(seg["duration"] for seg in timeline_segments) / len(timeline_segments)
            if timeline_segments else 0
        )
        
        return analysis
    
    def _generate_overall_statistics(self, result: TranscriptionResult) -> Dict[str, Any]:
        """Generate overall transcription statistics."""
        stats = {
            "transcription_quality": {
                "total_segments": len(result.segments),
                "total_duration": result.duration,
                "segments_per_minute": len(result.segments) / (result.duration / 60) if result.duration > 0 else 0
            }
        }
        
        if result.words:
            total_words = len(result.words)
            stats["transcription_quality"]["total_words"] = total_words
            stats["transcription_quality"]["words_per_segment"] = total_words / len(result.segments) if result.segments else 0
            
            # Average word duration
            word_durations = [word.end - word.start for word in result.words if word.end > word.start]
            stats["transcription_quality"]["average_word_duration"] = (
                sum(word_durations) / len(word_durations) if word_durations else 0
            )
        
        if result.speakers:
            stats["speaker_distribution"] = {
                "entropy": self._calculate_speaker_entropy(result.speakers),
                "balance_score": self._calculate_speaker_balance_score(result.speakers)
            }
        
        return stats
    
    def _calculate_speaker_entropy(self, speakers: List[Speaker]) -> float:
        """Calculate entropy of speaker distribution (higher = more balanced)."""
        import math
        
        if not speakers:
            return 0.0
        
        total_duration = sum(speaker.total_duration for speaker in speakers)
        if total_duration == 0:
            return 0.0
        
        entropy = 0.0
        for speaker in speakers:
            proportion = speaker.total_duration / total_duration
            if proportion > 0:
                entropy -= proportion * math.log2(proportion)
        
        return round(entropy, 3)
    
    def _calculate_speaker_balance_score(self, speakers: List[Speaker]) -> float:
        """Calculate balance score (0-1, where 1 is perfectly balanced)."""
        if not speakers or len(speakers) < 2:
            return 1.0
        
        percentages = [speaker.percentage for speaker in speakers]
        ideal_percentage = 100.0 / len(speakers)
        
        # Calculate deviation from ideal balance
        deviations = [abs(p - ideal_percentage) for p in percentages]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to 0-1 score (lower deviation = higher score)
        balance_score = max(0, 1 - (avg_deviation / ideal_percentage))
        return round(balance_score, 3)


def format_json(result: TranscriptionResult, filename: str, 
               pretty: bool = True, include_words: bool = True) -> str:
    """Convenience function to format as JSON."""
    formatter = JSONFormatter(pretty=pretty, include_words=include_words)
    return formatter.format(result, filename)