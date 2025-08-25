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
    
    def __init__(self, pretty: bool = True, include_words: bool = True):
        self.pretty = pretty
        self.include_words = include_words
    
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
        
        # Add words if available and requested
        if self.include_words and result.words:
            data["words"] = [self._word_to_dict(word) for word in result.words]
            data["metadata"]["word_count"] = len(result.words)
        
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


def format_json(result: TranscriptionResult, filename: str, 
               pretty: bool = True, include_words: bool = True) -> str:
    """Convenience function to format as JSON."""
    formatter = JSONFormatter(pretty=pretty, include_words=include_words)
    return formatter.format(result, filename)