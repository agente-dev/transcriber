from typing import List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from engines.base import TranscriptionResult, Segment, Speaker


class MarkdownFormatter:
    """Format transcription results as Markdown."""
    
    def __init__(self, include_timestamps: bool = False, include_confidence: bool = False):
        self.include_timestamps = include_timestamps
        self.include_confidence = include_confidence
    
    def format(self, result: TranscriptionResult, filename: str) -> str:
        """Format transcription result as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# Transcription: {filename}")
        lines.append("")
        
        # Metadata section
        lines.extend(self._format_metadata(result, filename))
        lines.append("")
        
        # Speaker summary (if diarization was used)
        if result.speakers:
            lines.extend(self._format_speaker_summary(result.speakers))
            lines.append("")
        
        # Main transcript
        lines.append("## Transcript")
        lines.append("")
        
        if result.speakers and len(result.speakers) > 1:
            # Multi-speaker format
            lines.extend(self._format_multi_speaker_transcript(result))
        else:
            # Single speaker format
            lines.extend(self._format_single_speaker_transcript(result))
        
        return "\n".join(lines)
    
    def _format_metadata(self, result: TranscriptionResult, filename: str) -> List[str]:
        """Format metadata section."""
        lines = []
        metadata = result.metadata or {}
        
        lines.append(f"**File:** {filename}")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Duration:** {result.duration:.1f}s")
        lines.append(f"**Language:** {result.language}")
        
        engine = metadata.get('engine', 'unknown')
        model = metadata.get('model', 'unknown')
        lines.append(f"**Engine:** {engine}")
        lines.append(f"**Model:** {model}")
        
        if engine == 'whisperx':
            device = metadata.get('device', 'unknown')
            compute_type = metadata.get('compute_type', 'unknown')
            lines.append(f"**Device:** {device}")
            lines.append(f"**Compute Type:** {compute_type}")
            
            if metadata.get('diarization_enabled'):
                lines.append(f"**Speaker Diarization:** Enabled")
            
            if metadata.get('alignment_enabled'):
                lines.append(f"**Word Alignment:** Enabled")
        
        return lines
    
    def _format_speaker_summary(self, speakers: List[Speaker]) -> List[str]:
        """Format speaker summary section."""
        lines = ["## Speaker Summary"]
        lines.append("")
        
        for speaker in speakers:
            duration_min = speaker.total_duration / 60
            lines.append(
                f"- **{speaker.label}**: {duration_min:.1f} minutes "
                f"({speaker.percentage:.1f}% of conversation)"
            )
        
        return lines
    
    def _format_single_speaker_transcript(self, result: TranscriptionResult) -> List[str]:
        """Format transcript for single speaker."""
        lines = []
        
        if self.include_timestamps and result.segments:
            # Segment-by-segment with timestamps
            for segment in result.segments:
                timestamp = self._format_timestamp(segment.start, segment.end)
                confidence = f" (confidence: {segment.confidence:.2f})" if self.include_confidence else ""
                lines.append(f"**{timestamp}**{confidence}")
                lines.append(f"{segment.text.strip()}")
                lines.append("")
        else:
            # Simple paragraph format
            lines.append(result.text)
            lines.append("")
        
        return lines
    
    def _format_multi_speaker_transcript(self, result: TranscriptionResult) -> List[str]:
        """Format transcript with speaker labels."""
        lines = []
        current_speaker = None
        
        for segment in result.segments:
            speaker_label = segment.speaker or "UNKNOWN"
            
            # Add speaker header if changed
            if speaker_label != current_speaker:
                if current_speaker is not None:
                    lines.append("")  # Add spacing between speakers
                
                lines.append(f"### {speaker_label}")
                lines.append("")
                current_speaker = speaker_label
            
            # Add segment content
            if self.include_timestamps:
                timestamp = self._format_timestamp(segment.start, segment.end)
                confidence = f" (confidence: {segment.confidence:.2f})" if self.include_confidence else ""
                lines.append(f"**{timestamp}**{confidence}")
            
            lines.append(f"{segment.text.strip()}")
            lines.append("")
        
        return lines
    
    def _format_timestamp(self, start: float, end: float) -> str:
        """Format timestamp in MM:SS format."""
        def seconds_to_mmss(seconds: float) -> str:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        
        start_str = seconds_to_mmss(start)
        end_str = seconds_to_mmss(end)
        return f"{start_str} - {end_str}"


def format_markdown(result: TranscriptionResult, filename: str, 
                   include_timestamps: bool = False, 
                   include_confidence: bool = False) -> str:
    """Convenience function to format as Markdown."""
    formatter = MarkdownFormatter(include_timestamps, include_confidence)
    return formatter.format(result, filename)