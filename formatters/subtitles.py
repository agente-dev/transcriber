"""Subtitle formatters for SRT and VTT output formats."""

from typing import List, Optional
from datetime import timedelta
import sys
from pathlib import Path
import re

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from engines.base import TranscriptionResult, Segment, Speaker


class SubtitleFormatter:
    """Base class for subtitle formatters."""
    
    def __init__(self, max_chars_per_line: int = 42, max_lines_per_subtitle: int = 2, 
                 include_speakers: bool = True, speaker_prefix: str = ""):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self.include_speakers = include_speakers
        self.speaker_prefix = speaker_prefix
    
    def _format_time(self, seconds: float, format_type: str = "srt") -> str:
        """Format time for subtitle formats."""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        milliseconds = int((td.total_seconds() % 1) * 1000)
        
        if format_type == "srt":
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        elif format_type == "vtt":
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        else:
            raise ValueError(f"Unknown time format: {format_type}")
    
    def _split_text_for_subtitles(self, text: str, speaker: Optional[str] = None) -> List[str]:
        """Split text into subtitle-appropriate lines."""
        # Clean and prepare text
        text = text.strip()
        if not text:
            return []
        
        # Add speaker prefix if needed
        if self.include_speakers and speaker and speaker != "UNKNOWN":
            if self.speaker_prefix:
                text = f"{self.speaker_prefix}{speaker}: {text}"
            else:
                text = f"{speaker}: {text}"
        
        # Simple word-based line breaking
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed line length
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= self.max_chars_per_line:
                current_line = test_line
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = word
                
                # If a single word is too long, we have to keep it anyway
                if len(current_line) > self.max_chars_per_line:
                    lines.append(current_line)
                    current_line = ""
        
        # Add remaining text
        if current_line:
            lines.append(current_line)
        
        # Limit to max lines per subtitle
        if len(lines) > self.max_lines_per_subtitle:
            # Join excess lines with the last allowed line
            combined_last = " ".join(lines[self.max_lines_per_subtitle-1:])
            lines = lines[:self.max_lines_per_subtitle-1] + [combined_last]
        
        return lines
    
    def _merge_consecutive_segments(self, segments: List[Segment], 
                                  max_duration: float = 5.0, 
                                  max_chars: int = 80) -> List[Segment]:
        """Merge consecutive segments from the same speaker for better subtitles."""
        if not segments:
            return []
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            # Check if we should merge with current segment
            should_merge = (
                # Same speaker (or both None/unknown)
                (current.speaker == next_segment.speaker or 
                 (not current.speaker and not next_segment.speaker)) and
                # Duration not too long
                (next_segment.end - current.start) <= max_duration and
                # Combined text not too long
                len(current.text + " " + next_segment.text) <= max_chars and
                # Gap between segments is small (less than 1 second)
                (next_segment.start - current.end) <= 1.0
            )
            
            if should_merge:
                # Merge segments
                current = Segment(
                    id=current.id,
                    start=current.start,
                    end=next_segment.end,
                    text=f"{current.text.strip()} {next_segment.text.strip()}",
                    speaker=current.speaker or next_segment.speaker,
                    confidence=min(current.confidence, next_segment.confidence),
                    words=(current.words or []) + (next_segment.words or [])
                )
            else:
                # Save current and start new one
                merged.append(current)
                current = next_segment
        
        # Add the last segment
        merged.append(current)
        return merged


class SRTFormatter(SubtitleFormatter):
    """Format transcription results as SRT subtitles."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def format(self, result: TranscriptionResult, filename: str) -> str:
        """Format transcription result as SRT."""
        if not result.segments:
            return ""
        
        # Merge segments for better subtitle readability
        segments = self._merge_consecutive_segments(result.segments)
        
        lines = []
        subtitle_index = 1
        
        for segment in segments:
            # Skip empty segments
            if not segment.text.strip():
                continue
            
            # Format timestamps
            start_time = self._format_time(segment.start, "srt")
            end_time = self._format_time(segment.end, "srt")
            
            # Split text into appropriate lines
            text_lines = self._split_text_for_subtitles(segment.text, segment.speaker)
            
            if text_lines:
                # Add subtitle entry
                lines.append(str(subtitle_index))
                lines.append(f"{start_time} --> {end_time}")
                lines.extend(text_lines)
                lines.append("")  # Empty line separator
                subtitle_index += 1
        
        return "\n".join(lines)


class VTTFormatter(SubtitleFormatter):
    """Format transcription results as WebVTT subtitles."""
    
    def __init__(self, include_header: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_header = include_header
    
    def format(self, result: TranscriptionResult, filename: str) -> str:
        """Format transcription result as VTT."""
        lines = []
        
        # Add WebVTT header
        if self.include_header:
            lines.append("WEBVTT")
            lines.append("")
            
            # Add optional metadata
            lines.append("NOTE")
            lines.append(f"Generated from: {filename}")
            lines.append(f"Language: {result.language}")
            if result.speakers and len(result.speakers) > 1:
                lines.append(f"Speakers: {len(result.speakers)}")
            lines.append("")
        
        if not result.segments:
            return "\n".join(lines) if lines else "WEBVTT"
        
        # Merge segments for better subtitle readability
        segments = self._merge_consecutive_segments(result.segments)
        
        for segment in segments:
            # Skip empty segments
            if not segment.text.strip():
                continue
            
            # Format timestamps
            start_time = self._format_time(segment.start, "vtt")
            end_time = self._format_time(segment.end, "vtt")
            
            # Split text into appropriate lines
            text_lines = self._split_text_for_subtitles(segment.text, segment.speaker)
            
            if text_lines:
                # Add subtitle entry (VTT doesn't require index numbers)
                lines.append(f"{start_time} --> {end_time}")
                lines.extend(text_lines)
                lines.append("")  # Empty line separator
        
        return "\n".join(lines)


class AdvancedSRTFormatter(SRTFormatter):
    """Enhanced SRT formatter with speaker colors and positioning."""
    
    def __init__(self, use_speaker_colors: bool = True, speaker_colors: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.use_speaker_colors = use_speaker_colors
        self.speaker_colors = speaker_colors or {
            'SPEAKER_1': '#00FF00',  # Green
            'SPEAKER_2': '#FF0000',  # Red
            'SPEAKER_3': '#0000FF',  # Blue
            'SPEAKER_4': '#FFFF00',  # Yellow
            'SPEAKER_5': '#FF00FF',  # Magenta
            'SPEAKER_6': '#00FFFF',  # Cyan
        }
    
    def _split_text_for_subtitles(self, text: str, speaker: Optional[str] = None) -> List[str]:
        """Enhanced text splitting with speaker styling."""
        lines = super()._split_text_for_subtitles(text, speaker)
        
        # Add color coding for speakers if enabled
        if self.use_speaker_colors and speaker and speaker in self.speaker_colors:
            color = self.speaker_colors[speaker]
            # Wrap each line in font color tags
            lines = [f'<font color="{color}">{line}</font>' for line in lines]
        
        return lines


# Convenience functions
def format_srt(result: TranscriptionResult, filename: str, 
               include_speakers: bool = True, **kwargs) -> str:
    """Convenience function to format as SRT."""
    formatter = SRTFormatter(include_speakers=include_speakers, **kwargs)
    return formatter.format(result, filename)


def format_vtt(result: TranscriptionResult, filename: str, 
               include_speakers: bool = True, **kwargs) -> str:
    """Convenience function to format as VTT."""
    formatter = VTTFormatter(include_speakers=include_speakers, **kwargs)
    return formatter.format(result, filename)


def format_advanced_srt(result: TranscriptionResult, filename: str, 
                       include_speakers: bool = True, use_speaker_colors: bool = True, **kwargs) -> str:
    """Convenience function to format as advanced SRT with colors."""
    formatter = AdvancedSRTFormatter(
        include_speakers=include_speakers, 
        use_speaker_colors=use_speaker_colors, 
        **kwargs
    )
    return formatter.format(result, filename)