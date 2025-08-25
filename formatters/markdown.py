from typing import List, Optional, Dict, Any
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from engines.base import TranscriptionResult, Segment, Speaker, Word


class MarkdownFormatter:
    """Format transcription results as Markdown."""
    
    def __init__(self, include_timestamps: bool = False, include_confidence: bool = False,
                 include_word_timestamps: bool = False, include_speaker_stats: bool = True,
                 include_conversation_flow: bool = True):
        self.include_timestamps = include_timestamps
        self.include_confidence = include_confidence
        self.include_word_timestamps = include_word_timestamps
        self.include_speaker_stats = include_speaker_stats
        self.include_conversation_flow = include_conversation_flow
    
    def format(self, result: TranscriptionResult, filename: str) -> str:
        """Format transcription result as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# Transcription: {filename}")
        lines.append("")
        
        # Metadata section
        lines.extend(self._format_metadata(result, filename))
        lines.append("")
        
        # Enhanced speaker summary (if diarization was used)
        if result.speakers and self.include_speaker_stats:
            lines.extend(self._format_enhanced_speaker_summary(result.speakers, result.duration))
            lines.append("")
        
        # Conversation flow analysis
        if result.speakers and len(result.speakers) > 1 and self.include_conversation_flow:
            lines.extend(self._format_conversation_flow(result))
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
        
        # Word-level timestamps section if requested
        if self.include_word_timestamps and result.words:
            lines.append("")
            lines.extend(self._format_word_timeline(result.words))
        
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
    
    def _format_enhanced_speaker_summary(self, speakers: List[Speaker], total_duration: float) -> List[str]:
        """Format enhanced speaker summary with detailed statistics."""
        lines = ["## 📊 Speaker Analysis"]
        lines.append("")
        
        # Overview table
        lines.append("| Speaker | Duration | Percentage | Segments |")
        lines.append("|---------|----------|------------|----------|")
        
        for speaker in speakers:
            duration_min = speaker.total_duration / 60
            segment_count = len(speaker.segments) if hasattr(speaker, 'segments') and speaker.segments else "N/A"
            lines.append(
                f"| **{speaker.label}** | {duration_min:.1f}m | {speaker.percentage:.1f}% | {segment_count} |"
            )
        
        lines.append("")
        
        # Speaking patterns analysis
        lines.append("### Speaking Patterns")
        lines.append("")
        
        for speaker in speakers:
            duration_min = speaker.total_duration / 60
            avg_segment_length = speaker.total_duration / len(speaker.segments) if hasattr(speaker, 'segments') and speaker.segments else 0
            
            lines.append(f"**{speaker.label}**:")
            lines.append(f"- Total speaking time: {duration_min:.1f} minutes")
            lines.append(f"- Average segment length: {avg_segment_length:.1f} seconds")
            
            # Determine speaking style
            if speaker.percentage > 60:
                style = "Dominant speaker"
            elif speaker.percentage > 40:
                style = "Active participant"
            elif speaker.percentage > 20:
                style = "Regular contributor"
            else:
                style = "Occasional speaker"
            
            lines.append(f"- Speaking style: {style}")
            lines.append("")
        
        return lines
    
    def _format_conversation_flow(self, result: TranscriptionResult) -> List[str]:
        """Format conversation flow analysis."""
        lines = ["## 💬 Conversation Flow"]
        lines.append("")
        
        if not result.segments:
            return lines
        
        # Analyze speaker transitions
        transitions = []
        prev_speaker = None
        
        for segment in result.segments:
            if segment.speaker and segment.speaker != prev_speaker:
                if prev_speaker is not None:
                    transitions.append((prev_speaker, segment.speaker))
                prev_speaker = segment.speaker
        
        if transitions:
            lines.append("### Speaker Transitions")
            lines.append("")
            
            # Count transitions
            transition_counts = {}
            for from_speaker, to_speaker in transitions:
                key = f"{from_speaker} → {to_speaker}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            # Sort by frequency
            sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
            
            lines.append("| Transition | Count |")
            lines.append("|------------|-------|")
            
            for transition, count in sorted_transitions[:10]:  # Top 10 transitions
                lines.append(f"| {transition} | {count} |")
            
            lines.append("")
        
        # Analyze conversation segments
        lines.append("### Conversation Timeline")
        lines.append("")
        
        timeline_segments = []
        current_speaker = None
        segment_start = 0
        
        for segment in result.segments:
            if segment.speaker != current_speaker:
                if current_speaker is not None:
                    timeline_segments.append({
                        'speaker': current_speaker,
                        'start': segment_start,
                        'end': segment.start,
                        'duration': segment.start - segment_start
                    })
                current_speaker = segment.speaker
                segment_start = segment.start
        
        # Add final segment
        if current_speaker and result.segments:
            final_end = result.segments[-1].end
            timeline_segments.append({
                'speaker': current_speaker,
                'start': segment_start,
                'end': final_end,
                'duration': final_end - segment_start
            })
        
        # Format timeline
        for i, seg in enumerate(timeline_segments[:20]):  # First 20 segments
            start_time = self._format_timestamp_simple(seg['start'])
            end_time = self._format_timestamp_simple(seg['end'])
            duration = seg['duration']
            
            lines.append(f"{i+1:2d}. **{seg['speaker']}** ({start_time} - {end_time}) - {duration:.1f}s")
        
        if len(timeline_segments) > 20:
            lines.append(f"... and {len(timeline_segments) - 20} more segments")
        
        lines.append("")
        
        return lines
    
    def _format_word_timeline(self, words: List[Word]) -> List[str]:
        """Format word-level timeline section."""
        lines = ["## 🔤 Word Timeline"]
        lines.append("")
        
        if not words:
            lines.append("No word-level timing information available.")
            return lines
        
        lines.append("*Showing first 50 words with precise timestamps*")
        lines.append("")
        
        # Group words by time intervals (every 10 seconds)
        time_groups = {}
        for word in words[:50]:  # Limit to first 50 words
            time_key = int(word.start // 10) * 10  # Round down to nearest 10s
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(word)
        
        # Format grouped words
        for time_key in sorted(time_groups.keys()):
            time_range = f"{time_key}s - {time_key + 10}s"
            lines.append(f"### {time_range}")
            lines.append("")
            
            word_list = time_groups[time_key]
            current_line = ""
            
            for word in word_list:
                word_text = word.word
                timestamp = f"[{word.start:.1f}s]"
                speaker_info = f" ({word.speaker})" if word.speaker else ""
                confidence = f" ({word.confidence:.2f})" if self.include_confidence else ""
                
                word_entry = f"`{word_text}`{timestamp}{speaker_info}{confidence}"
                
                if len(current_line) + len(word_entry) + 2 > 100:  # Line length limit
                    if current_line:
                        lines.append(current_line)
                    current_line = word_entry
                else:
                    current_line = f"{current_line}, {word_entry}" if current_line else word_entry
            
            if current_line:
                lines.append(current_line)
            lines.append("")
        
        return lines
    
    def _format_timestamp_simple(self, seconds: float) -> str:
        """Format timestamp in simple MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
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