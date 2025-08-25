#!/usr/bin/env python3
"""
Test script for enhanced output formatters.
Creates sample transcription data and tests all formatter outputs.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from engines.base import TranscriptionResult, Segment, Speaker, Word
from formatters import MarkdownFormatter, JSONFormatter, SRTFormatter, VTTFormatter, AdvancedSRTFormatter


def create_sample_transcription_data() -> TranscriptionResult:
    """Create sample transcription data for testing."""
    
    # Create sample words
    words = [
        Word("Hello", 0.5, 0.8, 0.95, "SPEAKER_1"),
        Word("everyone,", 0.9, 1.2, 0.92, "SPEAKER_1"),
        Word("thanks", 1.3, 1.7, 0.89, "SPEAKER_1"),
        Word("for", 1.8, 2.0, 0.94, "SPEAKER_1"),
        Word("joining", 2.1, 2.6, 0.87, "SPEAKER_1"),
        Word("the", 2.7, 2.9, 0.96, "SPEAKER_1"),
        Word("meeting.", 3.0, 3.8, 0.91, "SPEAKER_1"),
        Word("Great", 5.2, 5.6, 0.93, "SPEAKER_2"),
        Word("to", 5.7, 5.9, 0.88, "SPEAKER_2"),
        Word("be", 6.0, 6.2, 0.94, "SPEAKER_2"),
        Word("here!", 6.3, 6.8, 0.90, "SPEAKER_2"),
        Word("Let's", 7.0, 7.4, 0.92, "SPEAKER_2"),
        Word("get", 7.5, 7.8, 0.89, "SPEAKER_2"),
        Word("started.", 7.9, 8.5, 0.94, "SPEAKER_2"),
        Word("I", 10.1, 10.3, 0.95, "SPEAKER_1"),
        Word("think", 10.4, 10.8, 0.91, "SPEAKER_1"),
        Word("we", 10.9, 11.1, 0.93, "SPEAKER_1"),
        Word("should", 11.2, 11.6, 0.88, "SPEAKER_1"),
        Word("discuss", 11.7, 12.3, 0.90, "SPEAKER_1"),
        Word("the", 12.4, 12.6, 0.95, "SPEAKER_1"),
        Word("quarterly", 12.7, 13.4, 0.87, "SPEAKER_1"),
        Word("results", 13.5, 14.1, 0.92, "SPEAKER_1"),
        Word("first.", 14.2, 14.8, 0.89, "SPEAKER_1"),
    ]
    
    # Create sample segments
    segments = [
        Segment(
            id=0,
            start=0.5,
            end=3.8,
            text="Hello everyone, thanks for joining the meeting.",
            speaker="SPEAKER_1",
            confidence=0.91,
            words=words[:7]
        ),
        Segment(
            id=1,
            start=5.2,
            end=8.5,
            text="Great to be here! Let's get started.",
            speaker="SPEAKER_2",
            confidence=0.92,
            words=words[7:14]
        ),
        Segment(
            id=2,
            start=10.1,
            end=14.8,
            text="I think we should discuss the quarterly results first.",
            speaker="SPEAKER_1",
            confidence=0.90,
            words=words[14:]
        )
    ]
    
    # Create sample speakers
    speakers = [
        Speaker(
            id="SPEAKER_1",
            label="SPEAKER_1",
            segments=[0, 2],
            total_duration=7.2,  # (3.8-0.5) + (14.8-10.1)
            percentage=65.5
        ),
        Speaker(
            id="SPEAKER_2", 
            label="SPEAKER_2",
            segments=[1],
            total_duration=3.3,  # 8.5-5.2
            percentage=34.5
        )
    ]
    
    # Create metadata
    metadata = {
        "engine": "whisperx",
        "model": "large-v2",
        "device": "cpu",
        "compute_type": "float16",
        "language_requested": "en",
        "diarization_enabled": True,
        "alignment_enabled": True,
        "timestamp": datetime.now().isoformat(),
        "file_name": "test_sample.m4a"
    }
    
    # Create full transcription result
    full_text = "Hello everyone, thanks for joining the meeting. Great to be here! Let's get started. I think we should discuss the quarterly results first."
    
    return TranscriptionResult(
        text=full_text,
        language="en",
        duration=14.8,
        segments=segments,
        speakers=speakers,
        words=words,
        metadata=metadata
    )


def test_all_formatters():
    """Test all output formatters with sample data."""
    print("🧪 Testing Enhanced Output Formatters")
    print("=" * 50)
    
    # Create sample data
    result = create_sample_transcription_data()
    filename = "test_sample.m4a"
    
    # Test Markdown Formatter
    print("\n📝 Testing Enhanced Markdown Formatter...")
    md_formatter = MarkdownFormatter(
        include_timestamps=True,
        include_confidence=True,
        include_word_timestamps=True,
        include_speaker_stats=True,
        include_conversation_flow=True
    )
    md_content = md_formatter.format(result, filename)
    
    with open("test_output_enhanced.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("✅ Enhanced Markdown saved to: test_output_enhanced.md")
    
    # Test JSON Formatter
    print("\n📊 Testing Enhanced JSON Formatter...")
    json_formatter = JSONFormatter(
        pretty=True,
        include_words=True,
        include_analysis=True,
        include_statistics=True
    )
    json_content = json_formatter.format(result, filename)
    
    with open("test_output_enhanced.json", "w", encoding="utf-8") as f:
        f.write(json_content)
    print("✅ Enhanced JSON saved to: test_output_enhanced.json")
    
    # Test SRT Formatter
    print("\n🎬 Testing SRT Subtitle Formatter...")
    srt_formatter = SRTFormatter(include_speakers=True, max_chars_per_line=50)
    srt_content = srt_formatter.format(result, filename)
    
    with open("test_output.srt", "w", encoding="utf-8") as f:
        f.write(srt_content)
    print("✅ SRT subtitles saved to: test_output.srt")
    
    # Test Advanced SRT Formatter
    print("\n🌈 Testing Advanced SRT Formatter with Speaker Colors...")
    advanced_srt_formatter = AdvancedSRTFormatter(
        include_speakers=True, 
        use_speaker_colors=True,
        max_chars_per_line=50
    )
    advanced_srt_content = advanced_srt_formatter.format(result, filename)
    
    with open("test_output_advanced.srt", "w", encoding="utf-8") as f:
        f.write(advanced_srt_content)
    print("✅ Advanced SRT saved to: test_output_advanced.srt")
    
    # Test VTT Formatter
    print("\n📺 Testing VTT (WebVTT) Formatter...")
    vtt_formatter = VTTFormatter(include_speakers=True, include_header=True, max_chars_per_line=50)
    vtt_content = vtt_formatter.format(result, filename)
    
    with open("test_output.vtt", "w", encoding="utf-8") as f:
        f.write(vtt_content)
    print("✅ VTT subtitles saved to: test_output.vtt")
    
    print("\n🎉 All formatters tested successfully!")
    print("\nGenerated files:")
    print("- test_output_enhanced.md (Enhanced Markdown with analysis)")
    print("- test_output_enhanced.json (JSON with statistics)")
    print("- test_output.srt (Basic SRT subtitles)")
    print("- test_output_advanced.srt (SRT with speaker colors)")
    print("- test_output.vtt (WebVTT subtitles)")


if __name__ == "__main__":
    test_all_formatters()