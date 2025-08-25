# Test Directory

This directory contains test outputs and temporary files generated during development and testing of the transcription system.

## Directory Structure

```
tests/
├── output/          # Generated test outputs (git ignored)
│   ├── *.md         # Markdown transcription outputs
│   ├── *.json       # JSON structured outputs
│   ├── *.srt        # SRT subtitle files
│   ├── *.vtt        # VTT (WebVTT) subtitle files
│   └── *.m4a        # Test audio files
└── README.md        # This file
```

## Test Files

### Formatter Test Outputs
- `test_output_enhanced.md` - Enhanced Markdown with conversation analysis
- `test_output_enhanced.json` - JSON with comprehensive statistics
- `test_output.srt` - Basic SRT subtitles
- `test_output_advanced.srt` - Advanced SRT with speaker colors
- `test_output.vtt` - WebVTT subtitles

### Real-World Test Outputs
- `Diane Meeting 21:8:25.*` - Outputs from 22-minute Hebrew audio test
  - `.md` - Rich Markdown transcription
  - `.json` - Structured JSON with analytics
  - `.srt` - Professional subtitles
  - `.m4a` - Original audio file (21.4 MB)

## Usage

These files are automatically generated when:

1. **Running the formatter test script:**
   ```bash
   python test_formatters.py
   ```

2. **Processing real audio files:**
   ```bash
   python transcribe.py --engine whisperx --output-format [format] audio_file.m4a
   ```

## Git Ignore

All files in `tests/output/` are ignored by git to:
- Keep the repository clean and lightweight
- Avoid committing large binary audio files
- Prevent temporary test outputs from cluttering commits
- Focus repository on source code and documentation

## File Formats Tested

- **Markdown**: Enhanced formatting with speaker analysis and conversation flow
- **JSON**: Comprehensive structured data with statistics and analytics
- **SRT**: Professional subtitle formatting with timing
- **VTT**: WebVTT subtitles with metadata headers
- **Audio**: Multiple formats supported (M4A, MP3, WAV, FLAC)