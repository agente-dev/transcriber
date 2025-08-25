# Hebrew Audio Transcriber

Advanced CLI tool for Hebrew audio transcription with **WhisperX** integration, featuring speaker diarization, word-level timestamps, and local processing capabilities.

## 🚀 Features

### Dual Engine Support
- **OpenAI API**: Cloud-based transcription (default, backward compatible)
- **WhisperX**: Local processing with advanced features

### Advanced Capabilities (WhisperX)
- 🎯 **Speaker Diarization**: Automatically identify and label different speakers
- ⏱️ **Word-Level Timestamps**: Precise timing for each word
- 🔒 **Local Processing**: Keep sensitive data private (no cloud upload)
- 🚀 **GPU Acceleration**: Fast processing on compatible hardware
- 📁 **Large Files**: No 25MB limit for local processing
- 🌍 **Multi-Language**: Hebrew, English, Arabic, and more

### Output Formats
- 📝 **Markdown**: Rich formatting with speaker sections
- 📊 **JSON**: Structured data with full metadata
- 🎬 **SRT/VTT**: Subtitle formats (planned)

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Check

Verify your installation and hardware capabilities:

```bash
python transcribe.py --check-setup
```

### 3. Create Configuration (Optional)

```bash
python transcribe.py --create-config
```

### 4. Set API Keys

For OpenAI API:
```bash
export OPENAI_API_KEY="your-api-key"
```

For diarization features:
```bash
export HUGGINGFACE_TOKEN="your-hf-token"
```

**Hugging Face Setup**:
1. Create a **READ** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept these model licenses:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## 📖 Usage

### Basic Transcription (OpenAI API)
```bash
python transcribe.py audio.wav
```

### Local Processing with WhisperX
```bash
python transcribe.py audio.wav --engine whisperx
# or
python transcribe.py audio.wav --local
```

### Speaker Diarization
```bash
python transcribe.py meeting.wav --local --diarize --min-speakers 2 --max-speakers 4
```

### Word-Level Timestamps
```bash
python transcribe.py interview.wav --engine whisperx --word-timestamps
```

### JSON Output with Full Data
```bash
python transcribe.py audio.wav --engine whisperx --diarize --output-format json
```

### Advanced Configuration
```bash
python transcribe.py audio.wav --engine whisperx \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --batch-size 24 \
  --language he
```

## 🔧 Command Line Options

### Engine Selection
- `--engine {openai,whisperx}`: Choose transcription engine
- `--local`: Force local processing (equivalent to --engine whisperx)

### Model & Performance
- `--model MODEL`: Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
- `--device {auto,cuda,cpu}`: Processing device
- `--compute-type {float16,float32,int8}`: Computation precision
- `--batch-size SIZE`: Batch size for processing

### Diarization Features
- `--diarize`: Enable speaker diarization
- `--min-speakers N`: Minimum number of speakers
- `--max-speakers N`: Maximum number of speakers

### Timestamps & Alignment
- `--word-timestamps`: Include word-level timestamps
- `--no-alignment`: Disable word alignment
- `--no-vad`: Disable voice activity detection

### Output Control
- `--output-format {markdown,json,srt,vtt}`: Output format
- `--output PATH`: Custom output file path
- `--include-confidence`: Include confidence scores

### Configuration
- `--config FILE`: Use custom configuration file
- `--huggingface-token TOKEN`: Hugging Face access token
- `--verbose`: Enable detailed output
- `--quiet`: Suppress non-essential output

### Utilities
- `--check-setup`: Check installation and configuration
- `--create-config`: Create default configuration file

## 📋 Supported Formats

**Audio Files**: MP3, M4A, WAV, WEBM, MP4, MPGA, MPEG, FLAC, OGG

**Output Formats**:
- **Markdown** (`.md`): Rich text with speaker sections and timestamps
- **JSON** (`.json`): Structured data with segments, words, and speaker information
- **SRT** (`.srt`): Subtitle format (planned)
- **VTT** (`.vtt`): WebVTT subtitle format (planned)

## 🎛️ Configuration

The system uses a hierarchical configuration system:

1. **Command line arguments** (highest priority)
2. **Custom config file** (`--config path/to/config.yaml`)
3. **User config** (`~/.whisperx/config/default.yaml`)
4. **Environment variables**
5. **Built-in defaults**

Example configuration file:
```yaml
whisperx:
  model: large-v2
  device: auto
  compute_type: float16
  batch_size: 16

diarization:
  enabled: false
  min_speakers: null
  max_speakers: null

alignment:
  enabled: true
  language_models:
    he: ivrit-ai/wav2vec2-large-xlsr-53-hebrew
    en: jonatasgrosman/wav2vec2-large-xlsr-53-english

output:
  default_format: markdown
  include_word_timestamps: false
  include_confidence_scores: false
```

## 🔍 Performance & Hardware

### GPU Requirements
- **CUDA-compatible GPU** for optimal performance
- **8GB+ VRAM** recommended for large models
- **Automatic CPU fallback** if GPU unavailable

### Processing Speed (GPU)
- **70x realtime** typical speed with WhisperX
- **5-minute audio** processes in ~10 seconds
- **Memory efficient** with batch processing

### Hebrew Language Support
- Optimized for Hebrew transcription
- RTL text handling
- Mixed Hebrew-English support
- Specialized alignment models

## 📊 Output Examples

### Markdown with Speakers
```markdown
# Transcription: meeting.wav

**Duration:** 180.5s  
**Language:** he  
**Engine:** whisperx  
**Speaker Diarization:** Enabled

## Speaker Summary
- **SPEAKER_0**: 2.1 minutes (68.9% of conversation)
- **SPEAKER_1**: 0.9 minutes (31.1% of conversation)

## Transcript

### SPEAKER_0
שלום לכולם, ברוכים הבאים לפגישה...

### SPEAKER_1
תודה רבה על ההזמנה...
```

### JSON with Full Data
```json
{
  "metadata": {
    "filename": "meeting.wav",
    "duration": 180.5,
    "language": "he",
    "speaker_count": 2,
    "engine_info": {
      "engine": "whisperx",
      "model": "large-v2",
      "device": "cuda"
    }
  },
  "text": "שלום לכולם, ברוכים הבאים לפגישה...",
  "segments": [...],
  "speakers": [...],
  "words": [...]
}
```

## 🚨 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Install CUDA toolkit
# Check with:
python transcribe.py --check-setup
```

**WhisperX not available:**
```bash
pip install whisperx
```

**Diarization errors:**
```bash
# Set Hugging Face token
export HUGGINGFACE_TOKEN="your-token"
```

**Memory issues:**
```bash
# Use smaller model or CPU
python transcribe.py audio.wav --model base --device cpu
```

## 📈 Migration from Previous Version

The new version is **100% backward compatible**:
- Existing scripts continue to work unchanged
- Default behavior remains the same (OpenAI API)
- Enhanced features are opt-in only

To use new features:
```bash
# Old command (still works)
python transcribe.py audio.wav

# New features
python transcribe.py audio.wav --engine whisperx --diarize
```

## 🔗 Related Projects

- [WhisperX](https://github.com/m-bain/whisperX): Fast automatic speech recognition with word-level timestamps
- [pyannote-audio](https://github.com/pyannote/pyannote-audio): Neural building blocks for speaker diarization
- [OpenAI Whisper](https://github.com/openai/whisper): Robust speech recognition via large-scale weak supervision

## 📄 License

MIT License - see LICENSE file for details.