# Design Document

## Overview
This document outlines the technical architecture for integrating WhisperX with speaker diarization capabilities into the existing transcription service. The design maintains backward compatibility while adding powerful local processing, speaker identification, and enhanced output formats.

## Architecture

### High-Level Architecture
```
┌─────────────────┐
│   CLI Interface │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Router  │──────► Config Manager
    └────┬────┘
         │
    ┌────▼────────────────┐
    │  Engine Selection   │
    └──┬──────────────┬───┘
       │              │
   ┌───▼───┐    ┌────▼────┐
   │OpenAI │    │WhisperX │
   │  API  │    │ Engine  │
   └───┬───┘    └────┬────┘
       │             │
       │        ┌────▼─────────┐
       │        │ Diarization  │
       │        │   Pipeline   │
       │        └────┬─────────┘
       │             │
    ┌──▼─────────────▼───┐
    │  Output Formatter  │
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │  File Writer       │
    └────────────────────┘
```

### Component Architecture

#### 1. CLI Router Component
```python
# cli/router.py
class TranscriptionRouter:
    def __init__(self, config: Config):
        self.config = config
        self.engine = self._select_engine()
    
    def _select_engine(self) -> TranscriptionEngine:
        if self.config.engine == "whisperx" or self.config.local:
            return WhisperXEngine(self.config)
        return OpenAIEngine(self.config)
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        return self.engine.transcribe(audio_path)
```

#### 2. WhisperX Engine Component
```python
# engines/whisperx_engine.py
class WhisperXEngine(TranscriptionEngine):
    def __init__(self, config: Config):
        self.model = self._load_model(config)
        self.diarization_pipeline = self._init_diarization(config)
        self.alignment_model = self._load_alignment_model(config)
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        # Load and preprocess audio
        audio = self.load_audio(audio_path)
        
        # Transcribe with WhisperX
        result = self.model.transcribe(audio)
        
        # Align for word-level timestamps
        if self.config.word_timestamps:
            result = self.align_words(audio, result)
        
        # Apply diarization if enabled
        if self.config.diarize:
            result = self.apply_diarization(audio, result)
        
        return result
```

## Components and Interfaces

### Core Interfaces

#### TranscriptionEngine Interface
```typescript
interface TranscriptionEngine {
  transcribe(audioPath: string): Promise<TranscriptionResult>;
  validateFile(filePath: string): ValidationResult;
  getCapabilities(): EngineCapabilities;
}
```

#### TranscriptionResult Interface
```typescript
interface TranscriptionResult {
  text: string;
  language: string;
  duration: number;
  segments: Segment[];
  speakers?: Speaker[];
  words?: Word[];
  metadata: Metadata;
}

interface Segment {
  id: number;
  start: number;
  end: number;
  text: string;
  speaker?: string;
  confidence: number;
}

interface Word {
  word: string;
  start: number;
  end: number;
  confidence: number;
  speaker?: string;
}

interface Speaker {
  id: string;
  label: string;
  segments: number[];
  totalDuration: number;
  percentage: number;
}
```

#### Configuration Interface
```typescript
interface WhisperXConfig {
  engine: 'whisperx' | 'openai';
  model: 'tiny' | 'base' | 'small' | 'medium' | 'large' | 'large-v2' | 'large-v3';
  device: 'cuda' | 'cpu' | 'auto';
  computeType: 'float16' | 'float32' | 'int8';
  language?: string;
  diarize: boolean;
  minSpeakers?: number;
  maxSpeakers?: number;
  wordTimestamps: boolean;
  vadFilter: boolean;
  batchSize: number;
  outputFormat: 'markdown' | 'json' | 'srt' | 'vtt';
  huggingfaceToken?: string;
}
```

### Module Structure

```
transcriber/
├── cli/
│   ├── __init__.py
│   ├── router.py           # Main routing logic
│   └── parser.py           # Argument parsing
├── engines/
│   ├── __init__.py
│   ├── base.py            # Abstract base engine
│   ├── openai_engine.py   # OpenAI API implementation
│   └── whisperx_engine.py # WhisperX implementation
├── diarization/
│   ├── __init__.py
│   ├── pipeline.py        # Diarization pipeline
│   └── models.py          # Model management
├── alignment/
│   ├── __init__.py
│   ├── word_aligner.py    # Word-level alignment
│   └── language_models.py # Language-specific models
├── formatters/
│   ├── __init__.py
│   ├── markdown.py        # Markdown formatter
│   ├── json.py           # JSON formatter
│   └── subtitles.py      # SRT/VTT formatter
├── utils/
│   ├── __init__.py
│   ├── audio.py          # Audio processing utilities
│   ├── progress.py       # Progress indicators
│   └── validation.py     # Input validation
├── config/
│   ├── __init__.py
│   ├── settings.py       # Configuration management
│   └── models.py         # Model configurations
└── transcribe.py         # Main entry point
```

## Data Models

### Database Schema Changes
No database changes required for initial implementation (file-based system).

### File System Structure
```
~/.whisperx/
├── models/
│   ├── whisper/
│   │   ├── large-v2.pt
│   │   └── large-v3.pt
│   ├── alignment/
│   │   ├── wav2vec2-he.pt
│   │   └── wav2vec2-en.pt
│   └── diarization/
│       ├── pyannote-segmentation.pt
│       └── pyannote-embedding.pt
└── config/
    └── default.yaml
```

### Configuration Schema
```yaml
# config/default.yaml
whisperx:
  model: large-v2
  device: auto
  compute_type: float16
  batch_size: 16
  
diarization:
  enabled: false
  min_speakers: null
  max_speakers: null
  clustering: AgglomerativeClustering
  
alignment:
  enabled: true
  language_models:
    he: ivrit-ai/wav2vec2-large-xlsr-53-hebrew
    en: jonatasgrosman/wav2vec2-large-xlsr-53-english
    
vad:
  enabled: true
  threshold: 0.5
  min_speech_duration: 0.25
  
output:
  default_format: markdown
  include_word_timestamps: false
  include_confidence_scores: false
```

## Error Handling

### Error Categories and Strategies

#### 1. GPU/Memory Errors
```python
class GPUMemoryHandler:
    def handle_oom_error(self, error: Exception) -> RecoveryStrategy:
        strategies = [
            ReduceBatchSize(),
            SwitchToSmallerModel(),
            FallbackToCPU(),
            SplitAudioChunks()
        ]
        return self.select_strategy(error, strategies)
```

#### 2. Model Loading Errors
```python
class ModelLoadHandler:
    def handle_load_error(self, error: Exception) -> RecoveryAction:
        if isinstance(error, ModelNotFoundError):
            return DownloadModel()
        elif isinstance(error, CorruptModelError):
            return RedownloadModel()
        elif isinstance(error, IncompatibleModelError):
            return SuggestCompatibleModel()
        return RaiseError()
```

#### 3. Diarization Failures
```python
class DiarizationHandler:
    def handle_diarization_error(self, error: Exception) -> FallbackAction:
        logger.warning(f"Diarization failed: {error}")
        return ContinueWithoutDiarization()
```

### Error Response Format
```json
{
  "status": "error",
  "error_type": "GPUMemoryError",
  "message": "Insufficient GPU memory for large-v2 model",
  "suggestions": [
    "Try using --model base for lower memory usage",
    "Use --device cpu for CPU processing",
    "Reduce batch size with --batch-size 8"
  ],
  "fallback_action": "Switched to CPU processing",
  "partial_result": null
}
```

## Testing Strategy

### Unit Tests
```python
# tests/test_whisperx_engine.py
class TestWhisperXEngine:
    def test_model_loading(self):
        """Test model initialization and loading"""
    
    def test_transcription_accuracy(self):
        """Test transcription accuracy against baseline"""
    
    def test_diarization_output(self):
        """Test speaker diarization format and accuracy"""
    
    def test_word_alignment(self):
        """Test word-level timestamp accuracy"""
    
    def test_hebrew_support(self):
        """Test Hebrew language transcription"""
```

### Integration Tests
```python
# tests/test_integration.py
class TestEndToEnd:
    def test_cli_with_diarization(self):
        """Test complete CLI flow with diarization"""
    
    def test_fallback_mechanisms(self):
        """Test error recovery and fallbacks"""
    
    def test_output_formats(self):
        """Test all output format generation"""
    
    def test_performance_benchmarks(self):
        """Test processing speed requirements"""
```

### Performance Tests
```python
# tests/test_performance.py
class TestPerformance:
    def test_gpu_speed(self):
        """Verify 70x realtime processing speed"""
    
    def test_memory_usage(self):
        """Verify <8GB GPU memory constraint"""
    
    def test_cpu_fallback_speed(self):
        """Benchmark CPU processing speeds"""
```

## Security Considerations

### API Key Management
```python
class SecureConfig:
    def __init__(self):
        self.hf_token = self._load_secure_token()
    
    def _load_secure_token(self) -> str:
        # Priority order:
        # 1. Environment variable
        # 2. .env file
        # 3. Secure keyring
        # 4. Prompt user
        return self._get_token_securely()
```

### File Processing Security
- Input validation for file paths and formats
- Sandboxed audio processing
- Temporary file cleanup
- No logging of transcription content
- Local processing option for sensitive data

### Privacy Considerations
- Clear indication when using cloud vs local processing
- No telemetry or usage tracking
- Secure deletion of temporary files
- Option to disable all network connections

## Performance Optimization

### GPU Optimization
```python
class GPUOptimizer:
    def optimize_for_hardware(self, gpu_info: GPUInfo):
        if gpu_info.memory < 4096:  # 4GB
            return ModelConfig(model="base", batch_size=8)
        elif gpu_info.memory < 8192:  # 8GB
            return ModelConfig(model="small", batch_size=16)
        else:  # 8GB+
            return ModelConfig(model="large-v2", batch_size=24)
```

### Batch Processing
```python
class BatchProcessor:
    def process_long_audio(self, audio: Audio) -> Result:
        chunks = self.chunk_audio(audio, chunk_size=30)  # 30-second chunks
        results = []
        
        with tqdm(total=len(chunks)) as pbar:
            for chunk in chunks:
                result = self.process_chunk(chunk)
                results.append(result)
                pbar.update(1)
        
        return self.merge_results(results)
```

### Caching Strategy
```python
class ModelCache:
    def __init__(self):
        self.cache_dir = Path.home() / ".whisperx" / "cache"
        self.models = {}
    
    def get_model(self, model_name: str) -> Model:
        if model_name not in self.models:
            self.models[model_name] = self.load_model(model_name)
        return self.models[model_name]
```

## Migration Strategy

### Phase 1: Parallel Implementation
- Add WhisperX as optional engine
- Maintain OpenAI API as default
- Feature flag: `--engine whisperx`

### Phase 2: Testing & Validation
- A/B testing with sample files
- Performance benchmarking
- Accuracy comparison
- User acceptance testing

### Phase 3: Gradual Rollout
- Enable WhisperX for specific file types
- Automatic selection based on file size
- User preference storage

### Phase 4: Default Switch
- Make WhisperX default for local processing
- OpenAI API for cloud preference
- Full documentation update

## Monitoring and Metrics

### Performance Metrics
- Processing speed (x realtime)
- Memory usage (GPU/CPU)
- Model load time
- Transcription accuracy (WER)
- Diarization accuracy (DER)

### Usage Metrics (Optional)
- Engine selection distribution
- Feature usage (diarization, word timestamps)
- Error rates and types
- Processing time by file size

### Health Checks
```python
class HealthCheck:
    def verify_installation(self) -> HealthStatus:
        checks = [
            self.check_cuda_availability(),
            self.check_model_files(),
            self.check_dependencies(),
            self.check_disk_space(),
            self.check_memory_availability()
        ]
        return HealthStatus(checks)
```