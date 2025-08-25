#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from cli import create_argument_parser, TranscriptionRouter
from cli.parser import args_to_config
from config import load_config
from config.settings import create_default_config_file
from formatters import MarkdownFormatter, JSONFormatter, SRTFormatter, VTTFormatter, AdvancedSRTFormatter
from utils.progress import ProgressBar, set_global_quiet

load_dotenv()


def check_setup():
    """Check installation and configuration."""
    print("🔍 Checking WhisperX setup...")
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("! GPU not available - will use CPU processing")
    except ImportError:
        print("✗ PyTorch not installed")
    
    try:
        import whisperx
        print("✓ WhisperX installed")
    except ImportError:
        print("✗ WhisperX not installed - only OpenAI API will be available")
        print("  Install with: pip install whisperx")
    
    try:
        from pyannote.audio import Pipeline
        print("✓ Pyannote.audio available for diarization")
    except ImportError:
        print("! Pyannote.audio not available - diarization will be disabled")
        print("  Install with: pip install pyannote.audio")
    
    # Check configuration
    config_dir = Path.home() / '.whisperx'
    if config_dir.exists():
        print(f"✓ Configuration directory: {config_dir}")
    else:
        print(f"! Configuration directory not found: {config_dir}")
        print("  Run with --create-config to create default configuration")
    
    print("\n🔍 Setup check complete!")


def get_output_path(input_path: str, output_format: str, custom_path: str = None) -> Path:
    """Generate appropriate output path based on format."""
    input_path_obj = Path(input_path)
    
    if custom_path:
        return Path(custom_path)
    
    # Default extensions by format
    extensions = {
        'markdown': '.md',
        'json': '.json',
        'srt': '.srt',
        'vtt': '.vtt'
    }
    
    extension = extensions.get(output_format, '.md')
    return input_path_obj.with_suffix(extension)


def format_and_save_result(result, filename: str, output_path: Path, output_format: str, 
                          include_timestamps: bool = False, include_confidence: bool = False,
                          include_word_timestamps: bool = False, include_speaker_stats: bool = True):
    """Format and save transcription result."""
    
    if output_format == 'markdown':
        formatter = MarkdownFormatter(
            include_timestamps=include_timestamps, 
            include_confidence=include_confidence,
            include_word_timestamps=include_word_timestamps,
            include_speaker_stats=include_speaker_stats
        )
        content = formatter.format(result, filename)
        
    elif output_format == 'json':
        formatter = JSONFormatter(
            pretty=True, 
            include_words=True,
            include_analysis=True,
            include_statistics=include_speaker_stats
        )
        content = formatter.format(result, filename)
        
    elif output_format == 'srt':
        # Use enhanced SRT formatter with speaker support
        has_speakers = result.speakers and len(result.speakers) > 1
        if has_speakers:
            formatter = AdvancedSRTFormatter(
                include_speakers=True,
                use_speaker_colors=True,
                max_chars_per_line=42
            )
        else:
            formatter = SRTFormatter(include_speakers=False)
        content = formatter.format(result, filename)
        
    elif output_format == 'vtt':
        formatter = VTTFormatter(
            include_speakers=result.speakers and len(result.speakers) > 1,
            max_chars_per_line=42
        )
        content = formatter.format(result, filename)
        
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Transcription saved to: {output_path}")
    
    # Show format-specific information
    if output_format in ['srt', 'vtt'] and result.speakers and len(result.speakers) > 1:
        print(f"  📺 Generated {output_format.upper()} subtitles with {len(result.speakers)} speaker labels")
    elif output_format == 'json' and include_speaker_stats:
        print(f"  📊 JSON output includes detailed speaker statistics and conversation analysis")
    elif output_format == 'markdown' and include_speaker_stats:
        print(f"  📝 Markdown output includes enhanced speaker analysis and conversation flow")


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle utility commands
    if args.check_setup:
        check_setup()
        return
    
    if args.create_config:
        create_default_config_file()
        return
    
    # Validate required arguments
    if not hasattr(args, 'audio_file') or not args.audio_file:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        config_overrides = args_to_config(args)
        config = load_config(
            config_path=getattr(args, 'config', None),
            **config_overrides
        )
        
        if args.verbose:
            print(f"Using engine: {config.engine}")
            if config.engine == 'whisperx':
                print(f"WhisperX model: {config.whisperx.model}")
                print(f"Device: {config.whisperx.device}")
                if config.diarize:
                    print("Speaker diarization: Enabled")
                if config.word_timestamps:
                    print("Word timestamps: Enabled")
        
        # Set global progress settings
        set_global_quiet(args.quiet)
        
        # Initialize router
        router = TranscriptionRouter(config)
        
        # Validate file
        validation = router.validate_file(args.audio_file)
        if not validation.is_valid:
            print(f"Error: {validation.error_message}")
            sys.exit(1)
        
        if validation.error_message:  # Warning message
            print(f"Warning: {validation.error_message}")
        
        # Perform transcription
        print(f"🎤 Starting transcription...")
        result = router.transcribe(args.audio_file)
        
        # Determine output path
        output_path = get_output_path(
            args.audio_file, 
            config.output_format,
            config.output_path
        )
        
        # Format and save result
        format_and_save_result(
            result=result,
            filename=Path(args.audio_file).name,
            output_path=output_path,
            output_format=config.output_format,
            include_timestamps=config.word_timestamps or config.output.include_word_timestamps,
            include_confidence=config.output.include_confidence_scores,
            include_word_timestamps=config.word_timestamps,
            include_speaker_stats=True  # Always include speaker stats unless specifically disabled
        )
        
        # Summary
        print(f"✅ Transcription complete!")
        
        if result.speakers:
            print(f"📊 Detected {len(result.speakers)} speakers")
            for speaker in result.speakers:
                print(f"   - {speaker.label}: {speaker.percentage:.1f}% of conversation")
        
        print(f"⏱️  Duration: {result.duration:.1f}s")
        print(f"📝 Language: {result.language}")
        
        if not args.quiet:
            print(f"\n--- Preview ---")
            preview_text = result.text[:200] + "..." if len(result.text) > 200 else result.text
            print(preview_text)
        
    except KeyboardInterrupt:
        print("\n⚠️  Transcription interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()