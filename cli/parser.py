import argparse
from typing import Dict, Any, Optional


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Hebrew Audio Transcription with WhisperX and Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription (OpenAI API - default)
  python transcribe.py audio.wav
  
  # Use WhisperX for local processing
  python transcribe.py audio.wav --engine whisperx
  
  # Local processing with speaker diarization
  python transcribe.py audio.wav --local --diarize --min-speakers 2 --max-speakers 4
  
  # Word-level timestamps with JSON output
  python transcribe.py audio.wav --engine whisperx --word-timestamps --output-format json
  
  # Custom model and device selection
  python transcribe.py audio.wav --engine whisperx --model large-v3 --device cuda
        """
    )
    
    # Positional argument (optional for utility commands)
    parser.add_argument(
        'audio_file',
        nargs='?',
        help='Path to the audio file to transcribe'
    )
    
    # Engine selection
    parser.add_argument(
        '--engine',
        choices=['openai', 'whisperx'],
        default='openai',
        help='Transcription engine to use (default: openai)'
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Force local processing (equivalent to --engine whisperx)'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        default='large-v2',
        help='Model to use. For OpenAI: whisper-1. For WhisperX: tiny, base, small, medium, large, large-v2, large-v3 (default: large-v2)'
    )
    
    # Device and performance
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use for processing (default: auto)'
    )
    
    parser.add_argument(
        '--compute-type',
        choices=['float16', 'float32', 'int8'],
        default='float16',
        help='Compute type for processing (default: float16)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    
    # Language settings
    parser.add_argument(
        '--language',
        default='he',
        help='Language code for transcription (default: he for Hebrew)'
    )
    
    # Diarization options
    parser.add_argument(
        '--diarize',
        action='store_true',
        help='Enable speaker diarization (requires WhisperX)'
    )
    
    parser.add_argument(
        '--min-speakers',
        type=int,
        help='Minimum number of speakers for diarization'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        help='Maximum number of speakers for diarization'
    )
    
    # Timestamps and alignment
    parser.add_argument(
        '--word-timestamps',
        action='store_true',
        help='Include word-level timestamps (requires WhisperX)'
    )
    
    parser.add_argument(
        '--no-alignment',
        action='store_true',
        help='Disable word alignment (WhisperX only)'
    )
    
    parser.add_argument(
        '--alignment-method',
        choices=['nearest', 'linear'],
        default='nearest',
        help='Word alignment interpolation method (default: nearest)'
    )
    
    parser.add_argument(
        '--min-word-confidence',
        type=float,
        default=0.1,
        help='Minimum confidence threshold for word alignment (default: 0.1)'
    )
    
    parser.add_argument(
        '--char-alignments',
        action='store_true',
        help='Include character-level alignments (advanced, slower)'
    )
    
    parser.add_argument(
        '--no-vad',
        action='store_true',
        help='Disable voice activity detection (WhisperX only)'
    )
    
    # Output options
    parser.add_argument(
        '--output-format',
        choices=['markdown', 'json', 'srt', 'vtt'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (default: same as input with appropriate extension)'
    )
    
    parser.add_argument(
        '--include-confidence',
        action='store_true',
        help='Include confidence scores in output'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--huggingface-token',
        help='Hugging Face access token for diarization models'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    # Diagnostic commands
    parser.add_argument(
        '--check-setup',
        action='store_true',
        help='Check installation and configuration'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file'
    )
    
    return parser


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed arguments to configuration dictionary."""
    config_dict = {}
    
    # Engine selection
    if args.local or args.diarize or args.word_timestamps:
        # Force WhisperX for local processing, diarization, or word timestamps
        config_dict['engine'] = 'whisperx'
        config_dict['local'] = True
        if args.engine == 'openai' and (args.diarize or args.word_timestamps):
            print("Note: Switching to WhisperX engine for diarization/word-timestamps support")
    else:
        config_dict['engine'] = args.engine
    
    # Model selection
    if args.engine == 'openai' and not args.local:
        config_dict['openai_model'] = args.model if args.model != 'large-v2' else 'whisper-1'
    else:
        config_dict['whisperx'] = {'model': args.model}
    
    # Language
    config_dict['language'] = args.language
    
    # WhisperX specific settings
    whisperx_config = {}
    if hasattr(args, 'device') and args.device != 'auto':
        whisperx_config['device'] = args.device
    if hasattr(args, 'compute_type') and args.compute_type != 'float16':
        whisperx_config['compute_type'] = args.compute_type
    if hasattr(args, 'batch_size') and args.batch_size != 16:
        whisperx_config['batch_size'] = args.batch_size
    
    if whisperx_config:
        config_dict['whisperx'] = {**config_dict.get('whisperx', {}), **whisperx_config}
    
    # Diarization settings
    if args.diarize:
        config_dict['diarize'] = True
        diarization_config = {'enabled': True}
        if args.min_speakers:
            diarization_config['min_speakers'] = args.min_speakers
        if args.max_speakers:
            diarization_config['max_speakers'] = args.max_speakers
        config_dict['diarization'] = diarization_config
    
    # Alignment settings
    alignment_config = {}
    if args.no_alignment:
        alignment_config['enabled'] = False
    elif args.word_timestamps:
        alignment_config['enabled'] = True
    
    # Advanced alignment options
    if hasattr(args, 'alignment_method') and args.alignment_method != 'nearest':
        alignment_config['interpolate_method'] = args.alignment_method
    if hasattr(args, 'min_word_confidence') and args.min_word_confidence != 0.1:
        alignment_config['min_word_confidence'] = args.min_word_confidence
    if hasattr(args, 'char_alignments') and args.char_alignments:
        alignment_config['return_char_alignments'] = True
    
    if alignment_config:
        config_dict['alignment'] = alignment_config
    
    # VAD settings
    if args.no_vad:
        config_dict['vad'] = {'enabled': False}
    
    # Word timestamps
    if args.word_timestamps:
        config_dict['word_timestamps'] = True
        # Ensure alignment is enabled for word timestamps
        if 'alignment' not in config_dict:
            config_dict['alignment'] = {'enabled': True}
        else:
            config_dict['alignment']['enabled'] = True
    
    # Output settings
    config_dict['output_format'] = args.output_format
    if args.output:
        config_dict['output_path'] = args.output
    
    output_config = {}
    if args.include_confidence:
        output_config['include_confidence_scores'] = True
    if args.word_timestamps:
        output_config['include_word_timestamps'] = True
    
    if output_config:
        config_dict['output'] = output_config
    
    # API tokens
    if args.huggingface_token:
        config_dict['huggingface_token'] = args.huggingface_token
    
    # Verbosity
    config_dict['verbose'] = args.verbose
    
    return config_dict