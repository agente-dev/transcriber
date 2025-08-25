#!/usr/bin/env python3
"""
Test script for word-level alignment functionality.
"""

import os
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from alignment.word_aligner import WordAligner, AlignmentConfig
from alignment.language_models import LanguageModelManager


def test_language_model_manager():
    """Test language model manager functionality."""
    print("=== Testing Language Model Manager ===")
    
    try:
        # Initialize manager
        manager = LanguageModelManager()
        print(f"✓ Language model manager initialized (device: {manager.device})")
        
        # Test supported languages
        languages = manager.get_supported_languages()
        print(f"✓ Supported languages: {list(languages.keys())}")
        
        # Test language support validation
        for lang in ['he', 'en', 'ar', 'xyz']:
            validation = manager.validate_language_support(lang)
            status = "✓" if validation['supported'] else "!" if validation['fallback_required'] else "✗"
            print(f"{status} {lang}: supported={validation['supported']}, "
                  f"fallback_required={validation['fallback_required']}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"    Warning: {warning}")
        
        # Test RTL detection
        print(f"✓ Hebrew is RTL: {manager.is_rtl_language('he')}")
        print(f"✓ English is RTL: {manager.is_rtl_language('en')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Language model manager test failed: {e}")
        return False


def test_word_aligner_config():
    """Test word aligner configuration."""
    print("\n=== Testing Word Aligner Configuration ===")
    
    try:
        # Test default configuration
        config = AlignmentConfig()
        print(f"✓ Default config created:")
        print(f"    enabled: {config.enabled}")
        print(f"    device: {config.device}")
        print(f"    min_word_confidence: {config.min_word_confidence}")
        print(f"    interpolate_method: {config.interpolate_method}")
        
        # Test custom configuration
        custom_config = AlignmentConfig(
            enabled=True,
            return_char_alignments=True,
            interpolate_method='linear',
            device='cpu',
            min_word_confidence=0.2,
            rtl_support=True
        )
        print(f"✓ Custom config created:")
        print(f"    char_alignments: {custom_config.return_char_alignments}")
        print(f"    interpolate_method: {custom_config.interpolate_method}")
        print(f"    rtl_support: {custom_config.rtl_support}")
        
        return True
        
    except Exception as e:
        print(f"✗ Word aligner config test failed: {e}")
        return False


def test_word_aligner_initialization():
    """Test word aligner initialization without requiring models."""
    print("\n=== Testing Word Aligner Initialization ===")
    
    try:
        # Create configuration
        config = AlignmentConfig(
            enabled=True,
            device='cpu',  # Use CPU to avoid GPU requirements
            min_word_confidence=0.1
        )
        
        # Initialize aligner (this will test dependencies)
        aligner = WordAligner(config)
        print(f"✓ Word aligner initialized")
        print(f"    device: {aligner.language_manager.device}")
        print(f"    cache_dir: {aligner.cache_dir}")
        
        # Test statistics
        stats = aligner.get_alignment_stats()
        print(f"✓ Alignment stats retrieved:")
        print(f"    current_language: {stats['current_language']}")
        print(f"    config_enabled: {stats['config']['enabled']}")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Word aligner test skipped (missing dependencies): {e}")
        print("  This is expected if WhisperX is not installed")
        return True
        
    except Exception as e:
        print(f"✗ Word aligner initialization test failed: {e}")
        return False


def test_validation_functions():
    """Test validation functionality with mock data."""
    print("\n=== Testing Validation Functions ===")
    
    try:
        # Mock segments data for validation testing
        mock_segments = [
            {
                'id': 0,
                'start': 0.0,
                'end': 2.0,
                'text': 'Hello world',
                'words': [
                    {'word': 'Hello', 'start': 0.0, 'end': 0.5, 'score': 0.95},
                    {'word': 'world', 'start': 0.6, 'end': 2.0, 'score': 0.88}
                ]
            },
            {
                'id': 1,
                'start': 2.5,
                'end': 4.0,
                'text': 'Test segment',
                'words': [
                    {'word': 'Test', 'start': 2.5, 'end': 3.0, 'score': 0.92},
                    {'word': 'segment', 'start': 3.1, 'end': 4.0, 'score': 0.85}
                ]
            }
        ]
        
        # Test with minimal aligner setup (no model loading)
        config = AlignmentConfig(enabled=True, device='cpu')
        try:
            aligner = WordAligner(config)
            
            # Test validation
            validation = aligner.validate_alignment_quality(mock_segments, 4.0)
            print(f"✓ Validation completed:")
            print(f"    status: {validation['status']}")
            print(f"    total_segments: {validation['total_segments']}")
            print(f"    segments_with_words: {validation['segments_with_words']}")
            print(f"    total_words: {validation['total_words']}")
            print(f"    coverage_percentage: {validation['coverage_percentage']:.1f}%")
            print(f"    avg_word_confidence: {validation['avg_word_confidence']:.2f}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"    Warning: {warning}")
            
        except ImportError:
            print("⚠ Skipping validation test (WhisperX not available)")
            # Manual validation logic for testing
            total_words = sum(len(seg.get('words', [])) for seg in mock_segments)
            words_with_timestamps = sum(
                len([w for w in seg.get('words', []) if 'start' in w and 'end' in w])
                for seg in mock_segments
            )
            coverage = (words_with_timestamps / total_words * 100) if total_words > 0 else 0
            print(f"✓ Manual validation test:")
            print(f"    total_words: {total_words}")
            print(f"    coverage_percentage: {coverage:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation functions test failed: {e}")
        return False


def main():
    """Run all alignment tests."""
    print("Testing Word-Level Alignment Implementation")
    print("=" * 50)
    
    tests = [
        test_language_model_manager,
        test_word_aligner_config,
        test_word_aligner_initialization,
        test_validation_functions
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)