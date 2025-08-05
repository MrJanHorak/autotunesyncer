#!/usr/bin/env python3
"""
Test video processing pipeline to reproduce the "list indices must be integers or slices, not str" error
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path

# Add backend path to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'python'))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_midi_data():
    """Create minimal test MIDI data structure"""
    return {
        "tracks": [
            {
                "index": 0,
                "instrument": {
                    "name": "piano",
                    "isDrum": False
                },
                "notes": [
                    {
                        "midi": 60,
                        "time": 0.0,
                        "duration": 1.0,
                        "velocity": 100
                    }
                ]
            }
        ],
        "header": {
            "format": 1,
            "numTracks": 1,
            "ticksPerQuarter": 480
        },
        "duration": 5.0,
        "gridArrangement": {
            "0": {"row": 0, "column": 0}
        }
    }

def create_test_video_files():
    """Create minimal test video files structure"""
    return {
        "tracks": {
            "piano": {
                "path": "/test/path/piano.mp4",
                "notes": {
                    "60": "/test/path/piano_note_60.mp4"
                }
            }
        }
    }

def test_video_processor():
    """Test the video processor directly"""
    print("=== Testing Video Processor Directly ===")
    
    try:
        # Import video processor using correct path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'utils'))
        from video_processor import main as video_processor_main
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_file = os.path.join(temp_dir, 'midi_data.json')
            video_files_file = os.path.join(temp_dir, 'video_files.json')
            output_file = os.path.join(temp_dir, 'output.mp4')
            
            # Write test data
            with open(midi_file, 'w') as f:
                json.dump(create_test_midi_data(), f)
            
            with open(video_files_file, 'w') as f:
                json.dump(create_test_video_files(), f)
            
            # Test with sys.argv simulation
            original_argv = sys.argv
            sys.argv = ['video_processor.py', midi_file, video_files_file, output_file]
            
            try:
                video_processor_main()
            except Exception as e:
                print(f"Video processor error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                sys.argv = original_argv
                
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

def test_video_composer():
    """Test the video composer directly"""
    print("=== Testing Video Composer Directly ===")
    
    try:
        from video_composer import VideoComposer
        
        # Create temporary directory structure that matches expected layout
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the expected directory structure:
            # temp_dir/backend/uploads/
            # temp_dir/backend/session/uuid/processed/
            backend_dir = Path(temp_dir) / "backend"
            backend_dir.mkdir()
            
            uploads_dir = backend_dir / "uploads"
            uploads_dir.mkdir()
            
            session_dir = backend_dir / "session" / "test-uuid"
            session_dir.mkdir(parents=True)
            
            processed_dir = session_dir / "processed"
            processed_dir.mkdir()
            
            # Create minimal output
            output_file = session_dir / "output.mp4"
            
            # Test VideoComposer initialization with corrected path
            midi_data = create_test_midi_data()
            
            composer = VideoComposer(
                processed_videos_dir=str(processed_dir),
                midi_data=midi_data,
                output_path=str(output_file)
            )
            
            print("VideoComposer initialized successfully")
            
    except Exception as e:
        print(f"VideoComposer error: {e}")
        import traceback
        traceback.print_exc()

def test_ffmpeg_params():
    """Test the get_optimized_ffmpeg_params function specifically"""
    print("=== Testing FFmpeg Params Function ===")
    
    try:
        from video_utils import get_optimized_ffmpeg_params
        
        # Test function call
        params = get_optimized_ffmpeg_params()
        print(f"FFmpeg params type: {type(params)}")
        print(f"FFmpeg params content: {params}")
        
        # Test parameter access like preprocessing code does
        if isinstance(params, dict):
            print("Testing dictionary access:")
            for key in ['video_codec', 'preset', 'crf', 'gpu_options']:
                if key in params:
                    print(f"  {key}: {params[key]}")
                else:
                    print(f"  {key}: NOT FOUND")
        else:
            print(f"ERROR: Expected dict, got {type(params)}")
            
    except Exception as e:
        print(f"FFmpeg params error: {e}")
        import traceback
        traceback.print_exc()

def test_preprocess_videos():
    """Test preprocess_videos.py function directly"""
    print("=== Testing Preprocess Videos Function ===")
    
    try:
        from preprocess_videos import preprocess_video, VideoPreprocessor
        
        # Test VideoPreprocessor initialization
        preprocessor = VideoPreprocessor(performance_mode=True)
        print("VideoPreprocessor initialized successfully")
          # Create a dummy input/output path (won't actually process)
        test_input = "test_input_real.mp4"
        test_output = "test_output_preprocessed3.mp4"
        test_dimensions = (640, 480)
        
        # This should fail early but show us where the error occurs
        try:
            result = preprocess_video(test_input, test_output, test_dimensions)
            print(f"Preprocess result: {result}")
        except FileNotFoundError:
            print("File not found (expected) - function structure OK")
        except Exception as e:
            print(f"Preprocess error (this might be our target): {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Preprocess import/test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("AutoTuneSyncer Video Processing Pipeline Test")
    print("=" * 50)
    
    # Test individual components to isolate the error
    test_ffmpeg_params()
    print()
    test_preprocess_videos()
    print()
    test_video_composer()
    print()
    test_video_processor()
    
    print("\nTest completed!")
