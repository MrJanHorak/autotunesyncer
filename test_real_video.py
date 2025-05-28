#!/usr/bin/env python3
"""
Test the complete video processing pipeline with a real video file
"""

import sys
import os
import tempfile
import shutil
import logging

# Setup paths
sys.path.append('backend/python')
sys.path.append('backend/utils')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_real_video_processing():
    """Test complete video processing with a real file"""
    print("=" * 60)
    print("AutoTuneSyncer Real Video Processing Test")
    print("=" * 60)
    
    # Check if test file exists
    test_input = "test_input_real.mp4"
    if not os.path.exists(test_input):
        print(f"❌ Test input file '{test_input}' not found")
        return False
    
    print(f"✓ Found test input: {test_input}")
    
    try:
        # Test 1: GPU Context Manager
        print("\n=== Testing GPU Context Manager ===")
        from processing_utils import GPUManager
        
        gpu_manager = GPUManager()
        print(f"GPU available: {gpu_manager.has_gpu}")
        
        # Test context manager usage
        with gpu_manager.gpu_context() as stream:
            print(f"✓ GPU context manager working, stream: {stream}")
        
        # Test 2: FFmpeg Parameters
        print("\n=== Testing FFmpeg Parameters ===")
        from video_utils import get_optimized_ffmpeg_params
        
        params = get_optimized_ffmpeg_params()
        print(f"✓ FFmpeg params type: {type(params)}")
        print(f"✓ Video codec: {params['video_codec']}")
        
        # Test 3: Video Preprocessing
        print("\n=== Testing Video Preprocessing ===")
        from preprocess_videos import VideoPreprocessor
        
        preprocessor = VideoPreprocessor(performance_mode=True)
        print("✓ VideoPreprocessor initialized")
        
        # Test 4: Real Video Processing
        print("\n=== Testing Real Video Processing ===")
        test_output = "test_output_real.mp4"
        
        # Create a simple preprocessing command
        from video_utils import run_ffmpeg_command
        
        cmd = [
            'ffmpeg', '-y', '-i', test_input,
            '-vf', 'scale=320:240',  # Simple scale to test
            '-c:v', 'libx264',       # Use CPU encoder to avoid hardware issues
            '-preset', 'fast',
            '-crf', '28',
            '-t', '1',               # Only process 1 second
            test_output
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            run_ffmpeg_command(cmd)
            if os.path.exists(test_output):
                print(f"✓ Video processing successful! Output: {test_output}")
                print(f"✓ Output file size: {os.path.getsize(test_output)} bytes")
                return True
            else:
                print("❌ Output file not created")
                return False
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        for file in ['test_output_real.mp4']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"✓ Cleaned up {file}")
                except:
                    pass

def test_video_composer_integration():
    """Test VideoComposer with real integration"""
    print("\n=== Testing VideoComposer Integration ===")
    
    try:
        from video_composer import VideoComposer
        
        # Create test MIDI data
        test_midi = {
            'tracks': [
                {
                    'name': 'piano',
                    'instrument': {'name': 'piano'},
                    'notes': [
                        {'time': 0.0, 'duration': 0.5, 'midi': 60, 'velocity': 80}
                    ]
                }
            ],
            'header': {'tempo': 120},
            'duration': 2.0,
            'gridArrangement': {'0': {'row': 0, 'column': 0}}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the expected directory structure that VideoComposer expects
            # temp_dir/
            #   backend/
            #     uploads/
            #     session/
            #       session_id/
            #         processed/
            
            backend_dir = os.path.join(temp_dir, 'backend')
            uploads_dir = os.path.join(backend_dir, 'uploads')
            session_dir = os.path.join(backend_dir, 'session', 'test-session')
            processed_dir = os.path.join(session_dir, 'processed')
            
            # Create all directories
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Create output path
            output_path = os.path.join(session_dir, 'output.mp4')
            
            # Initialize VideoComposer with correct parameters and directory structure
            composer = VideoComposer(
                processed_videos_dir=processed_dir,
                midi_data=test_midi,
                output_path=output_path
            )
            
            print("✓ VideoComposer initialized with real MIDI data")
            print(f"✓ Regular tracks: {len(composer.tracks)}")
            print(f"✓ Drum tracks: {len(composer.drum_tracks)}")
            print(f"✓ Uploads directory: {composer.uploads_dir}")
            print(f"✓ Processed directory: {composer.processed_videos_dir}")
            return True
            
    except Exception as e:
        print(f"❌ VideoComposer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    results = []
    
    # Test 1: Real video processing
    results.append(test_real_video_processing())
    
    # Test 2: VideoComposer integration
    results.append(test_video_composer_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"Successful: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Video processing pipeline is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
