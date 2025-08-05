#!/usr/bin/env python3
"""
Debug preprocessing to track down why processed files aren't being detected
"""
import os
import sys
import logging
import json
from pathlib import Path

# Add backend/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'python'))

from preprocess_videos import preprocess_video, VideoPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_data():
    """Create test data files similar to the real pipeline"""
    # Create test video (black screen with audio)
    test_video = "debug_test.mp4"
      if not os.path.exists(test_video):
        from video_utils import run_ffmpeg_command
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=black:size=640x480:duration=2',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-pix_fmt', 'yuv420p', test_video
        ]
        run_ffmpeg_command(cmd)
        print(f"Created test video: {test_video}")
    
    return test_video

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline"""
    print("=== DEBUG: Testing Preprocessing Pipeline ===")
    
    # Create test data
    test_video = create_test_data()
    
    # Test legacy preprocessing function
    print("\n1. Testing legacy preprocess_video() function:")
    output_legacy = "debug_output_legacy.mp4"
    
    try:
        preprocess_video(test_video, output_legacy, target_size="640x480")
        
        if os.path.exists(output_legacy):
            size = os.path.getsize(output_legacy)
            print(f"   ✓ Legacy preprocessing SUCCESS: {output_legacy} ({size} bytes)")
        else:
            print(f"   ✗ Legacy preprocessing FAILED: File not created")
            
    except Exception as e:
        print(f"   ✗ Legacy preprocessing ERROR: {e}")
    
    # Test enhanced preprocessing
    print("\n2. Testing VideoPreprocessor class:")
    preprocessor = VideoPreprocessor()
    output_enhanced = "debug_output_enhanced.mp4"
    
    try:
        result = preprocessor.preprocess_video_enhanced(test_video, output_enhanced, target_size="640x480")
        
        if os.path.exists(output_enhanced):
            size = os.path.getsize(output_enhanced)
            print(f"   ✓ Enhanced preprocessing SUCCESS: {output_enhanced} ({size} bytes)")
        else:
            print(f"   ✗ Enhanced preprocessing FAILED: File not created")
            
    except Exception as e:
        print(f"   ✗ Enhanced preprocessing ERROR: {e}")
    
    # Test file detection logic
    print("\n3. Testing file detection:")
    test_files = {
        'piano': output_legacy,
        'guitar': output_enhanced
    }
    
    for instrument, file_path in test_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            abs_path = os.path.abspath(file_path)
            print(f"   ✓ {instrument}: {file_path} ({size} bytes)")
            print(f"     Absolute path: {abs_path}")
            
            # Test video validation
            try:
                from video_utils import validate_video
                validate_video(file_path)
                print(f"     ✓ Validation: PASSED")
            except Exception as e:
                print(f"     ✗ Validation: FAILED - {e}")
        else:
            print(f"   ✗ {instrument}: {file_path} - FILE NOT FOUND")

def test_path_resolution():
    """Test path resolution similar to video composer"""
    print("\n=== DEBUG: Testing Path Resolution ===")
    
    # Simulate the video composer's path logic
    current_dir = os.getcwd()
    uploads_dir = os.path.join(current_dir, "uploads")
    
    print(f"Current directory: {current_dir}")
    print(f"Uploads directory: {uploads_dir}")
    
    # Create test structure
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Create a test processed file
    test_file = os.path.join(uploads_dir, "processed_piano.mp4")
    test_video = create_test_data()
    
    try:
        from shutil import copy2
        copy2(test_video, test_file)
        print(f"Created test file: {test_file}")
        
        # Test different path resolution methods
        methods = [
            ("os.path.exists", lambda p: os.path.exists(p)),
            ("Path.exists", lambda p: Path(p).exists()),
            ("os.path.isfile", lambda p: os.path.isfile(p)),
        ]
        
        for method_name, method_func in methods:
            result = method_func(test_file)
            print(f"   {method_name}({test_file}): {result}")
            
        # Test absolute vs relative paths
        abs_path = os.path.abspath(test_file)
        rel_path = os.path.relpath(test_file)
        
        print(f"Absolute path: {abs_path}")
        print(f"Relative path: {rel_path}")
        print(f"Absolute exists: {os.path.exists(abs_path)}")
        print(f"Relative exists: {os.path.exists(rel_path)}")
        
    except Exception as e:
        print(f"Path resolution test failed: {e}")

if __name__ == "__main__":
    test_preprocessing_pipeline()
    test_path_resolution()
    print("\n=== DEBUG COMPLETE ===")
