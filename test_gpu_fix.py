#!/usr/bin/env python3
"""
Test GPU encoding fix with actual video files
"""
import os
import sys
import subprocess
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from python.video_composer import gpu_subprocess_run

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gpu_complex_filter():
    """Test GPU encoding with complex filter (xstack)"""
    print("=== Testing GPU Complex Filter ===")
    
    # Find some test video files
    uploads_dir = "uploads"
    video_files = []
    
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.endswith('.mp4'):
                video_files.append(os.path.join(uploads_dir, file))
                if len(video_files) >= 2:
                    break
    
    if len(video_files) < 2:
        print("❌ Need at least 2 video files for test")
        return False
    
    # Create a complex filter command (2x1 grid)
    output_file = "test_gpu_complex.mp4"
    
    complex_cmd = [
        'ffmpeg', '-y',
        '-i', video_files[0],
        '-i', video_files[1],
        '-filter_complex', 
        '[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[v0][v1]xstack=inputs=2:layout=0_0|640_0[video_out];[0:a][1:a]amix=inputs=2:duration=longest[audio_out]',
        '-map', '[video_out]',
        '-map', '[audio_out]',
        '-t', '2',  # Limit to 2 seconds for quick test
        output_file
    ]
    
    print(f"Testing complex filter with files: {video_files}")
    print(f"Command: {' '.join(complex_cmd[:5])}... (truncated)")
    
    try:
        result = gpu_subprocess_run(complex_cmd, capture_output=True, text=True, check=True)
        print("✅ GPU complex filter test successful!")
        
        # Check if output file was created
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✅ Output file created: {output_file} ({size} bytes)")
            return True
        else:
            print("❌ Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU complex filter test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_gpu_simple():
    """Test GPU encoding with simple command"""
    print("\n=== Testing GPU Simple Encoding ===")
    
    # Find a test video file
    uploads_dir = "uploads"
    video_file = None
    
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.endswith('.mp4'):
                video_file = os.path.join(uploads_dir, file)
                break
    
    if not video_file:
        print("❌ No video file found for test")
        return False
    
    # Create a simple command
    output_file = "test_gpu_simple.mp4"
    
    simple_cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-t', '2',  # Limit to 2 seconds
        output_file
    ]
    
    print(f"Testing simple encoding with file: {video_file}")
    
    try:
        result = gpu_subprocess_run(simple_cmd, capture_output=True, text=True, check=True)
        print("✅ GPU simple encoding test successful!")
        
        # Check if output file was created
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✅ Output file created: {output_file} ({size} bytes)")
            return True
        else:
            print("❌ Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU simple encoding test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing GPU encoding fixes...")
    
    success_count = 0
    total_tests = 2
    
    if test_gpu_simple():
        success_count += 1
    
    if test_gpu_complex_filter():
        success_count += 1
    
    print(f"\n=== Results ===")
    print(f"✅ {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All GPU tests passed! The fix should work.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
