#!/usr/bin/env python3
"""
Test video concatenation fix
"""
import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_concat_fix():
    """Test that FFmpeg concat works correctly"""
    print("=== Testing FFmpeg Concat Fix ===")
    
    # Find test video files
    uploads_dir = "uploads"
    video_files = []
    
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.endswith('.mp4'):
                video_files.append(os.path.join(uploads_dir, file))
                if len(video_files) >= 2:
                    break
    
    if len(video_files) < 2:
        print("❌ Need at least 2 video files for concat test")
        return False
    
    # Create temporary concat file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for video_file in video_files:
            # Use absolute path for FFmpeg compatibility
            absolute_path = Path(video_file).resolve()
            f.write(f"file '{absolute_path}'\n")
    
    print(f"Created concat file: {concat_file}")
    
    # Read and display concat file contents
    with open(concat_file, 'r') as f:
        concat_contents = f.read()
        print(f"Concat file contents:\n{concat_contents}")
    
    # Test concat command
    output_file = "test_concat_output.mp4"
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        output_file
    ]
    
    print(f"Running concat command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Concat command successful!")
        
        # Check if output file was created
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✅ Output file created: {output_file} ({size} bytes)")
            
            # Check file properties
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'v:0',
                output_file
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                if 'codec_name=h264' in probe_result.stdout:
                    print("✅ Output contains proper H.264 video stream")
                else:
                    print("⚠️  Output may not contain proper video stream")
                    print(f"Probe output: {probe_result.stdout[:200]}...")
            
            # Test frame extraction
            frame_cmd = [
                'ffmpeg', '-i', output_file, '-vframes', '1', '-y', 'test_concat_frame.png'
            ]
            
            frame_result = subprocess.run(frame_cmd, capture_output=True, text=True)
            if frame_result.returncode == 0 and os.path.exists('test_concat_frame.png'):
                frame_size = os.path.getsize('test_concat_frame.png')
                print(f"✅ Frame extraction successful: test_concat_frame.png ({frame_size} bytes)")
            else:
                print("❌ Frame extraction failed")
                print(f"Frame command stderr: {frame_result.stderr}")
            
            return True
        else:
            print("❌ Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Concat command failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(concat_file):
            os.unlink(concat_file)

if __name__ == "__main__":
    print("🚀 Testing video concatenation fix...")
    
    if test_concat_fix():
        print("\n✅ Concat test passed! Video concatenation should work properly.")
    else:
        print("\n❌ Concat test failed. Check the logs above.")
