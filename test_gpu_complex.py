#!/usr/bin/env python3
"""
Test direct GPU encoding with complex filters
"""
import os
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gpu_complex_direct():
    """Test GPU encoding directly with complex filter"""
    print("=== Testing Direct GPU Complex Filter ===")
    
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
    
    # Create a complex filter command with GPU encoding
    output_file = "test_gpu_complex_direct.mp4"
    
    gpu_complex_cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', video_files[0],
        '-i', video_files[1],
        '-filter_complex', 
        '[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[v0][v1]xstack=inputs=2:layout=0_0|640_0[video_out];[0:a][1:a]amix=inputs=2:duration=longest[audio_out]',
        '-map', '[video_out]',
        '-map', '[audio_out]',
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        '-t', '2',  # Limit to 2 seconds for quick test
        output_file
    ]
    
    print(f"Testing GPU complex filter with files: {video_files}")
    print(f"Command: ffmpeg -y -hwaccel cuda -i {video_files[0]} -i {video_files[1]} -filter_complex ... -c:v h264_nvenc")
    
    try:
        result = subprocess.run(gpu_complex_cmd, capture_output=True, text=True, check=True)
        print("✅ Direct GPU complex filter test successful!")
        
        # Check if output file was created
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✅ Output file created: {output_file} ({size} bytes)")
            
            # Check if it used GPU encoding
            if 'h264_nvenc' in result.stderr:
                print("✅ Confirmed GPU encoding was used!")
                return True
            else:
                print("⚠️  GPU encoding may not have been used")
                return False
        else:
            print("❌ Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Direct GPU complex filter test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_cpu_fallback():
    """Test CPU fallback when GPU fails"""
    print("\n=== Testing CPU Fallback ===")
    
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
    
    # Create a command that will use CPU encoding
    output_file = "test_cpu_fallback.mp4"
    
    cpu_cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-c:v', 'libx264',  # Force CPU encoding
        '-preset', 'fast',
        '-t', '2',  # Limit to 2 seconds
        output_file
    ]
    
    print(f"Testing CPU fallback with file: {video_file}")
    
    try:
        result = subprocess.run(cpu_cmd, capture_output=True, text=True, check=True)
        print("✅ CPU fallback test successful!")
        
        # Check if output file was created
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"✅ Output file created: {output_file} ({size} bytes)")
            return True
        else:
            print("❌ Output file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ CPU fallback test failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing GPU encoding with complex filters...")
    
    success_count = 0
    total_tests = 2
    
    if test_gpu_complex_direct():
        success_count += 1
    
    if test_cpu_fallback():
        success_count += 1
    
    print(f"\n=== Results ===")
    print(f"✅ {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! GPU encoding with complex filters should work.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
