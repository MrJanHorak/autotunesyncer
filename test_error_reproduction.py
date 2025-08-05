#!/usr/bin/env python3
"""
Create a simple end-to-end test to reproduce the original error
"""

import os
import tempfile
import json
import subprocess
import sys
from pathlib import Path

def create_test_video(output_path, duration=5):
    """Create a simple test video using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'testsrc=duration={duration}:size=640x480:rate=30',
            '-f', 'lavfi', 
            '-i', 'sine=frequency=1000:duration=5',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-t', str(duration),
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error creating test video: {e}")
        return False

def test_video_composition_error():
    """Test if the 'No valid video data for track' error still occurs"""
    print("🎬 Testing Video Composition Error")
    print("=" * 50)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test video
        test_video = temp_path / "test_input.mp4"
        print(f"📹 Creating test video: {test_video}")
        
        if not create_test_video(str(test_video)):
            print("❌ Failed to create test video")
            return False
        
        print(f"✅ Test video created: {test_video.stat().st_size} bytes")
        
        # Create MIDI data
        midi_data = {
            "tracks": {
                "1": {
                    "notes": [
                        {"time": 0, "duration": 1, "midi": 60, "velocity": 1.0},
                        {"time": 1, "duration": 1, "midi": 62, "velocity": 1.0}
                    ]
                }
            },
            "header": {"tempo": 120},
            "gridArrangement": {
                "piano": {"row": 0, "column": 0}
            }
        }
        
        # Create video files data  
        video_files = {
            "piano": {
                "path": str(test_video),
                "isDrum": False,
                "notes": [],
                "layout": {"x": 0, "y": 0, "width": 640, "height": 480}
            }
        }
        
        # Save data to JSON files
        midi_json = temp_path / "midi_data.json"
        videos_json = temp_path / "video_files.json"
        output_video = temp_path / "output.mp4"
        
        with open(midi_json, 'w') as f:
            json.dump(midi_data, f, indent=2)
        
        with open(videos_json, 'w') as f:
            json.dump(video_files, f, indent=2)
        
        print(f"📝 Created test files:")
        print(f"  MIDI: {midi_json}")
        print(f"  Videos: {videos_json}")
        print(f"  Output: {output_video}")
        
        # Test Python processor directly
        print("\n🐍 Testing Python video processor...")
        
        # Navigate to backend directory
        backend_dir = Path(__file__).parent / "backend"
        processor_script = backend_dir / "utils" / "video_processor.py"
        
        if not processor_script.exists():
            print(f"❌ Processor script not found: {processor_script}")
            return False
        
        # Run the processor
        cmd = [
            sys.executable, str(processor_script),
            str(midi_json),
            str(videos_json),
            str(output_video),
            "--performance-mode"
        ]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,
                cwd=str(backend_dir)
            )
            
            print(f"\n📊 Process Result:")
            print(f"  Return code: {result.returncode}")
            print(f"  Stdout: {result.stdout}")
            if result.stderr:
                print(f"  Stderr: {result.stderr}")
            
            # Check for the specific error
            if "No valid video data for track" in result.stderr:
                print("\n❌ ERROR REPRODUCED: 'No valid video data for track' error still occurs!")
                print("The fix is not complete yet.")
                return False
            elif result.returncode == 0:
                print("\n✅ SUCCESS: Video processing completed without the error!")
                if output_video.exists():
                    print(f"  Output file created: {output_video.stat().st_size} bytes")
                return True
            else:
                print(f"\n⚠️  Process failed with different error (return code: {result.returncode})")
                print("This might be a different issue not related to the 'No valid video data' error.")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n⏰ Process timed out")
            return False
        except Exception as e:
            print(f"\n❌ Error running processor: {e}")
            return False

def main():
    """Run the test"""
    print("🚀 AutoTuneSyncer Error Reproduction Test")
    print("Testing if 'No valid video data for track' error still occurs")
    print("=" * 60)
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
    except:
        print("❌ FFmpeg not found. Please install FFmpeg to run this test.")
        return False
    
    success = test_video_composition_error()
    
    print("\n" + "=" * 60)
    print("📊 ERROR REPRODUCTION TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 SUCCESS! The 'No valid video data for track' error has been FIXED!")
        print("The video composition pipeline is working correctly.")
    else:
        print("⚠️  ERROR STILL EXISTS or other issues found.")
        print("The fix may not be complete or there are other problems.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
