#!/usr/bin/env python3
"""
Comprehensive validation that the fix is complete and working
"""

import os
import tempfile
import json
import subprocess
import sys
from pathlib import Path

def create_test_video(output_path, duration=2, resolution="640x480"):
    """Create a simple test video using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'testsrc=duration={duration}:size={resolution}:rate=30',
            '-f', 'lavfi', 
            '-i', f'sine=frequency=1000:duration={duration}',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-t', str(duration),
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"Error creating test video: {e}")
        return False

def test_single_video_composition():
    """Test composition with a single video"""
    print("🎬 Testing Single Video Composition")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test video
        test_video = temp_path / "piano.mp4"
        if not create_test_video(str(test_video)):
            return False
        
        # Create configuration
        midi_data = {
            "tracks": {"1": {"notes": [{"time": 0, "duration": 1, "midi": 60, "velocity": 1.0}]}},
            "header": {"tempo": 120},
            "gridArrangement": {"piano": {"row": 0, "column": 0}}
        }
        
        video_files = {
            "piano": {
                "path": str(test_video),
                "isDrum": False,
                "notes": [],
                "layout": {"x": 0, "y": 0, "width": 640, "height": 480}
            }
        }
        
        # Test processing
        return run_processor_test(temp_path, midi_data, video_files, "single_video")

def test_multiple_video_composition():
    """Test composition with multiple videos"""
    print("🎬 Testing Multiple Video Composition")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test videos
        piano_video = temp_path / "piano.mp4"
        drum_video = temp_path / "drum.mp4"
        
        if not create_test_video(str(piano_video), duration=2):
            return False
        if not create_test_video(str(drum_video), duration=2):
            return False
        
        # Create configuration
        midi_data = {
            "tracks": {
                "1": {"notes": [{"time": 0, "duration": 1, "midi": 60, "velocity": 1.0}]},
                "2": {"notes": [{"time": 0, "duration": 1, "midi": 36, "velocity": 1.0}]}
            },
            "header": {"tempo": 120},
            "gridArrangement": {
                "piano": {"row": 0, "column": 0},
                "drum_kick": {"row": 0, "column": 1}
            }
        }
        
        video_files = {
            "piano": {
                "path": str(piano_video),
                "isDrum": False,
                "notes": [],
                "layout": {"x": 0, "y": 0, "width": 640, "height": 480}
            },
            "drum_kick": {
                "path": str(drum_video),
                "isDrum": True,
                "notes": [],
                "layout": {"x": 640, "y": 0, "width": 640, "height": 480}
            }
        }
        
        # Test processing
        return run_processor_test(temp_path, midi_data, video_files, "multi_video")

def test_video_data_buffer():
    """Test composition with video buffer data (the old problematic format)"""
    print("🎬 Testing Video Buffer Data")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test video and read as buffer
        test_video = temp_path / "piano.mp4"
        if not create_test_video(str(test_video)):
            return False
        
        # Read video as buffer (simulating the old format)
        with open(test_video, 'rb') as f:
            video_buffer = f.read()
        
        # Create configuration with videoData key
        midi_data = {
            "tracks": {"1": {"notes": [{"time": 0, "duration": 1, "midi": 60, "velocity": 1.0}]}},
            "header": {"tempo": 120},
            "gridArrangement": {"piano": {"row": 0, "column": 0}}
        }
        
        video_files = {
            "piano": {
                "videoData": video_buffer,  # This is the old format that was causing the error
                "isDrum": False,
                "notes": [],
                "layout": {"x": 0, "y": 0, "width": 640, "height": 480}
            }
        }
        
        # Test processing
        return run_processor_test(temp_path, midi_data, video_files, "buffer_data")

def run_processor_test(temp_path, midi_data, video_files, test_name):
    """Run the processor test with given data"""
    # Save data to JSON files
    midi_json = temp_path / "midi_data.json"
    videos_json = temp_path / "video_files.json"
    output_video = temp_path / f"output_{test_name}.mp4"
    
    with open(midi_json, 'w') as f:
        json.dump(midi_data, f)
    
    with open(videos_json, 'w') as f:
        json.dump(video_files, f, default=lambda x: x.hex() if isinstance(x, bytes) else x)
    
    # Convert bytes to base64 if needed for proper JSON serialization
    import base64
    for track_id, track_data in video_files.items():
        if 'videoData' in track_data and isinstance(track_data['videoData'], bytes):
            video_files[track_id]['videoData'] = base64.b64encode(track_data['videoData']).decode()
    
    with open(videos_json, 'w') as f:
        json.dump(video_files, f)
    
    # Run processor
    backend_dir = Path(__file__).parent / "backend"
    processor_script = backend_dir / "utils" / "video_processor.py"
    
    cmd = [
        sys.executable, str(processor_script),
        str(midi_json),
        str(videos_json),
        str(output_video)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=str(backend_dir)
        )
        
        # Check for the specific error
        if "No valid video data for track" in result.stderr:
            print(f"❌ ERROR in {test_name}: 'No valid video data for track' error still occurs!")
            print(f"   Stderr: {result.stderr}")
            return False
        elif result.returncode == 0:
            output_size = output_video.stat().st_size if output_video.exists() else 0
            print(f"✅ SUCCESS {test_name}: Video processing completed ({output_size} bytes)")
            return True
        else:
            print(f"⚠️  {test_name}: Process failed with different error (code: {result.returncode})")
            if result.stderr:
                print(f"   Stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error in {test_name}: {e}")
        return False

def main():
    """Run comprehensive validation tests"""
    print("🚀 AutoTuneSyncer Comprehensive Fix Validation")
    print("Testing all scenarios to ensure the fix is complete")
    print("=" * 70)
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
    except:
        print("❌ FFmpeg not found. Please install FFmpeg to run this test.")
        return False
    
    tests = [
        ("Single Video Composition", test_single_video_composition),
        ("Multiple Video Composition", test_multiple_video_composition),
        ("Video Buffer Data (Legacy Format)", test_video_data_buffer),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ PASS: {test_name}")
            else:
                print(f"❌ FAIL: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 🎉 🎉 COMPLETE SUCCESS! 🎉 🎉 🎉")
        print("The 'No valid video data for track' error has been COMPLETELY FIXED!")
        print("All video composition scenarios are working correctly:")
        print("  ✅ Single video compositions")
        print("  ✅ Multiple video compositions") 
        print("  ✅ Legacy video buffer data format")
        print("  ✅ Current file path format")
        print("\nThe AutoTuneSyncer video composition pipeline is fully operational!")
    else:
        print(f"\n⚠️  {total-passed} tests failed - some issues may remain")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
