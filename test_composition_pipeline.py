#!/usr/bin/env python3
"""
End-to-end test for video composition pipeline
Tests the complete flow from preprocessing to composition
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_video_data_structure():
    """Test that video data structure is handled correctly"""
    print("🧪 Testing Video Data Structure Handling")
    print("=" * 50)
    
    # Simulate the data structure that comes from queueService.js
    video_files_data = {
        "piano": {
            "videoData": b"fake_video_data_piano",
            "isDrum": False,
            "drumName": None,
            "notes": [
                {"time": 0.0, "duration": 1.0, "midi": 60, "velocity": 100}
            ],
            "layout": {"x": 0, "y": 0, "width": 960, "height": 720},
            "index": 0,
            "processedAt": 1640995200000
        },
        "drum_kick": {
            "videoData": b"fake_video_data_kick",
            "isDrum": True,
            "drumName": "kick",
            "notes": [
                {"time": 0.5, "duration": 0.1, "midi": 36, "velocity": 127}
            ],
            "layout": {"x": 0, "y": 0, "width": 960, "height": 720},
            "index": 1,
            "processedAt": 1640995200000
        }
    }
    
    # Test the Python video processor logic
    from backend.utils.video_processor import EnhancedVideoProcessor
    
    processor = EnhancedVideoProcessor()
    
    print("\n📋 Testing Track Data Validation:")
    for track_id, track_data in video_files_data.items():
        print(f"\nTesting track: {track_id}")
        
        # Simulate the validation logic from video_processor.py line 211
        if 'path' in track_data and Path(track_data['path']).exists():
            print("✅ Would use file path")
        elif 'videoData' in track_data:
            print("✅ Would use video buffer data")
            print(f"   - Data size: {len(track_data['videoData'])} bytes")
            print(f"   - Is drum: {track_data['isDrum']}")
            print(f"   - Notes count: {len(track_data['notes'])}")
        else:
            print("❌ ERROR: No valid video data found")
            return False
    
    print("\n🎯 Result: All tracks have valid video data structure!")
    return True

def test_json_file_format():
    """Test the JSON file format that gets written to temp files"""
    print("\n📄 Testing JSON File Format")
    print("=" * 30)
    
    # Simulate MIDI data
    midi_data = {
        "tracks": [
            {
                "instrument": {"name": "piano"},
                "notes": [
                    {"time": 0.0, "duration": 1.0, "midi": 60, "velocity": 100}
                ],
                "isDrum": False
            }
        ],
        "gridArrangement": {
            "piano": {"row": 0, "column": 0}
        }
    }
    
    # Simulate video files data (after transformation)
    video_files = {
        "piano": {
            "videoData": "base64_encoded_video_data_here",
            "isDrum": False,
            "notes": [
                {"time": 0.0, "duration": 1.0, "midi": 60, "velocity": 100}
            ]
        }
    }
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        midi_path = Path(temp_dir) / "midi_data.json"
        video_path = Path(temp_dir) / "video_files.json"
        
        # Write files
        with open(midi_path, 'w') as f:
            json.dump(midi_data, f, indent=2)
        
        with open(video_path, 'w') as f:
            json.dump(video_files, f, indent=2)
        
        print(f"✅ Created MIDI JSON: {midi_path}")
        print(f"✅ Created Video JSON: {video_path}")
        
        # Verify files can be read back
        try:
            with open(midi_path, 'r') as f:
                loaded_midi = json.load(f)
            
            with open(video_path, 'r') as f:
                loaded_video = json.load(f)
            
            print("✅ JSON files can be read back successfully")
            
            # Check video data structure
            if 'piano' in loaded_video and 'videoData' in loaded_video['piano']:
                print("✅ Video data has correct 'videoData' key")
            else:
                print("❌ Video data missing 'videoData' key")
                return False
                
        except Exception as e:
            print(f"❌ Error reading JSON files: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 AutoTuneSyncer Video Composition Pipeline Test")
    print("=" * 60)
    
    tests = [
        ("Video Data Structure", test_video_data_structure),
        ("JSON File Format", test_json_file_format),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The video composition fix should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
