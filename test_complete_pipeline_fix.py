#!/usr/bin/env python3
"""
End-to-end test of the complete video composition pipeline fix
Tests the full chain: data transformation + video processing + FFmpeg filter fix
"""

import sys
import os
import tempfile
import json
import subprocess

# Add the backend paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'utils'))

def create_test_video(output_path, color="red", duration=3):
    """Create a test video file"""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'testsrc=duration={duration}:size=320x240:rate=30,format=yuv420p',
        '-f', 'lavfi', 
        '-i', f'sine=frequency=440:duration={duration}',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-t', str(duration),
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def test_complete_pipeline():
    """Test the complete pipeline with the fixes"""
    print("🔧 Complete Pipeline Test")
    print("Testing: Data Transformation + Video Processing + FFmpeg Filter Fix")
    print("=" * 70)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test videos
            test_videos = {}
            for i in range(8):  # Test with 8 videos to trigger the fixed filter
                video_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
                if create_test_video(video_path, duration=2):
                    test_videos[f"track_{i}"] = video_path
                    print(f"✅ Created test video {i}: {video_path}")
                else:
                    print(f"❌ Failed to create test video {i}")
                    return False
            
            # Create MIDI data (simplified)
            midi_data = {
                "tracks": list(test_videos.keys()),
                "duration": 2.0,
                "tempo": 120
            }
            
            # Create video files data with file paths (simulating the fixed queue service)
            video_files_data = {}
            for track_id, video_path in test_videos.items():
                video_files_data[track_id] = {
                    "path": video_path,  # This is what the fixed queue service provides
                    "isDrum": False,
                    "notes": []
                }
            
            # Save to JSON files
            midi_json_path = os.path.join(temp_dir, "midi_data.json")
            videos_json_path = os.path.join(temp_dir, "video_files.json")
            output_video_path = os.path.join(temp_dir, "final_composition.mp4")
            
            with open(midi_json_path, 'w') as f:
                json.dump(midi_data, f)
            
            with open(videos_json_path, 'w') as f:
                json.dump(video_files_data, f)
            
            print(f"📝 MIDI data saved: {midi_json_path}")
            print(f"📝 Video files data saved: {videos_json_path}")
            print(f"🎯 Target output: {output_video_path}")
            
            # Test the fixed video processor
            print("\n🎬 Testing Fixed Video Processor...")
            from video_processor import EnhancedVideoProcessor
            
            processor = EnhancedVideoProcessor(performance_mode=True)
            
            # Load data
            with open(midi_json_path) as f:
                midi_data = json.load(f)
            with open(videos_json_path) as f:
                video_files = json.load(f)
            
            print(f"📊 Processing {len(video_files)} videos...")
            
            # Run the video processing
            success = processor.process_videos(midi_data, video_files, output_video_path)
            
            if success:
                print("✅ Video processing completed successfully!")
                
                # Verify output
                if os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path)
                    print(f"✅ Output file created: {file_size} bytes")
                    
                    # Quick validation with ffprobe
                    probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', output_video_path]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    
                    if probe_result.returncode == 0:
                        format_info = json.loads(probe_result.stdout)
                        duration = float(format_info['format'].get('duration', 0))
                        print(f"✅ Video validation: {duration:.2f} seconds duration")
                        
                        if duration > 1.0:  # Should be around 2 seconds
                            print("🎉 COMPLETE PIPELINE TEST: SUCCESS!")
                            print("✅ All fixes are working together:")
                            print("   - Data structure transformation ✅")
                            print("   - Video data handling ✅") 
                            print("   - FFmpeg filter syntax ✅")
                            print("   - Video composition ✅")
                            return True
                        else:
                            print("❌ Video duration too short, composition may have failed")
                            return False
                    else:
                        print("❌ Video validation failed")
                        return False
                else:
                    print("❌ Output file not created")
                    return False
            else:
                print("❌ Video processing failed")
                return False
                
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 AutoTuneSyncer Complete Pipeline Fix Test")
    print("Testing all fixes working together end-to-end")
    print("=" * 60)
    
    success = test_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("📋 FINAL PIPELINE RESULT")
    print("=" * 60)
    
    if success:
        print("🎉 SUCCESS! Complete video composition pipeline is working!")
        print("✅ All critical errors have been resolved:")
        print("   1. 'No valid video data for track' - FIXED")
        print("   2. FFmpeg filter syntax error - FIXED")
        print("   3. Video combination stage - WORKING")
        print("🎬 Users can now create complete video compositions!")
    else:
        print("❌ FAILED! Pipeline still has issues")
        print("🔧 Additional debugging required")
    
    print(f"\nPipeline test result: {'PASS' if success else 'FAIL'}")
