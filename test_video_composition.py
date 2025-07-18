#!/usr/bin/env python3

import sys
import os
import json
import logging
from pathlib import Path

# Add the backend directory to the path
sys.path.append('backend/python')

from video_composer import VideoComposer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_video_composition():
    """Test video composition with the fixed note-triggered video sequence"""
    
    print("=== Testing Video Composition ===")
    
    try:
        # Create directories
        processed_videos_dir = Path('backend/processed_videos')
        processed_videos_dir.mkdir(exist_ok=True)
        
        # Create test MIDI data with a small number of notes
        test_midi = {
            'tracks': [
                {
                    'instrument': {'family': 'brass', 'number': 58, 'name': 'tuba'},
                    'notes': [
                        {'time': 0.0, 'duration': 1.0, 'midi': 60},
                        {'time': 1.0, 'duration': 1.0, 'midi': 62},
                        {'time': 2.0, 'duration': 1.0, 'midi': 64}
                    ]
                },
                {
                    'instrument': {'family': 'brass', 'number': 56, 'name': 'trumpet'},
                    'notes': [
                        {'time': 0.5, 'duration': 1.0, 'midi': 67},
                        {'time': 1.5, 'duration': 1.0, 'midi': 69},
                        {'time': 2.5, 'duration': 1.0, 'midi': 71}
                    ]
                }
            ]
        }
        
        output_path = 'test_composition_output.mp4'
        
        # Initialize VideoComposer
        composer = VideoComposer(processed_videos_dir, test_midi, output_path)
        print("✅ VideoComposer initialized successfully")
        
        # Test creating composition
        print("\n🎬 Starting video composition...")
        result = composer.create_composition()
        
        if result and os.path.exists(result):
            file_size = os.path.getsize(result)
            print(f"✅ Composition successful!")
            print(f"   📁 Output: {result}")
            print(f"   📏 Size: {file_size:,} bytes")
            
            # Check if the video is not blank by verifying it has reasonable size
            if file_size > 50000:  # More than 50KB suggests actual content
                print("✅ Video appears to have content (not blank)")
            else:
                print("⚠️  Video may be blank or very small")
                
        else:
            print("❌ Composition failed - no output file created")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_composition()
