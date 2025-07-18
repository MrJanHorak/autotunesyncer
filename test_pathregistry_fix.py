#!/usr/bin/env python3

import sys
import os
import json
import logging
from pathlib import Path

# Add the backend directory to the path
sys.path.append('backend/python')

from video_composer import VideoComposer
from path_registry import PathRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_pathregistry_fixes():
    """Test that the PathRegistry.get_instrument_path() fixes are working correctly"""
    
    print("=== Testing PathRegistry Fixes ===")
    
    # Test 1: Register paths and test PathRegistry calls
    print("\n1. Setting up PathRegistry with actual video files:")
    
    try:
        registry = PathRegistry.get_instance()
        
        # Find and register actual video files from uploads
        uploads_dir = Path('backend/uploads')
        
        # Register tuba videos
        for video_file in uploads_dir.glob('*tuba*.mp4'):
            registry.register_instrument('tuba', '60', str(video_file))
            print(f"✅ Registered tuba: {video_file.name}")
        
        # Register trumpet videos
        for video_file in uploads_dir.glob('*trumpet*.mp4'):
            registry.register_instrument('trumpet', '60', str(video_file))
            print(f"✅ Registered trumpet: {video_file.name}")
        
        # Register tenor_sax videos
        for video_file in uploads_dir.glob('*tenor_sax*.mp4'):
            registry.register_instrument('tenor_sax', '60', str(video_file))
            print(f"✅ Registered tenor_sax: {video_file.name}")
        
        # Register honky-tonk_piano videos
        for video_file in uploads_dir.glob('*honky-tonk_piano*.mp4'):
            registry.register_instrument('honky-tonk_piano', '60', str(video_file))
            print(f"✅ Registered honky-tonk_piano: {video_file.name}")
        
        print("\n2. Testing PathRegistry calls:")
        
        # Test with available instruments
        test_instruments = ['tuba', 'trumpet', 'tenor_sax', 'honky-tonk_piano']
        test_note = "60"  # Middle C
        
        for instrument in test_instruments:
            try:
                path = registry.get_instrument_path(instrument, test_note)
                if path:
                    print(f"✅ {instrument} + note {test_note}: {os.path.basename(path)}")
                else:
                    print(f"❌ {instrument} + note {test_note}: No path found")
            except Exception as e:
                print(f"❌ {instrument} + note {test_note}: ERROR - {e}")
        
        print("\n3. Testing track name extraction:")
        
        # Test track name extraction with dict format
        track_examples = [
            {'instrument': {'family': 'brass', 'number': 58, 'name': 'tuba'}, 'notes': []},
            {'instrument': {'family': 'brass', 'number': 56, 'name': 'trumpet'}, 'notes': []},
            {'instrument': {'family': 'reed', 'number': 66, 'name': 'tenor_sax'}, 'notes': []},
            {'instrument': {'family': 'piano', 'number': 3, 'name': 'honky-tonk_piano'}, 'notes': []}
        ]
        
        for track in track_examples:
            if isinstance(track.get('instrument'), dict):
                track_name = track['instrument'].get('name', 'unknown')
            else:
                track_name = track.get('instrument', 'unknown')
            
            print(f"✅ Track name extraction: {track_name}")
        
        print("\n4. Testing _process_instrument_track_for_chunk method:")
        
        # Create a mock VideoComposer instance
        uploads_dir = Path('backend/uploads')
        processed_videos_dir = Path('backend/processed_videos')
        output_path = 'test_output.mp4'
        
        # Create simple test MIDI data
        test_midi = {
            'tracks': [
                {
                    'instrument': {'family': 'brass', 'number': 58, 'name': 'tuba'},
                    'notes': [
                        {'time': 0.0, 'duration': 1.0, 'midi': 60},
                        {'time': 1.0, 'duration': 1.0, 'midi': 62}
                    ]
                }
            ]
        }
        
        # Test VideoComposer initialization
        try:
            composer = VideoComposer(processed_videos_dir, test_midi, output_path)
            print("✅ VideoComposer initialization successful")
            
            # Test the fixed _process_instrument_track_for_chunk method
            test_track = test_midi['tracks'][0]
            
            result = composer._process_instrument_track_for_chunk(
                track=test_track,
                chunk_start_time=0.0,
                chunk_duration=4.0,
                chunk_index=0
            )
            
            if result:
                print(f"✅ _process_instrument_track_for_chunk returned result: {result}")
            else:
                print("❌ _process_instrument_track_for_chunk returned None")
                
        except Exception as e:
            print(f"❌ VideoComposer test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n=== PathRegistry Fix Test Complete ===")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pathregistry_fixes()
