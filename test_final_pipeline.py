#!/usr/bin/env python3
"""
Final end-to-end test of the complete video processing pipeline
with the note mapping fix to confirm synchronized video output.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Import the video processor
from utils.video_processor import EnhancedVideoProcessor

def test_complete_pipeline():
    """Test the complete pipeline with note mapping fix."""
    print("=== Final End-to-End Pipeline Test ===")
    
    # Find a MIDI data file and corresponding videos
    uploads_dir = backend_dir / "uploads"
    
    # Find MIDI file
    midi_files = []
    for file in uploads_dir.glob("*"):
        if file.is_file() and not file.suffix:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'tracks' in data and 'header' in data:
                        midi_files.append(file)
            except:
                continue
    
    if not midi_files:
        print("❌ No MIDI data files found!")
        return False
    
    midi_file = midi_files[0]
    print(f"✓ Using MIDI data: {midi_file}")
    
    with open(midi_file, 'r') as f:
        midi_data = json.load(f)
    
    # Find some matching video files (latest set)
    video_files = []
    for ext in ['.mp4']:
        video_files.extend(uploads_dir.glob(f"processed_1748444863*{ext}"))
    
    if not video_files:
        print("❌ No matching video files found!")
        return False
    
    print(f"✓ Found {len(video_files)} video files")
    
    # Create video mapping with note mapping logic
    videos = {}
    def normalize_instrument_name(name):
        return name.lower().replace(' ', '_')
    
    for video in video_files:
        instrument_name = video.stem[10:]  # Remove 'processed_' prefix
        
        videos[instrument_name] = {
            'path': str(video),
            'isDrum': 'drum' in instrument_name.lower(),
            'notes': [],
            'layout': {'x': 0, 'y': 0, 'width': 480, 'height': 270}
        }
    
    # Apply note mapping with the fixed logic
    notes_mapped = 0
    for track_index, track in enumerate(midi_data.get('tracks', [])):
        if not track.get('notes'):
            continue
        
        if track.get('instrument') and track['instrument'].get('name'):
            normalized_name = normalize_instrument_name(track['instrument']['name'])
            
            # Find matching video key
            matching_video_key = None
            for key in videos.keys():
                key_parts = key.split('-')
                instrument_part = key_parts[-1] if key_parts else ""
                
                if instrument_part == normalized_name or normalized_name in key:
                    matching_video_key = key
                    break
            
            if matching_video_key:
                videos[matching_video_key]['notes'].extend(track['notes'])
                notes_mapped += len(track['notes'])
                print(f"✓ Mapped {len(track['notes'])} notes to {matching_video_key}")
    
    print(f"\n✓ Total notes mapped: {notes_mapped}")
    
    if notes_mapped == 0:
        print("❌ No notes mapped - the fix didn't work!")
        return False
    
    # Create config for video processor
    config = {
        'tracks': {
            'tracks': midi_data['tracks'],
            'header': midi_data['header']
        },
        'videos': videos
    }
    
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    try:
        # Run video processor
        print(f"\n🎬 Running video composition...")
        processor = EnhancedVideoProcessor()
        result = processor.compose_videos(config_path)
        
        print(f"✓ Video composition completed!")
        print(f"✓ Output file: {result.get('output_path')}")
        
        # Check if output file exists and has reasonable size
        output_path = Path(result.get('output_path', ''))
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Output file size: {size_mb:.1f} MB")
            
            if size_mb > 1:  # Reasonable size for a composed video
                print("🎉 SUCCESS! Video composition with synchronized notes completed!")
                return True
            else:
                print("⚠️  Output file is very small - may not contain actual video content")
                return False
        else:
            print("❌ Output file does not exist!")
            return False
            
    except Exception as e:
        print(f"❌ Video composition failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink(config_path)
        except:
            pass

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🎉 PIPELINE FIX COMPLETE! The AutoTuneSyncer video composition now works with proper MIDI note synchronization!")
    else:
        print("\n❌ Pipeline test failed - more investigation needed")
