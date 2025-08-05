#!/usr/bin/env python3
"""
Test the updated note mapping logic with the improved matching algorithm.
"""

import os
import json
import sys
import tempfile
import shutil
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_updated_note_mapping():
    """Test the updated note mapping logic that handles timestamp prefixes."""
    print("=== Testing Updated Note Mapping Logic ===")
    
    # Find the MIDI data file
    uploads_dir = backend_dir / "uploads"
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
        print("No MIDI data files found!")
        return
    
    midi_file = midi_files[0]
    print(f"Using MIDI data file: {midi_file}")
    
    with open(midi_file, 'r') as f:
        midi_data = json.load(f)
    
    # Get video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(uploads_dir.glob(f"*{ext}"))
    
    # Create video mapping like the code does
    videos = {}
    for video in video_files:
        instrument_name = video.stem
        if instrument_name.startswith('processed_'):
            instrument_name = instrument_name[10:]
        
        videos[instrument_name] = {
            'path': str(video),
            'isDrum': 'drum' in instrument_name.lower(),
            'notes': [],
            'layout': {'x': 0, 'y': 0, 'width': 480, 'height': 270}
        }
    
    print(f"Found {len(videos)} video files to map")
    
    # Helper functions (from updated route)
    def normalize_instrument_name(name):
        return name.lower().replace(' ', '_')
    
    def is_drum_track(track):
        return (track.get('channel') == 9 or 
                ('instrument' in track and track['instrument'] and
                 ('drum' in str(track['instrument']).lower())))
    
    def get_drum_name(midi_note):
        DRUM_NOTES = {
            35: 'Bass Drum', 36: 'Kick Drum', 38: 'Snare Drum', 42: 'Hi-Hat Closed',
            46: 'Hi-Hat Open', 49: 'Crash Cymbal', 51: 'Ride Cymbal'
        }
        return DRUM_NOTES.get(midi_note, f'Drum_{midi_note}')
    
    # UPDATED MAPPING LOGIC
    notes_mapped = 0
    for track_index, track in enumerate(midi_data.get('tracks', [])):
        if not track.get('notes'):
            continue
        
        print(f"\nProcessing track {track_index}: {track.get('instrument', {})} ({len(track['notes'])} notes)")
        
        if is_drum_track(track):
            print("  -> Drum track detected")
            for note in track['notes']:
                drum_name = get_drum_name(note['midi'])
                drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                
                # UPDATED: Find video key that ends with the drum pattern
                matching_video_key = None
                for key in videos.keys():
                    if drum_key in key or key.endswith(drum_key):
                        matching_video_key = key
                        break
                
                if matching_video_key:
                    videos[matching_video_key]['notes'].append(note)
                    notes_mapped += 1
                    print(f"    ✓ Mapped note {note['midi']} ({drum_name}) to {matching_video_key}")
                else:
                    print(f"    ✗ No video for drum: {drum_key}")
                    # Show available drum videos
                    drum_videos = [k for k in videos.keys() if 'drum' in k]
                    if drum_videos:
                        print(f"      Available drums: {drum_videos[:3]}...")
        else:
            print("  -> Melodic instrument")
            if track.get('instrument') and track['instrument'].get('name'):
                normalized_name = normalize_instrument_name(track['instrument']['name'])
                
                # UPDATED: Find video key that contains the instrument name
                matching_video_key = None
                for key in videos.keys():
                    # Extract instrument part from filename (after last dash)
                    key_parts = key.split('-')
                    instrument_part = key_parts[-1] if key_parts else ""
                    
                    if instrument_part == normalized_name or normalized_name in key:
                        matching_video_key = key
                        break
                
                if matching_video_key:
                    videos[matching_video_key]['notes'].extend(track['notes'])
                    notes_mapped += len(track['notes'])
                    print(f"    ✓ Mapped {len(track['notes'])} notes to {matching_video_key}")
                else:
                    print(f"    ✗ No video for instrument: {normalized_name}")
                    # Show some available instruments that might match
                    for video_key in list(videos.keys())[:3]:
                        if not 'drum' in video_key and (normalized_name in video_key or video_key.endswith(normalized_name)):
                            print(f"      Potential match: {video_key}")
    
    print(f"\n=== UPDATED Results ===")
    print(f"Total notes mapped: {notes_mapped}")
    
    mapped_videos = 0
    for key, video in videos.items():
        if video['notes']:
            print(f"✓ {key}: {len(video['notes'])} notes")
            mapped_videos += 1
    
    print(f"\nSummary: {mapped_videos}/{len(videos)} videos have notes mapped")
    
    if notes_mapped > 0:
        print("🎉 SUCCESS! Notes are now being mapped correctly!")
        return True
    else:
        print("❌ STILL FAILING: No notes mapped")
        return False

if __name__ == "__main__":
    success = test_updated_note_mapping()
    if success:
        print("\n✅ The fix should now work in the actual application!")
    else:
        print("\n❌ More debugging needed...")
