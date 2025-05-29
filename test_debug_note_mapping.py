#!/usr/bin/env python3
"""
Debug test to check what video file names are being processed
and what the MIDI data structure looks like.
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

def analyze_midi_structure():
    """Analyze the MIDI data structure to understand note mapping."""
    print("=== MIDI Data Structure Analysis ===")
    
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
        return None
    
    midi_file = midi_files[0]
    print(f"Found MIDI data file: {midi_file}")
    
    with open(midi_file, 'r') as f:
        midi_data = json.load(f)
    
    print(f"MIDI Header: {midi_data.get('header', {})}")
    print(f"Number of tracks: {len(midi_data.get('tracks', []))}")
    
    for i, track in enumerate(midi_data.get('tracks', [])):
        print(f"\nTrack {i}:")
        print(f"  Channel: {track.get('channel', 'N/A')}")
        print(f"  Instrument: {track.get('instrument', {})}")
        print(f"  Notes: {len(track.get('notes', []))}")
        
        if track.get('notes'):
            # Show first few notes
            for j, note in enumerate(track['notes'][:3]):
                print(f"    Note {j}: MIDI={note.get('midi')}, time={note.get('time')}, duration={note.get('duration')}")
    
    return midi_data

def check_video_files():
    """Check what video files are available in uploads."""
    print("\n=== Video Files Analysis ===")
    
    uploads_dir = backend_dir / "uploads"
    video_files = []
    
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(uploads_dir.glob(f"*{ext}"))
    
    print(f"Found {len(video_files)} video files:")
    for video in video_files:
        # Extract instrument name like the code does
        instrument_name = video.stem
        if instrument_name.startswith('processed_'):
            instrument_name = instrument_name[10:]  # Remove 'processed_' prefix
        print(f"  {video.name} -> instrument: '{instrument_name}'")
    
    return video_files

def simulate_note_mapping():
    """Simulate the note mapping process to see what's happening."""
    print("\n=== Simulating Note Mapping Process ===")
    
    midi_data = analyze_midi_structure()
    video_files = check_video_files()
    
    if not midi_data or not video_files:
        print("Missing MIDI data or video files!")
        return
    
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
    
    print(f"Video mapping keys: {list(videos.keys())}")
    
    # Helper functions (copied from the route)
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
    
    # Process tracks
    notes_mapped = 0
    for track_index, track in enumerate(midi_data.get('tracks', [])):
        if not track.get('notes'):
            print(f"Track {track_index}: No notes")
            continue
        
        print(f"\nProcessing track {track_index}: {track.get('instrument', {})} ({len(track['notes'])} notes)")
        
        if is_drum_track(track):
            print("  -> Drum track detected")
            for note in track['notes']:
                drum_name = get_drum_name(note['midi'])
                drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                
                if drum_key in videos:
                    videos[drum_key]['notes'].append(note)
                    notes_mapped += 1
                    print(f"    Mapped note {note['midi']} ({drum_name}) to {drum_key}")
                else:
                    print(f"    No video for drum: {drum_key}")
        else:
            print("  -> Melodic instrument")
            if track.get('instrument') and track['instrument'].get('name'):
                normalized_name = normalize_instrument_name(track['instrument']['name'])
                
                if normalized_name in videos:
                    videos[normalized_name]['notes'].extend(track['notes'])
                    notes_mapped += len(track['notes'])
                    print(f"    Mapped {len(track['notes'])} notes to {normalized_name}")
                else:
                    print(f"    No video for instrument: {normalized_name}")
                    # Try to find close matches
                    for video_key in videos.keys():
                        if normalized_name in video_key or video_key in normalized_name:
                            print(f"      Possible match: {video_key}")
    
    print(f"\n=== Final Results ===")
    print(f"Total notes mapped: {notes_mapped}")
    for key, video in videos.items():
        print(f"{key}: {len(video['notes'])} notes")
    
    if notes_mapped == 0:
        print("\n⚠️  NO NOTES WERE MAPPED! This explains why the final video has no synchronization.")
        print("Possible issues:")
        print("1. Video file names don't match instrument names in MIDI")
        print("2. MIDI instrument names are in different format than expected")
        print("3. Drum mapping logic isn't working correctly")

if __name__ == "__main__":
    simulate_note_mapping()
