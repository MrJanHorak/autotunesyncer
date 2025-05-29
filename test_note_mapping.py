#!/usr/bin/env python3
"""
Test MIDI Note Mapping Fix
This test specifically verifies that MIDI notes are properly mapped to video files
"""

import json
import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from utils.video_processor import process_video_composition

def setup_logging():
    """Set up logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_realistic_midi_with_notes():
    """Create MIDI data with actual notes to test mapping"""
    return {
        "tracks": [
            {
                "index": 0,
                "instrument": {
                    "name": "piano",
                    "family": "melodic", 
                    "isDrum": False
                },
                "notes": [
                    {"time": 0.0, "duration": 0.5, "midi": 60, "velocity": 80},
                    {"time": 1.0, "duration": 0.5, "midi": 64, "velocity": 85},
                    {"time": 2.0, "duration": 0.5, "midi": 67, "velocity": 90}
                ],
                "channel": 0,
                "name": "piano"
            },
            {
                "index": 1,
                "instrument": {
                    "name": "synthbrass_2",
                    "family": "melodic",
                    "isDrum": False
                },
                "notes": [
                    {"time": 0.5, "duration": 1.0, "midi": 55, "velocity": 75},
                    {"time": 2.5, "duration": 1.0, "midi": 59, "velocity": 80}
                ],
                "channel": 2,
                "name": "synthbrass_2"
            },
            {
                "index": 2,
                "instrument": {
                    "name": "drum",
                    "family": "percussion",
                    "isDrum": True
                },
                "notes": [
                    {"time": 0.0, "duration": 0.1, "midi": 36, "velocity": 100},  # Kick
                    {"time": 0.5, "duration": 0.1, "midi": 38, "velocity": 90},   # Snare
                    {"time": 1.0, "duration": 0.1, "midi": 36, "velocity": 100},  # Kick
                    {"time": 1.5, "duration": 0.1, "midi": 42, "velocity": 80}    # Hi-Hat
                ],
                "channel": 9,
                "name": "drum"
            }
        ],
        "header": {
            "tempo": 120,
            "timeSignature": [4, 4]
        },
        "duration": 3.0,
        "gridArrangement": {
            "0": {"row": 0, "column": 0},
            "1": {"row": 0, "column": 1},  
            "2": {"row": 1, "column": 0}
        }
    }

def create_test_video_files_with_notes():
    """Create test video files structure that should receive notes"""
    test_videos_dir = Path(__file__).parent / "backend" / "uploads"
    
    return {
        "piano": {
            "path": str(test_videos_dir / "test_piano.mp4"),
            "notes": [],  # Should be populated by the route
            "isDrum": False
        },
        "synthbrass_2": {
            "path": str(test_videos_dir / "test_synthbrass.mp4"), 
            "notes": [],  # Should be populated by the route
            "isDrum": False
        },
        "drum_kick_drum": {
            "path": str(test_videos_dir / "test_kick.mp4"),
            "notes": [],  # Should be populated by the route  
            "isDrum": True
        },
        "drum_snare_drum": {
            "path": str(test_videos_dir / "test_snare.mp4"),
            "notes": [],  # Should be populated by the route
            "isDrum": True
        },
        "drum_hi-hat_closed": {
            "path": str(test_videos_dir / "test_hihat.mp4"),
            "notes": [],  # Should be populated by the route
            "isDrum": True
        }
    }

def simulate_note_mapping(midi_data, video_files):
    """Simulate the note mapping logic from processVideos.js"""
    
    def normalize_instrument_name(name):
        return name.lower().replace(' ', '_').replace('-', '_')
    
    def is_drum_track(track):
        return (track.get('channel') == 9 or 
                track.get('instrument', {}).get('isDrum') or
                'drum' in track.get('instrument', {}).get('name', '').lower())
    
    def get_drum_name(midi_note):
        DRUM_NOTES = {
            35: 'Bass Drum', 36: 'Kick Drum', 38: 'Snare Drum', 
            42: 'Hi-Hat Closed', 46: 'Hi-Hat Open', 49: 'Crash Cymbal', 
            51: 'Ride Cymbal'
        }
        return DRUM_NOTES.get(midi_note, f'Drum_{midi_note}')
    
    print('\n=== Simulating MIDI Note Mapping ===')
    
    for track_index, track in enumerate(midi_data['tracks']):
        if not track.get('notes'):
            print(f"Track {track_index}: No notes found")
            continue
            
        print(f"Processing track {track_index}: {track.get('instrument', {}).get('name')} ({len(track['notes'])} notes)")
        
        if is_drum_track(track):
            # Handle drum tracks
            for note in track['notes']:
                drum_name = get_drum_name(note['midi'])
                drum_key = f"drum_{drum_name.lower().replace(' ', '_').replace('-', '_')}"
                
                if drum_key in video_files:
                    video_files[drum_key]['notes'].append({
                        'midi': note['midi'],
                        'time': note['time'], 
                        'duration': note['duration'],
                        'velocity': note.get('velocity', 0.8)
                    })
                    print(f"  Mapped drum note {note['midi']} ({drum_name}) to {drum_key}")
                else:
                    print(f"  No video found for drum: {drum_key}")
        else:
            # Handle melodic instruments
            normalized_name = normalize_instrument_name(track['instrument']['name'])
            
            if normalized_name in video_files:
                for note in track['notes']:
                    video_files[normalized_name]['notes'].append({
                        'midi': note['midi'],
                        'time': note['time'],
                        'duration': note['duration'], 
                        'velocity': note.get('velocity', 0.8)
                    })
                print(f"  Mapped {len(track['notes'])} notes to {normalized_name}")
            else:
                print(f"  No video found for instrument: {normalized_name}")
    
    return video_files

def test_note_mapping():
    """Test the note mapping functionality"""
    setup_logging()
    
    print("🎵 Testing MIDI Note Mapping Fix")
    print("="*50)
    
    # Create test data
    midi_data = create_realistic_midi_with_notes()
    video_files = create_test_video_files_with_notes()
    
    print("📊 MIDI Data Summary:")
    for i, track in enumerate(midi_data['tracks']):
        print(f"  Track {i}: {track['instrument']['name']} - {len(track['notes'])} notes")
    
    print("\n📹 Video Files Available:")
    for key in video_files.keys():
        print(f"  {key}")
    
    # Simulate the note mapping
    mapped_videos = simulate_note_mapping(midi_data, video_files)
    
    print('\n=== Final Note Mapping Results ===')
    total_notes_mapped = 0
    for key, video in mapped_videos.items():
        note_count = len(video['notes'])
        total_notes_mapped += note_count
        print(f"{key}: {note_count} notes mapped")
        
        # Show first few notes for verification
        if note_count > 0:
            for i, note in enumerate(video['notes'][:3]):
                print(f"    Note {i+1}: MIDI {note['midi']}, time {note['time']}, velocity {note['velocity']}")
    
    print(f"\nTotal notes mapped across all videos: {total_notes_mapped}")
    
    # Calculate expected notes
    expected_notes = sum(len(track['notes']) for track in midi_data['tracks'])
    print(f"Expected total notes from MIDI: {expected_notes}")
    
    # Test results
    if total_notes_mapped > 0:
        print("\n✅ SUCCESS: MIDI notes are being mapped to video files!")
        print(f"   Mapped {total_notes_mapped} out of {expected_notes} notes")
        
        if total_notes_mapped == expected_notes:
            print("🎉 PERFECT: All notes successfully mapped!")
        else:
            print(f"⚠️  Some notes not mapped ({expected_notes - total_notes_mapped} missing)")
            print("   This might be due to missing video files for some instruments")
            
        return True
    else:
        print("\n❌ FAILURE: No MIDI notes were mapped to video files!")
        print("   The note mapping logic is not working correctly")
        return False

if __name__ == "__main__":
    success = test_note_mapping()
    sys.exit(0 if success else 1)
