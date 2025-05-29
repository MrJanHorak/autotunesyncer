#!/usr/bin/env python3
"""
Test script to validate MIDI data transformation
"""

import sys
import os
import json

# Add the utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'utils'))

from video_composer_wrapper import VideoComposerWrapper

def test_transformation():
    """Test the MIDI data transformation logic"""
    
    # Create sample MIDI data with start/end fields (problematic format)
    sample_midi_data = {
        "tracks": {
            "0": {
                "instrument": {"name": "piano"},
                "notes": [
                    {"start": 0.0, "end": 1.0, "midi": 60, "velocity": 80},
                    {"start": 1.0, "end": 2.0, "midi": 62, "velocity": 90},
                    {"start": 2.0, "end": 3.0, "midi": 64, "velocity": 85}
                ]
            },
            "1": {
                "instrument": {"name": "guitar"},
                "notes": [
                    {"start": 0.5, "end": 1.5, "midi": 67, "velocity": 75},
                    {"start": 1.5, "end": 2.5, "midi": 69, "velocity": 80}
                ]
            }
        },
        "duration": 3.0
    }
    
    print("Original MIDI data:")
    print(json.dumps(sample_midi_data, indent=2))
    
    # Create wrapper and transform
    wrapper = VideoComposerWrapper()
    transformed_data = wrapper._transform_midi_data(sample_midi_data)
    
    print("\nTransformed MIDI data:")
    print(json.dumps(transformed_data, indent=2))
    
    # Validate transformation
    assert isinstance(transformed_data['tracks'], list), "Tracks should be converted to list"
    
    for track in transformed_data['tracks']:
        assert 'id' in track, "Track should have ID"
        for note in track.get('notes', []):
            assert 'time' in note, f"Note missing 'time' field: {note}"
            assert 'duration' in note, f"Note missing 'duration' field: {note}"
            # Check that transformation worked correctly
            if 'start' in note:
                assert note['time'] == note['start'], "Time should match start value"
            if 'end' in note and 'start' in note:
                expected_duration = note['end'] - note['start']
                assert abs(note['duration'] - expected_duration) < 0.001, f"Duration calculation incorrect: {note}"
    
    print("\n✅ Transformation validation passed!")
    return transformed_data

if __name__ == "__main__":
    transformed = test_transformation()
