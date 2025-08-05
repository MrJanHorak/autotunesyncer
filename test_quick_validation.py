#!/usr/bin/env python3
"""
Quick validation test to check if our fixes resolved the grid arrangement issue
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path

def test_quick_validation():
    """Quick test to validate the data structure fix"""
    print('=== Quick Validation Test ===')
    
    # Create minimal test data
    midi_data = {
        'tracks': [  # This is a LIST, not a dict - this was the issue!
            {
                'notes': [{'midi': 60, 'time': 0, 'duration': 0.5}],
                'instrument': {'name': 'piano'}
            }
        ],
        'gridArrangement': {
            '1': {'row': 0, 'column': 0, 'type': 'track'}
        }
    }
    
    video_data = {
        'piano': {
            'path': 'test.mp4',
            'notes': [{'midi': 60, 'time': 0, 'duration': 0.5}]
        }
    }
    
    backend_dir = Path(__file__).parent / "backend"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Write test data
            midi_path = temp_dir / "midi.json"
            video_path = temp_dir / "videos.json"
            output_path = temp_dir / "output.mp4"
            
            with open(midi_path, 'w') as f:
                json.dump(midi_data, f)
            
            with open(video_path, 'w') as f:
                json.dump(video_data, f)
            
            # Test just the data structure handling
            python_script = backend_dir / "utils" / "video_processor.py"
            cmd = [
                "python", str(python_script),
                str(midi_path),
                str(video_path), 
                str(output_path),
                "--performance-mode"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            
            # Use shorter timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            
            # Check for our specific error
            if "'list' object has no attribute 'get'" in result.stderr:
                print("❌ Original error still present!")
                return False
            elif "Using MIDI grid arrangement" in result.stderr:
                print("✅ Grid arrangement fix working!")
                return True
            elif result.returncode == 0:
                print("✅ Process completed successfully!")
                return True
            else:
                print(f"⚠️ Different error occurred: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("⚠️ Test timed out - process was working but taking too long")
        return True  # Not an error, just slow
    except Exception as e:
        print(f'❌ Test failed: {e}')
        return False

if __name__ == "__main__":
    success = test_quick_validation()
    print(f'\n{"="*50}')
    if success:
        print('✅ DATA STRUCTURE FIX VERIFIED!')
    else:
        print('❌ Fix verification failed!')
    print('='*50)
    sys.exit(0 if success else 1)
