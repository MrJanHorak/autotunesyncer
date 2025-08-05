#!/usr/bin/env python3
"""
Final test of the complete video processing pipeline
with the MIDI note mapping fix applied.
"""

import sys
import json
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from utils.video_processor import EnhancedVideoProcessor

def test_complete_pipeline():
    """Test the complete pipeline with mapped notes."""
    print('=== Testing Complete Pipeline with Mapped Notes ===')
    
    # Load the test config with mapped notes
    config_path = Path(__file__).parent / "test_mapping_result.json"
    
    if not config_path.exists():
        print(f'❌ Config file not found: {config_path}')
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f'✓ Videos: {len(config["videos"])}')
    total_notes = sum(len(video["notes"]) for video in config["videos"].values())
    print(f'✓ Total notes mapped: {total_notes}')
    
    # Show note distribution
    print('\n=== Note Distribution ===')
    for video_key, video_data in config["videos"].items():
        note_count = len(video_data["notes"])
        print(f'  {video_key}: {note_count} notes')
      # Test video processor using the same method as the API
    print('\n=== Running Video Composition ===')
    
    try:
        import tempfile
        import subprocess
        
        # Create temporary files like the API does
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Write MIDI data
            midi_path = temp_dir / "test-midi.json"
            with open(midi_path, 'w') as f:
                json.dump(config["tracks"], f)
            
            # Write video data
            video_path = temp_dir / "test-videos.json"
            with open(video_path, 'w') as f:
                json.dump(config["videos"], f)
            
            # Output path
            output_path = temp_dir / "test-output.mp4"
            
            # Call Python processor like the API does
            python_script = backend_dir / "utils" / "video_processor.py"
            cmd = [
                "python", str(python_script),
                str(midi_path),
                str(video_path), 
                str(output_path),
                "--performance-mode",
                "--parallel-tracks", "4",
                "--memory-limit", "4"
            ]
            
            print(f'Running command: {" ".join(cmd)}')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f'✅ Video composition successful!')
                print(f'✓ Output: {output_path}')
                
                # Check if output file exists
                if output_path.exists():
                    size = output_path.stat().st_size
                    print(f'✓ Output file size: {size:,} bytes')
                    
                    # Copy output to main directory for inspection
                    final_output = Path(__file__).parent / "test_final_output.mp4"
                    import shutil
                    shutil.copy2(output_path, final_output)
                    print(f'✓ Output copied to: {final_output}')
                    return True
                else:
                    print(f'❌ Output file not found: {output_path}')
                    return False
            else:
                print(f'❌ Video composition failed!')
                print(f'Error output: {result.stderr}')
                print(f'Standard output: {result.stdout}')
                return False
            
    except Exception as e:
        print(f'❌ Video composition failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    print(f'\n=== Final Result ===')
    print(f'{"✅ COMPLETE SUCCESS!" if success else "❌ PIPELINE FAILED"}')
    sys.exit(0 if success else 1)
