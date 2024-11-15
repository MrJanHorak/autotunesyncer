import sys
import json
import logging
import os
from pathlib import Path

# Use absolute imports
from backend.src.processing.segments import process_video_segments
from backend.src.utils.logging import setup_logging
from backend.src.utils.audio import validate_audio_stream

def validate_input_files(midi_path: str, video_path: str) -> bool:
    """Validate that input files exist and have required streams"""
    if not all(Path(p).exists() for p in [midi_path, video_path]):
        logging.error("Input files not found")
        return False
    
    if not validate_audio_stream(video_path):
        logging.error(f"No audio stream found in video file: {video_path}")
        return False
    
    return True

def main():
    if len(sys.argv) != 4:
        print("Usage: python -m backend.src.video_processor midi_data.json video_files.json output_path")
        sys.exit(1)

    setup_logging()
    midi_json_path = sys.argv[1]
    video_files_json_path = sys.argv[2]
    output_path = sys.argv[3]

    try:
        # Validate input files
        if not validate_input_files(midi_json_path, video_files_json_path):
            sys.exit(1)

        with open(midi_json_path, 'r') as f:
            midi_data = json.load(f)
        with open(video_files_json_path, 'r') as f:
            video_files = json.load(f)
            
        # Process video segments with audio handling
        success = process_video_segments(midi_data, video_files, output_path)
        if not success:
            logging.error("Video processing failed")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in input files: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error in video processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()