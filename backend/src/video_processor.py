import sys
import json
import logging
import os
import subprocess
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

def ensure_video_format(input_path: str, output_path: str = None) -> str:
    """
    Ensure video is in the correct format for processing.
    Returns path to the properly formatted video.
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.mp4'))
        
    try:
        # Check current format
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                    '-show_entries', 'stream=codec_name,width,height,r_frame_rate', 
                    '-of', 'json', input_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        stream_info = json.loads(probe_result.stdout)
        
        # If format is already correct, return original path
        if (stream_info.get('streams', [{}])[0].get('codec_name') == 'h264' and
            Path(input_path).suffix.lower() == '.mp4'):
            return input_path
            
        # Convert to standard format
        convert_cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-vf', 'scale=960:720',
            '-c:a', 'aac',
            '-ar', '48000',
            '-ac', '2',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        subprocess.run(convert_cmd, check=True)
        return output_path
        
    except Exception as e:
        logging.error(f"Error ensuring video format: {e}")
        return input_path

def process_video_segments(midi_data, video_files, output_path):
    """Process video segments with format standardization"""
    try:
        # Create temp directory for processed videos
        temp_dir = Path(output_path).parent / 'temp_processed'
        temp_dir.mkdir(exist_ok=True)
        
        # Process each video file
        processed_videos = {}
        for track_id, track_data in video_files.items():
            video_path = track_data['path']
            processed_path = temp_dir / f"{Path(video_path).stem}_processed.mp4"
            
            # Ensure video is in correct format
            standardized_path = ensure_video_format(video_path, str(processed_path))
            processed_videos[track_id] = {
                **track_data,
                'path': standardized_path
            }
        
        # Continue with existing processing logic using processed_videos
        # ...existing processing code...
        
        # Cleanup temp files
        for file in temp_dir.glob('*'):
            try:
                file.unlink()
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {file}: {e}")
        temp_dir.rmdir()
        
        return True
        
    except Exception as e:
        logging.error(f"Error in video processing: {e}")
        return False

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