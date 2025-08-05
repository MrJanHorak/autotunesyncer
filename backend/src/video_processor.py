import sys
import json
import logging
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import time

# Use absolute imports
from backend.src.processing.segments import process_video_segments
from backend.src.utils.logging import setup_logging
from backend.src.utils.audio import validate_audio_stream

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def process_video_segments_enhanced(midi_data, video_files, output_path):
    """Enhanced video segments processing with parallel format standardization"""
    try:
        start_time = time.time()
        logging.info(f"Starting enhanced video segments processing for {len(video_files)} files")
        
        # Create temp directory for processed videos
        temp_dir = Path(output_path).parent / 'temp_processed'
        temp_dir.mkdir(exist_ok=True)
        
        # Process videos in parallel with format standardization
        processed_videos = {}
        max_workers = min(4, len(video_files), psutil.cpu_count(logical=False))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_track = {}
            
            for track_id, track_data in video_files.items():
                video_path = track_data['path']
                processed_path = temp_dir / f"{Path(video_path).stem}_processed.mp4"
                
                future = executor.submit(
                    ensure_video_format, 
                    video_path, 
                    str(processed_path)
                )
                future_to_track[future] = (track_id, track_data, str(processed_path))
            
            # Collect results
            for future in as_completed(future_to_track):
                track_id, track_data, processed_path = future_to_track[future]
                try:
                    standardized_path = future.result()
                    processed_videos[track_id] = {
                        **track_data,
                        'path': standardized_path
                    }
                    logging.info(f"Successfully processed {track_id}")
                except Exception as e:
                    logging.error(f"Failed to process {track_id}: {e}")
                    # Use original path as fallback
                    processed_videos[track_id] = track_data
        
        # Continue with existing processing logic using processed_videos
        success = process_video_segments(midi_data, processed_videos, output_path)
        
        # Cleanup temp files
        cleanup_temp_files(temp_dir)
        
        total_time = time.time() - start_time
        logging.info(f"Enhanced video segments processing completed in {total_time:.2f}s")
        
        return success
        
    except Exception as e:
        logging.error(f"Error in enhanced video processing: {e}")
        return False

def cleanup_temp_files(temp_dir):
    """Cleanup temporary files with error handling"""
    try:
        for file in temp_dir.glob('*'):
            try:
                file.unlink()
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {file}: {e}")
        temp_dir.rmdir()
    except Exception as e:
        logging.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

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