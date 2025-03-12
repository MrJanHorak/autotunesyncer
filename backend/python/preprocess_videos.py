import subprocess
import sys
import logging
from video_utils import run_ffmpeg_command, validate_video

logging.basicConfig(level=logging.INFO)

def preprocess_video(input_path, output_path, target_size=None):
    """Convert video to standardized format with optional resizing"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path
        ]
        
        if target_size:
            width, height = target_size.split('x')
            scale_filter = (
                f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad=w={width}:h={height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black'
            )
            cmd.extend(['-vf', scale_filter])
            
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-rc', 'vbr',
            '-rc-lookahead', '32',
            '-gpu', '0',
            '-tune', 'hq',
            '-profile:v', 'high',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ])
        
        logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
        run_ffmpeg_command(cmd)
        validate_video(output_path)
        
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise

def normalize_track_indices(grid_arrangement):
    """Normalize sparse track indices to sequential ones for optimal grid layout"""
    if not grid_arrangement:
        return {}
    
    # Separate numeric and non-numeric keys
    numeric_indices = []
    non_numeric_keys = []
    
    for key in grid_arrangement.keys():
        try:
            numeric_indices.append(int(key))
        except ValueError:
            non_numeric_keys.append(key)
    
    # Sort numeric indices
    numeric_indices.sort()
    
    # Create mapping from original to normalized indices for numeric keys only
    mapping = {str(old_idx): str(new_idx) for new_idx, old_idx in enumerate(numeric_indices)}
    
    # Map positions using the new indices, preserving non-numeric keys
    normalized = {}
    for old_idx, position in grid_arrangement.items():
        if old_idx in non_numeric_keys:
            # Keep non-numeric keys as-is (like 'drum_crash_cymbal')
            normalized[old_idx] = position
        else:
            # Map numeric keys to new indices
            new_idx = mapping.get(old_idx, old_idx)
            normalized[new_idx] = position
    
    return normalized
    
if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    target_size = sys.argv[3] if len(sys.argv) > 3 else None
    preprocess_video(input_path, output_path, target_size)