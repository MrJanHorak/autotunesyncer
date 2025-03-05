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
    
if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    target_size = sys.argv[3] if len(sys.argv) > 3 else None
    preprocess_video(input_path, output_path, target_size)