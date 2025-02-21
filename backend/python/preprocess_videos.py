import sys
import subprocess
import logging
logging.basicConfig(level=logging.INFO)

# def preprocess_video(input_path, output_path):
#     cmd = [
#         'ffmpeg', '-y',
#         '-i', input_path,
#         '-c:v', 'h264_nvenc',
#         '-preset', 'p4',
#         '-crf', '23',
#         '-pix_fmt', 'yuv420p',
#         '-movflags', '+faststart',
#         output_path
#     ]
#     return subprocess.run(cmd, check=True, capture_output=True)
def preprocess_video(input_path, output_path, target_size=None):
    """Convert video to standardized format with optional resizing"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path
        ]
        
        if target_size:
            # Fix scale/pad syntax by splitting width and height
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result
        
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise
    
if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    target_size = sys.argv[3] if len(sys.argv) > 3 else None
    preprocess_video(input_path, output_path, target_size)