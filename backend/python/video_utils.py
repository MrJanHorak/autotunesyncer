import subprocess
import logging
from processing_utils import encoder_queue, GPUManager

gpu_manager = GPUManager()

def run_ffmpeg_command(cmd):
    logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
    try:
        # Only use GPU context if available
        if gpu_manager.has_gpu:
            with gpu_manager.gpu_context():
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            # Fall back to CPU processing
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg command failed: {e.stderr}")
        raise

def encode_video(cmd):
    logging.info(f"Encoding video with command: {' '.join(cmd)}")
    result = encoder_queue.encode(cmd)
    if result.returncode != 0:
        logging.error(f"Encoding failed: {result.stderr}")
        raise Exception(f"Encoding failed: {result.stderr}")
    return result

def validate_video(output_path):
    """Validate video file integrity"""
    validate_cmd = [
        'ffmpeg',
        '-v', 'error',
        '-i', output_path,
        '-f', 'null',
        '-'
    ]
    result = encoder_queue.encode(validate_cmd)
    if result.returncode != 0:
        logging.error(f"Validation failed: {result.stderr}")
        raise Exception(f"Validation failed: {result.stderr}")
    return result