import subprocess
import logging
from typing import Optional

def validate_audio_stream(video_path: str) -> bool:
    """Check if video file contains an audio stream"""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 'codec_type=audio' in result.stdout
    except Exception as e:
        logging.error(f"Error checking audio stream: {e}")
        return False

def get_audio_codec(video_path: str) -> Optional[str]:
    """Get the audio codec of the video file"""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0', 
               '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', 
               video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip() if result.stdout.strip() else None
    except Exception as e:
        logging.error(f"Error getting audio codec: {e}")
        return None
