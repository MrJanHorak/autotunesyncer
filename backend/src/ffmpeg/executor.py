import logging
import subprocess
import shutil
from typing import List

def execute_ffmpeg_command(ffmpeg_cmd: List[str]) -> None:
    """Execute FFmpeg command and log outputs."""
    try:
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            raise Exception("FFmpeg executable not found in system PATH.")

        ffmpeg_cmd[0] = ffmpeg_path
        logging.debug(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")

        process = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            shell=False
        )

        logging.debug(f"FFmpeg stdout: {process.stdout}")
        logging.debug(f"FFmpeg stderr: {process.stderr}")

        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {process.stderr.strip()}")

    except Exception as e:
        logging.error(f"Error executing FFmpeg command: {e}")
        raise