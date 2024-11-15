import logging
import os
import math
from typing import List, Dict
from ..ffmpeg.executor import execute_ffmpeg_command
from ..ffmpeg.combiner import combine_chunks

def process_track(track_clips: List[Dict], duration: float, segment_start: float, 
                 temp_dir: str, track_id: str) -> str:
    """Process clips belonging to a single track."""
    try:
        CHUNK_SIZE = 5
        chunk_outputs = []
        
        for i in range(0, len(track_clips), CHUNK_SIZE):
            chunk = track_clips[i:i + CHUNK_SIZE]
            chunk_out = f"{temp_dir}{os.sep}chunk_{track_id}_{i}.mp4"
            
            # Build FFmpeg command for chunk
            cmd = build_track_command(chunk, duration, segment_start, chunk_out)
            execute_ffmpeg_command(cmd)
            
            if os.path.exists(chunk_out):
                chunk_outputs.append(chunk_out)

        # Combine chunks if needed
        if len(chunk_outputs) > 1:
            final_output = f"{temp_dir}{os.sep}track_{track_id}.mp4"
            combine_chunks(chunk_outputs, final_output)
            return final_output
        elif chunk_outputs:
            return chunk_outputs[0]
        
        return None

    except Exception as e:
        logging.error(f"Track processing failed: {e}")
        raise