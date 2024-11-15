import os
import logging
from typing import List
from .executor import execute_ffmpeg_command

def combine_chunks(chunk_files: List[str], output: str) -> None:
    """Combine multiple chunk files into a single output."""
    try:
        concat_file = os.path.splitext(output)[0] + '_concat.txt'
        with open(concat_file, 'w') as f:
            for file in chunk_files:
                f.write(f"file '{file}'\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output
        ]
        
        execute_ffmpeg_command(cmd)
        
        try:
            os.remove(concat_file)
        except:
            pass
            
    except Exception as e:
        logging.error(f"Error combining chunks: {e}")
        raise

def combine_tracks(track_files: List[str], output: str, duration: float) -> None:
    """Combine multiple track videos with overlay and audio mixing."""
    try:
        if not track_files:
            return

        cmd = ['ffmpeg', '-y']
        
        # Add inputs
        for file in track_files:
            cmd.extend(['-i', file])
        
        # Create overlay chain
        filters = []
        current = '[0:v]'
        audio_inputs = []
        
        for i in range(1, len(track_files)):
            next_out = '[v]' if i == len(track_files) - 1 else f'[v{i}]'
            filters.append(f"{current}[{i}:v]overlay=shortest=1{next_out}")
            current = next_out
            audio_inputs.append(f"[{i}:a]")
        
        # Add audio mixing
        if len(track_files) > 1:
            filters.append(f"[0:a]{(''.join(audio_inputs))}amix=inputs={len(track_files)}:dropout_transition=0[a]")
        
        # Add filter complex
        if filters:
            cmd.extend(['-filter_complex', ';'.join(filters)])
            cmd.extend(['-map', '[v]'])
            if len(track_files) > 1:
                cmd.extend(['-map', '[a]'])
            else:
                cmd.extend(['-map', '0:a'])
        
        # Add output options
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-c:a', 'aac',
            '-ar', '44100',
            '-b:a', '192k',
            '-t', str(duration),
            output
        ])
        
        execute_ffmpeg_command(cmd)

    except Exception as e:
        logging.error(f"Track combination failed: {e}")
        raise

def combine_segments(segment_files: List[str], output: str, duration: float) -> None:
    """Combine multiple segment files into final output."""
    try:
        concat_file = os.path.splitext(output)[0] + '_concat.txt'
        with open(concat_file, 'w') as f:
            for file in segment_files:
                f.write(f"file '{file}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            '-t', str(duration),
            output
        ]

        execute_ffmpeg_command(cmd)

        try:
            os.remove(concat_file)
        except:
            pass

    except Exception as e:
        logging.error(f"Segment combination failed: {e}")
        raise
