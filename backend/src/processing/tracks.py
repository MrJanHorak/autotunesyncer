import logging
import os
import math
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import time
from ..ffmpeg.executor import execute_ffmpeg_command
from ..ffmpeg.combiner import combine_chunks

def build_track_command(chunk_clips: List[Dict], duration: float, segment_start: float, output_path: str) -> List[str]:
    """Build FFmpeg command for processing a chunk of track clips"""
    cmd = ['ffmpeg', '-y']
    
    # Add inputs for each clip
    for clip in chunk_clips:
        cmd.extend(['-i', clip['source_path']])
    
    # Build filter complex for pitch shifting and positioning
    filters = []
    
    for i, clip in enumerate(chunk_clips):
        # Calculate pitch shift
        target_freq = 440 * (2 ** ((clip['midi_note'] - 69) / 12.0))
        shift_semitones = 12 * math.log2(target_freq / 440)
        
        # Add pitch shift filter
        filters.append(f'[{i}:v]scale={clip["width"]}:{clip["height"]}[v{i}]')
        filters.append(f'[{i}:a]rubberband=pitch={shift_semitones:+.2f}[a{i}]')
    
    # Combine video streams with overlay
    if len(chunk_clips) > 1:
        video_overlay = f'[v0]'
        for i in range(1, len(chunk_clips)):
            video_overlay += f'[v{i}]overlay={chunk_clips[i]["x"]}:{chunk_clips[i]["y"]}'
            if i < len(chunk_clips) - 1:
                video_overlay += f'[tmp{i}];[tmp{i}]'
        filters.append(video_overlay + '[vout]')
        
        # Mix audio streams
        audio_inputs = ''.join(f'[a{i}]' for i in range(len(chunk_clips)))
        filters.append(f'{audio_inputs}amix=inputs={len(chunk_clips)}:duration=longest[aout]')
    else:
        filters.extend(['[v0][vout]', '[a0][aout]'])
    
    # Add filter complex to command
    cmd.extend(['-filter_complex', ';'.join(filters)])
    cmd.extend(['-map', '[vout]', '-map', '[aout]'])
    cmd.extend(['-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac'])
    cmd.append(output_path)
    
    return cmd

def process_track_parallel(track_clips: List[Dict], duration: float, segment_start: float, 
                         temp_dir: str, track_id: str, max_workers: int = None) -> str:
    """Process clips belonging to a single track with parallel chunk processing"""
    try:
        start_time = time.perf_counter()
        
        if max_workers is None:
            max_workers = min(4, psutil.cpu_count(logical=False))
        
        CHUNK_SIZE = 5
        chunk_outputs = []
        
        # Process chunks in parallel
        if len(track_clips) > CHUNK_SIZE and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {}
                
                for i in range(0, len(track_clips), CHUNK_SIZE):
                    chunk = track_clips[i:i + CHUNK_SIZE]
                    chunk_out = f"{temp_dir}{os.sep}chunk_{track_id}_{i}.mp4"
                    
                    future = executor.submit(
                        process_single_chunk,
                        chunk, duration, segment_start, chunk_out
                    )
                    future_to_chunk[future] = chunk_out
                
                # Collect results
                for future in as_completed(future_to_chunk):
                    chunk_out = future_to_chunk[future]
                    try:
                        future.result()  # Check for exceptions
                        if os.path.exists(chunk_out):
                            chunk_outputs.append(chunk_out)
                    except Exception as e:
                        logging.error(f"Chunk processing failed for {chunk_out}: {e}")
        else:
            # Sequential processing for small numbers of clips
            for i in range(0, len(track_clips), CHUNK_SIZE):
                chunk = track_clips[i:i + CHUNK_SIZE]
                chunk_out = f"{temp_dir}{os.sep}chunk_{track_id}_{i}.mp4"
                
                try:
                    process_single_chunk(chunk, duration, segment_start, chunk_out)
                    if os.path.exists(chunk_out):
                        chunk_outputs.append(chunk_out)
                except Exception as e:
                    logging.error(f"Sequential chunk processing failed: {e}")

        # Combine chunks if needed
        if len(chunk_outputs) > 1:
            final_output = f"{temp_dir}{os.sep}track_{track_id}.mp4"
            combine_chunks(chunk_outputs, final_output)
            
            # Cleanup intermediate chunks
            for chunk_file in chunk_outputs:
                try:
                    os.remove(chunk_file)
                except Exception as e:
                    logging.warning(f"Failed to cleanup chunk file {chunk_file}: {e}")
            
            processing_time = time.perf_counter() - start_time
            logging.info(f"Track {track_id} processed in {processing_time:.2f}s with {len(chunk_outputs)} chunks")
            return final_output
        elif chunk_outputs:
            processing_time = time.perf_counter() - start_time
            logging.info(f"Track {track_id} processed in {processing_time:.2f}s (single chunk)")
            return chunk_outputs[0]
        
        return None

    except Exception as e:
        logging.error(f"Parallel track processing failed: {e}")
        raise

def process_single_chunk(chunk_clips: List[Dict], duration: float, segment_start: float, output_path: str):
    """Process a single chunk of clips"""
    cmd = build_track_command(chunk_clips, duration, segment_start, output_path)
    execute_ffmpeg_command(cmd)

def process_track(track_clips: List[Dict], duration: float, segment_start: float, 
                 temp_dir: str, track_id: str) -> str:
    """Process clips belonging to a single track."""
    # Use parallel processing by default
    return process_track_parallel(track_clips, duration, segment_start, temp_dir, track_id)