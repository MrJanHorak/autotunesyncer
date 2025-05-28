import logging
import os
import asyncio
import time
import tempfile
import shutil
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import psutil

from backend.src.ffmpeg.executor import execute_ffmpeg_command
from backend.src.ffmpeg.combiner import combine_tracks
from backend.src.processing.tracks import process_track

class SegmentProcessor:
    """High-performance segment processor with parallel processing"""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(8, psutil.cpu_count(logical=False))
        self.temp_dirs = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logging.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")

    async def process_segment_async(self, segment_clips: List[Dict], duration: float, 
                                  segment_start: float, segment_duration: float, 
                                  temp_dir: str, segment_idx: int) -> str:
        """Process a single segment asynchronously"""
        try:
            start_time = time.perf_counter()
            
            # Group clips by track for parallel processing
            tracks = {}
            for clip in segment_clips:
                track_id = clip['track_id']
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(clip)

            # Process tracks in parallel
            track_tasks = []
            for track_id, track_clips in tracks.items():
                task = asyncio.create_task(
                    self._process_track_async(
                        track_clips, segment_duration, segment_start, 
                        temp_dir, f"{segment_idx}_{track_id}"
                    )
                )
                track_tasks.append(task)

            # Wait for all track processing to complete
            track_outputs = await asyncio.gather(*track_tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_outputs = [
                output for output in track_outputs 
                if not isinstance(output, Exception) and output is not None
            ]

            if not valid_outputs:
                logging.error(f"No valid track outputs for segment {segment_idx}")
                return None

            # Combine tracks
            out_file = f"{temp_dir}{os.sep}{segment_idx:02d}.mp4"
            await self._combine_tracks_async(valid_outputs, out_file, segment_duration)
            
            # Cleanup intermediate files
            for file in valid_outputs:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except Exception as e:
                    logging.warning(f"Failed to remove temp file {file}: {e}")

            duration = time.perf_counter() - start_time
            logging.info(f"Segment {segment_idx} processed in {duration:.2f}s")
            
            return out_file

        except Exception as e:
            logging.error(f"Segment {segment_idx} processing failed: {e}")
            raise

    async def _process_track_async(self, track_clips: List[Dict], segment_duration: float,
                                 segment_start: float, temp_dir: str, track_id: str) -> str:
        """Process a single track asynchronously"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor, 
                process_track, 
                track_clips, segment_duration, segment_start, temp_dir, track_id
            )

    async def _combine_tracks_async(self, track_outputs: List[str], out_file: str, 
                                  duration: float) -> None:
        """Combine tracks asynchronously"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(
                executor, 
                combine_tracks, 
                track_outputs, out_file, duration
            )

    async def process_segments_batch(self, segments_data: List[Dict]) -> List[str]:
        """Process multiple segments in parallel batches"""
        batch_size = min(4, self.max_workers)  # Don't overload system
        results = []
        
        for i in range(0, len(segments_data), batch_size):
            batch = segments_data[i:i+batch_size]
            batch_tasks = []
            
            for segment_data in batch:
                task = asyncio.create_task(
                    self.process_segment_async(**segment_data)
                )
                batch_tasks.append(task)
            
            # Process batch and collect results
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [
                result for result in batch_results 
                if not isinstance(result, Exception) and result is not None
            ]
            
            results.extend(valid_results)
            
            # Log progress
            logging.info(f"Completed batch {i//batch_size + 1}/{(len(segments_data) + batch_size - 1)//batch_size}")
        
        return results

def process_segment(segment_clips: List[Dict], duration: float, segment_start: float, 
                    segment_duration: float, temp_dir: str, segment_idx: int) -> str:
    """Process a single segment of video clips with pitch shifting."""
    try:
        start_time = time.perf_counter()
        
        # Group clips by track
        tracks = {}
        for clip in segment_clips:
            track_id = clip['track_id']
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(clip)

        # Process tracks in parallel using ThreadPoolExecutor
        max_workers = min(4, len(tracks), psutil.cpu_count(logical=False))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit track processing tasks
            future_to_track = {
                executor.submit(
                    process_track, 
                    track_clips, segment_duration, segment_start, temp_dir, f"{segment_idx}_{track_id}"
                ): track_id
                for track_id, track_clips in tracks.items()
            }
            
            # Collect results as they complete
            track_outputs = []
            for future in as_completed(future_to_track):
                track_id = future_to_track[future]
                try:
                    result = future.result()
                    if result:
                        track_outputs.append(result)
                        logging.info(f"Track {track_id} processed successfully")
                except Exception as e:
                    logging.error(f"Track {track_id} processing failed: {e}")

        if not track_outputs:
            logging.error(f"No valid track outputs for segment {segment_idx}")
            return None

        # Combine tracks
        out_file = f"{temp_dir}{os.sep}{segment_idx:02d}.mp4"
        combine_tracks(track_outputs, out_file, segment_duration)
        
        # Cleanup intermediate files
        for file in track_outputs:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                logging.warning(f"Failed to remove temp file {file}: {e}")

        duration = time.perf_counter() - start_time
        logging.info(f"Segment {segment_idx} processed in {duration:.2f}s")
        
        return out_file

    except Exception as e:
        logging.error(f"Segment processing failed: {e}")
        raise

async def process_video_segments_async(midi_data, video_files, output_path):
    """Process video segments asynchronously with optimized performance"""
    try:
        start_time = time.perf_counter()
        
        # Create main temp directory
        main_temp_dir = Path(output_path).parent / f"temp_segments_{int(time.time())}"
        main_temp_dir.mkdir(exist_ok=True)
        
        # Analyze MIDI data to create segments
        segments_data = _analyze_midi_for_segments(midi_data, video_files, str(main_temp_dir))
        
        if not segments_data:
            logging.warning("No segments to process")
            return False
        
        # Process segments using high-performance processor
        with SegmentProcessor() as processor:
            segment_files = await processor.process_segments_batch(segments_data)
        
        if not segment_files:
            logging.error("No segments were processed successfully")
            return False
        
        # Concatenate final segments
        await _concatenate_segments_async(segment_files, output_path)
        
        # Cleanup
        shutil.rmtree(main_temp_dir)
        
        total_time = time.perf_counter() - start_time
        logging.info(f"Video segments processing completed in {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in async video segments processing: {e}")
        return False

def process_video_segments(midi_data, video_files, output_path):
    """Process video segments with enhanced performance"""
    try:
        # Run async version if possible, fallback to sync
        if hasattr(asyncio, 'run'):
            return asyncio.run(process_video_segments_async(midi_data, video_files, output_path))
        else:
            # Fallback for older Python versions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    process_video_segments_async(midi_data, video_files, output_path)
                )
            finally:
                loop.close()
                
    except Exception as e:
        logging.error(f"Failed to run async processing, falling back to sync: {e}")
        return _process_video_segments_sync(midi_data, video_files, output_path)

def _process_video_segments_sync(midi_data, video_files, output_path):
    """Fallback synchronous processing with parallel optimization"""
    try:
        start_time = time.perf_counter()
        
        # Create temp directory
        temp_dir = Path(output_path).parent / f"temp_segments_{int(time.time())}"
        temp_dir.mkdir(exist_ok=True)
        
        # Analyze and process segments
        segments_data = _analyze_midi_for_segments(midi_data, video_files, str(temp_dir))
        
        if not segments_data:
            return False
        
        # Process segments in parallel
        max_workers = min(4, len(segments_data), psutil.cpu_count(logical=False))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_segment, **segment_data): idx
                for idx, segment_data in enumerate(segments_data)
            }
            
            segment_files = [None] * len(segments_data)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    segment_files[idx] = result
                except Exception as e:
                    logging.error(f"Segment {idx} failed: {e}")
        
        # Filter out failed segments
        valid_segments = [f for f in segment_files if f is not None]
        
        if not valid_segments:
            return False
        
        # Concatenate segments
        _concatenate_segments_sync(valid_segments, output_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        total_time = time.perf_counter() - start_time
        logging.info(f"Sync video segments processing completed in {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        logging.error(f"Sync video segments processing failed: {e}")
        return False

def _analyze_midi_for_segments(midi_data, video_files, temp_dir):
    """Analyze MIDI data to create segment processing data"""
    # This is a placeholder - implement based on your MIDI structure
    segments_data = []
    
    # Example implementation - adjust based on your MIDI data structure
    segment_duration = 4.0  # 4 second segments
    total_duration = midi_data.get('duration', 60.0)  # Default 60 seconds
    
    for i in range(int(total_duration / segment_duration)):
        segment_start = i * segment_duration
        segments_data.append({
            'segment_clips': _extract_clips_for_segment(midi_data, video_files, segment_start, segment_duration),
            'duration': total_duration,
            'segment_start': segment_start,
            'segment_duration': segment_duration,
            'temp_dir': temp_dir,
            'segment_idx': i
        })
    
    return segments_data

def _extract_clips_for_segment(midi_data, video_files, start_time, duration):
    """Extract clips needed for a specific segment"""
    # Placeholder implementation - adjust based on your data structure
    clips = []
    
    for track_id, track_data in enumerate(midi_data.get('tracks', [])):
        for note in track_data.get('notes', []):
            note_time = note.get('time', 0)
            if start_time <= note_time < start_time + duration:
                clips.append({
                    'track_id': track_id,
                    'note': note,
                    'video_file': video_files.get(str(track_id))
                })
    
    return clips

async def _concatenate_segments_async(segment_files, output_path):
    """Concatenate segments asynchronously"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, _concatenate_segments_sync, segment_files, output_path)

def _concatenate_segments_sync(segment_files, output_path):
    """Concatenate segments synchronously"""
    try:
        # Create concat file
        concat_file = Path(output_path).parent / "concat_list.txt"
        
        with open(concat_file, 'w') as f:
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    f.write(f"file '{segment_file}'\n")
        
        # Use FFmpeg to concatenate
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            output_path
        ]
        
        result = execute_ffmpeg_command(concat_cmd)
        
        # Cleanup
        os.remove(concat_file)
        
        if result.returncode != 0:
            raise Exception(f"Concatenation failed: {result.stderr}")
        
        logging.info(f"Successfully concatenated {len(segment_files)} segments to {output_path}")
        
    except Exception as e:
        logging.error(f"Concatenation failed: {e}")
        raise