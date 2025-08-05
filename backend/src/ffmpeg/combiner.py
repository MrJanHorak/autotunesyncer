import os
import logging
import time
import psutil
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import threading
from .executor import execute_ffmpeg_commands_parallel, get_ffmpeg_stats

# Thread-safe operation tracking
combiner_lock = threading.RLock()
combiner_stats = {
    'total_combinations': 0,
    'successful_combinations': 0,
    'failed_combinations': 0,
    'total_time': 0.0
}

class EnhancedCombiner:
    """Enhanced combiner with parallel processing and memory optimization"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, psutil.cpu_count(logical=False))
        self.temp_files = []  # Track temporary files for cleanup
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def combine_chunks_parallel(self, chunk_groups: List[List[str]], outputs: List[str]) -> List[bool]:
        """Combine multiple groups of chunks in parallel"""
        if len(chunk_groups) != len(outputs):
            raise ValueError("Number of chunk groups must match number of outputs")
        
        start_time = time.perf_counter()
        
        with combiner_lock:
            combiner_stats['total_combinations'] += len(chunk_groups)
        
        # Prepare commands for parallel execution
        commands = []
        concat_files = []
        
        for i, (chunks, output) in enumerate(zip(chunk_groups, outputs)):
            concat_file = self._create_concat_file(chunks, f"concat_{i}")
            concat_files.append(concat_file)
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output
            ]
            commands.append(cmd)
        
        # Execute in parallel
        operation_names = [f"combine_chunks_{i}" for i in range(len(commands))]
        results = execute_ffmpeg_commands_parallel(commands, operation_names)
        
        # Process results
        success_count = sum(1 for result in results if result is not None and result.returncode == 0)
        
        with combiner_lock:
            combiner_stats['successful_combinations'] += success_count
            combiner_stats['failed_combinations'] += len(chunk_groups) - success_count
            combiner_stats['total_time'] += time.perf_counter() - start_time
        
        # Cleanup concat files
        for concat_file in concat_files:
            try:
                os.remove(concat_file)
            except Exception as e:
                logging.warning(f"Failed to cleanup concat file {concat_file}: {e}")
        
        logging.info(f"Parallel chunk combination: {success_count}/{len(chunk_groups)} successful")
        return [result is not None and result.returncode == 0 for result in results]
    
    def _create_concat_file(self, chunk_files: List[str], prefix: str = "concat") -> str:
        """Create a temporary concat file for FFmpeg"""
        concat_fd, concat_file = tempfile.mkstemp(suffix='.txt', prefix=prefix)
        self.temp_files.append(concat_file)
        
        try:
            with os.fdopen(concat_fd, 'w') as f:
                for file in chunk_files:
                    if os.path.exists(file):
                        # Use absolute path for better compatibility
                        abs_path = os.path.abspath(file)
                        f.write(f"file '{abs_path}'\n")
                    else:
                        logging.warning(f"Chunk file not found: {file}")
            return concat_file
        except Exception as e:
            logging.error(f"Failed to create concat file: {e}")
            raise

# Global enhanced combiner instance
enhanced_combiner = EnhancedCombiner()

def combine_chunks(chunk_files: List[str], output: str) -> None:
    """Combine multiple chunk files into a single output (enhanced version)"""
    try:
        start_time = time.perf_counter()
        
        with combiner_lock:
            combiner_stats['total_combinations'] += 1
        
        # Filter out non-existent files
        valid_files = [f for f in chunk_files if os.path.exists(f)]
        if not valid_files:
            raise ValueError("No valid chunk files found")
        
        if len(valid_files) != len(chunk_files):
            logging.warning(f"Found {len(valid_files)}/{len(chunk_files)} valid chunk files")
        
        # Create concat file
        concat_file = os.path.splitext(output)[0] + '_concat.txt'
        enhanced_combiner.temp_files.append(concat_file)
        
        with open(concat_file, 'w') as f:
            for file in valid_files:
                abs_path = os.path.abspath(file)
                f.write(f"file '{abs_path}'\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output
        ]
        
        from .executor import execute_ffmpeg_command
        execute_ffmpeg_command(cmd)
        
        # Cleanup
        try:
            os.remove(concat_file)
            enhanced_combiner.temp_files.remove(concat_file)
        except Exception as e:
            logging.warning(f"Failed to cleanup concat file: {e}")
        
        processing_time = time.perf_counter() - start_time
        
        with combiner_lock:
            combiner_stats['successful_combinations'] += 1
            combiner_stats['total_time'] += processing_time
        
        logging.info(f"Combined {len(valid_files)} chunks in {processing_time:.2f}s")
            
    except Exception as e:
        with combiner_lock:
            combiner_stats['failed_combinations'] += 1
        logging.error(f"Error combining chunks: {e}")
        raise

def combine_tracks(track_files: List[str], output: str, duration: float) -> None:
    """Combine multiple track videos with overlay and audio mixing (enhanced version)"""
    try:
        start_time = time.perf_counter()
        
        if not track_files:
            raise ValueError("No track files provided")

        # Filter valid files
        valid_files = [f for f in track_files if os.path.exists(f)]
        if not valid_files:
            raise ValueError("No valid track files found")
        
        if len(valid_files) != len(track_files):
            logging.warning(f"Found {len(valid_files)}/{len(track_files)} valid track files")

        cmd = ['ffmpeg', '-y']
        
        # Add inputs
        for file in valid_files:
            cmd.extend(['-i', file])
        
        # Handle single track case
        if len(valid_files) == 1:
            cmd.extend(['-t', str(duration), '-c', 'copy', output])
        else:
            # Create overlay chain for multiple tracks
            filters = []
            current = '[0:v]'
            audio_inputs = []
            
            for i in range(1, len(valid_files)):
                overlay = f'{current}[{i}:v]overlay[v{i}]'
                filters.append(overlay)
                current = f'[v{i}]'
            
            # Audio mixing
            for i in range(len(valid_files)):
                audio_inputs.append(f'[{i}:a]')
            
            if len(audio_inputs) > 1:
                audio_mix = f'{"".join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=longest[aout]'
                filters.append(audio_mix)
                cmd.extend(['-filter_complex', ';'.join(filters)])
                cmd.extend(['-map', current, '-map', '[aout]'])
            else:
                cmd.extend(['-filter_complex', ';'.join(filters)])
                cmd.extend(['-map', current, '-map', '0:a'])
        
        # Add timing and quality settings
        cmd.extend([
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            output
        ])
        
        from .executor import execute_ffmpeg_command
        execute_ffmpeg_command(cmd)
        
        processing_time = time.perf_counter() - start_time
        logging.info(f"Combined {len(valid_files)} tracks in {processing_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Error combining tracks: {e}")
        raise

def combine_segments(segment_files: List[str], output: str, duration: float) -> None:
    """Combine multiple segment files into final output (enhanced version)"""
    try:
        start_time = time.perf_counter()
        
        if not segment_files:
            raise ValueError("No segment files provided")
        
        # Filter and sort valid files
        valid_files = [f for f in segment_files if os.path.exists(f)]
        if not valid_files:
            raise ValueError("No valid segment files found")
        
        # Sort by filename to ensure correct order
        valid_files.sort()
        
        if len(valid_files) != len(segment_files):
            logging.warning(f"Found {len(valid_files)}/{len(segment_files)} valid segment files")
        
        # Use combine_chunks for segment combination
        combine_chunks(valid_files, output)
        
        processing_time = time.perf_counter() - start_time
        logging.info(f"Combined {len(valid_files)} segments in {processing_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Error combining segments: {e}")
        raise

def get_combiner_stats() -> dict:
    """Get combiner performance statistics"""
    with combiner_lock:
        stats = combiner_stats.copy()
        if stats['total_combinations'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_combinations']
            stats['success_rate'] = stats['successful_combinations'] / stats['total_combinations']
        else:
            stats['average_time'] = 0.0
            stats['success_rate'] = 0.0
        return stats
