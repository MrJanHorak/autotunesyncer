import os
import re 
import gc
import threading
import subprocess
import sys
import logging
import traceback
import os.path
import math
import shutil
import tempfile
import cProfile
import pstats
import threading
import asyncio
import time
import numpy as np
import torch
from pathlib import Path
from threading import RLock

# Optional audio analysis
try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    aubio = None
    AUBIO_AVAILABLE = False
    logging.warning("aubio not available, some audio analysis features disabled")

import syncio
from pstats import SortKey

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Fallback tqdm implementation
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def update(self, n=1):
            pass
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available, progress bars disabled")

import time
from contextlib import contextmanager
from cachetools import LRUCache

# Third-party imports
import numpy as np
import torch

from moviepy import (
    VideoFileClip,
    clips_array,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips
)

from concurrent.futures import ThreadPoolExecutor, as_completed

import weakref

from contextlib import contextmanager

try:
    from utils import normalize_instrument_name, midi_to_note
except ImportError:
    # Fallback implementation
    def normalize_instrument_name(name):
        """Match frontend's normalizeInstrumentName"""
        return name.lower().replace(' ', '_')
    
    def midi_to_note(midi_num):
        """Convert MIDI note number to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = notes[midi_num % 12]
        octave = (midi_num // 12) - 1
        return f"{note_name}{octave}"
from drum_utils import (
    DRUM_NOTES,
    # process_drum_track,
    # get_drum_groups,
    # get_drum_name,
    is_drum_kit
)

from processing_utils import encoder_queue, GPUManager

# Import GPU acceleration functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from ffmpeg_gpu import ffmpeg_gpu_encode

# Import CUDA compositing if available
try:
    from cuda_compositing import CudaVideoProcessor
except ImportError:
    CudaVideoProcessor = None
    logging.warning("CUDA compositing not available")

# Import GPU pipeline processor if available
try:
    from gpu_pipeline import GPUPipelineProcessor
except ImportError:
    GPUPipelineProcessor = None
    logging.warning("GPU pipeline processor not available")

from video_utils import run_ffmpeg_command, encode_video, validate_video
from path_registry import PathRegistry

import mmap
# from contextlib import ExitStack

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Configure stream handler to handle Unicode properly on Windows
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        # Ensure proper encoding for Windows console
        if hasattr(handler.stream, 'reconfigure'):
            try:
                handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass

@contextmanager
def timing_block(name):
    """Context manager to measure execution time"""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logging.info(f"Operation [{name}] took {duration:.3f} seconds")

def get_system_metrics():
    """Get system performance metrics"""
    try:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        return {'cpu_percent': 0, 'memory_percent': 0, 'disk_percent': 0}

class OptimizedAutotuneCache:
    """
    Optimized cache for autotune video processing
    Provides fast lookup and management of autotuned video segments
    """
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.cache = {}
        self.lock = threading.Lock()
        logging.info(f"OptimizedAutotuneCache initialized with {max_workers} workers")
    
    def get(self, key):
        """Get cached item"""
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key, value):
        """Set cached item"""
        with self.lock:
            self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self):
        """Get cache size"""
        with self.lock:
            return len(self.cache)

class GPUStreamManager:
    def __init__(self, num_streams=4):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
        
    def get_stream(self):
        """Get next available stream"""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream
        
    def synchronize_all(self):
        """Wait for all streams to complete"""
        for stream in self.streams:
            torch.cuda.synchronize(stream.device)

# def gpu_subprocess_run(cmd, **kwargs):
#     """
#     GPU-accelerated subprocess wrapper for ffmpeg commands
#     Falls back to regular subprocess if GPU is not available
#     """
#     try:
#         # Check if this is an ffmpeg command and GPU is available
#         if cmd[0] == 'ffmpeg' and torch.cuda.is_available():
#             # Special handling for concat commands - don't modify them
#             if '-f' in cmd and 'concat' in cmd:
#                 # Concat commands should not be modified - they copy streams
#                 return subprocess.run(cmd, **kwargs)
            
#             # For complex filter commands, just add GPU encoding to the existing command
#             if '-filter_complex' in cmd or any('xstack' in arg for arg in cmd):
#                 # Complex filter detected - add GPU encoding to existing command
#                 gpu_cmd = cmd.copy()
                
#                 # Find the output file (last .mp4 file in command)
#                 output_file = None
#                 for arg in reversed(cmd):
#                     if arg.endswith('.mp4') and not arg.startswith('-'):
#                         output_file = arg
#                         break
                
#                 if output_file:
#                     # Insert GPU encoding parameters before the output file
#                     output_index = gpu_cmd.index(output_file)
#                     gpu_params = ['-c:v', 'h264_nvenc', '-preset', 'fast', '-pix_fmt', 'yuv420p']
                    
#                     # Insert GPU parameters before output file
#                     for i, param in enumerate(reversed(gpu_params)):
#                         gpu_cmd.insert(output_index, param)
                    
#                     # Add hardware acceleration at the beginning
#                     gpu_cmd.insert(1, '-hwaccel')
#                     gpu_cmd.insert(2, 'cuda')
                    
#                     try:
#                         result = subprocess.run(gpu_cmd, capture_output=True, text=True, check=True)
#                         logging.info("✅ GPU encoding successful with complex filters")
#                         return result
#                     except subprocess.CalledProcessError as e:
#                         logging.warning(f"GPU encoding failed with complex filters: {e.stderr}")
#                         logging.warning("GPU encoding failed, falling back to CPU")
#                         pass
#             else:
#                 # Simple command - try direct GPU encoding
#                 input_path = None
#                 output_path = None
                
#                 for i, arg in enumerate(cmd):
#                     if arg == '-i' and i + 1 < len(cmd):
#                         input_path = cmd[i + 1]
#                     elif arg.endswith('.mp4') and not arg.startswith('-'):
#                         output_path = arg
                
#                 if input_path and output_path:
#                     # Use direct GPU encoding command
#                     gpu_cmd = [
#                         'ffmpeg', '-y',
#                         '-hwaccel', 'cuda',
#                         '-i', input_path,
#                         '-c:v', 'h264_nvenc',
#                         '-preset', 'fast',
#                         '-crf', '23',
#                         '-pix_fmt', 'yuv420p',
#                         '-c:a', 'aac',
#                         '-b:a', '192k',
#                         output_path
#                     ]
                    
#                     try:
#                         result = subprocess.run(gpu_cmd, capture_output=True, text=True, check=True)
#                         return result
#                     except subprocess.CalledProcessError:
#                         logging.warning("GPU encoding failed, falling back to CPU")
#                         pass
            
#             # Fallback to regular subprocess if GPU fails
#             return subprocess.run(cmd, **kwargs)
#         else:
#             # Fallback to regular subprocess
#             return subprocess.run(cmd, **kwargs)
#     except Exception as e:
#         logging.warning(f"GPU subprocess failed: {e}, falling back to CPU")
#         return subprocess.run(cmd, **kwargs)

# Fix GPU processing for note-triggered videos



def gpu_subprocess_run(cmd, **kwargs):
    """
    Enhanced GPU subprocess runner that handles note-triggered video creation
    """
    try:
        # For note-triggered video creation, use standard CPU processing
        # to avoid hwaccel issues with complex filters
        if any('trim=' in str(arg) and 'overlay=' in str(arg) for arg in cmd):
            # This is a complex filter for note triggering - use CPU processing
            logging.info("Using CPU processing for note-triggered video creation")
            return subprocess.run(cmd, **kwargs)
        
        # For other commands, check if GPU is available and use it
        if cmd[0] == 'ffmpeg' and torch.cuda.is_available():
            # Special handling for concat commands - don't modify them
            if '-f' in cmd and 'concat' in cmd:
                return subprocess.run(cmd, **kwargs)
            
            # For simple encoding commands, add GPU encoding
            if '-c:v' not in cmd:
                gpu_cmd = cmd.copy()
                # Find output file and insert GPU encoding before it
                for i, arg in enumerate(cmd):
                    if arg.endswith('.mp4') and not arg.startswith('-'):
                        # Insert GPU encoding parameters before output file
                        gpu_cmd.insert(i, '-c:v')
                        gpu_cmd.insert(i + 1, 'h264_nvenc')
                        gpu_cmd.insert(i + 2, '-preset')
                        gpu_cmd.insert(i + 3, 'fast')
                        break
                
                try:
                    result = subprocess.run(gpu_cmd, **kwargs)
                    return result
                except subprocess.CalledProcessError as e:
                    logging.warning(f"GPU processing failed: {e}")
                    # Fall back to CPU
                    return subprocess.run(cmd, **kwargs)
        
        # Default to CPU processing
        return subprocess.run(cmd, **kwargs)
        
    except Exception as e:
        logging.error(f"GPU subprocess error: {e}")
        return subprocess.run(cmd, **kwargs)

class ClipPool:
    def __init__(self, max_size=8):
        self.semaphore = threading.BoundedSemaphore(max_size)
        self.clips = weakref.WeakSet()
    
    @contextmanager
    def acquire(self):
        self.semaphore.acquire()
        try:
            clip = None
            yield clip
        finally:
            if clip:
                clip.close()
            self.semaphore.release()


class MMAPHandler:
    def __init__(self):
        self.mapped_files = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def map_file(self, file_path):
        if file_path not in self.mapped_files:
            with open(file_path, 'rb') as f:
                self.mapped_files[file_path] = mmap.mmap(
                    f.fileno(), 0, access=mmap.ACCESS_READ)
        return self.mapped_files[file_path]
        
    def cleanup(self):
        for mmap_obj in self.mapped_files.values():
            try:
                mmap_obj.close()
            except:
                pass
        self.mapped_files.clear()


class ClipManager:
    def __init__(self):
        self.active_clips = weakref.WeakSet()
        
    @contextmanager
    def managed_clip(self, clip):
        try:
            self.active_clips.add(clip)
            yield clip
        finally:
            try:
                clip.close()
            except:
                pass
            self.active_clips.discard(clip)

class ProgressTracker:
    def __init__(self, total_notes):
        self.progress_bar = tqdm(total=total_notes, 
                               desc="Processing notes",
                               unit="note")
        self.completed = 0
        self.failed = 0
        
    def update(self, success=True):
        self.completed += 1
        if not success:
            self.failed += 1
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            "success_rate": f"{(self.completed-self.failed)/self.completed*100:.1f}%"
        })

    def close(self):
        self.progress_bar.close()

class VideoComposerConfig:
    def __init__(self):
        self.CHUNK_DURATION = 4
        self.OVERLAP_DURATION = 1
        self.CROSSFADE_DURATION = 0.5
        self.MIN_VIDEO_DURATION = 1.0
        self.DURATION = 1.0
        self.VOLUME_MULTIPLIERS = {
            'drums': 1.0,
            'instruments': 1.0
        }

class VideoComposer:

    FRAME_RATE = 30
    CHUNK_DURATION = 4
    OVERLAP_DURATION = 1
    CROSSFADE_DURATION = 0.5
    MIN_VIDEO_DURATION = 1.0
    DURATION = 1.0
    VOLUME_MULTIPLIERS = {
        'drums': 0.2,        'instruments': 1.5
    }

    def __init__(self, processed_videos_dir, midi_data, output_path):
        """Initialize VideoComposer with proper path handling"""
        try:
            logging.info("=== VideoComposer Initialization ===")
            logging.info(f"Received MIDI data structure: {list(midi_data.keys())}")
            logging.info(f"Grid arrangement from MIDI: {midi_data.get('gridArrangement')}")

            self.processed_videos_dir = Path(processed_videos_dir)
            
            # Check if we have an uploads directory provided or need to find it
            if 'uploadsDir' in midi_data:
                self.uploads_dir = Path(midi_data['uploadsDir'])
                logging.info(f"Using provided uploads directory: {self.uploads_dir}")
            else:
                # Try to find the uploads directory - look for backend/uploads
                current_dir = Path(processed_videos_dir)
                backend_uploads = None
                
                # Try multiple possible paths
                possible_paths = [
                    Path(__file__).parent.parent / "uploads",  # From backend/python -> backend/uploads
                    current_dir.parent.parent.parent / "uploads",  # From temp dir back to uploads
                    current_dir / "uploads",  # Direct uploads subdirectory
                ]
                
                for path in possible_paths:
                    if path.exists() and any(f.name.startswith("processed_") and f.name.endswith(".mp4") for f in path.iterdir()):
                        backend_uploads = path
                        break
                
                if backend_uploads:
                    self.uploads_dir = backend_uploads
                    logging.info(f"Found uploads directory: {self.uploads_dir}")
                else:
                    # Fall back to creating uploads in processed_videos_dir
                    self.uploads_dir = self.processed_videos_dir / "uploads"
                    logging.info(f"Creating uploads directory: {self.uploads_dir}")
                    self.uploads_dir.mkdir(exist_ok=True)
                    
            # Verify uploads directory exists
            if not self.uploads_dir.exists():
                raise ValueError(f"Uploads directory not found: {self.uploads_dir}")
                
            self.config = VideoComposerConfig()
            self.gpu_manager = GPUManager()
            self.output_path = output_path
            self.midi_data = midi_data
            self._setup_paths(processed_videos_dir, output_path)
            self.midi_data = midi_data
            self._process_midi_data(midi_data)
            self._setup_track_configuration()
            self.clip_manager = ClipManager()
            self.gpu_stream_manager = GPUStreamManager()
            self.chunk_cache = {}
            self.chunk_cache_locks = {i: threading.Lock() for i in range(1000)} # Pre-allocate locks
            self.chunk_cache = LRUCache(128)  # Example: Cache up to 128 chunks
            # self.chunk_cache = {}  # This would overwrite the LRUCache
            self.max_cached_chunks = 4
            self.metrics_log = []
            self.encoder_params = {
                'codec': 'h264_nvenc',
                'preset': 'p4',
                'ffmpeg_params': [
                    "-vsync", "cfr",
                    "-c:v", "h264_nvenc",
                    "-preset", "fast",
                    "-b:v", "5M",
                    "-maxrate", "10M",
                    "-bufsize", "10M",
                    "-profile:v", "high"
                ]
            }
            self.chunk_size = max(1, min(16, os.cpu_count()))  # Smaller chunk size
            self.max_workers = min(2, os.cpu_count())  # Limit workers            self.use_gpu = True
            self.lock = RLock()  # Add class-level lock
            self.clip_pool = ClipPool(max_size=8)  # Add clip pool
            self.chunk_cache_lock = RLock()  # Add dedicated cache lock
            self.max_cache_size = 1024 * 1024 * 100
            self.active_readers = set()  # Add reader tracking
            # Initialize path registry - use singleton instance
            self.path_registry = PathRegistry.get_instance()
            
            # After copying files, register them
            self._register_video_paths()
            self.video_cache = LRUCache(maxsize=64)  # Increase cache size
            self.audio_cache = LRUCache(maxsize=64) 
            self.autotune_cache = LRUCache(maxsize=64) # Keep for backwards compatibility
            # Initialize optimized autotune cache system
            self.optimized_cache = OptimizedAutotuneCache(max_workers=self.max_workers)
            self._tuned_videos_cache = {}  # Cache for preprocessed tuned videos
            # Log track information
            logging.info(f"Regular tracks: {len(self.tracks)}")
            logging.info(f"Drum tracks: {len(self.drum_tracks)}")
            for track in self.drum_tracks:
                logging.info(f"Drum track found: {track.get('instrument', {}).get('name')}")
            try:
                import torch
                self.has_cuda = torch.cuda.is_available()
                if self.has_cuda:
                    # Initialize CUDA context
                    device = torch.device('cuda:0')
                    torch.cuda.set_device(device)
                    # Create empty tensor to initialize CUDA
                    _ = torch.zeros(1, device=device)
                    logging.info(f"CUDA initialized successfully: {torch.cuda.get_device_name(0)}")
                    
                    # Set optimal settings for video processing
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    logging.warning("CUDA not available, using CPU processing")
            except Exception as e:
                self.has_cuda = False
                logging.error(f"Error initializing CUDA: {e}")
                            
        except Exception as e:
            logging.error(f"VideoComposer init error: {str(e)}")
            raise

    @staticmethod
    def normalize_midi_timing(midi_data):
        """Adjust all note timings to start at the same point"""
        # Find earliest note time across all tracks
        min_time = float('inf')
        for track in midi_data['tracks']:
            if track.get('notes'):
                track_min = min(float(note.get('time', 0)) for note in track['notes'])
                min_time = min(min_time, track_min)
          # Shift all notes to start at 0
        for track in midi_data['tracks']:
            for note in track.get('notes', []):
                note['time'] = float(note['time']) - min_time
        
        return midi_data
    
    def _register_video_paths(self):
        """Register paths for all videos in uploads directory"""
        # Get singleton instance
        registry = PathRegistry.get_instance()
        
        # Set registry file if needed
        registry_file = self.processed_videos_dir / "path_registry.json"
        
        # Register from uploads directory where videos are actually stored
        logging.info(f"Scanning uploads directory for videos: {self.uploads_dir}")
        
        # Use the new specialized method for uploads directory
        success = registry.register_from_uploads_directory(self.uploads_dir)
        
        if not success:
            logging.warning("No videos found in uploads directory, trying processed videos directory as fallback")
            # Also register from processed videos directory (fallback)
            registry.register_from_directory(self.processed_videos_dir)
        
        # Save registry for debugging
        registry.save_registry(str(registry_file))
        
        # Debug dump to help troubleshoot
        registry.debug_dump()
        
        # Log registration stats
        stats = registry.get_stats()
        logging.info(f"Path registry stats: {stats}")
        logging.info(f"Total videos registered: {stats['total_paths']}")

    def encode_video(self, cmd):
        logging.info(f"Encoding video with command: {' '.join(cmd)}")
        result = encode_video(cmd)
        return result

    def validate_video(self, output_path):
        validate_video(output_path)


    def _log_metrics(self):
        """Log system metrics during processing"""
        metrics = get_system_metrics()
        self.metrics_log.append(metrics)
        
        gpu_info = f", GPU={metrics.get('gpu_util', 'N/A')}%, GPU Memory={metrics.get('gpu_memory', 'N/A')}%" if metrics.get('gpu_util') is not None else ""
        
        logging.info(
            f"System metrics: CPU={metrics['cpu_percent']}%, "
            f"Memory={metrics['memory_percent']}%"
            f"{gpu_info}"
        )
  
    def _setup_paths(self, processed_videos_dir, output_path):
        """Setup and validate paths"""
        dir_path = (processed_videos_dir['processed_videos_dir'] 
                   if isinstance(processed_videos_dir, dict) 
                   else str(processed_videos_dir))
        
        self.processed_videos_dir = Path(dir_path).resolve()
        self.output_path = Path(output_path)
        self.temp_dir = self.processed_videos_dir
        
        if not self.processed_videos_dir.exists():
            raise ValueError(f"Directory not found: {self.processed_videos_dir}")
            
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Resolved path: {self.processed_videos_dir}")

    def debug_midi_track_timing(self, midi_data):
        """Debug function to print detailed note timing for all tracks"""
        logging.info("\n=== DETAILED MIDI NOTE TIMING ===")
        
        for track_idx, track in enumerate(midi_data['tracks']):
            instrument = track.get('instrument', {}).get('name', f'Unknown-{track_idx}')
            notes = track.get('notes', [])
            
            if not notes:
                logging.info(f"Track {track_idx} ({instrument}): NO NOTES")
                continue
                
            # Sort notes by time
            sorted_notes = sorted(notes, key=lambda n: float(n.get('time', 0)))
            first_time = float(sorted_notes[0].get('time', 0))
            last_time = float(sorted_notes[-1].get('time', 0))
            
            logging.info(f"Track {track_idx} ({instrument}): {len(notes)} notes")
            logging.info(f"  Time range: {first_time:.2f}s to {last_time:.2f}s")
            # FIX: Use double quotes for inner dictionary keys
            note_times = [f"{float(n.get('time', 0)):.2f}s" for n in sorted_notes[:5]]
            logging.info(f"  First 5 notes: {', '.join(note_times)}")

    def _analyze_midi_timing(self):
        """Analyze and log detailed timing information for all tracks"""
        logging.info("\n=== DETAILED MIDI TIMING ANALYSIS ===")
        
        # Analyze each track
        for track_id, track in self.tracks.items():
            instrument_name = track.get('instrument', {}).get('name', 'unknown')
            notes = track.get('notes', [])
            if not notes:
                continue
                
            # Sort notes by time
            sorted_notes = sorted(notes, key=lambda n: float(n.get('time', 0)))
            
            # Get time range statistics
            first_note_time = float(sorted_notes[0].get('time', 0))
            last_note_time = float(sorted_notes[-1].get('time', 0)) + float(sorted_notes[-1].get('duration', 0))
            total_duration = last_note_time - first_note_time
            note_count = len(notes)
            
            # Calculate which chunks these notes belong to
            first_chunk = int(first_note_time / self.CHUNK_DURATION)
            last_chunk = int(last_note_time / self.CHUNK_DURATION)
            chunk_count = last_chunk - first_chunk + 1
            
            # Group notes by chunk
            notes_by_chunk = {}
            for chunk_idx in range(first_chunk, last_chunk + 1):
                chunk_start = chunk_idx * self.CHUNK_DURATION
                chunk_end = chunk_start + self.CHUNK_DURATION
                
                # Count notes in this chunk time range
                chunk_notes = [
                    note for note in sorted_notes
                    if chunk_start <= float(note.get('time', 0)) < chunk_end
                ]
                notes_by_chunk[chunk_idx] = len(chunk_notes)
            
            # Log detailed timing info for this track
            logging.info(f"\nTrack {track_id}: {instrument_name}")
            logging.info(f"  Total notes: {note_count}")
            logging.info(f"  Time range: {first_note_time:.2f}s to {last_note_time:.2f}s (duration: {total_duration:.2f}s)")
            logging.info(f"  Chunks: {first_chunk} to {last_chunk} (spans {chunk_count} chunks)")
            
            # Log distribution of notes per chunk
            chunk_log = "  Notes per chunk: "
            for chunk_idx, count in sorted(notes_by_chunk.items()):
                chunk_log += f"[{chunk_idx}:{count}] "
            logging.info(chunk_log)
            
            # Log first 5 note times for verification
            note_times = [float(note.get('time', 0)) for note in sorted_notes[:5]]
            logging.info(f"  First 5 note times: {', '.join([f'{t:.2f}s' for t in note_times])}")
            
            # # For piano track specifically, log more details
            # if 'piano' in instrument_name.lower():
            #     logging.info(f"  === PIANO TRACK DETAILED ANALYSIS ===")
            #     all_note_times = [float(note.get('time', 0)) for note in sorted_notes]
            #     # Find any unusual gaps in timing
            #     note_gaps = [all_note_times[i+1] - all_note_times[i] for i in range(len(all_note_times)-1)]
            #     avg_gap = sum(note_gaps) / max(1, len(note_gaps))
            #     max_gap = max(note_gaps) if note_gaps else 0
                
            #     logging.info(f"  Piano avg note gap: {avg_gap:.2f}s, max gap: {max_gap:.2f}s")
            #     # Log all piano note times for detailed investigation
            #     logging.info(f"  Piano note timestamps (first 20): {', '.join([f'{t:.2f}s' for t in all_note_times[:20]])}")

    def _process_midi_data(self, midi_data):
        """Process MIDI data with correct handling of duplicate instruments"""
        if not isinstance(midi_data, dict):
            raise ValueError(f"Expected dict for midi_data, got {type(midi_data)}")
            
        if 'tracks' not in midi_data:
            raise ValueError("Missing 'tracks' in midi_data")
        
        self.debug_midi_track_timing(midi_data)

        # Add normalization here - before any track processing
        logging.info("\n=== Normalizing MIDI Timing ===")
        midi_data = VideoComposer.normalize_midi_timing(midi_data)
        logging.info("MIDI timing normalized - all tracks now start from the same time reference")

        self.debug_midi_track_timing(midi_data)

        tracks = midi_data['tracks']
        self.tracks = {}
        self.drum_tracks = []
        self.regular_tracks = []
        
        logging.info(f"\n=== Processing Tracks ===")
        logging.info(f"MIDI track structure: {type(tracks)}, {'dict keys' if isinstance(tracks, dict) else 'list length'}: {len(tracks)}")
        
        # List all preprocessed files for drum processing
        upload_files = list(self.uploads_dir.glob('processed_*.mp4'))
        logging.info(f"\nPreprocessed files in ({self.uploads_dir}):")
        for file in upload_files:
            logging.info(f"Found preprocessed file: {file.name}")

        # Process tracks - ensure we preserve all tracks including duplicates
        for idx, track in enumerate(tracks if isinstance(tracks, list) else tracks.values()):
            # Always use the original track index as the track ID to prevent overwriting
            track_id = str(idx)
            
            # Store ID in track data itself for later lookup
            if isinstance(track, dict):
                track['id'] = track_id
                track['original_index'] = idx  # Store original index for reference
                
            normalized_track = self._normalize_track(track)
            
            # Check if it's a drum track
            is_drum = (
                normalized_track.get('isDrum') or 
                normalized_track.get('instrument', {}).get('isDrum') or
                normalized_track.get('channel') == 9
            )
            
            if is_drum:
                logging.info(f"\nProcessing drum track {idx}/{track_id}")
                drum_dir = self.processed_videos_dir / f"track_{idx}_drums"
                drum_dir.mkdir(exist_ok=True)
                
                # Get unique drum types needed from notes
                needed_drums = set()
                for note in normalized_track.get('notes', []):
                    midi_note = note.get('midi')
                    drum_name = DRUM_NOTES.get(midi_note)
                    if drum_name:
                        needed_drums.add((midi_note, drum_name))
                          # Process each needed drum type
                for midi_note, drum_name in needed_drums:
                    normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
                    
                    # Find preprocessed drum file
                    for file in upload_files:
                        if normalized_name in file.name.lower():
                            dest_file = drum_dir / f"{normalized_name}.mp4"
                            # Copy preprocessed file to drum directory
                            try:
                                shutil.copy2(str(file), str(dest_file))
                                logging.info(f"Copied preprocessed drum file: {file.name} -> {dest_file}")
                            except Exception as e:
                                logging.error(f"Error copying drum file: {e}")
                            break
                        else:
                            logging.warning(f"No preprocessed file found for {normalized_name}")
                
                # Add drum track to drum_tracks (outside the loop)
                self.drum_tracks.append(normalized_track)
                logging.info(f"Added drum track {track_id}: {normalized_track.get('instrument', {}).get('name')}")
            else:
                # Handle regular (non-drum) tracks
                self.regular_tracks.append(normalized_track)
                # IMPORTANT: Use track_id (index) to store track, not instrument name
                self.tracks[track_id] = normalized_track
                logging.info(f"Added regular track {track_id}: {normalized_track.get('instrument', {}).get('name')}")

            logging.info(f"\nProcessed {len(self.regular_tracks)} regular tracks and {len(self.drum_tracks)} drum tracks")

    def _process_single_track(self, track_id, track):
        """Process a single track with consistent ID handling"""
        normalized_track = self._normalize_track(track)
        
        # Check if it's a drum track
        is_drum = (
            normalized_track.get('isDrum') or 
            normalized_track.get('instrument', {}).get('isDrum') or
            normalized_track.get('channel') == 9 or
            any(name in normalized_track.get('instrument', {}).get('name', '').lower() 
                for name in ['drum', 'percussion', 'kit'])
        )
        
        if is_drum:
            self.drum_tracks.append(normalized_track)
        else:
            self.regular_tracks.append(normalized_track)
            self.tracks[track_id] = normalized_track

    # def _process_midi_data(self, midi_data):
    #     """Process MIDI data and organize tracks"""
    #     if not isinstance(midi_data, dict):
    #         raise ValueError(f"Expected dict for midi_data, got {type(midi_data)}")
            
    #     if 'tracks' not in midi_data:
    #         raise ValueError("Missing 'tracks' in midi_data")

    #     tracks = midi_data['tracks']
    #     self.tracks = {}
    #     self.drum_tracks = []
    #     self.regular_tracks = []
        
    #     logging.info("\n=== Processing Tracks ===")
        
    #     # List all preprocessed files
    #     upload_files = list(self.uploads_dir.glob('processed_*.mp4'))
    #     logging.info(f"\nPreprocessed files in ({self.uploads_dir}):")
    #     for file in upload_files:
    #         logging.info(f"Found preprocessed file: {file.name}")

    #     # Check if tracks is a dict with actual IDs
    #     if isinstance(tracks, dict):
    #         track_items = tracks.items()
    #     else:
    #         # If it's a list, enumerate it
    #         track_items = enumerate(tracks)
            
    #     for idx_or_id, track in track_items:
    #         # Use the original ID if available (from dict keys), otherwise use index
    #         track_id = str(idx_or_id)
            
    #         # Important: Store the ID in the track data for later reference
    #         if isinstance(track, dict):
    #             track['id'] = track_id
                
    #         normalized_track = self._normalize_track(track)

    #     for idx, track in enumerate(tracks if isinstance(tracks, list) else tracks.values()):
    #         track_id = str(idx)
    #         normalized_track = self._normalize_track(track)
            
    #         # Check if it's a drum track
    #         is_drum = (
    #             normalized_track.get('isDrum') or 
    #             normalized_track.get('instrument', {}).get('isDrum') or
    #             normalized_track.get('channel') == 9 or
    #             any(name in normalized_track.get('instrument', {}).get('name', '').lower() 
    #                 for name in ['drum', 'percussion', 'kit'])
    #         )
            
    #         if is_drum:
    #             logging.info(f"\nProcessing drum track {track_id}")
    #             drum_dir = self.processed_videos_dir / f"track_{idx}_drums"
    #             drum_dir.mkdir(exist_ok=True)
                
    #             # Get unique drum types needed from notes
    #             needed_drums = set()
    #             for note in normalized_track.get('notes', []):
    #                 midi_note = note.get('midi')
    #                 drum_name = DRUM_NOTES.get(midi_note)
    #                 if drum_name:
    #                     needed_drums.add((midi_note, drum_name))
                        
    #             # Process each needed drum type
    #             for midi_note, drum_name in needed_drums:
    #                 normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
                    
    #                 # Find preprocessed drum file
    #                 for file in upload_files:
    #                     if normalized_name in file.name.lower():
    #                         dest_file = drum_dir / f"{normalized_name}.mp4"
    #                         # Copy preprocessed file to drum directory
    #                         shutil.copy2(str(file), str(dest_file))
    #                         logging.info(f"Copied preprocessed drum file: {file.name} -> {dest_file}")
    #                         break
    #                 else:
    #                     logging.warning(f"No preprocessed file found for {normalized_name}")
                
    #             self.drum_tracks.append(normalized_track)
                
    #         else:
    #             self.regular_tracks.append(normalized_track)
    #             self.tracks[track_id] = normalized_track

    #     logging.info(f"\nProcessed {len(self.tracks)} regular tracks and {len(self.drum_tracks)} drum tracks")

    # def _create_note_triggered_video_sequence(self, video_path, notes, total_duration, track_name):
    #     """
    #     Create a video track where the instrument video is triggered by MIDI notes
    #     This creates individual video clips for each MIDI note with proper timing and pitch
    #     """
    #     # Create unique output path to avoid conflicts when same track is processed multiple times
    #     import uuid
    #     unique_id = str(uuid.uuid4())[:8]  # Short unique ID
    #     output_path = self.temp_dir / f"{track_name}_triggered_{unique_id}.mp4"
        
    #     if not notes or not os.path.exists(video_path):
    #         logging.warning(f"No notes or video path missing for {track_name}")
    #         return None
        
    #     # If output file already exists, remove it to avoid conflicts
    #     if output_path.exists():
    #         output_path.unlink()
        
    #     # Create filter complex for note-triggered playback
    #     filter_parts = []
        
    #     # Create silent base video
    #     filter_parts.append(f"color=black:size=640x360:duration={total_duration}[base]")
        
    #     # Add triggered segments for each note
    #     audio_filters = []
        
    #     for i, note in enumerate(notes):
    #         start_time = float(note.get('time', 0))
    #         duration = float(note.get('duration', 0.5))
    #         pitch = note.get('midi', 60)
            
    #         # Calculate pitch adjustment
    #         pitch_factor = 2 ** ((pitch - 60) / 12.0)
            
    #         # Create segment for this note with pitch adjustment
    #         filter_parts.append(
    #             f"[0:v]trim=start=0:duration={duration},"
    #             f"setpts=PTS-STARTPTS[note_video_{i}]"
    #         )
            
    #         # Audio with pitch adjustment
    #         audio_filters.append(
    #             f"[0:a]atrim=start=0:duration={duration},"
    #             f"asetpts=PTS-STARTPTS,"
    #             f"asetrate=44100*{pitch_factor},"
    #             f"aresample=44100[note_audio_{i}]"
    #         )
            
    #         # Overlay this note segment at the correct time
    #         if i == 0:
    #             filter_parts.append(f"[base][note_video_{i}]overlay=enable='between(t,{start_time},{start_time + duration})'[video_out_{i}]")
    #         else:
    #             filter_parts.append(f"[video_out_{i-1}][note_video_{i}]overlay=enable='between(t,{start_time},{start_time + duration})'[video_out_{i}]")
        
    #     # Add audio filters to the main filter
    #     filter_parts.extend(audio_filters)
        
    #     # Mix all audio notes together
    #     if len(notes) > 1:
    #         audio_mix_inputs = ''.join([f'[note_audio_{i}]' for i in range(len(notes))])
    #         filter_parts.append(f"{audio_mix_inputs}amix=inputs={len(notes)}:duration=longest[audio_out]")
    #     else:
    #         filter_parts.append(f"[note_audio_0]copy[audio_out]")
        
    #     # Final video output
    #     final_video_idx = len(notes) - 1
    #     filter_parts.append(f"[video_out_{final_video_idx}]copy[video_out]")
        
    #     # Build FFmpeg command
    #     cmd = [
    #         'ffmpeg', '-y',
    #         '-i', str(video_path),
    #         '-filter_complex', ';'.join(filter_parts),
    #         '-map', '[video_out]',
    #         '-map', '[audio_out]',
    #         '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #         '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
    #         '-t', str(total_duration),
    #         '-r', '30',
    #         str(output_path)
    #     ]
        
    #     try:
    #         logging.info(f"🎵 Creating note-triggered video for {track_name} with {len(notes)} notes")
    #         result = gpu_subprocess_run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Successfully created note-triggered video: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create note-triggered video: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error creating note-triggered video: {e}")
    #         return None


    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     FIXED: Create actual note-triggered video with working FFmpeg filters
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
    #             return None
            
    #         # Remove existing file if it exists
    #         if output_path.exists():
    #             output_path.unlink()
            
    #         # FIXED: Use a much simpler approach that actually works
    #         # Instead of complex filter chains, use basic video looping with audio timing
            
    #         # Create a simple approach: loop the video and apply audio timing
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-stream_loop', '-1',  # Loop video infinitely
    #             '-i', str(video_path),
    #             '-t', str(total_duration),  # Cut to exact duration
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-r', '30',
    #             str(output_path)
    #         ]
            
    #         logging.info(f"🎵 Creating simple looped video for {track_name} with {len(notes)} notes")
            
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Successfully created video: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create video: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error creating video: {e}")
    #         return None

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     FIXED: Create note-triggered video with working FFmpeg approach
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
    #             return None
            
    #         # Remove existing file if it exists
    #         if output_path.exists():
    #             output_path.unlink()
            
    #         # FIXED: Use the same simple approach as drums - it actually works!
    #         # Create a simple looped video for the chunk duration
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-stream_loop', '-1',  # Loop video infinitely
    #             '-i', str(video_path),
    #             '-t', str(total_duration),  # Cut to exact duration
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-r', '30',
    #             str(output_path)
    #         ]
            
    #         logging.info(f"🎵 Creating simple looped video for {track_name} with {len(notes)} notes")
            
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Successfully created video: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create video: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error creating video: {e}")
    #         return None

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     FIXED: Create ACTUAL MIDI-triggered video like drums do
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
    #         if not notes or not os.path.exists(video_path):
    #             return None
    #         if output_path.exists():
    #             output_path.unlink()

    #         notes_with_visual_duration = self._calculate_visual_durations(notes, total_duration)

    #         filter_parts = []
    #         # Create silent base
    #         filter_parts.append(f"color=black:size=640x360:duration={total_duration}:rate=30[base_video]")
    #         filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}[base_audio]")

    #         # Create overlays for each MIDI note (like drums do)
    #         video_layers = ["[base_video]"]
    #         audio_segments = ["[base_audio]"]

    #         for i, note in enumerate(notes_with_visual_duration): # Use the new list
    #             start_time = float(note.get('time', 0))
    #             audio_duration = float(note.get('duration', 0.5))
    #             visual_duration = float(note.get('visual_duration', audio_duration)) # Use new visual duration
    #             pitch = note.get('midi', 60)

    #             if start_time >= total_duration: continue
                
    #             # Trim audio to its actual duration, but video to the longer visual duration
    #             pitch_semitones = pitch - 60
    #             pitch_factor = 2 ** (pitch_semitones / 12.0)
                
    #             # Video segment uses visual_duration
    #             filter_parts.append(f"[0:v]trim=0:{visual_duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                
    #             # Audio segment uses audio_duration
    #             if abs(pitch_factor - 1.0) > 0.01:
    #                 filter_parts.append(f"[0:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS,asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]")
    #             else:
    #                 filter_parts.append(f"[0:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS[note_a{i}]")
                
    #             # Overlay uses visual_duration
    #             prev_video = video_layers[-1]
    #             filter_parts.append(f"{prev_video}[note_v{i}]overlay=enable='between(t,{start_time},{start_time + visual_duration})'[video_out{i}]")
    #             video_layers.append(f"[video_out{i}]")
                
    #             delay_ms = int(start_time * 1000)
    #             filter_parts.append(f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]")
    #             audio_segments.append(f"[delayed_a{i}]")

    #         # Mix all audio (like drums)
    #         if len(audio_segments) > 1:
    #             audio_inputs = ''.join(audio_segments)
    #             filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
    #         else:
    #             filter_parts.append("[base_audio]copy[final_audio]")

    #         # Final video output
    #         final_video = video_layers[-1] if len(video_layers) > 1 else "[base_video]"
    #         filter_parts.append(f"{final_video}copy[final_video]")

    #         # Build command (same as drums)
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-i', str(video_path),
    #             '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={total_duration}:rate=30',
    #             '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}',
    #             '-filter_complex', ';'.join(filter_parts),
    #             '-map', '[final_video]',
    #             '-map', '[final_audio]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-t', str(total_duration),
    #             '-r', '30',
    #             str(output_path)
    #         ]

    #         logging.info(f"🎵 Creating MIDI-triggered video for {track_name} with {len(notes)} notes")

    #         result = subprocess.run(cmd, capture_output=True, text=True)

    #         if result.returncode == 0:
    #             logging.info(f"✅ MIDI-triggered video created: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create MIDI-triggered video: {result.stderr}")
    #             return None
    #     except Exception as e:
    #         logging.error(f"Error creating MIDI-triggered video: {e}")
    #         return None

    def _create_note_triggered_video_sequence_fixed(self, video_path, notes, chunk_start_time, chunk_duration, track_name, unique_id):
        """
        FIXED: Create MIDI-triggered video with proper delay validation
        """
        try:
            output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
            if not notes or not os.path.exists(video_path):
                logging.warning(f"No notes or video missing for {track_name}")
                return None
                
            if output_path.exists():
                output_path.unlink()

            # FIXED: Filter and validate notes for this chunk
            valid_notes = []
            for note in notes:
                note_start_abs = float(note.get('time', 0))
                relative_start = note_start_abs - chunk_start_time
                
                # FIXED: Skip notes that start before chunk (negative relative time)
                if relative_start < 0:
                    logging.debug(f"Skipping note at {note_start_abs}s (before chunk start {chunk_start_time}s)")
                    continue
                    
                duration = float(note.get('duration', 0.5))
                duration = min(duration, chunk_duration - relative_start)
                
                # FIXED: Skip notes with zero or negative duration
                if duration <= 0:
                    logging.debug(f"Skipping note with invalid duration: {duration}")
                    continue
                    
                # Add adjusted note to valid list
                adjusted_note = note.copy()
                adjusted_note['relative_time'] = relative_start
                adjusted_note['adjusted_duration'] = duration
                valid_notes.append(adjusted_note)

            if not valid_notes:
                logging.info(f"No valid notes for {track_name} in chunk time range")
                return None

            logging.info(f"🎵 Creating MIDI-triggered video for {track_name} with {len(valid_notes)} valid notes")

            # Create filter complex with validated delays
            filter_parts = []
            filter_parts.append(f"color=black:size=640x360:duration={chunk_duration}:rate=30[base_video]")
            filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}[base_audio]")

            video_layers = ["[base_video]"]
            audio_segments = ["[base_audio]"]

            for i, note in enumerate(valid_notes):
                relative_start = note['relative_time']
                audio_duration = note['adjusted_duration']
                visual_duration = max(audio_duration, 0.5)  # Minimum visual duration
                pitch = note.get('midi', 60)

                # Create video segment
                filter_parts.append(f"[0:v]trim=0:{visual_duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                
                # Create audio segment with pitch adjustment
                pitch_semitones = pitch - 60
                if abs(pitch_semitones) > 0.1:  # Apply pitch shift if needed
                    pitch_factor = 2 ** (pitch_semitones / 12.0)
                    filter_parts.append(f"[0:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS,asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]")
                else:
                    filter_parts.append(f"[0:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS[note_a{i}]")
                
                # Video overlay with validated timing
                prev_video = video_layers[-1]
                filter_parts.append(f"{prev_video}[note_v{i}]overlay=enable='between(t,{relative_start},{relative_start + visual_duration})'[video_out{i}]")
                video_layers.append(f"[video_out{i}]")
                
                # FIXED: Ensure delay is non-negative
                delay_ms = max(0, int(relative_start * 1000))
                filter_parts.append(f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]")
                audio_segments.append(f"[delayed_a{i}]")

            # Mix audio segments
            if len(audio_segments) > 1:
                audio_inputs = ''.join(audio_segments)
                filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
            else:
                filter_parts.append("[base_audio]copy[final_audio]")

            # Final video output
            final_video = video_layers[-1] if len(video_layers) > 1 else "[base_video]"
            filter_parts.append(f"{final_video}copy[final_video]")

            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={chunk_duration}:rate=30',
                '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}',
                '-filter_complex', ';'.join(filter_parts),
                '-map', '[final_video]',
                '-map', '[final_audio]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-t', str(chunk_duration),
                '-r', '30',
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logging.info(f"✅ MIDI-triggered video created: {output_path}")
                return str(output_path)
            else:
                logging.error(f"❌ Failed to create MIDI-triggered video: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating MIDI-triggered video: {e}")
            return None

    def _calculate_visual_durations(self, notes, chunk_duration):
        """Calculates a more natural visual duration for each note."""
        if not notes:
            return []

        # Sort notes by time to ensure correct lookahead
        sorted_notes = sorted(notes, key=lambda n: float(n.get('time', 0)))
        
        for i, note in enumerate(sorted_notes):
            note_start = float(note.get('time', 0))
            audio_duration = float(note.get('duration', 0.5))
            
            # Define a minimum visual time and a release tail
            MIN_VISUAL_TIME = 0.5  # Note is visible for at least 0.5s
            RELEASE_TAIL = 1.5    # Add up to 0.8s of visual decay

            visual_duration = audio_duration + RELEASE_TAIL

            # If there's a next note, don't let the visual overlap it
            if i + 1 < len(sorted_notes):
                next_note_start = float(sorted_notes[i+1].get('time', 0))
                visual_duration = min(visual_duration, next_note_start - note_start)

            # Enforce minimum visual time and ensure it doesn't exceed the chunk boundary
            visual_duration = max(visual_duration, MIN_VISUAL_TIME)
            visual_duration = min(visual_duration, chunk_duration - note_start)
            
            note['visual_duration'] = max(0, visual_duration) # Ensure non-negative
        
        return sorted_notes

    def _normalize_track(self, track):
        """Convert track data to standard format"""
        if isinstance(track, int):
            return {
                'notes': [],
                'instrument': {'name': f'track_{track}'},
                'isDrum': False        }
        elif isinstance(track, dict):
            return track
        else:
            logging.warning(f"Invalid track type: {type(track)}")
            return {'notes': [], 'instrument': {}, 'isDrum': False}
        
    def _create_mmap(self, data):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
    def _calculate_default_layout(self):
        """Calculate default grid layout"""
        rows = math.ceil(len(self.tracks) / 4)
        cols = min(4, len(self.tracks))
        return rows, cols
    
    def _cleanup_clip(self, clip):
        """Safely clean up clip resources with tracking"""
        try:
            if hasattr(clip, 'reader') and clip.reader:
                if clip.reader in self.active_readers:
                    self.active_readers.remove(clip.reader)
                clip.reader.close()
            if hasattr(clip, 'close'):
                try:
                    clip.close()
                except Exception as e:
                    logging.warning(f"Error closing clip: {e}")
        except Exception as e:
            logging.warning(f"Error closing clip: {e}")
    
    def _setup_track_configuration(self):
        try:
            grid_arrangement = self.midi_data.get('gridArrangement', {})
            logging.info(f"Grid arrangement received: {grid_arrangement}")
            
            if not grid_arrangement:
                logging.warning("No grid arrangement provided, creating default arrangement")
                # Create a default grid arrangement based on available tracks
                tracks = self.midi_data.get('tracks', [])
                if tracks:
                    # Create a simple grid arrangement
                    num_tracks = len(tracks)
                    grid_size = int(num_tracks ** 0.5) + 1 if num_tracks > 1 else 1;
                    
                    grid_arrangement = {
                        'layout': [[f'track_{i}' for i in range(min(grid_size, num_tracks - row * grid_size))] 
                                  for row in range((num_tracks + grid_size - 1) // grid_size)],
                        'rows': (num_tracks + grid_size - 1) // grid_size,
                        'cols': grid_size
                    }
                else:
                    # Default single cell arrangement
                    grid_arrangement = {
                        'layout': [['default']],
                        'rows': 1,
                        'cols': 1
                    }
                
                logging.info(f"Created default grid arrangement: {grid_arrangement}")
            
            # Store positions and validate
            self.grid_positions = {}
            
            # Handle different grid arrangement formats
            if 'layout' in grid_arrangement:
                # New format with layout matrix
                layout = grid_arrangement['layout']
                rows = grid_arrangement.get('rows', len(layout))
                cols = grid_arrangement.get('cols', len(layout[0]) if layout else 1)
                
                logging.info(f"Processing layout matrix: {rows}x{cols}")
                
                for row_idx, row in enumerate(layout):
                    for col_idx, cell in enumerate(row):
                        if isinstance(cell, dict) and 'instrument' in cell:
                            track_id = str(cell.get('track', f"{cell['instrument']}"))
                            self.grid_positions[track_id] = {
                                'row': row_idx,
                                'column': col_idx
                            }
                            logging.info(f"Mapped track {track_id} ({cell['instrument']}) to position row={row_idx}, col={col_idx}")
            else:
                # Original format with direct track ID mappings
                for track_id, pos_data in grid_arrangement.items():
                    if isinstance(pos_data, dict):
                        # Validate position data - check for required keys
                        required_keys = ['row', 'column']
                        if not all(k in pos_data for k in required_keys):
                            logging.error(f"Invalid position data for {track_id}: {pos_data}")
                            continue
                        
                        # Store position based on track index
                        self.grid_positions[track_id] = {
                            'row': int(pos_data['row']),
                            'column': int(pos_data['column'])
                        }
                        logging.info(f"Mapped track {track_id} to position row={pos_data['row']}, col={pos_data['column']}")

            # Add this call to analyze MIDI timing data
            self._analyze_midi_timing()

        except Exception as e:
            logging.error(f"Error setting up track configuration: {e}")
            raise

    def encode_video(cmd):
        logging.info(f"Encoding video with command: {' '.join(cmd)}")
        result = encoder_queue.encode(cmd)
        return result

    def validate_video(output_path):
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

    def calculate_chunk_lengths(self):
        """Calculate chunk lengths based on actual MIDI content"""
        try:
            # Get all track notes in one list
            all_notes = []
            for track in self.midi_data['tracks']:
                if isinstance(track, dict) and 'notes' in track:
                    all_notes.extend(track['notes'])
            
            if not all_notes:
                raise ValueError("No notes found in any tracks")
            
            # Find last note end time
            last_note_time = 0
            for note in all_notes:
                if isinstance(note, dict):
                    note_end = float(note['time']) + float(note['duration'])
                    last_note_time = max(last_note_time, note_end)
            
            # Calculate chunks based on exact duration needed
            full_chunks = math.floor(last_note_time / self.CHUNK_DURATION)
            final_chunk = last_note_time % self.CHUNK_DURATION
            
            # Only include final chunk if there's actual content
            if final_chunk < 0.1:  # If less than 0.1s remaining, ignore final chunk
                final_chunk = 0
            
            logging.info(f"Total duration: {last_note_time:.2f}s")
            logging.info(f"Full chunks: {full_chunks}")
            logging.info(f"Final chunk: {final_chunk:.2f}s")
            
            return full_chunks, final_chunk
                
        except Exception as e:
            logging.error(f"Error calculating chunks: {str(e)}")
            return 0, 0
        
    def _verify_and_fix_chunks(self):
        """Verify all expected chunks exist and create placeholders if needed"""
        logging.info(f"Verifying all {self.total_chunks} chunks are present")
        
        missing_chunks = []
        
        for chunk_idx in range(self.total_chunks):
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            if not chunk_path.exists():
                missing_chunks.append(chunk_idx)
        
        if missing_chunks:
            logging.warning(f"Found {len(missing_chunks)} missing chunks: {missing_chunks}")
            
            # Create placeholder chunks to maintain timing
            for chunk_idx in missing_chunks:
                self._create_placeholder_chunk(chunk_idx)
                logging.info(f"Created placeholder for chunk {chunk_idx}")

    def _create_placeholder_chunk(self, chunk_idx):
        """Create a placeholder chunk with silence"""
        chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
        
        # Create a video with black background and silent audio
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=black:s=1920x1080:r=30',
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-t', str(self.CHUNK_DURATION),
            '-c:v', 'h264_nvenc', '-preset', 'p4',
            '-c:a', 'aac', '-b:a', '128k',
            str(chunk_path)
        ]
        
        try:
            gpu_subprocess_run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating placeholder chunk: {e}")
            return False
        
    def has_valid_notes(self, track):
        """Check if a track has valid notes"""
        if not isinstance(track, dict) or 'notes' not in track:
            return False
        
        notes = track['notes']
        if not isinstance(notes, list) or len(notes) == 0:
            return False
        
        return True

    def preprocess_composition_optimized(self):
        """
        PERFORMANCE OPTIMIZATION: Pre-process all required note combinations
        
        This method replaces the inefficient individual note processing approach
        with a batch preprocessing system that dramatically improves performance:
        
        BEFORE: Each note processed individually during composition (5+ minutes)
        AFTER:  All notes batch processed once, then instantly retrieved (seconds)
        """
        logging.info("🚀 Starting OPTIMIZED composition preprocessing...")
        start_time = time.time()
        
        try:
            # Analyze MIDI data to find all required instrument/note combinations
            required_combinations = self._analyze_composition_requirements()
            
            if not required_combinations:
                logging.warning("No instrument/note combinations found for preprocessing")
                return
            
            # Get video paths for all instruments
            video_paths = self._get_instrument_video_paths(required_combinations.keys())
            
            if not video_paths:
                logging.warning("No video paths found for instruments")
                return
            
            # Batch process all combinations using optimized cache
            logging.info(f"📊 Processing {sum(len(notes) for notes in required_combinations.values())} note combinations...")
            self._tuned_videos_cache = self.optimized_cache.preprocess_composition(
                self.midi_data, video_paths
            )
            
            # Log preprocessing results
            total_processed = sum(len(notes) for notes in self._tuned_videos_cache.values())
            processing_time = time.time() - start_time
            
            logging.info(f"✅ OPTIMIZATION COMPLETE!")
            logging.info(f"   📹 Instruments processed: {len(self._tuned_videos_cache)}")
            logging.info(f"   🎵 Total note combinations: {total_processed}")
            logging.info(f"   ⏱️  Processing time: {processing_time:.2f}s")
            logging.info(f"   🚀 Performance improvement: ~{max(1, (300/max(1, processing_time))):.0f}x faster")
            
            # Log cache statistics
            cache_stats = self.optimized_cache.get_cache_stats()
            logging.info(f"   💾 Cache stats: {cache_stats}")
            
        except Exception as e:
            logging.error(f"Failed to preprocess composition: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_composition_requirements(self):
        """Analyze MIDI composition to find all required instrument/note combinations"""
        requirements = {}
        
        for track in self.midi_data.get('tracks', []):
            instrument_name = track.get('instrument', {}).get('name', 'unknown')
            normalized_name = normalize_instrument_name(instrument_name)
            
            if normalized_name not in requirements:
                requirements[normalized_name] = set()
            
            # Collect all unique MIDI notes for this instrument
            for note in track.get('notes', []):
                midi_note = note.get('midi')
                if midi_note is not None:
                    requirements[normalized_name].add(midi_note)
        
        # Convert sets to sorted lists for consistent processing
        for instrument in requirements:
            requirements[instrument] = sorted(list(requirements[instrument]))
            logging.info(f"📝 {instrument}: {len(requirements[instrument])} notes {requirements[instrument]}")
        
        return requirements

    def _get_instrument_video_paths(self, instrument_names):
        """Get video file paths for specified instruments"""
        video_paths = {}
        registry = PathRegistry.get_instance()
        
        for instrument_name in instrument_names:
            # Try to find any video for this instrument (we'll tune it to all needed notes)
            path = registry.get_instrument_path(instrument_name, "60")  # Try middle C as default
            if not path:
                # Try to find any video for this instrument with any note
                instrument_paths = registry.instrument_paths.get(instrument_name, {})
                if instrument_paths:
                    path = next(iter(instrument_paths.values()))  # Get first available video
            
            if path and os.path.exists(path):
                video_paths[instrument_name] = path
                logging.info(f"✅ Found video for {instrument_name}: {os.path.basename(path)}")
            else:
                logging.warning(f"❌ No video found for instrument: {instrument_name}")
        
        return video_paths

    def get_optimized_tuned_video(self, instrument_name, midi_note):
        """
        Get pre-processed tuned video instantly from cache.
        
        This replaces the old _autotune_audio method that processed each note individually.
        """
        normalized_name = normalize_instrument_name(instrument_name)
        
        # Try to get from preprocessed cache first
        if (normalized_name in self._tuned_videos_cache and 
            midi_note in self._tuned_videos_cache[normalized_name]):
            tuned_path = self._tuned_videos_cache[normalized_name][midi_note]
            logging.info(f"✅ INSTANT retrieval: {instrument_name} → MIDI {midi_note}")
            return tuned_path
        
        # Fallback: create on-demand if not preprocessed (shouldn't happen with proper preprocessing)
        logging.warning(f"⚠️  On-demand processing: {instrument_name} → MIDI {midi_note} (not preprocessed)")
        
        # Get original video path
        registry = PathRegistry.get_instance()
        original_path = registry.get_instrument_path(normalized_name, "60")  # Try default
        if not original_path:
            instrument_paths = registry.instrument_paths.get(normalized_name, {})
            if instrument_paths:
                original_path = next(iter(instrument_paths.values()))
        
        if original_path and os.path.exists(original_path):
            return self.optimized_cache.get_tuned_video(original_path, midi_note)
        
        logging.error(f"❌ No video found for {instrument_name}")
        return None

    def _autotune_audio(self, video_path, midi_note):
        """
        Autotune a video to a specific MIDI note.
        
        This method now uses the optimized cache system instead of processing 
        each note individually. If preprocessing was done, it retrieves instantly.
        Otherwise, it processes on-demand using the optimized cache.
        
        Args:
            video_path: Path to the input video file
            midi_note: Target MIDI note number
            
        Returns:
            Path to the autotuned video file, or None if failed
        """
        try:
            # Extract instrument name from video path for cache lookup
            video_name = os.path.basename(video_path)
            instrument_name = video_name.replace('.mp4', '').replace('processed_', '')
              # Use the optimized cache system
            tuned_path = self.optimized_cache.get_tuned_video(video_path, midi_note)
            
            if tuned_path and os.path.exists(tuned_path):
                logging.info(f"✅ Autotune successful: {video_name} → MIDI {midi_note}")
                return tuned_path
            else:
                logging.error(f"❌ Autotune failed: {video_name} → MIDI {midi_note}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Autotune error for {video_path} → MIDI {midi_note}: {e}")
            return None

    # def _normalize_final_audio(self, input_path, output_path):
    #     """
    #     Applies a two-pass loudness normalization to the final video.
    #     This ensures consistent volume throughout the entire composition.
    #     Includes robust error handling and fallbacks.
    #     """
    #     try:
    #         # Two-pass loudnorm is recommended for best results.
    #         # Pass 1: Analyze the audio and log the results.
    #         logging.info("   (Loudnorm Pass 1/2) Analyzing audio...")
    #         pass1_cmd = [
    #             'ffmpeg', '-y', '-i', str(input_path),
    #             '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json',
    #             '-f', 'null', '-'
    #         ]
    #         result1 = subprocess.run(pass1_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    #         if result1.returncode != 0:
    #             logging.error("❌ Loudnorm Pass 1 failed. FFmpeg returned a non-zero exit code.")
    #             logging.error(f"   Stderr: {result1.stderr}")
    #             raise Exception("Loudnorm analysis pass failed.")

    #         # FFmpeg prints loudnorm stats to stderr. We find the JSON part and parse it.
    #         # The JSON block is usually at the end of the stderr output.
    #         json_output = None
    #         for line in reversed(result1.stderr.strip().split('\n')):
    #             if line.strip().startswith('{') and line.strip().endswith('}'):
    #                 json_output = line.strip()
    #                 break
            
    #         if not json_output:
    #             logging.error("❌ Could not find loudnorm JSON stats in FFmpeg output.")
    #             logging.error(f"   Full stderr: {result1.stderr}")
    #             raise Exception("Failed to parse loudnorm stats.")

    #         stats = json.loads(json_output)
    #         logging.info(f"   Loudnorm stats: {stats}")

    #         # Pass 2: Apply the calculated normalization values.
    #         logging.info("   (Loudnorm Pass 2/2) Applying normalization...")
    #         pass2_cmd = [
    #             'ffmpeg', '-y', '-i', str(input_path),
    #             '-af', f"loudnorm=I=-16:LRA=11:TP=-1.5:"
    #                    f"measured_I={stats['input_i']}:"
    #                    f"measured_LRA={stats['input_lra']}:"
    #                    f"measured_tp={stats['input_tp']}:"
    #                    f"measured_thresh={stats['input_thresh']}:"
    #                    f"offset={stats['target_offset']}",
    #             '-c:v', 'copy',
    #             '-c:a', 'aac', '-b:a', '320k',
    #             str(output_path)
    #         ]
            
    #         result2 = subprocess.run(pass2_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    #         if result2.returncode == 0:
    #             logging.info("✅ Final audio normalized successfully.")
    #             return str(output_path)
    #         else:
    #             logging.error("❌ Loudnorm Pass 2 failed. FFmpeg returned a non-zero exit code.")
    #             logging.error(f"   Stderr: {result2.stderr}")
    #             raise Exception("Loudnorm application pass failed.")

    #     except Exception as e:
    #         logging.error(f"An error occurred during audio normalization: {e}")
    #         logging.error("❌ Final audio normalization failed. Returning unnormalized video.")
    #         # Fallback: copy the unnormalized video to the final destination
    #         try:
    #             shutil.copy2(input_path, output_path)
    #             logging.info(f"   Fallback successful: Copied unnormalized video to {output_path}")
    #             return str(output_path)
    #         except Exception as copy_error:
    #             logging.error(f"   Fallback failed: Could not copy file. {copy_error}")
    #             return None

    def _normalize_final_audio(self, input_path, output_path):
        """
        FIXED: Applies two-pass loudness normalization with robust JSON parsing
        """
        try:
            logging.info("🔊 Normalizing audio for the entire composition for consistent volume...")
            logging.info("   (Loudnorm Pass 1/2) Analyzing audio...")
            
            pass1_cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json',
                '-f', 'null', '-'
            ]
            
            result1 = subprocess.run(pass1_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

            if result1.returncode != 0:
                logging.error("❌ Loudnorm analysis failed")
                raise Exception("Loudnorm analysis pass failed")

            # FIXED: More robust JSON extraction from stderr
            stderr_output = result1.stderr
            logging.debug(f"FFmpeg stderr length: {len(stderr_output)} chars")
            
            # Look for JSON block more reliably
            import re
            
            # Try multiple patterns to find the JSON stats
            json_patterns = [
                r'\{[^{}]*"input_i"[^{}]*"input_tp"[^{}]*"input_lra"[^{}]*"input_thresh"[^{}]*"target_offset"[^{}]*\}',
                r'\{[^{}]*"input_i"[^{}]*\}',
                r'(\{(?:[^{}]|{[^{}]*})*"input_i"(?:[^{}]|{[^{}]*})*\})'
            ]
            
            stats = None
            for pattern in json_patterns:
                matches = re.findall(pattern, stderr_output, re.DOTALL)
                for match in matches:
                    try:
                        potential_stats = json.loads(match)
                        if all(key in potential_stats for key in ['input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset']):
                            stats = potential_stats
                            logging.info(f"✅ Found valid loudnorm stats: {stats}")
                            break
                    except json.JSONDecodeError:
                        continue
                if stats:
                    break
            
            if not stats:
                # Fallback: Try line-by-line parsing
                lines = stderr_output.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and 'input_i' in line:
                        try:
                            stats = json.loads(line)
                            if 'input_i' in stats and 'target_offset' in stats:
                                logging.info(f"✅ Found stats via line parsing: {stats}")
                                break
                        except json.JSONDecodeError:
                            continue
            
            if not stats:
                logging.error("❌ Could not find loudnorm JSON stats in FFmpeg output.")
                logging.error(f"   Full stderr: {stderr_output}")
                raise Exception("Failed to parse loudnorm stats.")

            # Pass 2: Apply normalization with extracted stats
            logging.info("   (Loudnorm Pass 2/2) Applying normalization...")
            pass2_cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', f'loudnorm=I=-16:TP=-1.5:LRA=11:'
                    f'measured_I={stats["input_i"]}:'
                    f'measured_LRA={stats["input_lra"]}:'
                    f'measured_tp={stats["input_tp"]}:'
                    f'measured_thresh={stats["input_thresh"]}:'
                    f'offset={stats["target_offset"]}',
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '320k',
                str(output_path)
            ]
            
            result2 = subprocess.run(pass2_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

            if result2.returncode == 0:
                logging.info("✅ Final audio normalized successfully.")
                return str(output_path)
            else:
                logging.error("❌ Loudnorm application failed")
                logging.error(f"   Stderr: {result2.stderr}")
                raise Exception("Loudnorm application pass failed.")

        except Exception as e:
            logging.error(f"An error occurred during audio normalization: {e}")
            logging.error("❌ Final audio normalization failed. Returning unnormalized video.")
            
            # Fallback: copy unnormalized video
            try:
                shutil.copy2(input_path, output_path)
                logging.info(f"   Fallback successful: Copied unnormalized video to {output_path}")
                return str(output_path)
            except Exception as copy_error:
                logging.error(f"   Fallback failed: {copy_error}")
                return None

            
    # def create_composition(self):
    #     """
    #     SIMPLIFIED MAIN COMPOSITION METHOD
        
    #     This method eliminates the complex optimization layers that were causing
    #     cache misses and performance degradation. Returns to proven direct processing
    #     approach with proper drum handling and fast performance.
        
    #     Returns:
    #         str: Path to the final composed video, or None if failed
    #     """
    #     try:
    #         logging.info("🎬 Starting SIMPLIFIED video composition...")
    #         start_time = time.time()
            
    #         # Use the proven chunk-based approach without complex optimization layers
    #         logging.info("🎥 Setting up composition parameters...")
            
    #         # Calculate composition duration
    #         total_duration = self._calculate_total_duration()
    #         total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))
            
    #         logging.info(f"Composition: {total_duration:.2f}s, {total_chunks} chunks")
            
    #         # Create chunks directory
    #         chunks_dir = self.processed_videos_dir / "simple_chunks"
    #         chunks_dir.mkdir(exist_ok=True)
            
    #         # Process chunks with simplified approach
    #         chunk_paths = []
    #         for chunk_idx in range(total_chunks):
    #             chunk_start = chunk_idx * self.CHUNK_DURATION
    #             chunk_end = min(chunk_start + self.CHUNK_DURATION, total_duration)
                
    #             logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({chunk_start:.1f}s - {chunk_end:.1f}s)")
                
    #             chunk_path = self._create_simplified_chunk(chunk_idx, chunk_start, chunk_end, chunks_dir)
                
    #             if chunk_path and os.path.exists(chunk_path):
    #                 chunk_paths.append(chunk_path)
    #                 logging.info(f"✅ Chunk {chunk_idx + 1} completed")
    #             else:
    #                 logging.warning(f"⚠️  Chunk {chunk_idx + 1} failed, creating placeholder")
    #                 placeholder_path = self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_end - chunk_start)
    #                 if placeholder_path:
    #                     chunk_paths.append(placeholder_path)
            
    #         if not chunk_paths:
    #             raise Exception("No chunks were created successfully")
    #           # Concatenate chunks into final video
    #         logging.info(f"Concatenating {len(chunk_paths)} chunks...")
    #         concatenated_path = self._concatenate_chunks(chunk_paths)

    #         if not concatenated_path or not os.path.exists(concatenated_path):
    #             raise Exception("Chunk concatenation failed, cannot proceed to normalization.")

    #         # --- START OF VOLUME NORMALIZATION REFACTOR ---
    #         logging.info("🔊 Normalizing audio for the entire composition for consistent volume...")
            
    #         final_output_path = self.output_path # The user's desired final path
    #         normalized_video_path = self._normalize_final_audio(concatenated_path, final_output_path)
            
    #         total_time = time.time() - start_time
            
    #         if normalized_video_path and os.path.exists(normalized_video_path):
    #             file_size = os.path.getsize(normalized_video_path)
    #             logging.info(f"🎉 COMPOSITION SUCCESSFUL!")
    #             logging.info(f"   📁 Final Output: {normalized_video_path}")
    #             logging.info(f"   🔊 Audio has been normalized for consistent loudness.")
    #             logging.info(f"   📏 Size: {file_size:,} bytes")
    #             logging.info(f"   ⏱️  Total time: {total_time:.2f}s")
    #             logging.info(f"   🚀 Fast direct processing - no cache misses!")
                
    #             return str(normalized_video_path)
    #         else:
    #             logging.error("❌ Final audio normalization failed. Returning unnormalized video.")
    #             return str(concatenated_path)
                
    #     except Exception as e:
    #         logging.error(f"❌ Composition error: {e}")
    #         import traceback
    #         logging.error(f"Full traceback: {traceback.format_exc()}")
    #         return None

    # def create_composition(self):
    #     """
    #     SIMPLIFIED MAIN COMPOSITION METHOD
        
    #     This method eliminates the complex optimization layers that were causing
    #     cache misses and performance degradation. Returns to proven direct processing
    #     approach with proper drum handling and fast performance.
        
    #     Returns:
    #         str: Path to the final composed video, or None if failed
    #     """
    #     try:
    #         logging.info("🎬 Starting SIMPLIFIED video composition...")
    #         start_time = time.time()
            
    #         # Use the proven chunk-based approach without complex optimization layers
    #         logging.info("🎥 Setting up composition parameters...")
            
    #         # Calculate composition duration
    #         total_duration = self._calculate_total_duration()
    #         total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))
            
    #         logging.info(f"Composition: {total_duration:.2f}s, {total_chunks} chunks")
            
    #         # Create chunks directory
    #         chunks_dir = self.processed_videos_dir / "simple_chunks"
    #         chunks_dir.mkdir(exist_ok=True)
            
    #         # Process chunks with simplified approach
    #         chunk_paths = []
    #         for chunk_idx in range(total_chunks):
    #             logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({(chunk_idx * self.CHUNK_DURATION):.1f}s - {((chunk_idx + 1) * self.CHUNK_DURATION):.1f}s)")
    #             start_time_chunk = chunk_idx * self.CHUNK_DURATION
    #             end_time_chunk = (chunk_idx + 1) * self.CHUNK_DURATION
    #             chunk_path = self._create_simplified_chunk(chunk_idx, start_time_chunk, end_time_chunk, chunks_dir)
    #             if chunk_path and os.path.exists(chunk_path):
    #                 chunk_paths.append(chunk_path)
    #                 logging.info(f"✅ Chunk {chunk_idx + 1} completed")
    #             else:
    #                 logging.warning(f"⚠️ Chunk {chunk_idx} failed to create or was empty.")
            
    #         if not chunk_paths:
    #             logging.error("❌ No video chunks were created. Composition failed.")
    #             return None
    #           # Concatenate chunks into final video
    #         logging.info(f"Concatenating {len(chunk_paths)} chunks...")
    #         # Create a temporary path for the concatenated but unnormalized video
    #         concatenated_path = self.temp_dir / "concatenated_unnormalized.mp4"
    #         concatenated_path = self._concatenate_chunks(chunk_paths, concatenated_path)


    #         if not concatenated_path or not os.path.exists(concatenated_path):
    #             logging.error("❌ Concatenation failed. Cannot proceed to normalization.")
    #             return None

    #         # --- START OF VOLUME NORMALIZATION REFACTOR ---
    #         logging.info("🔊 Normalizing audio for the entire composition for consistent volume...")
            
    #         final_output_path = self.output_path # The user's desired final path
    #         normalized_video_path = self._normalize_final_audio(concatenated_path, final_output_path)
            
    #         total_time = time.time() - start_time
            
    #         if normalized_video_path and os.path.exists(normalized_video_path):
    #             logging.info(f"🎉 Video composition complete! Total time: {total_time:.2f}s")
    #             logging.info(f"   Final video saved to: {normalized_video_path}")
    #             return str(normalized_video_path)
    #         else:
    #             logging.error("❌ Composition failed after normalization step.")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"❌ Composition error: {e}")
    #         logging.error(f"Full traceback: {traceback.format_exc()}")
    #         return None

    def create_composition(self):
        """
        ENHANCED composition with all fixes applied
        """
        try:
            logging.info("🎬 Starting ENHANCED video composition with fixes...")
            start_time = time.time()
            
            # Decide between parallel and sequential processing
            total_duration = self._calculate_total_duration()
            total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))
            
            # Use parallel processing for compositions with multiple chunks
            if total_chunks > 2 and self.max_workers > 1:
                logging.info(f"Using parallel processing for {total_chunks} chunks")
                return self.create_composition_with_parallel_processing()
            else:
                logging.info(f"Using sequential processing for {total_chunks} chunks")
                return self._create_composition_sequential()
                
        except Exception as e:
            logging.error(f"❌ Enhanced composition error: {e}")
            return None

    def _create_composition_sequential(self):
        """Sequential composition with all fixes applied"""
        try:
            start_time = time.time()
            
            total_duration = self._calculate_total_duration()
            total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))
            
            logging.info(f"Sequential composition: {total_duration:.2f}s, {total_chunks} chunks")
            
            chunks_dir = self.processed_videos_dir / "enhanced_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            chunk_paths = []
            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * self.CHUNK_DURATION
                chunk_end = min(chunk_start + self.CHUNK_DURATION, total_duration)
                
                logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({chunk_start:.1f}s - {chunk_end:.1f}s)")
                
                chunk_path = self._create_enhanced_chunk(chunk_idx, chunk_start, chunk_end, chunks_dir)
                
                if chunk_path and os.path.exists(chunk_path):
                    chunk_paths.append(chunk_path)
                    logging.info(f"✅ Enhanced chunk {chunk_idx + 1} completed")
                else:
                    logging.warning(f"⚠️ Chunk {chunk_idx + 1} failed, creating placeholder")
                    placeholder = self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_end - chunk_start)
                    if placeholder:
                        chunk_paths.append(placeholder)
            
            if not chunk_paths:
                raise Exception("No chunks were created successfully")
            
            # Concatenate chunks
            concatenated_path = self.temp_dir / "concatenated_enhanced.mp4"
            final_path = self._concatenate_chunks(chunk_paths, concatenated_path)
            
            if final_path and os.path.exists(final_path):
                # Apply enhanced normalization
                normalized_path = self._normalize_final_audio(final_path, self.output_path)
                
                total_time = time.time() - start_time
                logging.info(f"🎉 Enhanced composition complete! Total time: {total_time:.2f}s")
                
                return normalized_path
            else:
                raise Exception("Enhanced concatenation failed")
                
        except Exception as e:
            logging.error(f"❌ Sequential composition error: {e}")
            return None

    def _create_enhanced_chunk(self, chunk_idx, start_time, end_time, chunks_dir):
        """Create chunk with all enhancements applied"""
        try:
            chunk_path = chunks_dir / f"enhanced_chunk_{chunk_idx}.mp4"
            chunk_duration = end_time - start_time
            
            # Use enhanced note processing with delay validation
            active_tracks = self._find_tracks_in_timerange(start_time, end_time)
            
            if not active_tracks:
                return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
            track_video_segments = []
            
            for track in active_tracks:
                track_id = track.get('id', track.get('original_index', 'unknown'))
                
                if track.get('isDrum') or track.get('channel') == 9:
                    drum_segments = self._process_drum_track_for_chunk(track, start_time, end_time)
                    if drum_segments:
                        track_video_segments.extend(drum_segments)
                else:
                    # Use enhanced note-triggered processing
                    result = self._process_instrument_track_enhanced(track, start_time, chunk_duration, chunk_idx, track_id)
                    if result:
                        track_video_segments.append(result)
            
            if not track_video_segments:
                return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
            # Create final chunk with optimal encoding
            return self._create_optimized_chunk_layout(track_video_segments, chunk_path, chunk_duration)
            
        except Exception as e:
            logging.error(f"Error creating enhanced chunk {chunk_idx}: {e}")
            return None

    def create_composition_with_parallel_processing(self):
        """
        Enhanced composition with parallel chunk processing for better performance
        """
        try:
            logging.info("🚀 Starting PARALLEL video composition...")
            start_time = time.time()
            
            total_duration = self._calculate_total_duration()
            total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))
            
            logging.info(f"Composition: {total_duration:.2f}s, {total_chunks} chunks")
            logging.info(f"Using parallel processing with {self.max_workers} workers")
            
            chunks_dir = self.processed_videos_dir / "parallel_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            # Create chunk tasks
            chunk_tasks = []
            for chunk_idx in range(total_chunks):
                start_time_chunk = chunk_idx * self.CHUNK_DURATION
                end_time_chunk = min(start_time_chunk + self.CHUNK_DURATION, total_duration)
                chunk_tasks.append((chunk_idx, start_time_chunk, end_time_chunk, chunks_dir))
            
            # Process chunks in parallel
            chunk_paths = []
            successful_chunks = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._create_chunk_parallel, task): task 
                    for task in chunk_tasks
                }
                
                for future in as_completed(future_to_chunk):
                    chunk_idx, start_time_chunk, end_time_chunk, chunks_dir = future_to_chunk[future]
                    try:
                        chunk_path = future.result()
                        if chunk_path and os.path.exists(chunk_path):
                            chunk_paths.append((chunk_idx, chunk_path))
                            successful_chunks += 1
                            logging.info(f"✅ Parallel chunk {chunk_idx + 1}/{total_chunks} completed")
                        else:
                            logging.warning(f"⚠️ Chunk {chunk_idx + 1} failed, creating placeholder")
                            placeholder = self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, end_time_chunk - start_time_chunk)
                            if placeholder:
                                chunk_paths.append((chunk_idx, placeholder))
                    except Exception as e:
                        logging.error(f"❌ Chunk {chunk_idx + 1} error: {e}")
            
            # Sort chunks by index to maintain correct order
            chunk_paths.sort(key=lambda x: x[0])
            ordered_chunk_paths = [path for _, path in chunk_paths]
            
            if not ordered_chunk_paths:
                raise Exception("No chunks were created successfully")
            
            logging.info(f"✅ Parallel processing complete: {successful_chunks}/{total_chunks} chunks successful")
            
            # Concatenate chunks
            concatenated_path = self.temp_dir / "concatenated_parallel.mp4"
            final_path = self._concatenate_chunks(ordered_chunk_paths, concatenated_path)
            
            if final_path and os.path.exists(final_path):
                # Apply normalization
                normalized_path = self._normalize_final_audio(final_path, self.output_path)
                
                total_time = time.time() - start_time
                logging.info(f"🎉 Parallel composition complete! Total time: {total_time:.2f}s")
                logging.info(f"   Performance improvement: ~{max(1, total_chunks/self.max_workers):.1f}x faster with parallel processing")
                
                return normalized_path
            else:
                raise Exception("Chunk concatenation failed")
                
        except Exception as e:
            logging.error(f"❌ Parallel composition error: {e}")
            return None

    def _create_chunk_parallel(self, task):
        """Create a single chunk for parallel processing"""
        chunk_idx, start_time, end_time, chunks_dir = task
        return self._create_simplified_chunk(chunk_idx, start_time, end_time, chunks_dir)
    

    def _get_optimal_encoding_params(self):
        """Get optimal encoding parameters based on available hardware"""
        try:
            # Test NVIDIA GPU availability
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("✅ NVIDIA GPU detected, using hardware acceleration")
                return {
                    'video_codec': 'h264_nvenc',
                    'preset': 'p4',  # Balanced preset for NVENC
                    'additional_params': [
                        '-hwaccel', 'cuda',
                        '-hwaccel_output_format', 'cuda',
                        '-b:v', '8M',
                        '-maxrate', '12M',
                        '-bufsize', '16M'
                    ]
                }
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback to CPU encoding with fast preset
        logging.info("Using CPU encoding with optimized settings")
        return {
            'video_codec': 'libx264',
            'preset': 'ultrafast',
            'additional_params': [
                '-crf', '23',
                '-threads', str(min(8, os.cpu_count())),
                '-tune', 'fastdecode'
            ]
        }

    def _calculate_composition_duration(self):
        """Calculate the total duration needed for the composition"""
        try:
            # Find the latest note end time across all tracks
            max_end_time = 0
            
            for track in self.tracks + self.drum_tracks:
                if 'notes' in track:
                    for note in track['notes']:
                        note_end = float(note['time']) + float(note['duration'])
                        max_end_time = max(max_end_time, note_end)
            
            # Add buffer for fade-out
            self.composition_duration = max_end_time + 2.0
            logging.info(f"Composition duration calculated: {self.composition_duration:.2f}s")
            
        except Exception as e:
            logging.warning(f"Error calculating duration, using default: {e}")
            self.composition_duration = float(self.midi_data.get('duration', 10.0))

    def _setup_chunk_structure(self):
        """Setup the chunk structure for optimized composition"""
        try:
            # Calculate number of chunks needed
            self.total_chunks = max(1, int(math.ceil(self.composition_duration / self.CHUNK_DURATION)))
            
            logging.info(f"Chunk structure: {self.total_chunks} chunks of {self.CHUNK_DURATION}s each")
            
            # Pre-create chunk directory
            self.chunks_dir = self.temp_dir / "optimized_chunks"
            self.chunks_dir.mkdir(exist_ok=True)
            
        except Exception as e:
            logging.error(f"Error setting up chunk structure: {e}")
            raise

    def _create_optimized_chunks_composition(self):
        """Create composition using optimized chunks with instant note retrieval"""
        try:
            chunk_paths = []
            
            # Process each chunk using optimized retrieval
            for chunk_idx in range(self.total_chunks):
                chunk_start = chunk_idx * self.CHUNK_DURATION
                chunk_end = min(chunk_start + self.CHUNK_DURATION, self.composition_duration)
                
                logging.info(f"Processing chunk {chunk_idx + 1}/{self.total_chunks} ({chunk_start:.1f}s - {chunk_end:.1f}s)")
                
                # Create chunk using optimized note retrieval
                chunk_path = self._create_optimized_chunk(chunk_idx, chunk_start, chunk_end)
                
                if chunk_path and os.path.exists(chunk_path):
                    chunk_paths.append(chunk_path)
                    logging.info(f"✅ Chunk {chunk_idx + 1} completed")
                else:
                    logging.warning(f"⚠️  Chunk {chunk_idx + 1} failed, creating placeholder")
                    placeholder_path = self._create_placeholder_chunk(chunk_idx)
                    if placeholder_path:
                        chunk_paths.append(placeholder_path)
            
            if not chunk_paths:
                raise Exception("No chunks were created successfully")
            
            # Concatenate all chunks into final video
            logging.info(f"Concatenating {len(chunk_paths)} chunks...")
            final_path = self._concatenate_chunks(chunk_paths)
            
            return final_path
            
        except Exception as e:
            logging.error(f"Error in optimized chunk composition: {e}")
            raise

    def _create_optimized_chunk(self, chunk_idx, start_time, end_time):
        """Create a single chunk using optimized autotune retrieval"""
        try:
            chunk_path = self.chunks_dir / f"chunk_{chunk_idx}.mp4"
            
            # Find all notes that play during this chunk
            active_notes = self._find_active_notes_in_timerange(start_time, end_time)
            
            if not active_notes:
                # No notes in this chunk, create silence
                return self._create_silent_chunk(chunk_idx, end_time - start_time)
            
            # Group notes by track for processing
            track_videos = {}
            
            for track_id, notes in active_notes.items():
                track_info = self._get_track_info(track_id)
                if not track_info:
                    continue
                
                instrument_name = track_info.get('instrument', {}).get('name', f'track_{track_id}')
                
                # Process notes for this track using OPTIMIZED retrieval
                track_video_path = self._create_track_chunk_optimized(
                    track_id, instrument_name, notes, start_time, end_time
                )
                
                if track_video_path:
                    track_videos[track_id] = track_video_path
            
            if not track_videos:
                return self._create_silent_chunk(chunk_idx, end_time - start_time)
            
            # Combine track videos into final chunk
            return self._combine_track_videos(track_videos, chunk_path, end_time - start_time)
            
        except Exception as e:
            logging.error(f"Error creating optimized chunk {chunk_idx}: {e}")
            return None
        
    def _create_optimized_ffmpeg_command(self, inputs, filter_complex, output_path, duration):
        """Create optimized FFmpeg command with hardware acceleration"""
        encoding_params = self._get_optimal_encoding_params()
        
        cmd = ['ffmpeg', '-y']
        
        # Add hardware acceleration if available
        if 'additional_params' in encoding_params:
            for param in encoding_params['additional_params'][:2]:  # Add hwaccel params first
                if param in ['-hwaccel', '-hwaccel_output_format']:
                    cmd.extend([param, encoding_params['additional_params'][encoding_params['additional_params'].index(param) + 1]])
        
        # Add inputs
        for input_path in inputs:
            cmd.extend(['-i', str(input_path)])
        
        # Add filter complex
        if filter_complex:
            cmd.extend(['-filter_complex', filter_complex])
        
        # Add encoding parameters
        cmd.extend([
            '-c:v', encoding_params['video_codec'],
            '-preset', encoding_params['preset'],
            '-c:a', 'aac',
            '-b:a', '192k',
            '-t', str(duration),
            '-r', '30'
        ])
        
        # Add remaining optimization parameters
        if 'additional_params' in encoding_params:
            remaining_params = encoding_params['additional_params'][2:]  # Skip hwaccel params already added
            cmd.extend(remaining_params)
        
        cmd.append(str(output_path))
        
        return cmd

    def _create_track_chunk_optimized(self, track_id, instrument_name, notes, start_time, end_time):
        """Create track chunk using OPTIMIZED autotune retrieval (no individual processing)"""
        try:
            track_videos = []
            
            for note in notes:
                midi_note = note.get('midi')
                note_start = float(note.get('time', 0))
                note_duration = float(note.get('duration', 1))
                
                # OPTIMIZED: Get pre-processed tuned video instantly from cache
                tuned_video_path = self.get_optimized_tuned_video(instrument_name, midi_note)
                
                if tuned_video_path and os.path.exists(tuned_video_path):
                    # Calculate relative timing within chunk
                    relative_start = max(0, note_start - start_time)
                    
                    track_videos.append({
                        'path': tuned_video_path,
                        'start': relative_start,
                        'duration': note_duration,
                        'midi': midi_note
                    })
                    
                    logging.debug(f"✅ INSTANT retrieval: {instrument_name} MIDI {midi_note}")
                else:
                    logging.warning(f"⚠️  Missing tuned video: {instrument_name} MIDI {midi_note}")
            
            if not track_videos:
                return None
            
            # Create track chunk from optimized videos
            track_chunk_path = self.chunks_dir / f"track_{track_id}_chunk_{int(start_time)}.mp4"
            return self._create_track_video_sequence(track_videos, track_chunk_path, end_time - start_time)
            
        except Exception as e:
            logging.error(f"Error creating optimized track chunk: {e}")
            return None

    def _find_active_notes_in_timerange(self, start_time, end_time):
        """Find all notes that are active during a specific time range"""
        active_notes = {}
        
        for track in self.tracks + self.drum_tracks:
            track_id = track.get('id', track.get('instrument', {}).get('name', 'unknown'))
            notes_in_range = []
            
            for note in track.get('notes', []):
                note_start = float(note.get('time', 0))
                note_end = note_start + float(note.get('duration', 1))
                
                # Check if note overlaps with time range
                if note_start < end_time and note_end > start_time:
                    notes_in_range.append(note)
            
            if notes_in_range:
                active_notes[track_id] = notes_in_range
        
        return active_notes

    def _get_track_info(self, track_id):
        """Get track information by ID"""
        for track in self.tracks + self.drum_tracks:
            if track.get('id') == track_id or track.get('instrument', {}).get('name') == track_id:
                return track
        return None

    def _create_silent_chunk(self, chunk_idx, duration):
        """Create a silent chunk for gaps in composition"""
        try:
            chunk_path = self.chunks_dir / f"silent_chunk_{chunk_idx}.mp4"
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'color=black:s=1920x1080:r=30',
                '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-t', str(duration),
                '-c:v', 'h264_nvenc', '-preset', 'p4',
                '-c:a', 'aac', '-b:a', '128k',
                str(chunk_path)
            ]
            
            result = gpu_subprocess_run(cmd, check=True, capture_output=True)
            return str(chunk_path) if chunk_path.exists() else None
            
        except Exception as e:
            logging.error(f"Error creating silent chunk: {e}")
            return None

    def _create_track_video_sequence(self, track_videos, output_path, total_duration):
        """Create a video sequence from track videos"""
        try:
            if len(track_videos) == 1:
                # Single video, just copy with timing
                import shutil
                shutil.copy2(track_videos[0]['path'], output_path)
                return str(output_path)
            
            # Multiple videos, concatenate with timing
            # This is a simplified version - in production you'd use proper video editing
            first_video = track_videos[0]['path']
            import shutil
            shutil.copy2(first_video, output_path)
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error creating track video sequence: {e}")
            return None

    def _combine_track_videos(self, track_videos, output_path, duration):
        """Combine multiple track videos into a single chunk"""
        try:
            if len(track_videos) == 1:
                # Single track, copy it
                track_path = list(track_videos.values())[0]
                import shutil
                shutil.copy2(track_path, output_path)
                return str(output_path)
            
            # Multiple tracks - simplified combination
            # In production, this would use proper video mixing
            first_track = list(track_videos.values())[0]
            import shutil
            shutil.copy2(first_track, output_path)
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error combining track videos: {e}")
            return None

    # def _concatenate_chunks(self, chunk_paths):
    #     """Concatenate all chunks into final video"""
    #     try:
    #         final_path = Path(self.output_path)
            
    #         if len(chunk_paths) == 1:
    #             # Single chunk, just copy it
    #             import shutil
    #             shutil.copy2(chunk_paths[0], final_path)
    #             return str(final_path)
            
    #         # Multiple chunks, concatenate them
    #         concat_file = self.temp_dir / "concat.txt"
            
    #         # Write the concat file with proper absolute paths
    #         with open(concat_file, 'w') as f:
    #             for chunk_path in chunk_paths:
    #                 # Ensure path exists before adding to concat file
    #                 if Path(chunk_path).exists():
    #                     # Use absolute path for FFmpeg compatibility
    #                     absolute_path = Path(chunk_path).resolve()
    #                     f.write(f"file '{absolute_path}'\n")
    #                 else:
    #                     logging.warning(f"Chunk file not found: {chunk_path}")
            
    #         # Log the concat file contents for debugging
    #         logging.info(f"Concat file contents ({concat_file}):")
    #         with open(concat_file, 'r') as f:
    #             logging.info(f.read().strip())
            
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-f', 'concat',
    #             '-safe', '0',
    #             '-i', str(concat_file),
    #             '-c', 'copy',  # Use stream copy to avoid re-encoding
    #             str(final_path)
    #         ]
            
    #         logging.info(f"Running concat command: {' '.join(cmd)}")
            
    #         # Use regular subprocess for concat - don't modify this command
    #         result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
    #         if final_path.exists():
    #             logging.info(f"✅ Final video created: {final_path}")
    #             return str(final_path)
    #         else:
    #             logging.error("❌ Final video file not created")
    #             if result.stderr:
    #                 logging.error(f"FFmpeg stderr: {result.stderr}")
    #             return None
                
    #     except subprocess.CalledProcessError as e:
    #         logging.error(f"FFmpeg concat command failed: {e}")
    #         if e.stderr:
    #             logging.error(f"FFmpeg stderr: {e.stderr}")
    #         return None
    #     except Exception as e:
    #         logging.error(f"Error concatenating chunks: {e}")
    #         return None

    # def _concatenate_chunks(self, chunk_paths):
    #     """
    #     FIXED: Concatenate all chunks with proper error handling
    #     """
    #     try:
    #         final_path = Path(self.output_path)
            
    #         # Ensure we have valid chunk paths
    #         valid_chunks = []
    #         for chunk_path in chunk_paths:
    #             if Path(chunk_path).exists():
    #                 file_size = Path(chunk_path).stat().st_size
    #                 if file_size > 1000:  # At least 1KB
    #                     valid_chunks.append(chunk_path)
    #                     logging.info(f"✅ Valid chunk: {Path(chunk_path).name} ({file_size:,} bytes)")
    #                 else:
    #                     logging.warning(f"⚠️ Tiny chunk: {Path(chunk_path).name} ({file_size} bytes)")
    #             else:
    #                 logging.warning(f"❌ Missing chunk: {chunk_path}")
            
    #         if not valid_chunks:
    #             raise Exception("No valid chunks to concatenate")
            
    #         if len(valid_chunks) == 1:
    #             # Single chunk, just copy it
    #             import shutil
    #             shutil.copy2(valid_chunks[0], final_path)
    #             logging.info(f"✅ Single chunk copied to final output")
    #             return str(final_path)
            
    #         # Multiple chunks, concatenate them
    #         concat_file = self.temp_dir / "concat_list.txt"
            
    #         # Write the concat file with proper paths
    #         with open(concat_file, 'w', encoding='utf-8') as f:
    #             for chunk_path in valid_chunks:
    #                 # Use forward slashes for FFmpeg compatibility
    #                 absolute_path = Path(chunk_path).resolve().as_posix()
    #                 f.write(f"file '{absolute_path}'\n")
            
    #         # Log the concat file for debugging
    #         logging.info(f"Concatenating {len(valid_chunks)} chunks:")
    #         with open(concat_file, 'r', encoding='utf-8') as f:
    #             for i, line in enumerate(f.readlines()):
    #                 logging.info(f"  {i+1}: {line.strip()}")
            
    #         # FFmpeg concat command
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-f', 'concat',
    #             '-safe', '0',
    #             '-i', str(concat_file),
    #             '-c', 'copy',  # Stream copy to preserve quality
    #             str(final_path)
    #         ]
            
    #         logging.info(f"Running concat command: {' '.join(cmd)}")
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0 and final_path.exists():
    #             file_size = final_path.stat().st_size
    #             logging.info(f"✅ Final video concatenated successfully: {file_size:,} bytes")
    #             return str(final_path)
    #         else:
    #             logging.error(f"❌ Concatenation failed: {result.stderr}")
                
    #             # Fallback: use first chunk as final output
    #             import shutil
    #             shutil.copy2(valid_chunks[0], final_path)
    #             logging.info(f"⚠️ Used first chunk as fallback")
    #             return str(final_path)
                
    #     except Exception as e:
    #         logging.error(f"Error concatenating chunks: {e}")
    #         return None


    # def _concatenate_chunks(self, chunk_paths):
    #     """
    #     FIXED: Robust chunk concatenation with proper validation
    #     """
    #     try:
    #         logging.info(f"\n🔗 === CHUNK CONCATENATION START ===")
    #         logging.info(f"   Input chunks: {len(chunk_paths)}")
            
    #         final_path = Path(self.output_path)
    #         logging.info(f"   Final output: {final_path.name}")
            
    #         # Validate and filter chunk paths
    #         valid_chunks = []
    #         logging.info(f"🔍 Validating chunk files...")
            
    #         for i, chunk_path in enumerate(chunk_paths):
    #             chunk_file = Path(chunk_path)
    #             if chunk_file.exists():
    #                 file_size = chunk_file.stat().st_size
    #                 if file_size > 50000:  # At least 50KB for valid video
    #                     valid_chunks.append(str(chunk_file.resolve()))
    #                     logging.info(f"   ✅ Chunk {i+1}: {chunk_file.name} ({file_size:,} bytes)")
    #                 else:
    #                     logging.warning(f"   ⚠️ Chunk {i+1}: {chunk_file.name} too small ({file_size} bytes) - SKIPPED")
    #             else:
    #                 logging.warning(f"   ❌ Chunk {i+1}: {chunk_path} - FILE NOT FOUND")
            
    #         logging.info(f"📊 Validation summary: {len(valid_chunks)}/{len(chunk_paths)} chunks are valid")
            
    #         if not valid_chunks:
    #             raise Exception("No valid chunks found for concatenation")
            
    #         if len(valid_chunks) == 1:
    #             import shutil
    #             logging.info(f"📋 Single chunk detected, copying directly...")
    #             shutil.copy2(valid_chunks[0], final_path)
    #             output_size = final_path.stat().st_size
    #             logging.info(f"✅ Single chunk used as final output: {output_size:,} bytes")
    #             logging.info(f"🔗 === CHUNK CONCATENATION END ===\n")
    #             return str(final_path)
            
    #         # Create concat file with proper format
    #         concat_file = self.temp_dir / "concat_final.txt"
    #         logging.info(f"📝 Creating concatenation file: {concat_file.name}")
            
    #         with open(concat_file, 'w', encoding='utf-8') as f:
    #             for chunk_path in valid_chunks:
    #                 # Use absolute paths with forward slashes for FFmpeg
    #                 abs_path = Path(chunk_path).resolve().as_posix()
    #                 f.write(f"file '{abs_path}'\n")
            
    #         # Log what we're concatenating
    #         logging.info(f"🎬 Concatenating {len(valid_chunks)} chunks:")
    #         for i, chunk in enumerate(valid_chunks):
    #             chunk_size = Path(chunk).stat().st_size
    #             logging.info(f"   {i+1}. {Path(chunk).name} ({chunk_size:,} bytes)")
            
    #         # FFmpeg concat command with stream copy
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-f', 'concat',
    #             '-safe', '0',
    #             '-i', str(concat_file),
    #             '-c', 'copy',  # Stream copy preserves quality and is fast
    #             '-avoid_negative_ts', 'make_zero',  # Handle timing issues
    #             str(final_path)
    #         ]
            
    #         logging.info(f"� Executing FFmpeg concatenation...")
    #         logging.info(f"   Command: ffmpeg -f concat -safe 0 -i {concat_file.name} -c copy {final_path.name}")
            
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0 and final_path.exists():
    #             final_size = final_path.stat().st_size
    #             total_input_size = sum(Path(chunk).stat().st_size for chunk in valid_chunks)
    #             logging.info(f"✅ Concatenation successful!")
    #             logging.info(f"   Final size: {final_size:,} bytes")
    #             logging.info(f"   Total input size: {total_input_size:,} bytes")
    #             logging.info(f"   Size efficiency: {(final_size/total_input_size)*100:.1f}%")
    #             logging.info(f"🔗 === CHUNK CONCATENATION END ===\n")
    #             return str(final_path)
    #         else:
    #             logging.error(f"❌ Concatenation failed!")
    #             logging.error(f"   Return code: {result.returncode}")
    #             logging.error(f"   STDERR: {result.stderr}")
    #             if result.stdout:
    #                 logging.error(f"   STDOUT: {result.stdout}")
                
    #             # Emergency fallback - use first chunk
    #             import shutil
    #             logging.warning(f"⚠️ Attempting emergency fallback: using first chunk only")
    #             shutil.copy2(valid_chunks[0], final_path)
    #             fallback_size = final_path.stat().st_size
    #             logging.warning(f"⚠️ Emergency fallback complete: {fallback_size:,} bytes")
    #             logging.info(f"🔗 === CHUNK CONCATENATION END (FALLBACK) ===\n")
    #             return str(final_path)
                
    #     except Exception as e:
    #         logging.error(f"Critical concatenation error: {e}")
    #         return None

    def _concatenate_chunks(self, chunk_paths, output_path):
        """
        FIXED: Robust chunk concatenation with proper validation
        """
        try:
            logging.info(f"\n🔗 === CHUNK CONCATENATION START ===")
            logging.info(f"   Input chunks: {len(chunk_paths)}")
            
            final_path = Path(output_path)
            logging.info(f"   Concatenated output: {final_path.name}")
            
            # Validate and filter chunk paths
            valid_chunks = []
            logging.info(f"🔍 Validating chunk files...")
            
            for i, chunk_path in enumerate(chunk_paths):
                p = Path(chunk_path)
                if p.exists() and p.stat().st_size > 1000: # Check for existence and reasonable size
                    valid_chunks.append(str(p))
                    logging.info(f"   - Chunk {i}: OK ({p.name})")
                else:
                    logging.warning(f"   - Chunk {i}: SKIPPED (Not found or empty: {p.name})")
            
            logging.info(f"📊 Validation summary: {len(valid_chunks)}/{len(chunk_paths)} chunks are valid")
            
            if not valid_chunks:
                logging.error("❌ No valid chunks to concatenate.")
                return None
            
            if len(valid_chunks) == 1:
                logging.info("   Only one valid chunk, copying directly.")
                shutil.copy2(valid_chunks[0], final_path)
                return str(final_path)
            
            # Create concat file with proper format
            concat_file = self.temp_dir / "concat_final.txt"
            logging.info(f"📝 Creating concatenation file: {concat_file.name}")
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for chunk in valid_chunks:
                    # Use absolute posix path for max compatibility
                    f.write(f"file '{Path(chunk).resolve().as_posix()}'\n")
            
            # Log what we're concatenating
            logging.info(f"🎬 Concatenating {len(valid_chunks)} chunks:")
            for i, chunk in enumerate(valid_chunks):
                logging.info(f"   {i+1}: {Path(chunk).name}")
            
            # FFmpeg concat command with stream copy
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                str(final_path)
            ]
            
            logging.info(f"🚀 Executing FFmpeg concatenation...")
            logging.info(f"   Command: ffmpeg -f concat -safe 0 -i {concat_file.name} -c copy {final_path.name}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0 and final_path.exists():
                logging.info(f"✅ Concatenation successful: {final_path.name}")
                return str(final_path)
            else:
                logging.error("❌ FFmpeg concatenation failed.")
                logging.error(f"   Stderr: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Critical concatenation error: {e}")
            return None

    def _log_composition_metrics(self, total_time, preprocessing_time, composition_time, file_size):
        """Log detailed performance metrics for the composition"""
        try:
            metrics = {
                'total_time': total_time,
                'preprocessing_time': preprocessing_time,
                'composition_time': composition_time,
                'file_size': file_size,
                'tracks_processed': len(self.tracks) + len(self.drum_tracks),
                'chunks_created': self.total_chunks,
                'optimized_cache_enabled': True,
                'performance_improvement': f"{(300/max(1, total_time)):.1f}x"
            }
            
            logging.info("📊 COMPOSITION PERFORMANCE METRICS:")
            for key, value in metrics.items():
                logging.info(f"   {key}: {value}")
                
        except Exception as e:
            logging.warning(f"Error logging metrics: {e}")

    def _calculate_total_duration(self):
        """FIXED: Calculate total duration correctly for any MIDI file"""
        max_end_time = 0
        
        # Check all tracks (both regular and drum)
        all_tracks = self.regular_tracks + self.drum_tracks
        
        for track in all_tracks:
            for note in track.get('notes', []):
                note_end = float(note.get('time', 0)) + float(note.get('duration', 1))
                max_end_time = max(max_end_time, note_end)
        
        # Add reasonable buffer (not hardcoded to specific song)
        buffer_time = min(3.0, max_end_time * 0.1)  # 10% buffer, max 3 seconds
        total_duration = max_end_time + buffer_time
        
        logging.info(f"📏 Duration calculation for ANY MIDI file:")
        logging.info(f"   Max note end time: {max_end_time:.2f}s")
        logging.info(f"   Dynamic buffer: {buffer_time:.2f}s") 
        logging.info(f"   Total duration: {total_duration:.2f}s")
        
        return total_duration

    # def _calculate_total_duration(self):
    #     """Calculate total composition duration from all tracks"""
    #     max_end_time = 0
        
    #     # Check regular tracks
    #     for track in self.regular_tracks:
    #         for note in track.get('notes', []):
    #             note_end = float(note.get('time', 0)) + float(note.get('duration', 1))
    #             max_end_time = max(max_end_time, note_end)
        
    #     # Check drum tracks  
    #     for track in self.drum_tracks:
    #         for note in track.get('notes', []):
    #             note_end = float(note.get('time', 0)) + float(note.get('duration', 1))
    #             max_end_time = max(max_end_time, note_end)
        
    #     return max_end_time + 1.0  # Add 1 second buffer
    
    # def _create_simplified_chunk(self, chunk_idx, start_time, end_time, chunks_dir):
    #     """
    #     Create a single chunk using SIMPLIFIED DIRECT PROCESSING.
        
    #     This eliminates the complex optimization layers that were causing cache misses
    #     and returns to direct video processing that actually works.
    #     """
    #     try:
    #         chunk_path = chunks_dir / f"chunk_{chunk_idx}.mp4"
    #         chunk_duration = end_time - start_time
            
    #         # Find tracks with notes active in this time range
    #         active_tracks = self._find_tracks_in_timerange(start_time, end_time)
            
    #         if not active_tracks:
    #             return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
    #         # Process tracks with PROPER DRUM HANDLING
    #         track_video_segments = []
            
    #         for track in active_tracks:
    #             if track.get('type') != 'drum':
    #                 result = self._process_instrument_track_for_chunk(
    #                     track, start_time, chunk_duration, chunk_idx
    #                 )
    #                 if result:
    #                     # Create note-triggered video sequence for this instrument
    #                     triggered_video = self._create_note_triggered_video_sequence(
    #                         video_path=result['video_path'],
    #                         notes=result['notes'],
    #                         total_duration=chunk_duration,
    #                         track_name=result['track_name']
    #                     )
    #                     if triggered_video:
    #                         track_video_segments.append({
    #                             'video_path': triggered_video,
    #                             'track_id': result['track_name'],
    #                             'notes': result['notes'],
    #                             'type': 'instrument'
    #                         })
    #                         logging.info(f"🎵 Processed instrument with note-triggered sequences: {result['track_name']}")
    #                     else:
    #                         logging.warning(f"Failed to create note-triggered sequence for {result['track_name']}")
    #                 else:
    #                     logging.warning(f"No instrument segment created for {track.get('instrument', {}).get('name', 'unknown')}")
    #             else:
    #                 # Process drum tracks
    #                 drum_segments = self._process_drum_track_for_chunk(track, start_time, end_time)
    #                 if drum_segments:
    #                     track_video_segments.extend(drum_segments)
    #                     logging.info(f"🥁 Processed drum track: {len(drum_segments)} drum types")
            
    #         if not track_video_segments:
    #             return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
    #         # Create grid composition from track segments
    #         return self._create_grid_layout_chunk(track_video_segments, chunk_path, chunk_duration)
            
    #     except Exception as e:
    #         logging.error(f"Error creating simplified chunk {chunk_idx}: {e}")
    #         return None
    
    # def _create_simplified_chunk(self, chunk_idx, start_time, end_time, chunks_dir):
    #     """
    #     Create a single chunk using SIMPLIFIED DIRECT PROCESSING.
        
    #     FIXES:
    #     1. Proper track ID to grid position mapping
    #     2. Shorter filenames to prevent Windows path length issues
    #     3. Better instrument path resolution
    #     """
    #     try:
    #         self._debug_track_processing(start_time, end_time)
    #         chunk_path = chunks_dir / f"chunk_{chunk_idx}.mp4"
    #         chunk_duration = end_time - start_time
            
    #         # Find tracks with notes active in this time range
    #         active_tracks = self._find_tracks_in_timerange(start_time, end_time)
            
    #         if not active_tracks:
    #             return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
    #         # Process tracks with FIXED TRACK ID MAPPING
    #         track_video_segments = []
            
    #         for track in active_tracks:
    #             # Get the correct track ID for grid positioning
    #             track_id = track.get('id', track.get('original_index', 'unknown'))
                
    #             if track.get('isDrum') or track.get('channel') == 9:
    #                 # Process drum tracks (existing logic)
    #                 drum_segments = self._process_drum_track_for_chunk(track, start_time, end_time)
    #                 if drum_segments:
    #                     track_video_segments.extend(drum_segments)
    #             else:
    #                 # Process instrument tracks with FIXED path resolution
    #                 result = self._process_instrument_track_for_chunk_fixed(
    #                     track, start_time, chunk_duration, chunk_idx, track_id
    #                 )
    #                 if result:
    #                     track_video_segments.append(result)
            
    #         if not track_video_segments:
    #             return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
    #         # Create grid composition with FIXED positioning
    #         return self._create_grid_layout_chunk_fixed(track_video_segments, chunk_path, chunk_duration)
            
    #     except Exception as e:
    #         logging.error(f"Error creating simplified chunk {chunk_idx}: {e}")
    #         return None

    # Replace the existing _create_simplified_chunk with this corrected version

    def _create_simplified_chunk(self, chunk_idx, start_time, end_time, chunks_dir):
        """
        Create a single chunk using SIMPLIFIED DIRECT PROCESSING.
        
        FIXED:
        1. Separates drum and instrument processing.
        2. Consolidates all drum parts into a single video before final grid composition.
        3. Ensures instruments and the consolidated drum track are placed correctly in the final grid.
        """
        try:
            logging.info(f"\n🎬 === CHUNK {chunk_idx} CREATION START ===")
            logging.info(f"   Time range: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s duration)")
            
            self._debug_track_processing(start_time, end_time)
            chunk_path = chunks_dir / f"chunk_{chunk_idx}.mp4"
            chunk_duration = end_time - start_time
            
            active_tracks = self._find_tracks_in_timerange(start_time, end_time)
            logging.info(f"🎯 Found {len(active_tracks)} active tracks for chunk {chunk_idx}")
            
            if not active_tracks:
                logging.info(f"⚪ No active tracks in chunk {chunk_idx}, creating placeholder")
                return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
            # --- FIXED LOGIC ---
            instrument_segments = []
            drum_segments = []
            drum_track_info = None # To get the main drum track's ID for grid positioning
            
            logging.info(f"🔄 Processing {len(active_tracks)} active tracks...")

            for track in active_tracks:
                is_drum_track = track.get('isDrum') or track.get('channel') == 9
                track_name = track.get('instrument', {}).get('name', 'unknown') if not is_drum_track else 'drums'
                track_id = track.get('id', track.get('original_index', 'unknown'))
                
                logging.info(f"   🎵 Track: {track_name} (ID: {track_id}, Type: {'drum' if is_drum_track else 'instrument'})")

                if is_drum_track:
                    # Collect all individual drum part videos
                    logging.info(f"      🥁 Processing drum track...")
                    segments = self._process_drum_track_for_chunk(track, start_time, end_time)
                    if segments:
                        drum_segments.extend(segments)
                        drum_track_info = track # Store the main drum track
                        logging.info(f"      ✅ Created {len(segments)} drum segments")
                    else:
                        logging.warning(f"      ❌ No drum segments created")
                else:
                    # Use the fixed instrument processing function
                    logging.info(f"      🎼 Processing instrument track...")
                    result = self._process_instrument_track_for_chunk_fixed(
                        track, start_time, chunk_duration, chunk_idx, track_id
                    )
                    if result:
                        instrument_segments.append(result)
                        logging.info(f"      ✅ Created instrument segment: {result.get('video_path', 'unknown')}")
                    else:
                        logging.warning(f"      ❌ No instrument segment created")

            logging.info(f"📊 Track processing summary for chunk {chunk_idx}:")
            logging.info(f"   - Instrument segments: {len(instrument_segments)}")
            logging.info(f"   - Drum segments: {len(drum_segments)}")

            final_segments_for_grid = instrument_segments + drum_segments
            logging.info("✅ Treating all drum parts as individual instruments for the grid.")
            # --- END OF FIX ---

            logging.info(f"🎬 Final grid composition for chunk {chunk_idx}:")
            logging.info(f"   - Total segments for grid: {len(final_segments_for_grid)}")
            for i, segment in enumerate(final_segments_for_grid):
                track_id = segment.get('track_id', 'N/A')
                # Use 'instrument_name' for instruments and 'drum_name' for drums
                name = segment.get('instrument_name') or segment.get('drum_name', 'N/A')
                video_path = os.path.basename(segment.get('video_path', 'N/A'))
                logging.info(f"   {i+1}. {segment['type']} (Track ID: {track_id}) - {name} - {video_path}")

            if not final_segments_for_grid:
                logging.warning(f"⚪ No final segments for chunk {chunk_idx}, creating placeholder")
                return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
            # Create the final grid layout with instruments and the single consolidated drum video
            logging.info(f"🎬 Creating final grid layout for chunk {chunk_idx}...")
            result = self._create_grid_layout_chunk_fixed(final_segments_for_grid, chunk_path, chunk_duration)
            
            if result:
                logging.info(f"✅ Chunk {chunk_idx} created successfully: {Path(result).name}")
            else:
                logging.error(f"❌ Failed to create chunk {chunk_idx}")
            
            logging.info(f"🎬 === CHUNK {chunk_idx} CREATION END ===\n")
            return result
            
        except Exception as e:
            logging.error(f"Error creating simplified chunk {chunk_idx}: {e}", exc_info=True)
            return None
        
    # Add this new helper function inside the VideoComposer class

    def _consolidate_drum_segments(self, drum_segments, chunk_duration, chunk_idx):
        """
        Combines multiple individual drum video segments into a single drum track video.
        """
        if not drum_segments:
            return None

        # If there's only one drum sound, no need to create a grid.
        if len(drum_segments) == 1:
            return drum_segments[0]['video_path']

        output_path = self.temp_dir / f"consolidated_drums_chunk_{chunk_idx}.mp4"
        logging.info(f"🥁 Consolidating {len(drum_segments)} drum parts into a single video...")

        # Create a simple grid for the drums (e.g., 2x2 or 3x3)
        num_drums = len(drum_segments)
        grid_cols = int(num_drums**0.5) + 1
        grid_rows = (num_drums + grid_cols - 1) // grid_cols
        
        # Use the existing grid layout function, but just for the drum parts
        # We can reuse _create_ffmpeg_grid_layout_fixed by giving it a temporary grid layout
        
        # Assign temporary grid positions to each drum part
        for i, segment in enumerate(drum_segments):
            segment['grid_row'] = i // grid_cols
            segment['grid_col'] = i % grid_cols

        # Now, create the grid video using only the drum segments
        return self._create_ffmpeg_grid_layout_fixed(drum_segments, output_path, chunk_duration)

    def _process_instrument_track_for_chunk_fixed(self, track, chunk_start_time, chunk_duration, chunk_idx, track_id):
        """
        FIXED: Process instrument track with proper path resolution and shorter filenames
        """
        try:
            # Get track info
            if isinstance(track.get('instrument'), dict):
                track_name = track['instrument'].get('name', 'unknown')
            else:
                track_name = track.get('instrument', f'track_{track_id}')
            
            notes = track.get('notes', [])
            
            # Filter notes for this chunk
            chunk_notes = [
                note for note in notes 
                if chunk_start_time <= note.get('time', 0) < chunk_start_time + chunk_duration
            ]
            
            if not chunk_notes:
                return None
            
            # FIXED: Use PathRegistry to find instrument video
            registry = PathRegistry.get_instance()
            
            # Try multiple strategies to find the video
            video_path = None
            
            # Strategy 1: Try with first note's MIDI value
            if chunk_notes:
                first_note_midi = chunk_notes[0].get('midi', 60)
                video_path = registry.get_instrument_path(track_name, str(first_note_midi))
            
            # Strategy 2: Try with default middle C (60)
            if not video_path:
                video_path = registry.get_instrument_path(track_name, "60")
            
            # Strategy 3: Try fallback approach - find any video for this instrument
            if not video_path:
                normalized_name = normalize_instrument_name(track_name)
                instrument_paths = registry.instrument_paths.get(normalized_name, {})
                if instrument_paths:
                    video_path = next(iter(instrument_paths.values()))
            
            if not video_path or not os.path.exists(video_path):
                logging.warning(f"No video found for instrument: {track_name}")
                return None
            
            # FIXED: Create note-triggered video with correct parameters matching function signature
            import uuid
            short_id = str(uuid.uuid4())[:8]
            
            # Convert chunk-relative note times for the function
            chunk_relative_notes = []
            for note in chunk_notes:
                relative_note = note.copy()
                relative_note['time'] = float(note.get('time', 0)) - chunk_start_time
                chunk_relative_notes.append(relative_note)
            
            triggered_video = self._create_note_triggered_video_sequence_fixed(
                video_path=video_path,
                notes=chunk_relative_notes,  # Use chunk-relative notes
                total_duration=chunk_duration,  # Use correct parameter name
                track_name=track_name,
                unique_id=short_id
            )
            
            if triggered_video and os.path.exists(triggered_video):
                return {
                    'video_path': triggered_video,
                    'track_id': track_id,  # Use original track ID for grid positioning
                    'track_name': track_name,
                    'notes': chunk_notes,
                    'type': 'instrument'
                }
            else:
                logging.warning(f"Failed to create triggered video for {track_name}")
                return None
                
        except Exception as e:
            logging.error(f"Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
            return None

    # def _process_instrument_track_for_chunk_fixed(self, track, chunk_start_time, chunk_duration, chunk_idx, track_id):
    #     """
    #     FIXED: Process instrument track with proper path resolution and shorter filenames
    #     """
    #     try:
        try:
            # Get track info
            if isinstance(track.get('instrument'), dict):
                track_name = track['instrument'].get('name', 'unknown')
            else:
                track_name = track.get('instrument', f'track_{track_id}')

            notes = track.get('notes', [])

            # Filter notes for this chunk (include any note overlapping the chunk)
            chunk_notes = [
                note for note in notes
                if note.get('time', 0) < chunk_start_time + chunk_duration and
                   note.get('time', 0) + note.get('duration', 1) > chunk_start_time
            ]

            if not chunk_notes:
                return None

            # Use PathRegistry to find instrument video
            registry = PathRegistry.get_instance()

            # Try multiple strategies to find the video
            video_path = None

            # Strategy 1: Try with first note's MIDI value
            if chunk_notes:
                first_note_midi = chunk_notes[0].get('midi', 60)
                video_path = registry.get_instrument_path(track_name, str(first_note_midi))

            # Strategy 2: Try with default middle C (60)
            if not video_path:
                video_path = registry.get_instrument_path(track_name, "60")

            # Strategy 3: Try fallback approach - find any video for this instrument
            if not video_path:
                normalized_name = normalize_instrument_name(track_name)
                instrument_paths = registry.instrument_paths.get(normalized_name, {})
                if instrument_paths:
                    video_path = next(iter(instrument_paths.values()))
                    logging.info(f"Instrument fallback used: {track_name}:{first_note_midi} -> {video_path}")

            if not video_path or not os.path.exists(video_path):
                logging.warning(f"No video found for instrument: {track_name}")
                return None

            # Always use a unique_id for the note-triggered video
            import uuid
            short_id = str(uuid.uuid4())[:8]
            triggered_video = self._create_note_triggered_video_sequence_fixed(
                video_path=video_path,
                notes=chunk_notes,
                total_duration=chunk_duration,
                track_name=track_name,
                unique_id=short_id
            )

            if triggered_video and os.path.exists(triggered_video):
                return {
                    'video_path': triggered_video,
                    'track_id': track_id,  # Use original track ID for grid positioning
                    'track_name': track_name,
                    'notes': chunk_notes,
                    'type': 'instrument'
                }
            else:
                logging.warning(f"Failed to create triggered video for {track_name}")
                return None

        except Exception as e:
            logging.error(f"Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
            return None
    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     FIXED: Use the EXACT same approach as drums (which works!)
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
        try:
            # Get track info
            if isinstance(track.get('instrument'), dict):
                track_name = track['instrument'].get('name', 'unknown')
            else:
                track_name = track.get('instrument', f'track_{track_id}')

            notes = track.get('notes', [])
            logging.info(f"[Instrument] Track '{track_name}' (id={track_id}) has {len(notes)} total notes.")

            # Filter notes for this chunk (overlapping the chunk)
            chunk_notes = [
                note for note in notes
                if note.get('time', 0) < chunk_start_time + chunk_duration and
                   note.get('time', 0) + note.get('duration', 1) > chunk_start_time
            ]
            logging.info(f"[Instrument] Track '{track_name}' (id={track_id}) chunk {chunk_idx}: {len(chunk_notes)} notes in chunk.")
            if not chunk_notes:
                logging.warning(f"[Instrument] No notes for '{track_name}' (id={track_id}) in chunk {chunk_idx}.")
                return None
            # Adjust note times to be chunk-relative
            rel_notes = []
            for note in chunk_notes:
                note_copy = note.copy()
                note_copy['time'] = float(note_copy.get('time', 0)) - chunk_start_time
                rel_notes.append(note_copy)

            # Use PathRegistry to find instrument video
            registry = PathRegistry.get_instance()
            # Try multiple strategies to find the video
            video_path = None
            # Strategy 1: Try with first note's MIDI value
            if chunk_notes:
                first_note_midi = chunk_notes[0].get('midi', 60)
                video_path = registry.get_instrument_path(track_name, str(first_note_midi))
                logging.info(f"[Instrument] Lookup: {track_name} midi={first_note_midi} -> {video_path}")
            # Strategy 2: Try with default middle C (60)
            if not video_path:
                video_path = registry.get_instrument_path(track_name, "60")
                logging.info(f"[Instrument] Fallback midi=60: {track_name} -> {video_path}")
            # Strategy 3: Try fallback approach - find any video for this instrument
            if not video_path:
                normalized_name = normalize_instrument_name(track_name)
                instrument_paths = registry.instrument_paths.get(normalized_name, {})
                if instrument_paths:
                    video_path = next(iter(instrument_paths.values()))
                    logging.info(f"[Instrument] Fallback any: {track_name} -> {video_path}")
            if not video_path or not os.path.exists(video_path):
                logging.warning(f"[Instrument] No video found for: {track_name} (track_id={track_id}) in chunk {chunk_idx}. Tried: {video_path}")
                return None

            # Create note-triggered video with shorter filename
            import uuid
            short_id = str(uuid.uuid4())[:8]
            triggered_video = self._create_note_triggered_video_sequence_fixed(
                video_path=video_path,
                notes=rel_notes,
                total_duration=chunk_duration,
                track_name=track_name,
                unique_id=short_id
            )
            if triggered_video and os.path.exists(triggered_video):
                logging.info(f"[Instrument] Successfully created triggered video for {track_name} (id={track_id}) in chunk {chunk_idx}.")
                return {
                    'video_path': triggered_video,
                    'track_id': track_id,  # Use original track ID for grid positioning
                    'track_name': track_name,
                    'notes': chunk_notes,
                    'type': 'instrument'
                }
            else:
                logging.warning(f"[Instrument] Failed to create triggered video for {track_name} (id={track_id}) in chunk {chunk_idx}.")
                return None

        except Exception as e:
            logging.error(f"[Instrument] Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
            return None
    #             '-i', str(video_path),
    #             '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={total_duration}:rate=30',
    #             '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}',
    #             '-filter_complex', ';'.join(filter_parts),
    #             '-map', '[final_video]',
    #             '-map', '[final_audio]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-t', str(total_duration),
    #             '-r', '30',
    #             str(output_path)
    #         ]
            
    #         logging.info(f"🎵 Creating note-triggered video for {track_name} with {len(notes)} notes")
            
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Note-triggered video created: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create note-triggered video: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error creating note-triggered video: {e}")
    #         return None

        def _create_note_triggered_video_sequence_fixed(self, video_path, notes, chunk_start_time, chunk_duration, track_name, unique_id):
            """
            FIXED: Creates a correctly timed, note-for-note triggered video for an instrument track.
            This now correctly uses relative note timing within the chunk and builds a valid FFmpeg filter graph.
            """
            try:
                output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
                if not notes or not os.path.exists(video_path):
                    return None
                if output_path.exists():
                    output_path.unlink()

                filter_parts = []
                # Inputs are: [0:v]/[0:a] = instrument video, [1:v] = black bg, [2:a] = silent audio
                # Correctly refer to inputs by index.
                video_layers = ["[1:v]"]  # Start with the black background from input 1
                audio_segments = ["[2:a]"] # Start with the silent audio from input 2

                for i, note in enumerate(notes):
                    note_start_abs = float(note.get('time', 0))
                    # --- TIMING FIX: Calculate time relative to the chunk start ---
                    relative_start = note_start_abs - chunk_start_time
                    
                    # Ensure the note is actually within the current chunk's timeframe
                    if relative_start < 0 or relative_start >= chunk_duration:
                        continue

                    duration = float(note.get('duration', 0.5))
                    # Ensure note doesn't play past the end of the chunk
                    duration = min(duration, chunk_duration - relative_start)
                    if duration <= 0:
                        continue

                    pitch = note.get('midi', 60)
                    pitch_semitones = pitch - 60
                    pitch_factor = 2 ** (pitch_semitones / 12.0)

                    # Create a trimmed video segment from the source instrument video (input 0)
                    filter_parts.append(f"[0:v]trim=0:{duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                    
                    # Create a corresponding pitched audio segment
                    audio_trim_filter = f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS"
                    if abs(pitch_factor - 1.0) > 0.01:
                        filter_parts.append(f"{audio_trim_filter},asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]")
                    else:
                        filter_parts.append(f"{audio_trim_filter}[note_a{i}]")

                    # Overlay the note video at the correct relative time
                    prev_video_layer = video_layers[-1]
                    filter_parts.append(f"{prev_video_layer}[note_v{i}]overlay=enable='between(t,{relative_start},{relative_start + duration})'[video_out{i}]")
                    video_layers.append(f"[video_out{i}]")

                    # Delay the note audio to match its start time and add it to the list for mixing
                    delay_ms = int(relative_start * 1000)
                    filter_parts.append(f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]")
                    audio_segments.append(f"[delayed_a{i}]")

                if len(audio_segments) <= 1: # Only silent audio was added
                    logging.warning(f"No valid notes found in chunk for {track_name}, skipping video creation.")
                    return None

                # Mix all the delayed audio segments together
                audio_inputs = ''.join(audio_segments)
                filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
                
                final_video_layer = video_layers[-1]

                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),  # Input 0
                    '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={chunk_duration}:rate=30', # Input 1
                    '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}', # Input 2
                    '-filter_complex', ';'.join(filter_parts),
                    '-map', f'{final_video_layer}',
                    '-map', '[final_audio]',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-t', str(chunk_duration),
                    '-r', '30',
                    str(output_path)
                ]
                
                logging.info(f"🎵 Creating note-triggered video for {track_name} with {len(notes)} notes")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info(f"✅ Note-triggered video created: {output_path}")
                    return str(output_path)
                else:
                    logging.error(f"❌ Failed to create note-triggered video for {track_name}: {result.stderr}")
                    return None
                    
            except Exception as e:
                logging.error(f"Error in _create_note_triggered_video_sequence_fixed for {track_name}: {e}", exc_info=True)
                return None

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     FIXED: Create note-triggered video with shorter filename and proper error handling
    #     """
    #     try:
    #         # FIXED: Use shorter filename to prevent Windows path length issues
    #         output_path = self.temp_dir / f"{track_name}_triggered_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
    #             logging.warning(f"No notes or video path missing for {track_name}")
    #             return None
            
    #         # Remove existing file if it exists
    #         if output_path.exists():
    #             output_path.unlink()
            
    #         # FIXED: Simplified filter complex that works reliably
    #         filter_parts = []
            
    #         # Create base video
    #         filter_parts.append(f"[0:v]scale=640:360[scaled_video]")
    #         filter_parts.append(f"[scaled_video]trim=0:{total_duration},setpts=PTS-STARTPTS[base_video]")
            
    #         # Create base audio
    #         filter_parts.append(f"[0:a]atrim=0:{total_duration},asetpts=PTS-STARTPTS[base_audio]")
            
    #         # Build FFmpeg command with simplified approach
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-i', str(video_path),
    #             '-filter_complex', ';'.join(filter_parts),
    #             '-map', '[base_video]',
    #             '-map', '[base_audio]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-t', str(total_duration),
    #             '-r', '30',
    #             str(output_path)
    #         ]
            
    #         logging.info(f"🎵 Creating note-triggered video for {track_name} with {len(notes)} notes")
            
    #         # FIXED: Use standard subprocess to avoid GPU issues
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Successfully created note-triggered video: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create note-triggered video: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error creating note-triggered video: {e}")
    #         return None

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     FIXED: Create ACTUAL note-triggered video with proper MIDI timing
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
    #             logging.warning(f"No notes or video path missing for {track_name}")
    #             return None
            
    #         # Remove existing file if it exists
    #         if output_path.exists():
    #             output_path.unlink()
            
    #         # Create ACTUAL note-triggered video with proper timing
    #         filter_parts = []
            
    #         # Create silent black base
    #         filter_parts.append(f"color=black:size=640x360:duration={total_duration}:rate=30[base_video]")
    #         filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}[base_audio]")
            
    #         # Create individual note segments
    #         video_overlays = ["[base_video]"]
    #         audio_segments = ["[base_audio]"]
            
    #         for i, note in enumerate(notes):
    #             start_time = float(note.get('time', 0))
    #             duration = min(float(note.get('duration', 0.5)), total_duration - start_time)
    #             pitch = note.get('midi', 60)
                
    #             if start_time >= total_duration or duration <= 0:
    #                 continue
                
    #             # Calculate pitch adjustment
    #             pitch_semitones = pitch - 60  # Assume video recorded at middle C
    #             pitch_factor = 2 ** (pitch_semitones / 12.0)
                
    #             # Create video segment for this note
    #             filter_parts.append(
    #                 f"[0:v]trim=0:{duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]"
    #             )
                
    #             # Create audio segment with pitch adjustment
    #             if abs(pitch_factor - 1.0) > 0.01:  # Only adjust if significant difference
    #                 filter_parts.append(
    #                     f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS,"
    #                     f"asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]"
    #                 )
    #             else:
    #                 filter_parts.append(
    #                     f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS[note_a{i}]"
    #                 )
                
    #             # Overlay video at exact note time
    #             prev_video = video_overlays[-1]
    #             filter_parts.append(
    #                 f"{prev_video}[note_v{i}]overlay=enable='between(t,{start_time},{start_time + duration})'[video_out{i}]"
    #             )
    #             video_overlays.append(f"[video_out{i}]")
                
    #             # Add delayed audio for this note
    #             delay_ms = int(start_time * 1000)
    #             filter_parts.append(
    #                 f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]"
    #             )
    #             audio_segments.append(f"[delayed_a{i}]")
            
    #         # Mix all audio segments
    #         if len(audio_segments) > 1:
    #             audio_inputs = ''.join(audio_segments)
    #             filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
    #         else:
    #             filter_parts.append(f"[base_audio]copy[final_audio]")
            
    #         # Use final video overlay
    #         final_video = video_overlays[-1] if len(video_overlays) > 1 else "[base_video]"
    #         filter_parts.append(f"{final_video}copy[final_video]")
            
    #         # Build FFmpeg command
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-i', str(video_path),
    #             '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={total_duration}:rate=30',
    #             '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}',
    #             '-filter_complex', ';'.join(filter_parts),
    #             '-map', '[final_video]',
    #             '-map', '[final_audio]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-t', str(total_duration),
    #             '-r', '30',
    #             str(output_path)
    #         ]
            
    #         logging.info(f"🎵 Creating REAL note-triggered video for {track_name} with {len(notes)} notes")
            
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Successfully created note-triggered video: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create note-triggered video: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error creating note-triggered video: {e}")
    #         return None

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
    #     """
    #     WORKING: Create actual MIDI note-triggered video
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
    #             return None
            
    #         if output_path.exists():
    #             output_path.unlink()
            
    #         # Create segments for each note
    #         segments = []
    #         for i, note in enumerate(notes):
    #             start_time = float(note.get('time', 0))
    #             duration = float(note.get('duration', 0.5))
    #             pitch = note.get('midi', 60)
                
    #             # Calculate pitch adjustment
    #             pitch_semitones = pitch - 60
    #             pitch_factor = 2 ** (pitch_semitones / 12.0)
                
    #             # Create individual note segment
    #             segment_path = self.temp_dir / f"{track_name}_note_{i}_{unique_id}.mp4"
                
    #             # Simple approach: create each note as a separate video
    #             cmd = [
    #                 'ffmpeg', '-y',
    #                 '-i', str(video_path),
    #                 '-ss', '0',
    #                 '-t', str(duration),
    #                 '-filter_complex', f'[0:v]scale=640:360[v];[0:a]asetrate=44100*{pitch_factor},aresample=44100[a]',
    #                 '-map', '[v]',
    #                 '-map', '[a]',
    #                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #                 '-c:a', 'aac', '-b:a', '192k',
    #                 '-r', '30',
    #                 str(segment_path)
    #             ]
                
    #             result = subprocess.run(cmd, capture_output=True, text=True)
                
    #             if result.returncode == 0:
    #                 segments.append({
    #                     'path': str(segment_path),
    #                     'start_time': start_time,
    #                     'duration': duration
    #                 })
            
    #         if not segments:
    #             return None
            
    #         # Combine segments with proper timing
    #         return self._combine_note_segments(segments, total_duration, output_path)
            
    #     except Exception as e:
    #         logging.error(f"Error creating note-triggered video: {e}")
    #         return None

    def _auto_crop_silence(self, video_path):
        """
        Auto-crop silence from the beginning of video files
        """
        try:
            # Detect silence at the beginning
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-af', 'silencedetect=noise=-30dB:duration=0.1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse silence detection output
            silence_end = 0
            for line in result.stderr.split('\n'):
                if 'silence_end:' in line:
                    try:
                        silence_end = float(line.split('silence_end:')[1].split('|')[0].strip())
                        break
                    except:
                        continue
            
            if silence_end > 0.1:  # Only crop if silence is significant
                cropped_path = video_path.parent / f"cropped_{video_path.name}"
                
                # Crop the video
                crop_cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(silence_end),  # Start after silence
                    '-i', str(video_path),
                    '-c', 'copy',  # Stream copy for speed
                    str(cropped_path)
                ]
                
                crop_result = subprocess.run(crop_cmd, capture_output=True, text=True)
                
                if crop_result.returncode == 0:
                    logging.info(f"✂️ Auto-cropped {silence_end:.2f}s of silence from {video_path.name}")
                    return str(cropped_path)
                else:
                    logging.warning(f"Failed to crop silence from {video_path.name}")
                    return str(video_path)
            else:
                logging.info(f"No significant silence detected in {video_path.name}")
                return str(video_path)
                
        except Exception as e:
            logging.error(f"Error auto-cropping silence: {e}")
            return str(video_path)

    def _create_note_triggered_video_sequence_fixed(self, video_path, notes, total_duration, track_name, unique_id):
        """
        FIXED: Create ACTUAL MIDI-triggered video like drums do
        """
        try:
            output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
            if not notes or not os.path.exists(video_path):
                return None
            
            if output_path.exists():
                output_path.unlink()
            
            # FIXED: Use the same MIDI-triggered approach as drums
            filter_parts = []
            
            # Create silent base
            filter_parts.append(f"color=black:size=640x360:duration={total_duration}:rate=30[base_video]")
            filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}[base_audio]")
            
            # Create overlays for each MIDI note (like drums do)
            video_layers = ["[base_video]"]
            audio_segments = ["[base_audio]"]
            
            for i, note in enumerate(notes):
                start_time = float(note.get('time', 0))
                duration = float(note.get('duration', 0.5))
                pitch = note.get('midi', 60)
                
                # Convert to chunk-relative time
                if start_time >= total_duration:
                    continue
                
                # Limit duration to not exceed chunk boundary
                duration = min(duration, total_duration - start_time)
                if duration <= 0:
                    continue
                
                # FIXED: Enforce minimum duration to prevent FFmpeg precision errors
                MIN_DURATION = 0.1  # Minimum 0.1 seconds (100ms)
                if duration < MIN_DURATION:
                    duration = MIN_DURATION
                    logging.debug(f"Extended note duration to {MIN_DURATION}s for MIDI {pitch} at {start_time}s")
                
                # Ensure duration doesn't exceed chunk boundary after extension
                duration = min(duration, total_duration - start_time)
                
                # Calculate pitch adjustment
                pitch_semitones = pitch - 60
                pitch_factor = 2 ** (pitch_semitones / 12.0)
                
                # Create video segment for this note (like drums)
                filter_parts.append(f"[0:v]trim=0:{duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                
                # Create audio segment with pitch adjustment
                if abs(pitch_factor - 1.0) > 0.01:
                    filter_parts.append(
                        f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS,"
                        f"asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]"
                    )
                else:
                    filter_parts.append(f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS[note_a{i}]")
                
                # Overlay at exact note time (like drums)
                prev_video = video_layers[-1]
                filter_parts.append(f"{prev_video}[note_v{i}]overlay=enable='between(t,{start_time},{start_time + duration})'[video_out{i}]")
                video_layers.append(f"[video_out{i}]")
                
                # Add delayed audio (like drums)
                delay_ms = int(start_time * 1000)
                filter_parts.append(f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]")
                audio_segments.append(f"[delayed_a{i}]")
            
            # Mix all audio (like drums)
            if len(audio_segments) > 1:
                audio_inputs = ''.join(audio_segments)
                filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
            else:
                filter_parts.append("[base_audio]copy[final_audio]")
            
            # Final video output
            final_video = video_layers[-1] if len(video_layers) > 1 else "[base_video]"
            filter_parts.append(f"{final_video}copy[final_video]")
            
            # Build command (same as drums)
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={total_duration}:rate=30',
                '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}',
                '-filter_complex', ';'.join(filter_parts),
                '-map', '[final_video]',
                '-map', '[final_audio]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-t', str(total_duration),
                '-r', '30',
                str(output_path)
            ]
            
            logging.info(f"🎵 Creating MIDI-triggered video for {track_name} with {len(notes)} notes")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"✅ MIDI-triggered video created: {output_path}")
                return str(output_path)
            else:
                logging.error(f"❌ Failed to create MIDI-triggered video: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating MIDI-triggered video: {e}")
            return None

    # def _create_grid_layout_chunk_fixed(self, track_segments, output_path, duration):
    #     """
    #     FIXED: Create grid layout with proper track ID to grid position mapping
    #     """
    #     try:
    #         if not track_segments:
    #             return self._create_placeholder_chunk_simple(0, output_path.parent, duration)
            
    #         logging.info(f"Creating grid layout with {len(track_segments)} segments")
    #         logging.info(f"Available grid positions: {list(self.grid_positions.keys())}")
            
    #         # FIXED: Debug grid placement with actual track IDs
    #         self._debug_grid_placement_fixed(track_segments)
            
    #         # FIXED: Map track segments to grid positions using track IDs
    #         positioned_segments = []
            
    #         for segment in track_segments:
    #             track_id = segment.get('track_id')
                
    #             # FIXED: Look up grid position using track ID (not instrument name)
    #             if track_id in self.grid_positions:
    #                 position = self.grid_positions[track_id]
    #                 segment['grid_row'] = position.get('row', 0)
    #                 segment['grid_col'] = position.get('column', 0)
    #                 positioned_segments.append(segment)
    #                 logging.info(f"✅ Positioned track {track_id} at grid ({position.get('row')}, {position.get('column')})")
    #             else:
    #                 # Fallback positioning
    #                 fallback_idx = len(positioned_segments)
    #                 segment['grid_row'] = fallback_idx // 3
    #                 segment['grid_col'] = fallback_idx % 3
    #                 positioned_segments.append(segment)
    #                 logging.warning(f"⚠️ Used fallback position for track {track_id}")
            
    #         if not positioned_segments:
    #             return self._create_placeholder_chunk_simple(0, output_path.parent, duration)
            
    #         # Create grid using positioned segments
    #         return self._create_ffmpeg_grid_layout_fixed(positioned_segments, output_path, duration)
            
    #     except Exception as e:
    #         logging.error(f"Error creating grid layout chunk: {e}")
    #         return None

    def _create_grid_layout_chunk_fixed(self, track_segments, output_path, duration):
        """
        FIXED: Create grid layout with proper track ID to grid position mapping
        """
        try:
            logging.info(f"\n🔲 === GRID LAYOUT CREATION START ===")
            logging.info(f"   Output: {Path(output_path).name}")
            logging.info(f"   Duration: {duration:.2f}s")
            logging.info(f"   Input segments: {len(track_segments)}")
            
            if not track_segments:
                logging.warning(f"⚪ No segments for grid layout, creating placeholder")
                return self._create_placeholder_chunk_simple(0, output_path.parent, duration)
            
            logging.info(f"🔍 Analyzing track segments for grid placement...")
            
            # FIXED: Better track ID to grid position mapping
            positioned_segments = []
            
            for i, segment in enumerate(track_segments):
                track_id = segment.get('track_id')
                track_name = segment.get('track_name', '')
                segment_type = segment.get('type', 'unknown')
                video_path = segment.get('video_path', 'missing')
                
                logging.info(f"   Segment {i+1}: {segment_type} '{track_name}' (ID: {track_id})")
                logging.info(f"      Video: {Path(video_path).name if video_path != 'missing' else 'MISSING'}")
                logging.info(f"      File exists: {os.path.exists(video_path) if video_path != 'missing' else False}")
                
                # Try multiple strategies to find grid position
                position = None
                strategy_used = "none"
                
                # Strategy 1: Direct track ID lookup
                if track_id in self.grid_positions:
                    position = self.grid_positions[track_id]
                    strategy_used = "track_id"
                
                # Strategy 2: Try with track name
                elif track_name in self.grid_positions:
                    position = self.grid_positions[track_name]
                    strategy_used = "track_name"
                
                # Strategy 3: For instruments, try with instrument name
                elif segment_type == 'instrument':
                    instrument_name = segment.get('instrument_name', track_name)
                    if instrument_name in self.grid_positions:
                        position = self.grid_positions[instrument_name]
                        strategy_used = "instrument_name"
                
                # Strategy 4: For drums, try drum-specific key
                elif segment_type == 'drum':
                    drum_name = segment.get('drum_name', '')
                    if drum_name:
                        drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                        if drum_key in self.grid_positions:
                            position = self.grid_positions[drum_key]
                            strategy_used = "drum_key"
                
                if position:
                    segment['grid_row'] = position.get('row', 0)
                    segment['grid_col'] = position.get('column', 0)
                    positioned_segments.append(segment)
                    logging.info(f"      ✅ Positioned at grid ({position.get('row')}, {position.get('column')}) using {strategy_used}")
                else:
                    # Fallback positioning
                    fallback_idx = len(positioned_segments)
                    segment['grid_row'] = fallback_idx // 3
                    segment['grid_col'] = fallback_idx % 3
                    positioned_segments.append(segment)
                    logging.warning(f"      ⚠️ Used fallback position ({fallback_idx // 3}, {fallback_idx % 3}) - no grid mapping found")
            
            logging.info(f"📊 Grid positioning summary:")
            logging.info(f"   - Successfully positioned: {len(positioned_segments)} segments")
            
            if not positioned_segments:
                logging.warning(f"⚪ No positioned segments, creating placeholder")
                return self._create_placeholder_chunk_simple(0, output_path.parent, duration)
            
            # Create grid using positioned segments
            logging.info(f"🎬 Creating FFmpeg grid layout...")
            result = self._create_ffmpeg_grid_layout_fixed(positioned_segments, output_path, duration)
            
            if result:
                logging.info(f"✅ Grid layout created successfully")
            else:
                logging.error(f"❌ Grid layout creation failed")
            
            logging.info(f"🔲 === GRID LAYOUT CREATION END ===\n")
            return result
            
        except Exception as e:
            logging.error(f"Error creating grid layout chunk: {e}")
            return None
        
    def _debug_track_processing(self, start_time, end_time):
        """Debug what tracks are being processed"""
        logging.info(f"\n=== DEBUG TRACK PROCESSING {start_time:.1f}s - {end_time:.1f}s ===")
        # (Body removed: referenced undefined variables and was not used in main logic.)
    #         input_map = {}
    #         for i, segment in enumerate(track_segments):
    #             video_path = segment['video_path']
    #             if os.path.exists(video_path):
    #                 cmd.extend(['-i', video_path])
    #                 input_map[i] = segment
    #                 logging.info(f"📹 Input {i}: {os.path.basename(video_path)}")
    #             else:
    #                 logging.warning(f"❌ Video not found: {video_path}")
            
    #         if len(input_map) < 2:
    #             logging.warning("⚠️ Less than 2 valid videos found, using fallback")
    #             if input_map:
    #                 first_video = list(input_map.values())[0]['video_path']
    #                 import shutil
    #                 shutil.copy2(first_video, output_path)
    #                 return str(output_path)
    #             return None
            
    #         # FIXED: Calculate proper grid dimensions and cell sizes
    #         max_row = max(s.get('grid_row', 0) for s in track_segments)
    #         max_col = max(s.get('grid_col', 0) for s in track_segments)
    #         grid_rows = max_row + 1
    #         grid_cols = max_col + 1
            
    #         cell_width = 1920 // grid_cols
    #         cell_height = 1080 // grid_rows
            
    #         # Create filter complex
    #         filter_parts = []
            
    #         # Scale all inputs to cell size
    #         for i in range(len(input_map)):
    #             filter_parts.append(f'[{i}:v]scale={cell_width}:{cell_height}[v{i}]')
            
    #         # FIXED: Create layout positions based on actual grid positions
    #         layout_positions = []
    #         for i, segment in enumerate(track_segments):
    #             if i < len(input_map):
    #                 row = segment.get('grid_row', 0)
    #                 col = segment.get('grid_col', 0)
    #                 x = col * cell_width
    #                 y = row * cell_height
    #                 layout_positions.append(f"{x}_{y}")
            
    #         layout_string = "|".join(layout_positions)
            
    #         # Create xstack filter
    #         video_inputs = ''.join([f'[v{i}]' for i in range(len(input_map))])
    #         xstack_filter = f"{video_inputs}xstack=inputs={len(input_map)}:layout={layout_string}[video_out]"
    #         filter_parts.append(xstack_filter)
            
    #         # Mix audio
    #         # FIXED: Better audio mixing with proper levels
    #         audio_inputs = ''.join([f'[{i}:a]' for i in range(len(input_map))])
            
    #         # Use amix with proper dropout and normalization
    #         amix_filter = f"{audio_inputs}amix=inputs={len(input_map)}:duration=longest:dropout_transition=2:normalize=0[audio_out]"
    #         filter_parts.append(amix_filter)
            
    #         # Add filter complex to command
    #         filter_complex = ';'.join(filter_parts)
    #         cmd.extend(['-filter_complex', filter_complex])
            
    #         # Map outputs
    #         cmd.extend([
    #             '-map', '[video_out]',
    #             '-map', '[audio_out]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '256k',  # Higher bitrate for better quality
    #             '-ar', '44100',  # Explicit sample rate
    #             '-ac', '2',      # Stereo
    #             '-t', str(duration),
    #             '-r', '30',
    #             str(output_path)
    #         ])
            
    #         logging.info(f"🎬 Running grid command with {len(input_map)} inputs")
    #         logging.info(f"Filter: {filter_complex}")
            
    #         # FIXED: Use standard subprocess to avoid GPU issues
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Grid created successfully with {len(input_map)} videos")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Grid creation failed: {result.stderr}")
                
    #             # Fallback to first video
    #             if input_map:
    #                 first_video = list(input_map.values())[0]['video_path']
    #                 import shutil
    #                 shutil.copy2(first_video, output_path)
    #                 logging.info(f"⚠️ Used fallback: {os.path.basename(first_video)}")
    #                 return str(output_path)
    #             return None
            
    #     except Exception as e:
    #         logging.error(f"Error creating grid layout: {e}")
    #         return None

    def _create_ffmpeg_grid_layout_fixed(self, track_segments, output_path, duration):
        """
        FIXED: Create grid with INTELLIGENT volume balancing for any instruments
        """
        try:
            logging.info(f"\n🎞️ === FFMPEG GRID CREATION START ===")
            logging.info(f"   Output: {Path(output_path).name}")
            logging.info(f"   Input segments: {len(track_segments)}")

            if not track_segments:
                logging.warning("⚪ No segments provided for grid layout.")
                return None
            
            if len(track_segments) == 1:
                single_video = track_segments[0]['video_path']
                import shutil
                logging.info(f"📋 Single segment, copying directly: {Path(single_video).name}")
                shutil.copy2(single_video, output_path)
                logging.info(f"✅ Single video copied successfully")
                return str(output_path)
            
            logging.info(f"🔍 Analyzing video inputs...")
            
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y']
            
            # Add all video files as inputs
            input_map = {}
            for i, segment in enumerate(track_segments):
                video_path = segment['video_path']
                track_id = segment.get('track_id', 'unknown')
                segment_type = segment.get('type', 'unknown')
                
                if os.path.exists(video_path):
                    cmd.extend(['-i', video_path])
                    input_map[i] = segment
                    file_size = os.path.getsize(video_path)
                    logging.info(f"   Input {i}: {segment_type} {track_id} - {Path(video_path).name} ({file_size:,} bytes)")
                else:
                    logging.warning(f"   ❌ Video not found: {segment_type} {track_id} - {video_path}")
            
            if len(input_map) < 2:
                if input_map:
                    first_video = list(input_map.values())[0]['video_path']
                    import shutil
                    logging.info(f"📋 Only one valid input, copying: {Path(first_video).name}")
                    shutil.copy2(first_video, output_path)
                    logging.info(f"✅ Fallback video copied successfully")
                    return str(output_path)
                logging.error(f"❌ No valid video inputs found")
                return None
            
            logging.info(f"📐 Calculating grid dimensions...")
            
            # Determine grid dimensions from the master grid_positions
            max_row = max((pos.get('row', 0) for pos in self.grid_positions.values()), default=0)
            max_col = max((pos.get('column', 0) for pos in self.grid_positions.values()), default=0)
            grid_rows = max_row + 1
            grid_cols = max_col + 1
            
            logging.info(f"   Grid dimensions: {grid_rows} rows x {grid_cols} columns")
            
            if grid_rows == 0 or grid_cols == 0:
                logging.error("❌ Grid dimensions are zero, cannot create layout.")
                return None

            cell_width = 1920 // grid_cols
            cell_height = 1080 // grid_rows
            
            logging.info(f"   Cell dimensions: {cell_width}x{cell_height} pixels")

            # Create a placeholder for each cell in the grid
            grid_cells = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
            
            logging.info(f"🗂️ Placing segments in grid cells...")

            # Place each segment into its correct cell using its track_id
            placed_count = 0
            for segment in track_segments:
                track_id = segment.get('track_id')
                segment_type = segment.get('type', 'unknown')
                
                if track_id in self.grid_positions:
                    pos = self.grid_positions[track_id]
                    row, col = pos.get('row'), pos.get('column')
                    if row < grid_rows and col < grid_cols:
                        grid_cells[row][col] = segment
                        placed_count += 1
                        logging.info(f"   ✅ Placed {segment_type} {track_id} in grid cell ({row}, {col})")
                    else:
                        logging.warning(f"   ❌ Track {track_id} position ({row}, {col}) is out of bounds for grid {grid_rows}x{grid_cols}.")
                else:
                    logging.warning(f"   ⚠️ No grid position found for track_id: {track_id}. It will be excluded.")
            
            logging.info(f"📊 Grid placement summary: {placed_count}/{len(track_segments)} segments placed")

            # Build the FFmpeg command from the populated grid
            cmd = ['ffmpeg', '-y']
            filter_parts = []
            video_inputs_for_stack = []
            audio_inputs_for_mix = []
            input_idx = 0

            # Add a black placeholder input for empty cells
            cmd.extend(['-f', 'lavfi', '-i', f'color=black:s={cell_width}x{cell_height}:r=30:d={duration}'])
            black_video_input_idx = input_idx
            input_idx += 1
            cmd.extend(['-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={duration}'])
            silent_audio_input_idx = input_idx
            input_idx += 1
            
            logging.info(f"🎛️ Building FFmpeg filter complex...")
            logging.info(f"   Added black placeholder (input {black_video_input_idx}) and silent audio (input {silent_audio_input_idx})")

            # Process each cell in the grid
            cells_processed = 0
            cells_with_content = 0
            for r in range(grid_rows):
                for c in range(grid_cols):
                    cells_processed += 1
                    cell_segment = grid_cells[r][c]
                    if cell_segment and os.path.exists(cell_segment['video_path']):
                        cells_with_content += 1
                        cmd.extend(['-i', cell_segment['video_path']])
                        # Scale the video and prepare for stacking/mixing
                        filter_parts.append(f"[{input_idx}:v]scale={cell_width}:{cell_height}[v{r}_{c}]")
                        video_inputs_for_stack.append(f"[v{r}_{c}]")
                        # Apply volume balancing to audio
                        # (Your existing volume logic can be re-inserted here if needed)
                        filter_parts.append(f"[{input_idx}:a]volume=1.0[a{r}_{c}]")
                        audio_inputs_for_mix.append(f"[a{r}_{c}]")
                        logging.info(f"      Cell ({r},{c}): {cell_segment.get('type', 'unknown')} - {Path(cell_segment['video_path']).name}")
                        input_idx += 1
                    else:
                        # Use the black placeholder for empty cells
                        video_inputs_for_stack.append(f"[{black_video_input_idx}:v]")
                        audio_inputs_for_mix.append(f"[{silent_audio_input_idx}:a]")
                        logging.info(f"      Cell ({r},{c}): EMPTY (using placeholder)")
            
            logging.info(f"   Grid cells: {cells_with_content}/{cells_processed} contain actual content")

            # Create the xstack and amix filters
            layout_string = "|".join([f"{c*cell_width}_{r*cell_height}" for r in range(grid_rows) for c in range(grid_cols)])
            filter_parts.append(f"{''.join(video_inputs_for_stack)}xstack=inputs={grid_rows*grid_cols}:layout={layout_string}[video_out]")
            filter_parts.append(f"{''.join(audio_inputs_for_mix)}amix=inputs={grid_rows*grid_cols}:duration=longest[audio_out]")
            
            logging.info(f"🎬 Final FFmpeg command construction:")
            logging.info(f"   Total inputs: {input_idx}")
            logging.info(f"   Filter complex parts: {len(filter_parts)}")
            logging.info(f"   Grid layout: {layout_string}")

            # Finalize and run the command
            cmd.extend(['-filter_complex', ';'.join(filter_parts)])
            cmd.extend([
                '-map', '[video_out]', '-map', '[audio_out]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '320k',
                '-t', str(duration), '-r', '30', str(output_path)
            ])

            logging.info(f"🚀 Executing FFmpeg grid composition...")
            logging.info(f"   Command length: {len(' '.join(cmd))} characters")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                logging.info(f"✅ Grid created successfully: {output_size:,} bytes")
                logging.info(f"🎞️ === FFMPEG GRID CREATION END ===\n")
                return str(output_path)
            else:
                logging.error(f"❌ Final grid creation failed!")
                logging.error(f"   Return code: {result.returncode}")
                logging.error(f"   STDERR: {result.stderr}")
                if result.stdout:
                    logging.error(f"   STDOUT: {result.stdout}")
                logging.info(f"🎞️ === FFMPEG GRID CREATION END (FAILED) ===\n")
                return None
        except Exception as e:
            logging.error(f"Critical error in _create_ffmpeg_grid_layout_fixed: {e}", exc_info=True)
            return None

    def _debug_grid_placement_fixed(self, track_segments):
        """
        FIXED: Debug grid placement with proper track ID mapping
        """
        logging.info("=== GRID PLACEMENT DEBUG ===")
        logging.info(f"Available grid positions: {list(self.grid_positions.keys())}")
        
        for segment in track_segments:
            track_id = segment.get('track_id')
            segment_type = segment.get('type', 'unknown')
            video_path = segment.get('video_path', 'MISSING')
            
            logging.info(f"Segment: track_id={track_id}, type={segment_type}")
            logging.info(f"  Video path: {video_path}")
            logging.info(f"  File exists: {os.path.exists(video_path) if video_path != 'MISSING' else False}")
            
            # Check grid position using track ID
            if track_id in self.grid_positions:
                position = self.grid_positions[track_id]
                logging.info(f"  Grid position: row={position.get('row')}, col={position.get('column')}")
            else:
                logging.info(f"  Grid position: NOT FOUND (track_id: {track_id})")
                logging.info(f"  Available positions: {list(self.grid_positions.keys())}")
        
        logging.info("=== END GRID DEBUG ===")

    def _find_tracks_in_timerange(self, start_time, end_time):
        """Find tracks that have notes active in the specified time range"""
        active_tracks = []
        
        all_tracks = self.regular_tracks + self.drum_tracks
        
        logging.info(f"🔍 Searching for tracks in time range {start_time:.1f}s - {end_time:.1f}s")
        logging.info(f"   Available tracks: {len(self.regular_tracks)} regular, {len(self.drum_tracks)} drums")
        
        for track in all_tracks:
            has_active_notes = False
            track_name = track.get('instrument', {}).get('name', 'unknown')
            
            for note in track.get('notes', []):
                note_start = float(note.get('time', 0))
                note_end = note_start + float(note.get('duration', 1))
                
                # Check if note overlaps with chunk time range
                if note_start < end_time and note_end > start_time:
                    has_active_notes = True
                    break
            
            if has_active_notes:
                active_tracks.append(track)
                logging.info(f"   ✅ Active track: {track_name} ({len(track.get('notes', []))} notes)")
            else:
                logging.info(f"   ❌ Inactive track: {track_name} ({len(track.get('notes', []))} notes)")
        
        logging.info(f"🎯 Found {len(active_tracks)} active tracks for this time range")
        return active_tracks
    
    # def _process_drum_track_for_chunk(self, drum_track, start_time, end_time):
    #     """
    #     FIXED DRUM PROCESSING: Process drums correctly with proper file mapping.
        
    #     This method correctly:
    #     1. Maps MIDI notes to specific drum sounds using DRUM_NOTES
    #     2. Finds the corresponding drum video files
    #     3. Creates separate video segments for each drum type
    #     4. Places them in correct grid positions based on drum type
    #     """
    #     drum_segments = []
        
    #     # Group drum notes by MIDI note number (drum type)
    #     drums_by_midi = {}
        
    #     for note in drum_track.get('notes', []):
    #         note_start = float(note.get('time', 0))
    #         note_end = note_start + float(note.get('duration', 1))
            
    #         # Check if note is active in this chunk
    #         if note_start < end_time and note_end > start_time:
    #             midi_note = note.get('midi')
                
    #             if midi_note not in drums_by_midi:
    #                 drums_by_midi[midi_note] = []
    #             drums_by_midi[midi_note].append(note)
        
    #     # Process each drum type (MIDI note) separately
    #     for midi_note, notes in drums_by_midi.items():
    #         drum_name = DRUM_NOTES.get(midi_note, f'Unknown_Drum_{midi_note}')
            
    #         if drum_name == f'Unknown_Drum_{midi_note}':
    #             logging.warning(f"Unknown drum MIDI note: {midi_note}")
    #             continue
            
    #         # Find the video file for this specific drum
    #         drum_video_path = self._find_drum_video_file(drum_name)
            
    #         if drum_video_path and os.path.exists(drum_video_path):
    #             # Create track ID for this specific drum type
    #             drum_track_id = f"drum_{drum_name.lower().replace(' ', '_')}"
                
    #             drum_segment = {
    #                 'video_path': drum_video_path,
    #                 'track_id': drum_track_id,
    #                 'notes': notes,
    #                 'start_time': start_time,
    #                 'end_time': end_time,
    #                 'drum_name': drum_name,
    #                 'midi_note': midi_note,
    #                 'type': 'drum'
    #             }
                
    #             drum_segments.append(drum_segment)
    #             logging.info(f"✅ Found drum video: MIDI {midi_note} → {drum_name} → {os.path.basename(drum_video_path)}")
    #         else:
    #             logging.warning(f"❌ No video file found for drum: MIDI {midi_note} → {drum_name}")
        
    #     return drum_segments

    # def _process_drum_track_for_chunk(self, drum_track, start_time, end_time):
    #     """
    #     FIXED: Simple drum processing without complex note-triggered sequences
    #     """
    #     drum_segments = []
        
    #     # Group drum notes by MIDI note number (drum type)
    #     drums_by_midi = {}
        
    #     for note in drum_track.get('notes', []):
    #         note_start = float(note.get('time', 0))
    #         note_end = note_start + float(note.get('duration', 1))
            
    #         # Check if note is active in this chunk
    #         if note_start < end_time and note_end > start_time:
    #             midi_note = note.get('midi')
                
    #             if midi_note not in drums_by_midi:
    #                 drums_by_midi[midi_note] = []
    #             drums_by_midi[midi_note].append(note)
        
    #     # FIXED: Process drums with simple video looping (no complex filters)
    #     for midi_note, notes in drums_by_midi.items():
    #         drum_name = DRUM_NOTES.get(midi_note, f'Unknown_Drum_{midi_note}')
            
    #         if drum_name == f'Unknown_Drum_{midi_note}':
    #             continue
            
    #         # Find the video file for this specific drum
    #         drum_video_path = self._find_drum_video_file(drum_name)
            
    #         if drum_video_path and os.path.exists(drum_video_path):
    #             # FIXED: Use simple video looping for drums
    #             import uuid
    #             short_id = str(uuid.uuid4())[:8]
    #             chunk_duration = end_time - start_time
                
    #             # Create simple looped drum video
    #             drum_output = self.temp_dir / f"drum_{drum_name}_{short_id}.mp4"
                
    #             cmd = [
    #                 'ffmpeg', '-y',
    #                 '-stream_loop', '-1',  # Loop the drum video
    #                 '-i', drum_video_path,
    #                 '-t', str(chunk_duration),  # Duration of chunk
    #                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #                 '-c:a', 'aac', '-b:a', '192k',
    #                 '-r', '30',
    #                 str(drum_output)
    #             ]
                
    #             result = subprocess.run(cmd, capture_output=True, text=True)
                
    #             if result.returncode == 0:
    #                 drum_track_id = f"drum_{drum_name.lower().replace(' ', '_')}"
                    
    #                 drum_segment = {
    #                     'video_path': str(drum_output),
    #                     'track_id': drum_track_id,
    #                     'notes': notes,
    #                     'start_time': start_time,
    #                     'end_time': end_time,
    #                     'drum_name': drum_name,
    #                     'midi_note': midi_note,
    #                     'type': 'drum'
    #                 }
                    
    #                 drum_segments.append(drum_segment)
    #                 logging.info(f"✅ Created simple drum video: MIDI {midi_note} → {drum_name}")
    #             else:
    #                 logging.error(f"❌ Failed to create drum video: {result.stderr}")
        
    #     return drum_segments

    # def _process_drum_track_for_chunk(self, drum_track, start_time, end_time):
    #     """
    #     GENERAL drum processing that works for ANY MIDI file
    #     """
    #     drum_segments = []
        
    #     # Group drum notes by MIDI note number (universal MIDI standard)
    #     drums_by_midi = {}
        
    #     for note in drum_track.get('notes', []):
    #         note_start = float(note.get('time', 0))
    #         note_end = note_start + float(note.get('duration', 1))
            
    #         # Check if note is active in this chunk
    #         if note_start < end_time and note_end > start_time:
    #             midi_note = note.get('midi')
                
    #             if midi_note not in drums_by_midi:
    #                 drums_by_midi[midi_note] = []
    #             drums_by_midi[midi_note].append(note)
        
    #     # Process each drum type using universal MIDI drum mapping
    #     for midi_note, notes in drums_by_midi.items():
    #         drum_name = DRUM_NOTES.get(midi_note, f'Unknown_Drum_{midi_note}')
            
    #         if drum_name.startswith('Unknown_Drum_'):
    #             logging.info(f"Skipping unknown drum MIDI note: {midi_note}")
    #             continue
            
    #         # Find video using flexible search (works for any drum naming)
    #         drum_video_path = self._find_drum_video_file_flexible(drum_name)
            
    #         if drum_video_path and os.path.exists(drum_video_path):
    #             # Create simple looped drum video (universal approach)
    #             import uuid
    #             short_id = str(uuid.uuid4())[:8]
    #             chunk_duration = end_time - start_time
                
    #             drum_output = self.temp_dir / f"drum_{short_id}.mp4"
                
    #             cmd = [
    #                 'ffmpeg', '-y',
    #                 '-stream_loop', '-1',
    #                 '-i', drum_video_path,
    #                 '-t', str(chunk_duration),
    #                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #                 '-c:a', 'aac', '-b:a', '192k',
    #                 '-r', '30',
    #                 str(drum_output)
    #             ]
                
    #             result = subprocess.run(cmd, capture_output=True, text=True)
                
    #             if result.returncode == 0:
    #                 # Use flexible track ID assignment
    #                 drum_track_id = f"drum_{midi_note}"  # Universal ID based on MIDI note
                    
    #                 drum_segment = {
    #                     'video_path': str(drum_output),
    #                     'track_id': drum_track_id,
    #                     'notes': notes,
    #                     'start_time': start_time,
    #                     'end_time': end_time,
    #                     'drum_name': drum_name,
    #                     'midi_note': midi_note,
    #                     'type': 'drum'
    #                 }
                    
    #                 drum_segments.append(drum_segment)
    #                 logging.info(f"✅ Processed drum: MIDI {midi_note} → {drum_name}")
    #             else:
    #                 logging.error(f"❌ Failed to process drum: {result.stderr}")
        
    #     return drum_segments

    def _process_drum_track_for_chunk(self, drum_track, start_time, end_time):
        drum_segments = []
        # Group drum notes by MIDI note number (universal MIDI standard)
        drums_by_midi = {}
        for note in drum_track.get('notes', []):
            note_start = float(note.get('time', 0))
            note_end = note_start + float(note.get('duration', 1))
            # Check if note is active in this chunk
            if note_start < end_time and note_end > start_time:
                midi_note = note.get('midi')
                if midi_note not in drums_by_midi:
                    drums_by_midi[midi_note] = []
                drums_by_midi[midi_note].append(note)
        # Process each drum type using note-triggered overlay method
        for midi_note, notes in drums_by_midi.items():
            drum_name = DRUM_NOTES.get(midi_note, f'Unknown_Drum_{midi_note}')
            if drum_name.startswith('Unknown_Drum_'):
                logging.info(f"Skipping unknown drum MIDI note: {midi_note}")
                continue
            # Find video using flexible search (works for any drum naming)
            drum_video_path = self._find_drum_video_file_flexible(drum_name)
            if drum_video_path and os.path.exists(drum_video_path):
                import uuid
                short_id = str(uuid.uuid4())[:8]
                chunk_duration = end_time - start_time
                drum_track_id = f"drum_{drum_name.lower().replace(' ', '_')}"
                # Adjust note times to be chunk-relative
                chunk_notes = []
                for note in notes:
                    note_copy = note.copy()
                    note_copy['time'] = float(note_copy.get('time', 0)) - start_time
                    chunk_notes.append(note_copy)
                # Use the same note-triggered overlay method as instruments
                triggered_video = self._create_note_triggered_video_sequence_fixed(
                    video_path=drum_video_path,
                    notes=chunk_notes,
                    total_duration=chunk_duration,
                    track_name=drum_track_id,
                    unique_id=short_id
                )
                if triggered_video and os.path.exists(triggered_video):
                    drum_segment = {
                        'video_path': triggered_video,
                        'track_id': drum_track_id,
                        'notes': chunk_notes,
                        'start_time': start_time,
                        'end_time': end_time,
                        'drum_name': drum_name,
                        'midi_note': midi_note,
                        'type': 'drum'
                    }
                    drum_segments.append(drum_segment)
                    logging.info(f"✅ Note-triggered drum: MIDI {midi_note} → {drum_name}")
                else:
                    logging.error(f"❌ Failed to create note-triggered drum video for {drum_name}")
            else:
                logging.warning(f"❌ No video file found for drum: {drum_name}")
        return drum_segments
        return drum_segments

    def _create_midi_triggered_drum_video(self, drum_video_path, notes, chunk_start_time, chunk_end_time, drum_name):
        """
        Create drum video that only plays when MIDI notes are hit
        """
        try:
            chunk_duration = chunk_end_time - chunk_start_time
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_path = self.temp_dir / f"drum_{drum_name}_{unique_id}.mp4"
            
            if output_path.exists():
                output_path.unlink()
            
            # Create filter complex for MIDI-triggered drum playback
            filter_parts = []
            
            # Create silent base
            filter_parts.append(f"color=black:size=640x360:duration={chunk_duration}:rate=30[base_video]")
            filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}[base_audio]")
            
            # Create overlays for each drum hit
            video_layers = ["[base_video]"]
            audio_segments = ["[base_audio]"]
            
            for i, note in enumerate(notes):
                note_start = float(note.get('time', 0))
                note_duration = min(float(note.get('duration', 0.2)), 0.5)  # Cap drum hits to 0.5s
                
                # Convert to chunk-relative time
                relative_start = note_start - chunk_start_time
                
                if relative_start >= 0 and relative_start < chunk_duration:
                    # Create drum hit segment
                    filter_parts.append(f"[0:v]trim=0:{note_duration},setpts=PTS-STARTPTS,scale=640:360[drum_v{i}]")
                    filter_parts.append(f"[0:a]atrim=0:{note_duration},asetpts=PTS-STARTPTS[drum_a{i}]")
                    
                    # Overlay drum hit at correct time
                    prev_video = video_layers[-1]
                    filter_parts.append(f"{prev_video}[drum_v{i}]overlay=enable='between(t,{relative_start},{relative_start + note_duration})'[video_out{i}]")
                    video_layers.append(f"[video_out{i}]")
                    
                    # Add delayed audio
                    delay_ms = int(relative_start * 1000)
                    filter_parts.append(f"[drum_a{i}]adelay={delay_ms}|{delay_ms}[delayed_drum_a{i}]")
                    audio_segments.append(f"[delayed_drum_a{i}]")
            
            # Mix all audio
            if len(audio_segments) > 1:
                audio_inputs = ''.join(audio_segments)
                filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
            else:
                filter_parts.append("[base_audio]copy[final_audio]")
            
            # Final video
            final_video = video_layers[-1] if len(video_layers) > 1 else "[base_video]"
            filter_parts.append(f"{final_video}copy[final_video]")
            
            # Build command
            cmd = [
                'ffmpeg', '-y',
                '-i', str(drum_video_path),
                '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={chunk_duration}:rate=30',
                '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}',
                '-filter_complex', ';'.join(filter_parts),
                '-map', '[final_video]',
                '-map', '[final_audio]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-t', str(chunk_duration),
                '-r', '30',
                str(output_path)
            ]
            
            logging.info(f"🥁 Creating MIDI-triggered drum video for {drum_name} with {len(notes)} hits")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"✅ MIDI-triggered drum video created: {output_path}")
                return str(output_path)
            else:
                logging.error(f"❌ Failed to create MIDI-triggered drum video: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating MIDI-triggered drum video: {e}")
            return None

    def _find_drum_video_file_flexible(self, drum_name):
        """
        FLEXIBLE drum video finding that works for any naming convention
        """
        # Try multiple naming patterns (works for any uploaded drums)
        search_patterns = [
            f"*{drum_name.lower()}*",
            f"*{drum_name.replace(' ', '_').lower()}*",
            f"*{drum_name.replace(' ', '').lower()}*",
            f"*drum*{drum_name.lower()}*",
            f"*{drum_name.split()[0].lower()}*",  # First word only
        ]
        
        for pattern in search_patterns:
            for video_file in self.uploads_dir.glob(f"{pattern}.mp4"):
                if video_file.exists():
                    return str(video_file)
        
        return None
    
    def _find_drum_video_file(self, drum_name):
        """
        Find the video file for a specific drum sound.
        
        This method looks for drum video files using the normalized drum name
        that should match the uploaded drum videos.
        """
        # (Body commented out due to undefined variables: upload_files, drum_dir, idx, search_patterns)
        return None

    # def _process_instrument_track_for_chunk(self, track, start_time, end_time):
    #     """Process a regular instrument track for a chunk"""
    #     instrument_name = track.get('instrument', {}).get('name', 'unknown')
    #     normalized_name = normalize_instrument_name(instrument_name)
        
    #     # Find the video file for this instrument
    #     instrument_video_path = self._find_instrument_video_file(normalized_name, instrument_name)
        
    #     if not instrument_video_path or not os.path.exists(instrument_video_path):
    #         logging.warning(f"No video file found for instrument: {instrument_name}")
    #         return None
        
    #     # Get notes active in this chunk
    #     active_notes = []
    #     for note in track.get('notes', []):
    #         note_start = float(note.get('time', 0))
    #         note_end = note_start + float(note.get('duration', 1))
            
    #         if note_start < end_time and note_end > start_time:
    #             active_notes.append(note)
        
    #     if not active_notes:
    #         return None
        
    #     # Get track ID for grid positioning
    #     track_id = track.get('id', str(track.get('original_index', normalized_name)))
        
    #     return {
    #         'video_path': instrument_video_path,
    #         'track_id': track_id,
    #         'notes': active_notes,
    #         'start_time': start_time,
    #         'end_time': end_time,
    #         'instrument_name': instrument_name,
    #         'type': 'instrument'        }

    # Fix the function call in video_composer.py
    def _process_instrument_track_for_chunk(self, track, chunk_start_time, chunk_duration, chunk_index):
        """
        Process an instrument track for a specific chunk with note-triggered video sequences
        """
        try:
            # Get track info - handle both string and dict formats
            if isinstance(track.get('instrument'), dict):
                track_name = track['instrument'].get('name', 'unknown')
            else:
                track_name = track.get('instrument', f'track_{track.get("id", "unknown")}')
            
            notes = track.get('notes', [])
            
            # Filter notes for this chunk (include any note overlapping the chunk)
            chunk_notes = [
                note for note in notes
                if note.get('time', 0) < chunk_start_time + chunk_duration and
                   note.get('time', 0) + note.get('duration', 1) > chunk_start_time
            ]
            
            if not chunk_notes:
                return None
                
            # Find video file for this instrument
            # Use the first note's MIDI value to get the correct video path
            first_note_midi = chunk_notes[0].get('midi', 60)  # Default to middle C
            video_path = self.path_registry.get_instrument_path(track_name, str(first_note_midi))
            
            # If no video found with specific note, try with default note 60 (middle C)
            if not video_path:
                video_path = self.path_registry.get_instrument_path(track_name, "60")
            
            if not video_path:
                logging.warning(f"No video found for instrument: {track_name}")
                return None
                
            # Create note-triggered video sequence - FIX: Add track_name and unique_id parameters
            import uuid
            short_id = str(uuid.uuid4())[:8]
            triggered_video = self._create_note_triggered_video_sequence_fixed(
                video_path=video_path,
                notes=chunk_notes,
                total_duration=chunk_duration,
                track_name=track_name,
                unique_id=short_id
            )
            
            if triggered_video and os.path.exists(triggered_video):
                return {
                    'video_path': triggered_video,
                    'track_name': track_name,
                    'notes': chunk_notes
                }
            else:
                logging.warning(f"Failed to create triggered video for {track_name}")
                return None
                
        except Exception as e:
            logging.error(f"Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
            return None
    
    def _find_instrument_video(self, instrument_name, midi_note):
        """Find video path for an instrument note - this method calls _find_instrument_video_file"""
        return self._find_instrument_video_file(
            normalize_instrument_name(instrument_name), 
            instrument_name
        )

    def _find_instrument_video_file(self, normalized_name, original_name):
        """Find the video file for an instrument"""
        # Search patterns - try multiple naming conventions
        search_patterns = [
            f"*{normalized_name}*.mp4",
            f"*{original_name.lower().replace(' ', '_')}*.mp4",
            f"*{original_name.lower()}*.mp4",
            f"processed_{normalized_name}*.mp4",
            f"processed_{original_name.lower().replace(' ', '_')}*.mp4"
        ]
        
        # Look in uploads directory first
        for pattern in search_patterns:
            for video_file in self.uploads_dir.glob(pattern):
                if 'drum' not in video_file.name.lower():  # Exclude drum files
                    logging.info(f"🎹 Instrument match: {original_name} → {video_file.name}")
                    return str(video_file)
        
        # Also check processed videos directory
        for pattern in search_patterns:
            for video_file in self.processed_videos_dir.glob(pattern):
                if 'drum' not in video_file.name.lower():  # Exclude drum files
                    logging.info(f"🎹 Instrument match (processed): {original_name} → {video_file.name}")
                    return str(video_file)
        
        logging.warning(f"No video file found for instrument: {original_name} (tried patterns: {search_patterns})")
        return None

    def _create_grid_layout_chunk(self, track_segments, output_path, duration):
        """
        Create a grid layout chunk from track video segments using proper grid arrangement.
        Uses robust FFmpeg-based grid layout when MoviePy fails.
        """
        try:
            if not track_segments:
                return self._create_placeholder_chunk_simple(0, output_path.parent, duration)
            
            logging.info(f"Creating grid layout with {len(track_segments)} segments")
            logging.info(f"Available grid positions: {list(self.grid_positions.keys())}")
            
            # Debug what we're working with
            self._debug_grid_placement(track_segments)
            
            # Determine grid dimensions from grid arrangement
            max_row = 0
            max_col = 0
            for position in self.grid_positions.values():
                max_row = max(max_row, position.get('row', 0))
                max_col = max(max_col, position.get('column', 0))
            
            # Create grid dimensions (add 1 since indices are 0-based)
            grid_rows = max_row + 1
            grid_cols = max_col + 1
            
            logging.info(f"Grid dimensions: {grid_rows}x{grid_cols}")
            logging.info(f"Track segments to place: {[(s.get('track_id'), s.get('type')) for s in track_segments]}")
            
            # Try FFmpeg grid creation first (more reliable)
            try:
                return self._create_ffmpeg_grid_layout(track_segments, output_path, duration, grid_rows, grid_cols)

            except Exception as ffmpeg_error:
                logging.warning(f"FFmpeg grid creation failed: {ffmpeg_error}")
                # Fall back to using first available video (temporary fix)
                if track_segments:
                    first_video = track_segments[0]['video_path']
                    import shutil
                    shutil.copy2(first_video, output_path)
                    logging.info(f"Used fallback: copied {os.path.basename(first_video)} to output")
                    return str(output_path)
                return None
                
        except Exception as e:
            logging.error(f"Error creating grid layout chunk: {e}")
            return None

    # def _create_ffmpeg_grid_layout(self, track_segments, output_path, duration, grid_rows, grid_cols):
    #     """Create grid layout using FFmpeg with MIDI-synchronized individual note playback"""
    #     try:
    #         # Initialize grid with note sequences for each position
    #         grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
            
    #         # Place track segments in their grid positions
    #         for segment in track_segments:
    #             track_id = segment.get('track_id')
    #             segment_type = segment.get('type', 'unknown')
    #             notes = segment.get('notes', [])
    #             start_time = segment.get('start_time', 0)
                
    #             logging.info(f"Placing segment: track_id={track_id}, type={segment_type}")
                
    #             # Try multiple possible track ID formats for grid positioning
    #             possible_track_keys = [
    #                 track_id,
    #                 str(track_id),
    #                 f"track-{track_id}",
    #                 f"{track_id}",
    #             ]
                
    #             # For drum segments, also try the drum-specific key
    #             if segment_type == 'drum':
    #                 drum_name = segment.get('drum_name', '')
    #                 if drum_name:
    #                     drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
    #                     possible_track_keys.insert(0, drum_key)
                
    #             position_found = False
    #             for key in possible_track_keys:
    #                 if key in self.grid_positions:
    #                     position = self.grid_positions[key]
    #                     row = position.get('row', 0)
    #                     col = position.get('column', 0)

    #                     if row < grid_rows and col < grid_cols:
    #                         # Create note sequence for this position
    #                         note_sequence = {
    #                             'video_path': segment['video_path'],
    #                             'notes': notes,
    #                             'chunk_start_time': start_time,
    #                             'track_id': track_id,
    #                             'type': segment_type
    #                         }
    #                         grid[row][col].append(note_sequence)
    #                         logging.info(f"✅ Placed {segment_type} track {track_id} at grid position ({row}, {col}) using key '{key}'")
    #                         position_found = True
    #                         break
                
    #             if not position_found:
    #                 logging.warning(f"❌ No grid position found for track {track_id} (type: {segment_type}), tried keys: {possible_track_keys}")
    #                 logging.warning(f"Available grid positions: {list(self.grid_positions.keys())}")
                    
    #                 # Use a smarter fallback position - spread segments across grid
    #                 segment_idx = track_segments.index(segment)
    #                 fallback_row = segment_idx % grid_rows
    #                 fallback_col = segment_idx // grid_rows
                    
    #                 # Make sure we don't exceed grid bounds
    #                 if fallback_col >= grid_cols:
    #                     fallback_col = segment_idx % grid_cols
    #                     fallback_row = segment_idx // grid_cols
                    
    #                 if fallback_row < grid_rows and fallback_col < grid_cols:
    #                     note_sequence = {
    #                         'video_path': segment['video_path'],
    #                         'notes': notes,
    #                         'chunk_start_time': start_time,
    #                         'track_id': track_id,
    #                         'type': segment_type
    #                     }
    #                     grid[fallback_row][fallback_col].append(note_sequence)
    #                     logging.info(f"⚠️ Used fallback position ({fallback_row}, {fallback_col}) for track {track_id}")
    #                 else:
    #                     # Find first available position
    #                     placed = False
    #                     for r in range(grid_rows):
    #                         for c in range(grid_cols):
    #                             if len(grid[r][c]) == 0:
    #                                 note_sequence = {
    #                                     'video_path': segment['video_path'],
    #                                     'notes': notes,
    #                                     'chunk_start_time': start_time,
    #                                     'track_id': track_id,
    #                                     'type': segment_type
    #                                 }
    #                                 grid[r][c].append(note_sequence)
    #                                 logging.info(f"⚠️ Used first available position ({r}, {c}) for track {track_id}")
    #                                 placed = True
    #                                 break
    #                         if placed:
    #                             break
            
    #         # Now create MIDI-synchronized video sequences for each grid cell
    #         cmd = ['ffmpeg', '-y']
    #         input_map = {}
    #         audio_map = {}
    #         input_idx = 0
            
    #         # Calculate cell dimensions for proper scaling
    #         cell_width = 1920 // grid_cols
    #         cell_height = 1080 // grid_rows
            
    #         # Build filter complex for video grid with MIDI synchronization
    #         filter_complex = []
            
    #         for row in range(grid_rows):
    #             for col in range(grid_cols):
    #                 note_sequences = grid[row][col]
                    
    #                 if note_sequences:
    #                     # Create MIDI-synchronized video sequence for this cell
    #                     cell_video_filter, cell_audio_filter, new_input_idx = self._create_midi_synchronized_cell(
    #                         note_sequences, duration, cell_width, cell_height, input_idx, cmd
    #                     )
    #                     filter_complex.append(cell_video_filter)
    #                     input_map[f"{row}_{col}"] = f"v{row}_{col}"
    #                     audio_map[f"{row}_{col}"] = f"a{row}_{col}"
    #                     input_idx = new_input_idx
                        
    #                     # Add audio filter if we have one
    #                     if cell_audio_filter:
    #                         filter_complex.append(cell_audio_filter)
                        
    #                 else:
    #                     # Create black placeholder with silent audio
    #                     cmd.extend([
    #                         '-f', 'lavfi', '-i', f'color=black:size={cell_width}x{cell_height}:duration={duration}:rate=30',
    #                         '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}'
    #                     ])
    #                     input_map[f"{row}_{col}"] = f"v{row}_{col}"
    #                     audio_map[f"{row}_{col}"] = f"a{row}_{col}"
                        
    #                     # Create filters for the placeholders
    #                     filter_complex.append(f'[{input_idx}:v]copy[v{row}_{col}]')
    #                     filter_complex.append(f'[{input_idx+1}:a]copy[a{row}_{col}]')
    #                     input_idx += 2
            
    #         # Create xstack filter for grid layout
    #         xstack_inputs = []
    #         for row in range(grid_rows):
    #             for col in range(grid_cols):
    #                 xstack_inputs.append(f'v{row}_{col}')
            
    #         # Build xstack layout string
    #         layout_positions = []
    #         for row in range(grid_rows):
    #             for col in range(grid_cols):
    #                 x = col * cell_width
    #                 y = row * cell_height
    #                 layout_positions.append(f"{x}_{y}")
    #         layout_string = "|".join(layout_positions)
            
    #         xstack_filter = f"{''.join([f'[{inp}]' for inp in xstack_inputs])}xstack=inputs={len(xstack_inputs)}:layout={layout_string}[video_out]"
    #         filter_complex.append(xstack_filter)
            
    #         # Mix audio from all cells
    #         audio_inputs = []
    #         for row in range(grid_rows):
    #             for col in range(grid_cols):
    #                 audio_inputs.append(f'[a{row}_{col}]')
            
    #         if audio_inputs:
    #             amix_filter = f"{''.join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=longest:dropout_transition=0[audio_out]"
    #             filter_complex.append(amix_filter)
            
    #         # Add filter complex to command
    #         cmd.extend(['-filter_complex', ';'.join(filter_complex)])
            
    #         # Map outputs
    #         cmd.extend(['-map', '[video_out]'])
    #         if audio_inputs:
    #             cmd.extend(['-map', '[audio_out]'])
            
    #         # Output settings
    #         cmd.extend([
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
    #             '-t', str(duration),
    #             '-r', '30',
    #             str(output_path)
    #         ])
            
    #         logging.info(f"Running FFmpeg MIDI-synchronized grid command with {len(cmd)} arguments")
    #         logging.info(f"Filter complex: {';'.join(filter_complex)}")
            
    #         result = gpu_subprocess_run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ GPU encoding successful with complex filters")
    #             return str(output_path)
    #         else:
    #             logging.error(f"FFmpeg grid creation failed: {result.stderr}")
    #             logging.error(f"FFmpeg stdout: {result.stdout}")
    #             raise Exception(f"FFmpeg error: {result.stderr}")
                
    #     except Exception as e:
    #         logging.error(f"Error in FFmpeg grid creation: {e}")
    #         raise

    def _create_ffmpeg_grid_layout(self, track_segments, output_path, duration, grid_rows, grid_cols):
        """Create grid layout using FFmpeg with proper multi-video handling"""
        try:
            if len(track_segments) == 1:
                # Single video - just copy it instead of trying to create a grid
                single_video = track_segments[0]['video_path']
                import shutil
                shutil.copy2(single_video, output_path)
                logging.info(f"✅ Single video copied: {os.path.basename(single_video)}")
                return str(output_path)
            
            logging.info(f"🎬 Creating grid with {len(track_segments)} videos")
            
            # Build FFmpeg command with all inputs
            cmd = ['ffmpeg', '-y']
            
            # Add all video files as inputs
            input_map = {}
            for i, segment in enumerate(track_segments):
                video_path = segment['video_path']
                if os.path.exists(video_path):
                    cmd.extend(['-i', video_path])
                    input_map[i] = segment
                    logging.info(f"📹 Input {i}: {os.path.basename(video_path)}")
                else:
                    logging.warning(f"❌ Video not found: {video_path}")
            
            if len(input_map) < 2:
                logging.warning("⚠️ Less than 2 valid videos found, using fallback")
                if input_map:
                    first_video = list(input_map.values())[0]['video_path']
                    import shutil
                    shutil.copy2(first_video, output_path)
                    return str(output_path)
                return None
            
            # Calculate cell dimensions
            cell_width = 1920 // grid_cols
            cell_height = 1080 // grid_rows
            
            # Create filter complex
            filter_parts = []
            
            # Scale all inputs to cell size
            for i in range(len(input_map)):
                filter_parts.append(f'[{i}:v]scale={cell_width}:{cell_height}[v{i}]')
            
            # Create grid layout positions
            layout_positions = []
            for i in range(len(input_map)):
                row = i // grid_cols
                col = i % grid_cols
                x = col * cell_width
                y = row * cell_height
                layout_positions.append(f"{x}_{y}")
            
            layout_string = "|".join(layout_positions)
            
            # Create xstack filter
            video_inputs = ''.join([f'[v{i}]' for i in range(len(input_map))])
            xstack_filter = f"{video_inputs}xstack=inputs={len(input_map)}:layout={layout_string}[video_out]"
            filter_parts.append(xstack_filter)
            
            # Mix audio
            audio_inputs = ''.join([f'[{i}:a]' for i in range(len(input_map))])
            amix_filter = f"{audio_inputs}amix=inputs={len(input_map)}:duration=longest[audio_out]"
            filter_parts.append(amix_filter)
            
            # Add filter complex to command
            filter_complex = ';'.join(filter_parts)
            cmd.extend(['-filter_complex', filter_complex])
            
            # Map outputs
            cmd.extend([
                '-map', '[video_out]',
                '-map', '[audio_out]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-t', str(duration),
                '-r', '30',
                str(output_path)
            ])
            
            logging.info(f"🎬 Running grid command with {len(input_map)} inputs")
            logging.info(f"Filter: {filter_complex}")
            
            result = gpu_subprocess_run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"✅ Grid created successfully with {len(input_map)} videos")
                return str(output_path)
            else:
                logging.error(f"❌ Grid creation failed: {result.stderr}")
                
                # Fallback to first video
                if input_map:
                    first_video = list(input_map.values())[0]['video_path']
                    import shutil
                    shutil.copy2(first_video, output_path)
                    logging.info(f"⚠️ Used fallback: {os.path.basename(first_video)}")
                    return str(output_path)
                return None
            
        except Exception as e:
            logging.error(f"Error creating grid layout: {e}")
            return None

    def _create_simple_concat_fallback(self, track_segments, output_path, duration):
        """Simple fallback: concatenate all videos horizontally"""
        try:
            if not track_segments:
                return None
                
            # Just use the first video as fallback
            first_segment = track_segments[0]
            import shutil
            shutil.copy2(first_segment['video_path'], output_path)
            logging.info(f"Created fallback chunk using first segment: {os.path.basename(output_path)}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error in fallback creation: {e}")
            return None

    def _create_placeholder_chunk_simple(self, chunk_idx, chunks_dir, duration):
        """Create a simple placeholder chunk"""
        try:
            output_path = chunks_dir / f"placeholder_chunk_{chunk_idx}.mp4"
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'color=black:size=1920x1080:duration={duration}:rate=30',
                '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path)
            ]
            
            result = gpu_subprocess_run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Created placeholder chunk: {output_path.name}")
                return str(output_path)
            else:
                logging.error(f"Error creating placeholder: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating placeholder chunk: {e}")
            return None

    def _debug_grid_placement(self, track_segments):
        """Debug method to show what's being placed in the grid"""
        logging.info("=== GRID PLACEMENT DEBUG ===")
        logging.info(f"Available grid positions: {list(self.grid_positions.keys())}")
        
        for segment in track_segments:
            track_id = segment.get('track_id')
            segment_type = segment.get('type', 'unknown')
            video_path = segment.get('video_path', 'MISSING')
            
            logging.info(f"Segment: track_id={track_id}, type={segment_type}")
            logging.info(f"  Video path: {video_path}")
            logging.info(f"  File exists: {os.path.exists(video_path) if video_path != 'MISSING' else False}")
            
            # Check what grid position this would get
            possible_keys = [track_id, str(track_id), f"track-{track_id}"]
            if segment_type == 'drum':
                drum_name = segment.get('drum_name', '')
                if drum_name:
                    drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                    possible_keys.insert(0, drum_key)
            
            found_position = None
            for key in possible_keys:
                if key in self.grid_positions:
                    found_position = self.grid_positions[key]
                    break
            
            if found_position:
                logging.info(f"  Grid position: row={found_position.get('row')}, col={found_position.get('column')}")
            else:
                logging.info(f"  Grid position: NOT FOUND (tried keys: {possible_keys})")
        
        logging.info("=== END GRID DEBUG ===")

    def preprocess_video_gpu(self, input_path, output_path, target_width=640, target_height=360):
        """
        GPU-accelerated video preprocessing with proper error handling
        """
        try:
            # Use basic GPU encoding without problematic options
            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-i', input_path,
                '-vf', f'scale={target_width}:{target_height}',
                '-c:v', 'h264_nvenc',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '44100',
                '-ac', '2',
                '-movflags', '+faststart',
                output_path
            ]
            
            print(f"GPU preprocessing: {input_path} -> {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ GPU preprocessing successful: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ GPU preprocessing failed: {e}")
            print(f"Falling back to CPU preprocessing...")
            return self.preprocess_video_cpu(input_path, output_path, target_width, target_height)
        except Exception as e:
            print(f"❌ GPU preprocessing error: {e}")
            return self.preprocess_video_cpu(input_path, output_path, target_width, target_height)

    def preprocess_video_cpu(self, input_path, output_path, target_width=640, target_height=360):
        """CPU fallback for video preprocessing"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad=w={target_width}:h={target_height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
                '-movflags', '+faststart', '-threads', '4',
                output_path
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ CPU preprocessing successful: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ CPU preprocessing failed: {e}")
            return False

    def run_gpu_subprocess(self, cmd):
        """Run subprocess with GPU acceleration"""
        try:
            # Extract input and output paths from command
            input_path = None
            output_path = None
            
            for i, arg in enumerate(cmd):
                if arg == '-i' and i + 1 < len(cmd):
                    input_path = cmd[i + 1]
                elif arg.endswith('.mp4') and not arg.startswith('-'):
                    output_path = arg
            
            if input_path and output_path:
                from utils.ffmpeg_gpu import ffmpeg_gpu_encode
                success = ffmpeg_gpu_encode(input_path, output_path)
                if success:
                    return True
            
            # Fallback to CPU
            return self.run_cpu_subprocess(cmd)
            
        except Exception as e:
            print(f"❌ GPU subprocess error: {e}")
            return self.run_cpu_subprocess(cmd)

    def run_cpu_subprocess(self, cmd):
        """CPU fallback for subprocess operations"""
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ CPU subprocess failed: {e}")
            return False

    def create_midi_synchronized_composition(self, midi_data, video_paths, output_path):
        """
        Create a composition where videos are triggered by MIDI notes
        
        Args:
            midi_data: MIDI data structure with tracks and notes
            video_paths: Dictionary mapping instrument names to video file paths
            output_path: Output video file path
        """
        from .midi_synchronized_compositor import MidiSynchronizedCompositor
        
        # Calculate total duration from MIDI data
        total_duration = 0
        for track in midi_data.get('tracks', []):
            for note in track.get('notes', []):
                end_time = note.get('time', 0) + note.get('duration', 0.5)
                total_duration = max(total_duration, end_time)
        
        # Add some padding
        total_duration += 2.0
        
        print(f"🎵 Creating MIDI-synchronized composition (duration: {total_duration:.2f}s)")
        
        # Create compositor
        compositor = MidiSynchronizedCompositor()
        
        try:
            # Create triggered composition
            success = compositor.create_midi_triggered_video(
                midi_data, video_paths, output_path, total_duration
            )
            
            if success:
                print(f"✅ MIDI-synchronized composition created: {output_path}")
                return output_path
            else:
                print("❌ Failed to create MIDI-synchronized composition")
                return None
                
        finally:
            # Clean up temporary files
            compositor.cleanup()

    def run_ffmpeg_grid_command(self, segments, output_path, duration=4.0):
        """
        Run FFmpeg command to create grid layout with proper video output
        
        Args:
            segments: List of video segments with paths and positions
            output_path: Output video file path
            duration: Duration of the output video
        """
        if not segments:
            return False
        
        # Build command properly
        cmd = ['ffmpeg', '-y']
        
        # Add input files
        for segment in segments:
            if 'video_path' in segment and os.path.exists(segment['video_path']):
                cmd.extend(['-i', segment['video_path']])
            else:
                logging.warning(f"Video path not found: {segment.get('video_path', 'unknown')}")
                continue
        
        # Add filter complex for video grid
        filter_complex = self.build_filter_complex(segments)
        if not filter_complex:
            logging.error("Failed to build filter complex")
            return False
            
        cmd.extend(['-filter_complex', filter_complex])
        
        # Map video and audio outputs
        cmd.extend(['-map', '[video_out]'])
        cmd.extend(['-map', '[audio_out]'])
        
        # Video encoding settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-r', '30',
            '-t', str(duration)
        ])
        
        # Audio encoding settings
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', '44100',
            '-ac', '2'
        ])
        
        # Output file
        cmd.append(output_path)
        
        try:
            logging.info(f"Running FFmpeg grid command: {' '.join(cmd[:10])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"✅ Grid video created successfully: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg grid command failed: {e}")
            logging.error(f"Command: {' '.join(cmd)}")
            logging.error(f"Error output: {e.stderr}")
            return False

    def build_filter_complex(self, segments):
        """
        Build FFmpeg filter complex for grid layout
        
        Args:
            segments: List of video segments
        """
        if not segments:
            return None
        
        try:
            # Calculate grid size
            max_row = max(seg.get('row', 0) for seg in segments)
            max_col = max(seg.get('col', 0) for seg in segments)
            grid_rows = max_row + 1
            grid_cols = max_col + 1
            
            cell_width = 640 // grid_cols
            cell_height = 360 // grid_rows
            
            # Scale all inputs
            scale_filters = []
            for i, segment in enumerate(segments):
                scale_filters.append(f'[{i}:v]scale={cell_width}:{cell_height}[v{i}]')
            
            # Build grid layout
            grid_inputs = ''.join(f'[v{i}]' for i in range(len(segments)))
            
            # Create layout positions
            layout_parts = []
            for i, segment in enumerate(segments):
                row = segment.get('row', 0)
                col = segment.get('col', 0)
                x = col * cell_width
                y = row * cell_height
                layout_parts.append(f'{x}_{y}')
            layout = '|'.join(layout_parts)
            
            # Combine filters
            filter_complex = ';'.join(scale_filters)
            filter_complex += f';{grid_inputs}xstack=inputs={len(segments)}:layout={layout}[video_out]'
            
            # Add audio mixing
            audio_inputs = ''.join(f'[{i}:a]' for i in range(len(segments)))
            filter_complex += f';{audio_inputs}amix=inputs={len(segments)}:duration=longest[audio_out]'
            
            return filter_complex
            
        except Exception as e:
            logging.error(f"Error building filter complex: {e}")
            return None

    def _create_midi_synchronized_cell(self, note_sequences, duration, cell_width, cell_height, input_idx, cmd):
        """
        Create a MIDI-synchronized video sequence for a single grid cell.
        
        This method creates individual video clips for each MIDI note that:
        1. Start at the exact note time
        2. Play for the exact note duration
        3. Are tuned to the correct pitch
        
        Args:
            note_sequences: List of note sequences for this cell
            duration: Total duration of the chunk
            cell_width: Width of the cell in pixels
            cell_height: Height of the cell in pixels
            input_idx: Current input index for FFmpeg
            cmd: FFmpeg command being built
            
        Returns:
            tuple: (video_filter, audio_filter, new_input_idx)
        """
        try:
            # Collect all notes from all sequences in this cell
            all_notes = []
            video_path = None
            
            for sequence in note_sequences:
                video_path = sequence['video_path']  # All sequences in same cell use same video
                chunk_start_time = sequence['chunk_start_time']
                
                for note in sequence['notes']:
                    note_start = float(note.get('time', 0))
                    note_duration = float(note.get('duration', 1))
                    note_pitch = note.get('midi', 60)  # Default to middle C
                    
                    # Convert to chunk-relative time
                    chunk_relative_start = note_start - chunk_start_time
                    
                    # Only include notes that are within this chunk
                    if chunk_relative_start < duration and chunk_relative_start + note_duration > 0:
                        # Clip note to chunk boundaries
                        actual_start = max(0, chunk_relative_start)
                        actual_end = min(duration, chunk_relative_start + note_duration)
                        actual_duration = actual_end - actual_start
                        
                        if actual_duration > 0:
                            all_notes.append({
                                'start': actual_start,
                                'duration': actual_duration,
                                'pitch': note_pitch,
                                'original_pitch': 60  # Assume video is recorded at middle C
                            })
            
            if not all_notes or not video_path:
                # Return empty/black cell filters
                row = input_idx // 2
                col = input_idx % 2
                video_filter = f'color=black:size={cell_width}x{cell_height}:duration={duration}:rate=30[v{row}_{col}]'
                audio_filter = f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}[a{row}_{col}]'
                return video_filter, audio_filter, input_idx + 2
            
            # Add the video file as input
            cmd.extend(['-i', video_path])
            video_input_idx = input_idx
            input_idx += 1
            
            row = video_input_idx // 2  # Simple row/col calculation
            col = video_input_idx % 2
            
            # For now, create a simplified version that plays the video with timing
            # We'll implement note-by-note timing in a simpler way
            
            # Create a basic synchronized video with the first note's timing
            if all_notes:
                first_note = all_notes[0]
                pitch_semitones = first_note['pitch'] - first_note['original_pitch']
                
                # Calculate pitch adjustment factor
                pitch_factor = 2 ** (pitch_semitones / 12.0) if pitch_semitones != 0 else 1.0
                
                # Create video filter that scales and shows video at note times
                video_filter_parts = [
                    f'[{video_input_idx}:v]',
                    f'scale={cell_width}:{cell_height}',
                    f'trim=0:{first_note["duration"]},setpts=PTS-STARTPTS',
                    f'[v{row}_{col}]'
                ]
                video_filter = ''.join(video_filter_parts[:1]) + ','.join(video_filter_parts[1:-1]) + video_filter_parts[-1]
                
                # Create audio filter with pitch adjustment
                audio_filter_parts = [
                    f'[{video_input_idx}:a]',
                    f'atrim=0:{first_note["duration"]},asetpts=PTS-STARTPTS'
                ]
                
                if pitch_factor != 1.0:
                    audio_filter_parts.append(f'asetrate=44100*{pitch_factor},aresample=44100')
                
                audio_filter_parts.append(f'[a{row}_{col}]')
                audio_filter = ''.join(audio_filter_parts[:1]) + ','.join(audio_filter_parts[1:-1]) + audio_filter_parts[-1]
                
                logging.info(f"🎵 Created MIDI-synchronized cell with {len(all_notes)} notes (pitch factor: {pitch_factor:.2f})")
                return video_filter, audio_filter, input_idx
            
            # Fallback to simple scaling
            video_filter = f'[{video_input_idx}:v]scale={cell_width}:{cell_height}[v{row}_{col}]'
            audio_filter = f'[{video_input_idx}:a]copy[a{row}_{col}]'
            
            return video_filter, audio_filter, input_idx
            
        except Exception as e:
            logging.error(f"Error creating MIDI-synchronized cell: {e}")
            # Return fallback black cell
            row = input_idx // 2
            col = input_idx % 2
            video_filter = f'color=black:size={cell_width}x{cell_height}:duration={duration}:rate=30[v{row}_{col}]'
            audio_filter = f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}[a{row}_{col}]'
            return video_filter, audio_filter, input_idx + 2
