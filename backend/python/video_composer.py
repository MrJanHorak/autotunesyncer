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
import json
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
from collections import defaultdict

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
# Removed unresolved import; GPU encode is imported lazily from utils.ffmpeg_gpu where used

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
from optimized_autotune_cache import OptimizedAutotuneCache as _RealOptimizedAutotuneCache

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

# Use the real OptimizedAutotuneCache from optimized_autotune_cache.py
OptimizedAutotuneCache = _RealOptimizedAutotuneCache

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
        self.CHUNK_DURATION = 16
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
    FRAME_DURATION = 1.0 / FRAME_RATE
    MIN_NOTE_DURATION = max(0.05, FRAME_DURATION * 1.5)  # ~50ms
    TIME_QUANTUM = 0.01  # 10ms grid to reduce floating point noise
    CHUNK_DURATION = 16
    OVERLAP_DURATION = 1
    CROSSFADE_DURATION = 0.5
    MIN_VIDEO_DURATION = 1.0
    DURATION = 1.0
    VOLUME_MULTIPLIERS = {
        'drums': 0.2, 'instruments': 1.5
    }

    def __init__(self, processed_videos_dir, midi_data, output_path, preview_mode=False):
        """Initialize VideoComposer with proper path handling"""
        try:
            logging.info("=== VideoComposer Initialization ===")
            self.preview_mode = preview_mode
            
            # Initialize render configuration for preview vs production
            self.render_config = self._get_render_config()
            if self.preview_mode:
                logging.info(f"🚀 PREVIEW MODE ENABLED: {self.render_config}")

            logging.info(f"Received MIDI data structure: {list(midi_data.keys())}")
            logging.info(f"Grid arrangement from MIDI: {midi_data.get('gridArrangement')}")

            # Extract track volumes from midi_data
            self.track_volumes = midi_data.get('trackVolumes', {})
            logging.info(f"Initialized VideoComposer with track volumes: {self.track_volumes}")

            # ── Composition & clip style settings ────────────────────────────
            self.composition_style = midi_data.get('compositionStyle', {})
            self.clip_styles = midi_data.get('clipStyles', {})  # keyed by grid item id
            if self.composition_style:
                logging.info(f"Composition style loaded: {list(self.composition_style.keys())}")
            else:
                logging.info("Composition style: empty (no global effects)")
            if self.clip_styles:
                logging.info(f"Clip styles loaded for: {list(self.clip_styles.keys())}")
            else:
                logging.info("Clip styles: empty (no per-clip effects)")

            # Explicit session-specific video paths (prevents stale uploads from past sessions)
            self.explicit_video_files = midi_data.get('videoFiles', {})

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
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "44100",
                    "-ac", "2",
                    "-movflags", "+faststart",
                    "-threads", "4",
                    "-profile:v", "high"
                ]
            }
            self.chunk_size = max(1, min(16, os.cpu_count()))  # Smaller chunk size
            self.max_workers = max(2, min(os.cpu_count() or 2, 4))  # Up to 4 parallel workers
            self.use_gpu = True
            self.lock = RLock()  # Add class-level lock
            self.clip_pool = ClipPool(max_size=8)  # Add clip pool
            self.chunk_cache_lock = RLock()  # Add dedicated cache lock
            self.max_cache_size = 1024 * 1024 * 100
            self.active_readers = set()  # Add reader tracking
            # Onset alignment cache and toggle
            self.onset_trim_cache = {}
            self.trim_leading_silence = os.environ.get('ATS_TRIM_LEADING_SILENCE', '1') == '1'
            # Onset offset cache for non-destructive alignment — protected by per-path locks
            self.onset_offset_cache = {}
            self._onset_path_locks: dict = defaultdict(threading.Lock)  # one Lock per clip path
            # Media duration cache — avoids duplicate ffprobe calls from parallel stem threads
            self._duration_cache: dict = {}
            self._duration_cache_lock = threading.RLock()
            # Global FFmpeg concurrency cap — shared by stem sub-batches AND video chunks.
            # Caps simultaneous filter-complex FFmpeg processes to half the CPU count so
            # nested parallelism (outer stem workers + inner sub-batch workers) never
            # over-subscribes the machine.
            _cpu = os.cpu_count() or 4
            self._ffmpeg_semaphore = threading.Semaphore(max(4, _cpu // 2))
            self.max_concurrent_streams = int(os.environ.get('ATS_MAX_CONCURRENT_STREAMS', '16'))
            self.ffmpeg_hwaccel = self._detect_ffmpeg_hwaccel()
            logging.info(f"Selected FFmpeg hwaccel: {self.ffmpeg_hwaccel or 'none'}")
            # Initialize path registry - use singleton instance
            self.path_registry = PathRegistry.get_instance()
            
            # After copying files, register them
            self._register_video_paths()
            self.video_cache = LRUCache(maxsize=64)  # Increase cache size
            self.audio_cache = LRUCache(maxsize=64) 
            self.autotune_cache = LRUCache(maxsize=64) # Keep for backwards compatibility
            # Initialize optimized autotune cache system — use more workers for cold-start runs
            _preprocess_workers = min(os.cpu_count() or 4, 8)
            self.optimized_cache = OptimizedAutotuneCache(max_workers=_preprocess_workers)
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

    def _get_render_config(self):
        """
        Clean separation of rendering configurations for Preview vs Production.
        Returns a dictionary of encoding settings.
        """
        if self.preview_mode:
            return {
                'resolution': '640x360',
                'preset': 'ultrafast',  # CPU: ultrafast, GPU: p1/p2
                'crf': '28',            # Lower quality
                'audio_bitrate': '128k',
                'video_bitrate': '1M',
                'scale_filter': 'scale=640:360'
            }
        else:
            return {
                'resolution': '1920x1080',
                'preset': 'fast',       # CPU: fast, GPU: p4
                'crf': '26',            # Slightly lower quality for intermediate chunks; final pass re-encodes at 28
                'audio_bitrate': '192k',
                'video_bitrate': '3M',  # Reduced from 8M — GPU hard cap; CPU uses CRF only
                'scale_filter': 'scale=1920:1080'
            }

    def _get_encoding_settings(self):
        """Get FFmpeg encoding arguments based on configuration and hardware"""
        config = self.render_config

        if self.ffmpeg_hwaccel == 'cuda':
            return [
                '-c:v', 'h264_nvenc',
                '-preset', 'fast' if not self.preview_mode else 'p1',  # p1 for preview (fastest), fast for production
                '-rc', 'vbr',  # Variable bitrate mode
                '-cq', config['crf'],  # Use config CRF
                '-b:v', config['video_bitrate'],  # Use config bitrate
            ]
        if self.ffmpeg_hwaccel == 'videotoolbox':
            return [
                '-c:v', 'h264_videotoolbox',
                '-b:v', config['video_bitrate'],
            ]
        else:
            # CPU Settings
            return [
                '-c:v', 'libx264',
                '-preset', config['preset'],
                '-crf', config['crf'],
            ]

    def _detect_ffmpeg_hwaccel(self):
        """Probe ffmpeg hw acceleration methods with safe fallbacks."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-hwaccels'],
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or '') + '\n' + (result.stderr or '')
            lower = output.lower()

            if 'cuda' in lower:
                return 'cuda'
            if 'videotoolbox' in lower:
                return 'videotoolbox'
            return None
        except Exception as e:
            logging.warning(f"Failed to detect FFmpeg hwaccels: {e}")
            return None

    def _get_ffmpeg_decode_args(self):
        """Return decode args for ffmpeg -i based on selected hwaccel."""
        if self.ffmpeg_hwaccel == 'cuda':
            return ['-hwaccel', 'cuda']
        if self.ffmpeg_hwaccel == 'videotoolbox':
            return ['-hwaccel', 'videotoolbox']
        return []

    def _resolve_segment_volume(self, segment):
        """
        Resolve the volume (in dB) for a given segment using trackVolumes.
        Supports keys by track_id, normalized instrument name, and drum name.
        """
        try:
            volumes = self.track_volumes or {}

            # Try direct by track_id
            track_id = str(segment.get('track_id', '')).lower()
            if track_id:
                val = volumes.get(track_id)
                if isinstance(val, (int, float)):
                    logging.info(f"Resolved volume via track_id '{track_id}': {val} dB")
                    return float(val)

            seg_type = segment.get('type')
            # Drums: use 'drum_<normalized>' key
            if seg_type == 'drum':
                drum_name = segment.get('drum_name') or segment.get('track_name') or ''
                drum_key = f"drum_{str(drum_name).lower().replace(' ', '_')}"
                val = volumes.get(drum_key)
                if isinstance(val, (int, float)):
                    logging.info(f"Resolved volume via drum key '{drum_key}': {val} dB")
                    return float(val)

            # Instruments: normalized instrument name
            track_name = segment.get('track_name') or segment.get('instrument') or ''
            if isinstance(track_name, str) and track_name:
                norm_name = normalize_instrument_name(track_name)
                val = volumes.get(norm_name)
                if isinstance(val, (int, float)):
                    logging.info(f"Resolved volume via instrument '{norm_name}': {val} dB")
                    return float(val)

            # Fallback: 0 dB
            return 0.0
        except Exception as e:
            logging.warning(f"Volume resolve error: {e}; defaulting to 0 dB")
            return 0.0

    def _velocity_to_db(self, velocity, min_db=-18.0):
        """
        Convert MIDI velocity (1-127) to dB using an exponential mapping.
        127 -> 0 dB, 1 -> min_db.
        min_db=-18 maps the typical MIDI dynamic range naturally:
          v=1  -> -18 dB (pianissimo), v=64 -> -13.5 dB (mezzo-forte),
          v=100 -> -6.9 dB (forte),    v=127 -> 0 dB (fortissimo).
        The old min_db=-40 caused v=64 to be attenuated by 30 dB, which
        drove the mix to -47 LUFS and required a damaging +31 dB loudnorm boost.
        """
        try:
            if velocity is None:
                return 0.0
            v = max(1.0, min(127.0, float(velocity)))
            norm = (v - 1.0) / 126.0
            return float(min_db + (0.0 - min_db) * (norm ** 2.0))
        except Exception:
            return 0.0

    def _detect_leading_silence(self, media_path, threshold_db=-35, min_duration=0.08):
        """
        Detect leading silence at the start of the audio stream using ffmpeg's silencedetect.
        Returns trim_start seconds or 0.0 if no leading silence.
        """
        try:
            if not media_path or not os.path.exists(media_path):
                return 0.0

            cmd = [
                'ffmpeg', '-hide_banner', '-i', str(media_path),
                '-af', f'silencedetect=noise={threshold_db}dB:duration={min_duration}',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return 0.0

            silence_start = None
            for line in (result.stderr or '').splitlines():
                if 'silence_start:' in line and silence_start is None:
                    try:
                        silence_start = float(line.split('silence_start:')[1].strip())
                    except:
                        silence_start = None
                if 'silence_end:' in line and silence_start is not None:
                    try:
                        silence_end = float(line.split('silence_end:')[1].split('|')[0].strip())
                        if silence_start is not None and silence_start <= 0.2:
                            return max(0.0, silence_end)
                        else:
                            return 0.0
                    except:
                        return 0.0
            return 0.0
        except Exception as e:
            logging.debug(f"Silence detect failed: {e}")
            return 0.0

    def _auto_crop_silence(self, media_path, threshold_db=-35, min_duration=0.08):
        """
        Trim leading silence from the media using stream copy for speed.
        Returns path to trimmed file or original path on failure.
        """
        try:
            trim_start = self._detect_leading_silence(media_path, threshold_db, min_duration)
            if trim_start <= 0.0:
                return str(media_path)

            media_path = Path(media_path)
            cropped_path = media_path.parent / f"cropped_{media_path.name}"
            crop_cmd = [
                'ffmpeg', '-y',
                '-ss', f'{trim_start}',
                '-i', str(media_path),
                '-c', 'copy',
                str(cropped_path)
            ]
            crop_result = subprocess.run(crop_cmd, capture_output=True, text=True)
            if crop_result.returncode == 0 and cropped_path.exists():
                logging.info(f"Trimmed leading silence ({trim_start:.3f}s) for {media_path.name}")
                return str(cropped_path)
            return str(media_path)
        except Exception as e:
            logging.debug(f"Auto-crop failed: {e}")
            return str(media_path)

    def _get_onset_aligned_video(self, video_path):
        """
        Return onset-aligned version of video by trimming initial silence; cached per path.
        Controlled by self.trim_leading_silence.
        """
        try:
            if not self.trim_leading_silence:
                return video_path
            if not video_path or not os.path.exists(video_path):
                return video_path
            if video_path in self.onset_trim_cache and os.path.exists(self.onset_trim_cache[video_path]):
                return self.onset_trim_cache[video_path]

            aligned = self._auto_crop_silence(video_path)
            self.onset_trim_cache[video_path] = aligned
            return aligned
        except Exception as e:
            logging.debug(f"Onset align error: {e}")
            return video_path

    def _extract_temp_audio(self, video_path):
        """Extract mono 44.1kHz WAV to temp for onset analysis."""
        try:
            if not video_path or not os.path.exists(video_path):
                return None
            if not hasattr(self, 'temp_dir') or self.temp_dir is None:
                import tempfile
                self.temp_dir = Path(tempfile.mkdtemp(prefix='composer_tmp_'))
            # Use MD5 of the full path so concurrent threads never share a temp file,
            # even when two clips happen to have the same basename.
            import hashlib
            path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
            out_wav = Path(self.temp_dir) / f"onset_{path_hash}.wav"
            cmd = [
                'ffmpeg', '-hide_banner', '-y',
                '-i', str(video_path),
                '-vn', '-ac', '1', '-ar', '44100',
                '-acodec', 'pcm_s16le',
                str(out_wav)
            ]
            subprocess.run(cmd, capture_output=True)
            return str(out_wav) if out_wav.exists() else None
        except Exception:
            return None

    def _get_onset_offset(self, video_path):
        """
        Detect first audible onset time (seconds) using librosa; fallback to RMS.
        Cached per video_path with per-path locking so parallel stem threads never
        race to compute the same clip's onset or write the same temp WAV.
        """
        try:
            if not video_path or not os.path.exists(video_path):
                return 0.0
            # Fast path: already cached (pure read, GIL-safe for dict lookup)
            if video_path in self.onset_offset_cache:
                return self.onset_offset_cache[video_path]

            # Per-path lock: only one thread runs onset detection per clip path.
            # defaultdict(Lock) creation is GIL-protected under CPython.
            with self._onset_path_locks[video_path]:
                # Re-check: another thread may have populated while we waited.
                if video_path in self.onset_offset_cache:
                    return self.onset_offset_cache[video_path]

                wav_path = self._extract_temp_audio(video_path)
                if not wav_path:
                    self.onset_offset_cache[video_path] = 0.0
                    return 0.0

                import librosa, numpy as np
                y, sr = librosa.load(wav_path, sr=44100, mono=True)

                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
                if onsets is not None and len(onsets) > 0:
                    offset = float(onsets[0])
                    if offset < 0.02 and len(onsets) > 1:
                        offset = float(onsets[1])
                    offset = max(0.0, min(offset, 5.0))
                    self.onset_offset_cache[video_path] = offset
                    logging.info(f"Detected onset offset {offset:.3f}s for {Path(video_path).name}")
                    return offset

                rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
                times = librosa.times_like(rms, sr=sr, hop_length=512)
                thresh = max(0.01, float(np.median(rms) * 3.0))
                sustain_frames = max(1, int((0.05 * sr) / 512))
                for i in range(len(rms) - sustain_frames):
                    if np.all(rms[i:i + sustain_frames] > thresh):
                        offset = float(times[i])
                        offset = max(0.0, min(offset, 5.0))
                        self.onset_offset_cache[video_path] = offset
                        logging.info(f"Fallback onset offset {offset:.3f}s for {Path(video_path).name}")
                        return offset

                self.onset_offset_cache[video_path] = 0.0
                return 0.0

        except Exception as e:
            logging.debug(f"Onset offset detect error: {e}")
            try:
                if video_path:
                    self.onset_offset_cache[video_path] = 0.0
            except Exception:
                pass
            return 0.0

    def _get_media_duration(self, media_path):
        """Return media duration in seconds using ffprobe, or 0.0 on failure.
        Results are cached so parallel stem threads don't duplicate ffprobe calls."""
        try:
            if not media_path or not os.path.exists(media_path):
                return 0.0
            with self._duration_cache_lock:
                if media_path in self._duration_cache:
                    return self._duration_cache[media_path]
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(media_path)
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return 0.0
            import json as _json
            data = _json.loads(r.stdout or '{}')
            # Prefer audio stream duration if present, else format duration
            duration = 0.0
            streams = data.get('streams', [])
            for s in streams:
                if s.get('codec_type') == 'audio' and s.get('duration'):
                    try:
                        duration = float(s['duration'])
                        break
                    except:
                        pass
            if duration <= 0.0:
                fmt = data.get('format', {})
                try:
                    duration = float(fmt.get('duration', 0.0))
                except:
                    duration = 0.0
            result = max(0.0, duration)
            with self._duration_cache_lock:
                self._duration_cache[media_path] = result
            return result
        except Exception:
            return 0.0

    def _normalize_name_token(self, name: str) -> str:
        """Normalize instrument names for robust filename matching."""
        try:
            import re
            s = (name or '').lower()
            s = s.replace(' ', '_')
            s = re.sub(r'[^a-z0-9_]+', '_', s)
            s = re.sub(r'_+', '_', s).strip('_')
            return s
        except Exception:
            return (name or '').lower()

    def _find_instrument_video_file(self, instrument_name: str) -> str | None:
        """
        Find a preprocessed instrument video by matching normalized tokens against files in uploads.
        Prefers files starting with 'processed_'.
        """
        try:
            norm = self._normalize_name_token(instrument_name)
            candidates = list(self.uploads_dir.glob('processed_*.mp4')) or list(self.uploads_dir.glob('*.mp4'))
            # Prefer processed files
            for f in candidates:
                stem = self._normalize_name_token(f.stem)
                if norm in stem or stem.endswith(norm):
                    return str(f)
            # Fallback: partial token match
            tokens = [t for t in norm.split('_') if t]
            for f in candidates:
                stem = self._normalize_name_token(f.stem)
                if any(tok in stem for tok in tokens):
                    return str(f)
            avail = [p.name for p in candidates]
            logging.warning(f"❌ No instrument file match for '{instrument_name}' (norm='{norm}'). Samples: {avail[:8]}...")
            return None
        except Exception as e:
            logging.warning(f"Error finding instrument file for '{instrument_name}': {e}")
            return None

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
        """Register paths for all videos.

        When the caller supplies explicit session-specific video paths (via
        midi_data['videoFiles']), those are used exclusively — the shared
        uploads directory is NOT scanned.  This prevents videos from past
        sessions from contaminating the current composition.

        Falls back to a directory scan only when no explicit paths are provided
        (e.g. direct VideoComposer usage without the Node.js wrapper).
        """
        registry = PathRegistry.get_instance()
        registry_file = self.processed_videos_dir / "path_registry.json"

        if self.explicit_video_files:
            logging.info(
                f"Using {len(self.explicit_video_files)} explicit session video paths "
                "(shared uploads scan skipped to prevent stale-video contamination)"
            )
            self._register_explicit_video_files(registry)
        else:
            logging.info(f"Scanning uploads directory for videos: {self.uploads_dir}")
            success = registry.register_from_uploads_directory(self.uploads_dir)
            if not success:
                logging.warning(
                    "No videos found in uploads directory, trying processed videos directory as fallback"
                )
                registry.register_from_directory(self.processed_videos_dir)

        registry.save_registry(str(registry_file))
        registry.debug_dump()
        stats = registry.get_stats()
        logging.info(f"Path registry stats: {stats}")
        logging.info(f"Total videos registered: {stats['total_paths']}")

    def _register_explicit_video_files(self, registry):
        """Register session-specific video paths into PathRegistry.

        Registers each instrument for every unique MIDI note present in the
        composition so note-based lookups always resolve to the correct file.
        Uses the isDrum flag from the payload rather than name heuristics.
        """
        count = 0
        for instrument_name, video_info in self.explicit_video_files.items():
            if isinstance(video_info, dict):
                path = video_info.get('path', '')
                is_drum = video_info.get('isDrum', False)
                midi_notes = video_info.get('notes', [])
            else:
                path = str(video_info)
                is_drum = instrument_name.lower().startswith('drum_')
                midi_notes = []

            if not path or not os.path.exists(path):
                logging.warning(f"Explicit video not found, skipping: {instrument_name} → {path}")
                continue

            norm_name = instrument_name.lower().replace(' ', '_').replace('-', '_')

            if is_drum:
                # Strip leading 'drum_' prefix that the grid uses but PathRegistry doesn't store
                drum_name = norm_name[5:] if norm_name.startswith('drum_') else norm_name
                registry.register_drum(drum_name, path, validate=False)
            else:
                # Register note "60" as a guaranteed default fallback
                registry.register_instrument(norm_name, "60", path, validate=False)
                # Also register for every unique MIDI note actually used
                for midi_note in set(int(n) for n in midi_notes if isinstance(n, (int, float))):
                    registry.register_instrument(norm_name, str(midi_note), path, validate=False)

            logging.info(f"✅ Pinned explicit video: {instrument_name} → {os.path.basename(path)}")
            count += 1

        logging.info(f"Registered {count} explicit session videos (stale uploads scan bypassed)")

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
                    
                    # Find preprocessed drum file (FIX: avoid per-file warning spam)
                    found = False
                    for file in upload_files:
                        if normalized_name in file.name.lower():
                            dest_file = drum_dir / f"{normalized_name}.mp4"
                            try:
                                shutil.copy2(str(file), str(dest_file))
                                logging.info(f"Copied preprocessed drum file: {file.name} -> {dest_file}")
                            except Exception as e:
                                logging.error(f"Error copying drum file: {e}")
                            found = True
                            break
                    
                    if not found:
                        # Single warning only after exhausting all candidates
                        logging.warning(f"No preprocessed file found for {normalized_name} (searched {len(upload_files)} files)")
                
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

    def _sanitize_note_events(note_events, chunk_start, chunk_end):
        """
        note_events: list of dicts with 'start', 'end' (absolute or chunk-relative)
        Returns sanitized list (chunk-relative).
        """
        sanitized = []
        for ev in note_events:
            start = ev.get('start', 0.0)
            end = ev.get('end', start)

            # Constrain to chunk window
            if end < chunk_start or start > chunk_end:
                continue

            start = max(start, chunk_start)
            end = min(end, chunk_end)

            # Quantize
            start = round(start / VideoComposer.TIME_QUANTUM) * VideoComposer.TIME_QUANTUM
            end = round(end / VideoComposer.TIME_QUANTUM) * VideoComposer.TIME_QUANTUM

            # Enforce ordering
            if end <= start:
                end = start + VideoComposer.MIN_NOTE_DURATION

            # Clamp again to chunk_end
            if end > chunk_end:
                end = chunk_end

            # Enforce minimum duration
            if (end - start) < VideoComposer.MIN_NOTE_DURATION:
                end = min(start + VideoComposer.MIN_NOTE_DURATION, chunk_end)

            duration = end - start
            if duration < VideoComposer.MIN_NOTE_DURATION * 0.5:
                # Still too tiny after attempts → skip to avoid ffmpeg trim errors
                continue

            sanitized.append({
                **ev,
                'start': start,
                'end': end,
                'duration': duration
            })
        return sanitized

    # Helper to build safe delay (ms) values for adelay (must be non-negative int)
    def _safe_delay_ms(t_seconds):
        if t_seconds < 0:
            t_seconds = 0.0
        # Align to quantum & convert to ms
        return int(round(t_seconds * 1000.0))

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

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, chunk_start_time, chunk_duration, track_name, unique_id):
    #     """
    #     FIXED: Create MIDI-triggered video with proper delay validation
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    #         if not notes or not os.path.exists(video_path):
    #             logging.warning(f"No notes or video missing for {track_name}")
    #             return None
                
    #         if output_path.exists():
    #             output_path.unlink()

    #         # FIXED: Filter and validate notes for this chunk
    #         valid_notes = []
    #         for note in notes:
    #             note_start_abs = float(note.get('time', 0))
    #             relative_start = note_start_abs - chunk_start_time
                
    #             # FIXED: Skip notes that start before chunk (negative relative time)
    #             if relative_start < 0:
    #                 logging.debug(f"Skipping note at {note_start_abs}s (before chunk start {chunk_start_time}s)")
    #                 continue
                    
    #             duration = float(note.get('duration', 0.5))
    #             duration = min(duration, chunk_duration - relative_start)
                
    #             # FIXED: Skip notes with zero or negative duration
    #             if duration <= 0:
    #                 logging.debug(f"Skipping note with invalid duration: {duration}")
    #                 continue
                    
    #             # Add adjusted note to valid list
    #             adjusted_note = note.copy()
    #             adjusted_note['relative_time'] = relative_start
    #             adjusted_note['adjusted_duration'] = duration
    #             valid_notes.append(adjusted_note)

    #         if not valid_notes:
    #             logging.info(f"No valid notes for {track_name} in chunk time range")
    #             return None

    #         logging.info(f"🎵 Creating MIDI-triggered video for {track_name} with {len(valid_notes)} valid notes")

    #         # Create filter complex with validated delays
    #         filter_parts = []
    #         filter_parts.append(f"color=black:size=640x360:duration={chunk_duration}:rate=30[base_video]")
    #         filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}[base_audio]")

    #         video_layers = ["[base_video]"]
    #         audio_segments = ["[base_audio]"]

    #         for i, note in enumerate(valid_notes):
    #             relative_start = note['relative_time']
    #             audio_duration = note['adjusted_duration']
    #             visual_duration = max(audio_duration, 0.5)  # Minimum visual duration
    #             pitch = note.get('midi', 60)

    #             # Create video segment
    #             filter_parts.append(f"[0:v]trim=0:{visual_duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                
    #             # Create audio segment with pitch adjustment
    #             pitch_semitones = pitch - 60
    #             if abs(pitch_semitones) > 0.1:  # Apply pitch shift if needed
    #                 pitch_factor = 2 ** (pitch_semitones / 12.0)
    #                 filter_parts.append(f"[0:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS,asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]")
    #             else:
    #                 filter_parts.append(f"[0:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS[note_a{i}]")
                
    #             # Video overlay with validated timing
    #             prev_video = video_layers[-1]
    #             filter_parts.append(f"{prev_video}[note_v{i}]overlay=enable='between(t,{relative_start},{relative_start + visual_duration})'[video_out{i}]")
    #             video_layers.append(f"[video_out{i}]")
                
    #             # FIXED: Ensure delay is non-negative
    #             delay_ms = max(0, int(relative_start * 1000))
    #             filter_parts.append(f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]")
    #             audio_segments.append(f"[delayed_a{i}]")

    #         # Mix audio segments
    #         if len(audio_segments) > 1:
    #             audio_inputs = ''.join(audio_segments)
    #             filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
    #         else:
    #             filter_parts.append("[base_audio]copy[final_audio]")

    #         # Final video output
    #         final_video = video_layers[-1] if len(video_layers) > 1 else "[base_video]"
    #         filter_parts.append(f"{final_video}copy[final_video]")

    #         # Build FFmpeg command
    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-i', str(video_path),
    #             '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={chunk_duration}:rate=30',
    #             '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}',
    #             '-filter_complex', ';'.join(filter_parts),
    #             '-map', '[final_video]',
    #             '-map', '[final_audio]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-t', str(chunk_duration),
    #             '-r', '30',
    #             str(output_path)
    #         ]

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

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, chunk_start, chunk_duration, track_name, unique_id):
    #     """
    #     Stable single implementation. Notes have absolute times.
    #     """
    #     try:
    #         out = self.temp_dir / f"{track_name}_{unique_id}.mp4"
    #         if out.exists():
    #             out.unlink()
    #         if not os.path.exists(video_path):
    #             return None
    #         # Build relative sanitized notes
    #         rel = []
    #         for n in notes:
    #             abs_start = float(n.get("time", 0.0))
    #             dur = float(n.get("duration", 0.0))
    #             start = abs_start - chunk_start
    #             if dur <= 0: dur = 0.1
    #             if start >= chunk_duration: continue
    #             end = start + dur
    #             if end <= 0: continue
    #             if end > chunk_duration:
    #                 dur = chunk_duration - start
    #                 end = start + dur
    #             if dur <= 0: continue
    #             rel.append((round(start,3), round(dur,3), int(n.get("midi",60))))
    #         if not rel:
    #             return self._create_simple_loop(video_path, out, chunk_duration)
    #         rel.sort()
    #         filter_parts = [
    #             f"[1:v]trim=0:{chunk_duration},setpts=PTS-STARTPTS[base_v]",
    #             f"[2:a]atrim=0:{chunk_duration},asetpts=PTS-STARTPTS[base_a]"
    #         ]
    #         video_chain = "[base_v]"
    #         audio_inputs = ["[base_a]"]
    #         for i,(start,dur,midi_note) in enumerate(rel):
    #             pitch = 2 ** ((midi_note - 60) / 12.0)
    #             filter_parts.append(f"[0:v]trim=0:{dur},setpts=PTS-STARTPTS,scale=640:360[v{i}]")
    #             if abs(pitch-1.0) > 0.01:
    #                 filter_parts.append(f"[0:a]atrim=0:{dur},asetpts=PTS-STARTPTS,asetrate=44100*{pitch},aresample=44100[a{i}]")
    #             else:
    #                 filter_parts.append(f"[0:a]atrim=0:{dur},asetpts=PTS-STARTPTS[a{i}]")
    #             filter_parts.append(f"{video_chain}[v{i}]overlay=enable='between(t,{start},{start+dur})'[ov{i}]")
    #             video_chain = f"[ov{i}]"
    #             delay_ms = int(start*1000)
    #             filter_parts.append(f"[a{i}]adelay={delay_ms}|{delay_ms}[ad{i}]")
    #             audio_inputs.append(f"[ad{i}]")
    #         if len(audio_inputs) == 1:
    #             filter_parts.append(f"{audio_inputs[0]}anull[final_a]")
    #         else:
    #             filter_parts.append("".join(audio_inputs)+f"amix=inputs={len(audio_inputs)}:duration=longest:dropout_transition=0[final_a]")
    #         filter_parts.append(f"{video_chain}format=yuv420p[final_v]")
    #         cmd = [
    #             "ffmpeg","-y",
    #             "-i", str(video_path),
    #             "-f","lavfi","-i", f"color=black:size=640x360:rate=30:duration={chunk_duration}",
    #             "-f","lavfi","-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}",
    #             "-filter_complex",";".join(filter_parts),
    #             "-map","[final_v]","-map","[final_a]",
    #             "-t", f"{chunk_duration:.3f}",
    #             "-c:v","libx264","-preset","fast","-crf","23",
    #             "-c:a","aac","-b:a","192k",
    #             "-r","30",
    #             str(out)
    #         ]
    #         r = subprocess.run(cmd, capture_output=True, text=True)
    #         if r.returncode != 0:
    #             logging.error(f"Instrument note-trigger failed (fallback loop) {track_name}: {r.stderr}")
    #             return self._create_simple_loop(video_path, out, chunk_duration)
    #         return str(out)
    #     except Exception as e:
    #         logging.error(f"Note trigger exception {track_name}: {e}")
    #         return self._create_simple_loop(video_path, self.temp_dir / f"{track_name}_{unique_id}.mp4", chunk_duration)

    @staticmethod
    def _build_atempo_chain(rate: float) -> str:
        """
        Build an FFmpeg atempo filter chain for a given playback rate.

        FFmpeg's atempo filter is restricted to [0.5, 100] per stage.
        For values outside that range we chain multiple stages whose
        product equals the target rate.

        For best quality, each stage is kept inside [0.5, 2.0].
        """
        STAGE_MIN, STAGE_MAX = 0.5, 2.0
        if rate <= 0:
            rate = 1.0
        filters = []
        while rate < STAGE_MIN:
            filters.append(f"atempo={STAGE_MIN:.6f}")
            rate /= STAGE_MIN  # remaining factor
        while rate > STAGE_MAX:
            filters.append(f"atempo={STAGE_MAX:.6f}")
            rate /= STAGE_MAX
        filters.append(f"atempo={rate:.6f}")
        return ",".join(filters)

    def _create_note_triggered_video_sequence_fixed(
        self,
        video_path,
        notes,
        total_duration,
        track_name,
        unique_id,
        chunk_start_time=0.0,
        onset_offset=0.0,
        note_audio_map=None,
    ):
        """
        WORKING unified note-triggered clip builder.

        Args:
            video_path: source instrument video (single reference performance)
            notes: list of note dicts (absolute or chunk-relative 'time', 'duration', 'midi')
            total_duration: target output duration for this chunk
            track_name: for filename/logging
            unique_id: short id to avoid collisions
            chunk_start_time: absolute start of this chunk (so we can convert absolute note times)
            note_audio_map: optional Dict[int, str] mapping midi_note → path of pre-tuned video.
                When provided the pre-tuned file's audio stream is used for that note (higher
                quality than the asetrate fallback).

        Returns:
            str path or None
        """
        try:
            if not video_path or not os.path.exists(video_path):
                logging.warning(f"[NoteTrigger] Missing video for {track_name}")
                return None

            out_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            if out_path.exists():
                try:
                    out_path.unlink()
                except:
                    pass

            # Sanitize & normalize notes
            MIN_DUR = 0.10  # 100 ms min to avoid ffmpeg micro durations
            valid = []
            for n in notes or []:
                raw_start = float(n.get("time", 0.0))
                dur = float(n.get("duration", 0.0))
                if dur <= 0:
                    continue

                # Convert to chunk-relative
                rel_start = raw_start - chunk_start_time
                # If notes were already relative (e.g. small start while chunk_start_time>0),
                # allow negative tolerance then clamp.
                if rel_start < -0.001:
                    # Starts before this chunk; trim head
                    head_trim = -rel_start
                    dur -= head_trim
                    rel_start = 0.0
                if rel_start >= total_duration:
                    continue

                # Clamp to chunk boundary
                if rel_start + dur > total_duration:
                    dur = total_duration - rel_start
                if dur <= 0:
                    continue
                if dur < MIN_DUR:
                    dur = min(MIN_DUR, max(0.0, total_duration - rel_start))
                    if dur <= 0:
                        continue

                midi_note = int(n.get("midi", 60))
                valid.append((round(rel_start, 3), round(dur, 3), midi_note))

            if not valid:
                # Fallback: simple loop (keeps something visible)
                return self._create_simple_loop(video_path, out_path, total_duration)

            valid.sort(key=lambda x: x[0])

            # Build the mapping from midi_note → FFmpeg input index for pre-tuned audio.
            # Inputs 0, 1, 2 are: source video, black frame, silent audio.
            # Pre-tuned video files (audio only) start at index 3.
            note_input_index: dict = {}  # midi_note -> ffmpeg input index
            extra_audio_inputs: list = []  # paths appended as additional -i args
            BASE_EXTRA_IDX = 3
            if note_audio_map:
                unique_cached = sorted({mn for (_, _, mn) in valid if mn in note_audio_map})
                for mn in unique_cached:
                    note_input_index[mn] = BASE_EXTRA_IDX + len(extra_audio_inputs)
                    extra_audio_inputs.append(note_audio_map[mn])

            # Build filter parts
            filter_parts = [
                # Base black video & silent audio come from inputs 1 & 2
                f"[1:v]trim=0:{total_duration},setpts=PTS-STARTPTS[base_v]",
                f"[2:a]atrim=0:{total_duration},asetpts=PTS-STARTPTS[base_a]"
            ]

            video_chain = "[base_v]"
            audio_streams = ["[base_a]"]

            # Base onset offset
            onset_base = 0.0 if onset_offset is None else max(0.0, min(float(onset_offset), 5.0))
            source_duration = self._get_media_duration(video_path)

            for i, (start, dur, midi_note) in enumerate(valid):
                # Pitch factor relative to C4 (60)
                pitch_factor = 2 ** ((midi_note - 60) / 12.0)

                # Compute safe onset per note so trim doesn't overshoot
                safe_onset = onset_base
                if source_duration > 0.0:
                    max_start = max(0.0, source_duration - dur - 0.01)
                    safe_onset = min(onset_base, max_start)
                if safe_onset < 0.0:
                    safe_onset = 0.0
                logging.info(f"[NoteTrigger] {track_name} note {i}: onset_base={onset_base:.3f}s, safe_onset={safe_onset:.3f}s, dur={dur:.3f}s")

                # Video always comes from the original source (input 0)
                filter_parts.append(f"[0:v]trim=start={safe_onset}:duration={dur},setpts=PTS-STARTPTS,scale=640:360[v{i}]")

                # Audio: prefer pre-tuned file → fall back to asetrate+atempo
                if midi_note in note_input_index:
                    idx = note_input_index[midi_note]
                    filter_parts.append(
                        f"[{idx}:a]atrim=start={safe_onset}:duration={dur},asetpts=PTS-STARTPTS[a{i}]"
                    )
                elif abs(pitch_factor - 1.0) > 0.01:
                    # asetrate shifts pitch but compresses/stretches duration by 1/pitch_factor.
                    # atempo chain compensates to restore the original duration.
                    atempo = self._build_atempo_chain(1.0 / pitch_factor)
                    filter_parts.append(
                        f"[0:a]atrim=start={safe_onset}:duration={dur},asetpts=PTS-STARTPTS,"
                        f"asetrate=44100*{pitch_factor},aresample=44100,{atempo}[a{i}]"
                    )
                else:
                    filter_parts.append(f"[0:a]atrim=start={safe_onset}:duration={dur},asetpts=PTS-STARTPTS[a{i}]")

                # Overlay enable window
                end = start + dur
                filter_parts.append(
                    f"{video_chain}[v{i}]overlay=enable='between(t,{start:.3f},{end:.3f})'[ov{i}]"
                )
                video_chain = f"[ov{i}]"

                # Delay audio
                delay_ms = int(start * 1000)
                filter_parts.append(f"[a{i}]adelay={delay_ms}|{delay_ms}[ad{i}]")
                audio_streams.append(f"[ad{i}]")

            if len(audio_streams) == 1:
                filter_parts.append(f"{audio_streams[0]}anull[final_a]")
            else:
                filter_parts.append(
                    f"{''.join(audio_streams)}amix=inputs={len(audio_streams)}:"
                    f"duration=longest:dropout_transition=0[final_a]"
                )

            filter_parts.append(f"{video_chain}format=yuv420p[final_v]")

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-f", "lavfi", "-i", f"color=black:size=640x360:rate=30:duration={total_duration}",
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}",
            ]
            # Add pre-tuned audio sources (one per unique cached midi_note)
            for tuned_path in extra_audio_inputs:
                cmd += ["-i", str(tuned_path)]

            tail_args = [
                "-map", "[final_v]", "-map", "[final_a]",
                "-t", f"{total_duration:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-r", "30",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                str(out_path)
            ]
            r = self._run_ffmpeg_with_filter_script(cmd, filter_parts, tail_args)
            if r.returncode != 0:
                logging.error(f"[NoteTrigger] ffmpeg failed for {track_name}: {r.stderr[-2000:]}")
                return self._create_simple_loop(video_path, out_path, total_duration)
            return str(out_path)
        except Exception as e:
            logging.error(f"[NoteTrigger] Exception for {track_name}: {e}", exc_info=True)
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
        """Analyze MIDI composition to find all required instrument/note combinations.
        Drum tracks are excluded because they don't need pitch tuning."""
        requirements = {}
        
        for track in self.midi_data.get('tracks', []):
            # Skip drum tracks — pitch tuning doesn't apply to percussion
            if track.get('isDrum') or track.get('channel') == 9:
                continue

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
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:print_format=json',
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
            # Skip if audio is silent (input_i = -inf) — loudnorm can't handle it
            if stats.get('input_i') in ('-inf', 'inf'):
                logging.warning("⚠️  Audio is silent (-inf), skipping loudnorm pass 2.")
                shutil.copy2(input_path, output_path)
                return str(output_path)

            logging.info("   (Loudnorm Pass 2/2) Applying normalization...")
            pass2_cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-af', f'loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:'
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

    # ── Audio-first pipeline ─────────────────────────────────────────────────

    def _parse_loudnorm_stats(self, stderr_output: str):
        """Extract loudnorm JSON stats block from FFmpeg stderr."""
        import re as _re, json as _json
        patterns = [
            r'\{[^{}]*"input_i"[^{}]*"input_tp"[^{}]*"input_lra"[^{}]*"input_thresh"[^{}]*"target_offset"[^{}]*\}',
            r'\{[^{}]*"input_i"[^{}]*\}',
            r'(\{(?:[^{}]|{[^{}]*})*"input_i"(?:[^{}]|{[^{}]*})*\})',
        ]
        for pattern in patterns:
            for match in _re.findall(pattern, stderr_output, _re.DOTALL):
                try:
                    stats = _json.loads(match)
                    if all(k in stats for k in ['input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset']):
                        return stats
                except _json.JSONDecodeError:
                    continue
        for line in stderr_output.split('\n'):
            line = line.strip()
            if line.startswith('{') and 'input_i' in line:
                try:
                    stats = _json.loads(line)
                    if 'input_i' in stats and 'target_offset' in stats:
                        return stats
                except _json.JSONDecodeError:
                    continue
        return None

    def _normalize_audio_file(self, input_path: str, output_path: str,
                               total_duration: float = None) -> 'str | None':
        """
        Two-pass loudnorm + alimiter on an audio-only file.
        Returns output_path on success, or None on failure.
        """
        try:
            logging.info("🔊 Normalising audio timeline (pass 1/2)…")
            p1 = subprocess.run(
                ['ffmpeg', '-y', '-i', input_path,
                 '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:print_format=json',
                 '-f', 'null', '-'],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if p1.returncode != 0:
                raise Exception("loudnorm pass 1 failed")

            stats = self._parse_loudnorm_stats(p1.stderr)
            if not stats:
                raise Exception("Failed to parse loudnorm stats")

            if stats.get('input_i') in ('-inf', 'inf'):
                logging.warning("Audio is silent — encoding without loudnorm")
                args = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'aac', '-b:a', '320k']
                if total_duration:
                    args += ['-t', str(total_duration + 0.5)]
                args.append(str(output_path))
                subprocess.run(args, capture_output=True, check=True)
                return str(output_path)

            # If the mix is already within ±3 LUFS of the -16 target AND the true
            # peak is safe, skip the loudnorm boost to preserve the user's balance.
            try:
                input_lufs = float(stats['input_i'])
                input_tp   = float(stats['input_tp'])
            except (KeyError, ValueError, TypeError):
                input_lufs = -100.0
                input_tp   = -100.0

            near_target = (-19.0 <= input_lufs <= -13.0)
            peak_safe   = (input_tp <= -0.5)

            if near_target and peak_safe:
                logging.info(
                    f"🔊 Mix already at {input_lufs:.1f} LUFS (peak {input_tp:.1f} dBTP) — "
                    f"applying limiter only (skip loudnorm boost to preserve user balance)"
                )
                af_passthru = "alimiter=limit=0.95:attack=20:release=200"
                args = ['ffmpeg', '-y', '-i', input_path, '-af', af_passthru, '-c:a', 'aac', '-b:a', '320k']
                if total_duration:
                    args += ['-t', str(total_duration + 0.5)]
                args.append(str(output_path))
                p2 = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
                if p2.returncode == 0 and os.path.exists(output_path):
                    logging.info("✅ Audio timeline encoded (limiter only)")
                    return str(output_path)
                # fall through to full loudnorm if this fails

            logging.info(f"🔊 Normalising audio timeline (pass 2/2) — input {input_lufs:.1f} LUFS…")
            af = (
                f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:"
                f"measured_I={stats['input_i']}:"
                f"measured_LRA={stats['input_lra']}:"
                f"measured_tp={stats['input_tp']}:"
                f"measured_thresh={stats['input_thresh']}:"
                f"offset={stats['target_offset']},"
                f"alimiter=limit=0.95:attack=20:release=200"
            )
            args = ['ffmpeg', '-y', '-i', input_path, '-af', af, '-c:a', 'aac', '-b:a', '320k']
            if total_duration:
                args += ['-t', str(total_duration + 0.5)]
            args.append(str(output_path))
            p2 = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if p2.returncode == 0 and os.path.exists(output_path):
                logging.info("✅ Audio timeline normalised")
                return str(output_path)
            raise Exception(f"loudnorm pass 2 failed: {p2.stderr[-300:]}")
        except Exception as e:
            logging.error(f"_normalize_audio_file failed: {e}")
            try:
                args = ['ffmpeg', '-y', '-i', input_path, '-c:a', 'aac', '-b:a', '320k']
                if total_duration:
                    args += ['-t', str(total_duration + 0.5)]
                args.append(str(output_path))
                subprocess.run(args, capture_output=True, check=True)
                return str(output_path)
            except Exception:
                return None

    def _batch_amix(self, filter_parts: list, labels: list, batch_size: int = 500) -> list:
        """
        Split audio labels into sub-mixes to stay within FFmpeg's ~1024 filter
        input limit. Appends new filter entries to filter_parts in-place and
        returns a list of sub-mix output labels.
        """
        batches = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
        batch_labels = []
        for b_idx, batch in enumerate(batches):
            bl = f"[submix_{b_idx}]"
            if len(batch) == 1:
                filter_parts.append(f"{batch[0]}anull{bl}")
            else:
                filter_parts.append(
                    f"{''.join(batch)}amix=inputs={len(batch)}:normalize=0{bl}"
                )
            batch_labels.append(bl)
        return batch_labels

    def _run_ffmpeg_with_filter_script(self, cmd: list, filter_parts: list,
                                        tail_args: list) -> 'subprocess.CompletedProcess':
        """
        Run FFmpeg using -filter_complex_script to avoid Windows' 32,767-character
        command-line length limit when filter graphs are large.

        Acquires self._ffmpeg_semaphore before launching so nested parallelism
        (outer stem workers + inner sub-batch workers + video chunk workers) never
        over-subscribes the machine.

        Writes the filter graph to a temp file inside self.temp_dir, then
        replaces the inline -filter_complex argument with -filter_complex_script.
        The temp file is always deleted in the finally block.
        """
        script_path = None
        try:
            import tempfile as _tf
            fd, script_path = _tf.mkstemp(suffix='_fc.txt', dir=str(self.temp_dir))
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(';'.join(filter_parts))
            full_cmd = cmd + ['-filter_complex_script', script_path] + tail_args
            with self._ffmpeg_semaphore:
                return subprocess.run(
                    full_cmd, capture_output=True, text=True,
                    encoding='utf-8', errors='replace'
                )
        finally:
            if script_path and os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

    # ── Sub-batch tuning constants ─────────────────────────────────────────
    # These are data-driven: sparse MIDIs (<= threshold) build monolithically;
    # dense MIDIs auto-split into parallel sub-batches.
    _SUBBATCH_THRESHOLD = 400   # total note occurrences; below → monolithic
    _SUBBATCH_TARGET    = 350   # target occurrences per sub-batch
    _MAX_SUBBATCHES     = 4     # cap per stem (avoid spawning too many FFmpeg)

    def _mix_stem_wavs(self, input_paths: list, output_path: str) -> 'str | None':
        """Mix N per-stem WAV files into one via FFmpeg amix, then delete the inputs."""
        cmd = ['ffmpeg', '-y']
        for p in input_paths:
            cmd.extend(['-i', p])
        n = len(input_paths)
        mix_filter = (
            f"[0:a]anull[out]" if n == 1
            else "".join(f"[{i}:a]" for i in range(n)) + f"amix=inputs={n}:normalize=0[out]"
        )
        cmd += ['-filter_complex', mix_filter, '-map', '[out]',
                '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_path]
        with self._ffmpeg_semaphore:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            for p in input_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass
            return output_path
        logging.error(f"❌ _mix_stem_wavs failed: {r.stderr[-300:]}")
        return None

    def _build_instrument_stem(self, track_name: str, notes: list,
                                stem_path) -> 'str | None':
        """
        Build a per-instrument WAV stem — public entry point.

        For small tracks (≤ _SUBBATCH_THRESHOLD occurrences) the stem is built
        in one FFmpeg call (monolithic).  For dense tracks the note occurrence
        list is split into parallel sub-batches of _SUBBATCH_TARGET notes each,
        each sub-batch runs as its own FFmpeg job, then the results are mixed.
        Splitting is by occurrence count (not unique-MIDI-note count) so a
        single high-density pitch is also split correctly.

        The shared_onset is computed once from the full clip set and passed into
        every sub-batch so all clips trim from the same attack position.
        """
        if not notes:
            return None

        normalized_name = normalize_instrument_name(track_name)
        track_vol_db = float(self._resolve_segment_volume(
            {'track_name': track_name, 'type': 'instrument'}
        ))

        # Group occurrences by MIDI note; resolve cached clip paths
        by_midi: dict = {}
        for note in notes:
            midi = note.get('midi')
            if midi is None:
                continue
            cached_path = self._tuned_videos_cache.get(normalized_name, {}).get(midi)
            if not cached_path or not os.path.exists(cached_path):
                logging.debug(f"No cached clip for {track_name} MIDI {midi} — note skipped")
                continue
            by_midi.setdefault(midi, {'path': cached_path, 'occurrences': []})['occurrences'].append(note)

        if not by_midi:
            logging.warning(f"No cached clips for {track_name} — stem skipped")
            return None

        # Compute shared_onset ONCE from the full clip set so all sub-batches
        # (and the monolithic path) trim from the same attack position.
        first_path = next(iter(by_midi.values()))['path']
        shared_onset = self._get_onset_offset(first_path)

        # Flatten to (midi, path, note) triples for occurrence-count splitting
        flat = [
            (midi, info['path'], note)
            for midi, info in by_midi.items()
            for note in info['occurrences']
        ]
        total_occ = len(flat)

        if total_occ <= self._SUBBATCH_THRESHOLD:
            return self._build_instrument_stem_inner(
                track_name, track_vol_db, by_midi, shared_onset, stem_path)

        # ── Sub-batch path ────────────────────────────────────────────────
        n_batches = min(self._MAX_SUBBATCHES, max(2, math.ceil(total_occ / self._SUBBATCH_TARGET)))
        chunk_size = math.ceil(total_occ / n_batches)
        chunks = [flat[i:i + chunk_size] for i in range(0, total_occ, chunk_size)]
        chunks = [c for c in chunks if c]  # drop empty tail
        stem_path = Path(stem_path)
        temp_paths = [stem_path.with_suffix(f'.sub{i}.wav') for i in range(len(chunks))]

        logging.info(
            f"🔀 Sub-batching {track_name}: {total_occ} notes → "
            f"{len(chunks)} batches of ~{chunk_size}"
        )

        def _build_sub(idx):
            chunk = chunks[idx]
            sub_by_midi: dict = {}
            for midi, path, note in chunk:
                sub_by_midi.setdefault(midi, {'path': path, 'occurrences': []})['occurrences'].append(note)
            return self._build_instrument_stem_inner(
                track_name, track_vol_db, sub_by_midi, shared_onset, temp_paths[idx])

        sub_results = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            futures = {ex.submit(_build_sub, i): i for i in range(len(chunks))}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r:
                        sub_results.append(r)
                except Exception as exc:
                    logging.error(f"❌ Sub-batch error for {track_name}: {exc}")

        if len(sub_results) < len(chunks):
            # Partial failure — clean up and fall back to monolithic
            logging.warning(
                f"⚠️ {track_name}: {len(chunks) - len(sub_results)} sub-batch(es) failed "
                f"— falling back to monolithic build"
            )
            for p in sub_results:
                try:
                    os.unlink(p)
                except Exception:
                    pass
            return self._build_instrument_stem_inner(
                track_name, track_vol_db, by_midi, shared_onset, stem_path)

        if len(sub_results) == 1:
            os.replace(sub_results[0], str(stem_path))
            return str(stem_path)
        return self._mix_stem_wavs(sub_results, str(stem_path))

    def _build_instrument_stem_inner(self, track_name: str, track_vol_db: float,
                                      by_midi: dict, forced_onset: float,
                                      stem_path) -> 'str | None':
        """
        Build one WAV stem from a by_midi dict using a single FFmpeg invocation.
        Called both by the monolithic path and by each sub-batch worker.
        Uses forced_onset for all trim positions so sub-batches stay in sync.
        """
        cmd = ['ffmpeg', '-y']
        input_index: dict = {}
        for idx, (midi, info) in enumerate(by_midi.items()):
            cmd.extend(['-i', info['path']])
            input_index[midi] = idx

        filter_parts: list = []
        final_labels: list = []

        for midi, info in by_midi.items():
            i_idx = input_index[midi]
            occurrences = info['occurrences']
            clip_dur = self._get_media_duration(info['path'])
            n = len(occurrences)

            if n > 1:
                split_lbls = [f"sp{i_idx}_{j}" for j in range(n)]
                filter_parts.append(
                    f"[{i_idx}:a]asplit={n}" + "".join(f"[{l}]" for l in split_lbls)
                )
            else:
                split_lbls = [f"sp{i_idx}_0"]
                filter_parts.append(f"[{i_idx}:a]anull[{split_lbls[0]}]")

            for j, note in enumerate(occurrences):
                note_dur = float(note.get('duration', 1.0))
                t_ms = int(float(note.get('time', 0)) * 1000)
                velocity = (note.get('velocity') or note.get('midi_velocity')
                            or note.get('vel') or 100)
                vol_db = track_vol_db + self._velocity_to_db(velocity)
                vol_linear = max(0.01, 10 ** (vol_db / 20.0))

                trim_start = forced_onset
                trim_end = trim_start + note_dur
                if clip_dur and trim_end > clip_dur:
                    if trim_end - clip_dur > 0.05:
                        logging.debug(
                            f"{track_name} MIDI {midi} note {j}: dur {note_dur:.2f}s "
                            f"exceeds clip {clip_dur:.2f}s — audio will truncate"
                        )
                    trim_end = clip_dur

                lbl_out = f"n{i_idx}_{j}"
                seg_dur = trim_end - trim_start
                fade_in  = min(0.003, seg_dur * 0.05)
                fade_out = min(0.005, seg_dur * 0.05)
                fade_out_st = max(0.0, seg_dur - fade_out)
                filter_parts.append(
                    f"[{split_lbls[j]}]"
                    f"atrim={trim_start:.4f}:{trim_end:.4f},"
                    f"asetpts=PTS-STARTPTS,"
                    f"volume={vol_linear:.4f},"
                    f"afade=t=in:st=0:d={fade_in:.4f},"
                    f"afade=t=out:st={fade_out_st:.4f}:d={fade_out:.4f},"
                    f"adelay={t_ms}|{t_ms}"
                    f"[{lbl_out}]"
                )
                final_labels.append(f"[{lbl_out}]")

        if not final_labels:
            return None

        total = len(final_labels)
        if total > 900:
            batch_labels = self._batch_amix(filter_parts, final_labels)
            filter_parts.append(
                f"{''.join(batch_labels)}amix=inputs={len(batch_labels)}:normalize=0[stem]"
            )
        elif total == 1:
            filter_parts.append(f"{final_labels[0]}anull[stem]")
        else:
            filter_parts.append(f"{''.join(final_labels)}amix=inputs={total}:normalize=0[stem]")

        r = self._run_ffmpeg_with_filter_script(
            cmd, filter_parts,
            ['-map', '[stem]', '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(stem_path)]
        )
        if r.returncode == 0 and os.path.exists(stem_path) and os.path.getsize(stem_path) > 100:
            logging.info(f"✅ Instrument stem: {track_name} ({total} note events)")
            return str(stem_path)
        logging.error(f"❌ Instrument stem failed for {track_name}: {r.stderr[-500:]}")
        return None

    def _build_drum_stem(self, drum_track: dict, stem_path) -> 'str | None':
        """
        Build a per-drum-track WAV stem — public entry point.

        Maps MIDI notes to named drum sounds via DRUM_NOTES, then delegates to
        _build_drum_stem_inner.  For dense drum tracks (> _SUBBATCH_THRESHOLD
        total hits) the hit list is split into parallel sub-batches and the
        results are mixed — identical to the instrument sub-batch strategy.
        Each drum type computes its own onset independently (no shared_onset
        coordination needed, unlike instruments).
        """
        notes = drum_track.get('notes', [])
        if not notes:
            return None

        # Map all notes to drum names and resolve video paths
        by_drum: dict = {}
        for note in notes:
            midi_note = note.get('midi')
            if midi_note is None:
                continue
            drum_name = DRUM_NOTES.get(midi_note, f'Unknown_Drum_{midi_note}')
            if drum_name.startswith('Unknown_Drum_'):
                continue
            if drum_name not in by_drum:
                drum_path = self._find_drum_video_file_flexible(drum_name)
                if not drum_path or not os.path.exists(drum_path):
                    logging.debug(f"No video for drum {drum_name} — skipped")
                    continue
                by_drum[drum_name] = {'path': drum_path, 'occurrences': []}
            by_drum[drum_name]['occurrences'].append(note)

        if not by_drum:
            return None

        # Flatten for occurrence-count-based splitting
        flat = [
            (drum_name, info['path'], note)
            for drum_name, info in by_drum.items()
            for note in info['occurrences']
        ]
        total_hits = len(flat)

        if total_hits <= self._SUBBATCH_THRESHOLD:
            return self._build_drum_stem_inner(by_drum, stem_path)

        # ── Sub-batch path ────────────────────────────────────────────────
        n_batches = min(self._MAX_SUBBATCHES, max(2, math.ceil(total_hits / self._SUBBATCH_TARGET)))
        chunk_size = math.ceil(total_hits / n_batches)
        chunks = [flat[i:i + chunk_size] for i in range(0, total_hits, chunk_size)]
        chunks = [c for c in chunks if c]
        stem_path = Path(stem_path)
        temp_paths = [stem_path.with_suffix(f'.dsub{i}.wav') for i in range(len(chunks))]

        logging.info(
            f"🥁 Sub-batching drums: {total_hits} hits → "
            f"{len(chunks)} batches of ~{chunk_size}"
        )

        def _build_drum_sub(idx):
            chunk = chunks[idx]
            sub_by_drum: dict = {}
            for drum_name, path, note in chunk:
                sub_by_drum.setdefault(drum_name, {'path': path, 'occurrences': []})['occurrences'].append(note)
            return self._build_drum_stem_inner(sub_by_drum, temp_paths[idx])

        sub_results = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            futures = {ex.submit(_build_drum_sub, i): i for i in range(len(chunks))}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r:
                        sub_results.append(r)
                except Exception as exc:
                    logging.error(f"❌ Drum sub-batch error: {exc}")

        if len(sub_results) < len(chunks):
            logging.warning(
                f"⚠️ Drums: {len(chunks) - len(sub_results)} sub-batch(es) failed "
                f"— falling back to monolithic drum build"
            )
            for p in sub_results:
                try:
                    os.unlink(p)
                except Exception:
                    pass
            return self._build_drum_stem_inner(by_drum, stem_path)

        if len(sub_results) == 1:
            os.replace(sub_results[0], str(stem_path))
            return str(stem_path)
        return self._mix_stem_wavs(sub_results, str(stem_path))

    def _build_drum_stem_inner(self, by_drum: dict, stem_path) -> 'str | None':
        """Build one drum WAV stem from a by_drum group using a single FFmpeg call."""
        cmd = ['ffmpeg', '-y']
        drum_input_idx: dict = {}
        for idx, (drum_name, info) in enumerate(by_drum.items()):
            cmd.extend(['-i', info['path']])
            drum_input_idx[drum_name] = idx

        filter_parts: list = []
        final_labels: list = []

        for drum_name, info in by_drum.items():
            d_idx = drum_input_idx[drum_name]
            occurrences = info['occurrences']
            onset = self._get_onset_offset(info['path'])
            clip_dur = self._get_media_duration(info['path'])
            n = len(occurrences)
            track_vol_db = float(self._resolve_segment_volume(
                {'type': 'drum', 'drum_name': drum_name, 'track_name': drum_name}
            ))

            if n > 1:
                split_lbls = [f"ds{d_idx}_{j}" for j in range(n)]
                filter_parts.append(
                    f"[{d_idx}:a]asplit={n}" + "".join(f"[{l}]" for l in split_lbls)
                )
            else:
                split_lbls = [f"ds{d_idx}_0"]
                filter_parts.append(f"[{d_idx}:a]anull[{split_lbls[0]}]")

            for j, note in enumerate(occurrences):
                note_dur = float(note.get('duration', 0.25))
                t_ms = int(float(note.get('time', 0)) * 1000)
                velocity = (note.get('velocity') or note.get('midi_velocity')
                            or note.get('vel') or 100)
                vol_db = track_vol_db + self._velocity_to_db(velocity)
                vol_linear = max(0.01, 10 ** (vol_db / 20.0))

                trim_start = onset
                trim_end = min(trim_start + note_dur, clip_dur) if clip_dur else trim_start + note_dur

                lbl_out = f"dn{d_idx}_{j}"
                seg_dur = trim_end - trim_start
                # Drums: no fade-in (preserve transient attack), only fade-out to avoid tail clicks
                fade_out = min(0.005, seg_dur * 0.05)
                fade_out_st = max(0.0, seg_dur - fade_out)
                filter_parts.append(
                    f"[{split_lbls[j]}]"
                    f"atrim={trim_start:.4f}:{trim_end:.4f},"
                    f"asetpts=PTS-STARTPTS,"
                    f"volume={vol_linear:.4f},"
                    f"afade=t=out:st={fade_out_st:.4f}:d={fade_out:.4f},"
                    f"adelay={t_ms}|{t_ms}"
                    f"[{lbl_out}]"
                )
                final_labels.append(f"[{lbl_out}]")

        if not final_labels:
            return None

        total = len(final_labels)
        if total > 900:
            batch_labels = self._batch_amix(filter_parts, final_labels)
            filter_parts.append(
                f"{''.join(batch_labels)}amix=inputs={len(batch_labels)}:normalize=0[dstem]"
            )
        elif total == 1:
            filter_parts.append(f"{final_labels[0]}anull[dstem]")
        else:
            filter_parts.append(f"{''.join(final_labels)}amix=inputs={total}:normalize=0[dstem]")

        r = self._run_ffmpeg_with_filter_script(
            cmd, filter_parts,
            ['-map', '[dstem]', '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(stem_path)]
        )
        if r.returncode == 0 and os.path.exists(stem_path) and os.path.getsize(stem_path) > 100:
            logging.info(f"✅ Drum stem built ({total} hits)")
            return str(stem_path)
        logging.error(f"❌ Drum stem failed: {r.stderr[-500:]}")
        return None

    def _build_full_audio_timeline(self, total_duration: float) -> 'str | None':
        """
        Phase 1: Build one mastered audio file covering the entire song.
        All instruments and drums are mixed with a single loudnorm + alimiter
        pass — eliminating per-chunk boundary pumping/phasing.

        Stems are built in parallel (one FFmpeg process per track).  Each stem
        writes to its own unique file, so there are no write conflicts.
        Thread-safety is guaranteed by per-path locks in _get_onset_offset and
        _get_media_duration's cache lock.
        """
        audio_dir = self.temp_dir / "audio_first"
        audio_dir.mkdir(exist_ok=True)

        # Collect all stem tasks (instruments + drums) before launching threads.
        tasks: list = []
        for entry_id, track in self.tracks.items():
            instr = track.get('instrument', {})
            track_name = (instr.get('name', 'unknown') if isinstance(instr, dict)
                          else str(instr or 'unknown'))
            notes = track.get('notes', [])
            if not notes:
                continue
            stem_path = audio_dir / f"stem_{entry_id}_{normalize_instrument_name(track_name)}.wav"
            tasks.append(('instrument', entry_id, track_name, notes, stem_path))

        for drum_idx, drum_track in enumerate(self.drum_tracks):
            instr = drum_track.get('instrument', {})
            tag = instr.get('name', 'drums') if isinstance(instr, dict) else 'drums'
            stem_path = audio_dir / f"stem_drums_{drum_idx}_{normalize_instrument_name(tag)}.wav"
            tasks.append(('drum', drum_idx, drum_track, None, stem_path))

        if not tasks:
            logging.error("❌ No tracks — audio-first cannot continue")
            return None

        # Use up to half the CPU cores for parallel stem building.
        # Keep some headroom for the concurrent video phase (Phase 2).
        cpu = os.cpu_count() or 4
        stem_workers = min(len(tasks), max(4, cpu // 2))
        logging.info(f"🎵 Building {len(tasks)} stems with {stem_workers} parallel workers…")

        def _build_one(task):
            kind, idx, arg, notes, stem_path = task
            if kind == 'instrument':
                result = self._build_instrument_stem(arg, notes, stem_path)
                if not result:
                    logging.warning(f"⚠️ Stem skipped for {arg} (id={idx})")
                return result
            else:  # drum
                return self._build_drum_stem(arg, stem_path)

        stem_paths: list = []
        with ThreadPoolExecutor(max_workers=stem_workers) as executor:
            futures = {executor.submit(_build_one, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        stem_paths.append(result)
                except Exception as exc:
                    task = futures[future]
                    logging.error(f"❌ Stem task {task[2]} raised: {exc}")

        if not stem_paths:
            logging.error("❌ No stems built — audio-first cannot continue")
            return None

        unmastered_path = audio_dir / "unmastered_mix.wav"
        cmd = ['ffmpeg', '-y']
        for sp in stem_paths:
            cmd.extend(['-i', str(sp)])
        n = len(stem_paths)
        mix_filter = (
            "[0:a]anull[premix]" if n == 1
            else "".join(f"[{i}:a]" for i in range(n)) + f"amix=inputs={n}:normalize=0[premix]"
        )
        cmd += [
            '-filter_complex', mix_filter,
            '-map', '[premix]',
            '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
            '-t', str(total_duration + 2.0),
            str(unmastered_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode != 0 or not os.path.exists(unmastered_path):
            logging.error(f"❌ Stem mix failed: {r.stderr[-500:]}")
            return None

        mastered_path = audio_dir / "mastered_audio.aac"
        return self._normalize_audio_file(str(unmastered_path), str(mastered_path), total_duration)

    def _build_silent_video(self, total_duration: float, total_chunks: int,
                             work_dir) -> 'str | None':
        """
        Phase 2: Render the video grid and strip audio so the mux step can
        attach the mastered audio track.

        Chunks are built in parallel — each writes to its own unique file and
        only reads shared data (track lists, cached clip paths on disk) so
        there are no write conflicts.  Results are sorted by index before
        concatenation to preserve playback order.
        """
        try:
            chunks_dir = work_dir / "video_chunks"
            chunks_dir.mkdir(exist_ok=True)

            chunk_tasks = [
                (idx, idx * self.CHUNK_DURATION,
                 min((idx + 1) * self.CHUNK_DURATION, total_duration), chunks_dir)
                for idx in range(total_chunks)
            ]

            # Audio finishes at ~t+136s; video runs until ~t+163s on this hardware.
            # cpu//3 gives 5 workers on 16-core, reducing waves from 4 to 3 (~40s saved).
            cpu = os.cpu_count() or 4
            chunk_workers = min(total_chunks, max(2, cpu // 3))
            logging.info(f"🎬 Building {total_chunks} video chunks with {chunk_workers} parallel workers…")

            def _build_one_chunk(task):
                idx, start, end, out_dir = task
                path = self._create_enhanced_chunk(idx, start, end, out_dir)
                return idx, path

            chunk_map: dict = {}
            with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
                futures = {executor.submit(_build_one_chunk, task): task for task in chunk_tasks}
                for future in as_completed(futures):
                    try:
                        idx, chunk_path = future.result()
                        if chunk_path and os.path.exists(chunk_path):
                            chunk_map[idx] = chunk_path
                        else:
                            task = futures[future]
                            duration = task[2] - task[1]
                            placeholder = self._create_placeholder_chunk_simple(
                                idx, chunks_dir, duration)
                            if placeholder:
                                chunk_map[idx] = placeholder
                    except Exception as exc:
                        logging.error(f"❌ Video chunk error: {exc}")

            # Reconstruct ordered list for concatenation
            chunk_paths = [chunk_map[i] for i in range(total_chunks) if i in chunk_map]

            if not chunk_paths:
                raise Exception("No video chunks were created")

            concat_path = work_dir / "video_concat_raw.mp4"
            concat_result = self._concatenate_chunks(chunk_paths, concat_path)
            if not concat_result or not os.path.exists(concat_result):
                raise Exception("Chunk concatenation failed")

            silent_path = work_dir / "silent_video.mp4"
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', str(concat_result),
                 '-c:v', 'copy', '-an', str(silent_path)],
                capture_output=True, text=True
            )
            if r.returncode == 0 and os.path.exists(silent_path):
                logging.info(f"✅ Silent video: {os.path.getsize(silent_path):,} bytes")
                return str(silent_path)
            logging.error(f"❌ Audio strip failed: {r.stderr[-200:]}")
            return None
        except Exception as e:
            logging.error(f"❌ _build_silent_video: {e}")
            return None

    def _mux_video_audio(self, silent_video: str, audio_path: str,
                          output_path: str, total_duration: float) -> 'str | None':
        """
        Phase 3: Combine the silent video grid with the mastered audio track.
        Uses explicit -t instead of -shortest to expose duration bugs early.
        """
        try:
            video_dur = self._get_media_duration(silent_video)
            audio_dur = self._get_media_duration(audio_path)
            if video_dur and audio_dur and abs(video_dur - audio_dur) > 1.5:
                logging.warning(
                    f"Mux duration mismatch: video={video_dur:.2f}s "
                    f"audio={audio_dur:.2f}s (Δ={abs(video_dur - audio_dur):.2f}s)"
                )
            r = subprocess.run(
                ['ffmpeg', '-y',
                 '-i', str(silent_video),
                 '-i', str(audio_path),
                 '-map', '0:v', '-map', '1:a',
                 '-c:v', 'copy', '-c:a', 'copy',
                 '-t', str(total_duration),
                 str(output_path)],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if r.returncode == 0 and os.path.exists(output_path):
                logging.info(f"✅ Mux complete: {os.path.getsize(output_path):,} bytes")
                return str(output_path)
            logging.error(f"❌ Mux failed: {r.stderr[-300:]}")
            return None
        except Exception as e:
            logging.error(f"❌ _mux_video_audio: {e}")
            return None

    def _compress_final_output(self, input_path: str, output_path: str) -> 'str | None':
        """
        Final re-encode pass: compress the muxed output and apply any global
        post-processing (intro card overlay) in a single FFmpeg pass.
        Falls back to a stream-copy rename if re-encode fails.
        """
        tmp = output_path + '.tmp_compress.mp4'
        try:
            enc = self._get_encoding_settings()
            if '-c:v' in enc and enc[enc.index('-c:v') + 1] == 'h264_nvenc':
                final_enc = [
                    '-c:v', 'h264_nvenc', '-preset', 'p5',
                    '-rc', 'vbr', '-cq', '28', '-b:v', '2M', '-maxrate', '4M',
                ]
            else:
                final_enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '28']

            # ── Intro card overlay (applied as vf filter during compression) ──
            cs = getattr(self, 'composition_style', {}) or {}
            vf_filters = []
            if cs.get('introCardEnabled'):
                intro_dur = max(1.0, float(cs.get('introCardDuration', 3)))
                bg_col = self._hex_to_ffmpeg_color(cs.get('introCardBg', '#000000'))
                txt_col = self._hex_to_ffmpeg_color(cs.get('introCardTextColor', '#ffffff'))
                animated = bool(cs.get('introCardAnimated', True))
                alpha_expr = f'min(t/0.5,1)*gt(t,0)*lt(t,{intro_dur-0.3})+max(0,1-(t-({intro_dur-0.3}))/0.3)*lt(t,{intro_dur})' \
                             if animated else '1'

                def _ic_esc(t):
                    """Escape text for single-quoted FFmpeg drawtext text= value.
                    Use '\\'' to include apostrophes (backslash-escape outside quotes)."""
                    return (t or '').replace('\r', '').replace('\n', ' ') \
                                    .replace("'", "'\\''")

                # Solid colour fill for intro duration
                vf_filters.append(
                    f"drawbox=x=0:y=0:w=iw:h=ih:color={bg_col}@1:t=fill"
                    f":enable='lt(t,{intro_dur})'"
                )
                title = _ic_esc(cs.get('introCardText', ''))
                sub = _ic_esc(cs.get('introCardSubtext', ''))
                if title:
                    vf_filters.append(
                        f"drawtext=text='{title}':expansion=none"
                        f":x=(w-text_w)/2:y=(h-text_h)/2:fontsize=72"
                        f":fontcolor={txt_col}:alpha='{alpha_expr}'"
                        f":enable='lt(t,{intro_dur})'"
                    )
                if sub:
                    vf_filters.append(
                        f"drawtext=text='{sub}':expansion=none"
                        f":x=(w-text_w)/2:y=h*0.62:fontsize=36"
                        f":fontcolor={txt_col}:alpha='{alpha_expr}'"
                        f":enable='lt(t,{intro_dur})'"
                    )
                logging.info(f'🎬 Intro card: {intro_dur}s, bg={bg_col}, '
                             f'title={bool(title)}, subtitle={bool(sub)}')

            cmd = ['ffmpeg', '-y', '-i', input_path]
            if vf_filters:
                cmd.extend(['-vf', ','.join(vf_filters)])
            cmd.extend([
                *final_enc,
                '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                tmp,
            ])
            logging.info('🗜️  Final compression pass (CRF 28 / medium)…')
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if r.returncode == 0 and os.path.exists(tmp):
                orig_size = os.path.getsize(input_path)
                new_size = os.path.getsize(tmp)
                os.replace(tmp, output_path)
                logging.info(
                    f'✅ Compression complete: {orig_size:,} → {new_size:,} bytes '
                    f'({100*(1-new_size/orig_size):.0f}% reduction)'
                )
                return output_path
            logging.warning(f'⚠️  Compression pass failed, keeping original: {r.stderr[-400:]}')
        except Exception as e:
            logging.warning(f'⚠️  Compression pass error: {e}')
        finally:
            if os.path.exists(tmp):
                try: os.unlink(tmp)
                except Exception: pass
        return input_path  # fall back to uncompressed mux

    def _create_audio_first_composition(self, total_duration: float,
                                         total_chunks: int) -> 'str | None':
        """
        Orchestrates the 3-phase audio-first pipeline.
        Phases 1 (audio) and 2 (video) run concurrently; Phase 3 muxes them.
        """
        work_dir = self.temp_dir / "audio_first_work"
        work_dir.mkdir(exist_ok=True)
        logging.info("🎵 Audio-first: launching Phase 1 (audio) + Phase 2 (video) concurrently")

        with ThreadPoolExecutor(max_workers=2) as pool:
            audio_future = pool.submit(self._build_full_audio_timeline, total_duration)
            video_future = pool.submit(
                self._build_silent_video, total_duration, total_chunks, work_dir
            )
            full_audio = audio_future.result()
            silent_video = video_future.result()

        if not full_audio:
            logging.error("❌ Phase 1 (audio) failed")
            return None
        if not silent_video:
            logging.error("❌ Phase 2 (video) failed")
            return None

        logging.info("🎬 Phase 3: muxing video + audio…")
        mux_path = str(self.output_path) + '.mux_raw.mp4'
        muxed = self._mux_video_audio(silent_video, full_audio, mux_path, total_duration)
        if not muxed:
            return None

        if not self.preview_mode:
            logging.info("🗜️  Phase 4: final compression pass…")
            result = self._compress_final_output(mux_path, str(self.output_path))
            try: os.unlink(mux_path)
            except Exception: pass
            return result

        # Preview mode: just rename the mux output
        import shutil
        shutil.move(mux_path, str(self.output_path))
        return str(self.output_path)

    def create_composition(self):
        """
        ENHANCED composition with audio-first pipeline.

        Tries the 3-phase audio-first approach first (eliminates per-chunk
        loudnorm/limiter resets and chunk-boundary pumping). Falls back to the
        legacy sequential/parallel chunk pipeline when running in preview mode
        or if the audio-first path fails.
        """
        try:
            logging.info("🎬 Starting ENHANCED video composition with fixes...")
            start_time = time.time()

            # Pre-process all unique note/instrument combos so each phase can
            # retrieve pitch-shifted audio instantly from the cache.
            self.preprocess_composition_optimized()

            total_duration = self._calculate_total_duration()
            total_chunks = max(1, math.ceil(total_duration / self.CHUNK_DURATION))

            # Audio-first pipeline (production only — preview keeps legacy for speed)
            if not self.preview_mode and self._tuned_videos_cache:
                try:
                    logging.info(
                        f"🎵 Audio-first pipeline: {total_duration:.2f}s, {total_chunks} chunks"
                    )
                    result = self._create_audio_first_composition(total_duration, total_chunks)
                    if result and os.path.exists(result):
                        elapsed = time.time() - start_time
                        logging.info(f"🎉 Audio-first composition complete! {elapsed:.2f}s")
                        return result
                    logging.warning("⚠️ Audio-first returned no output — falling back to legacy")
                except Exception as e:
                    logging.warning(f"⚠️ Audio-first failed ({e}) — falling back to legacy")

            # Legacy chunk-based pipeline (fallback / preview mode)
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
                if self.preview_mode:
                    # Skip 2-pass loudnorm in preview — shaves ~1-2s off quick renders
                    import shutil
                    shutil.move(str(final_path), str(self.output_path))
                    normalized_path = self.output_path
                else:
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
                    # Use the same fixed instrument processing used by the simplified path
                    result = self._process_instrument_track_for_chunk_fixed(track, start_time, chunk_duration, chunk_idx, track_id)
                    if result:
                        track_video_segments.append(result)
            
            if not track_video_segments:
                return self._create_placeholder_chunk_simple(chunk_idx, chunks_dir, chunk_duration)
            
            # Create final chunk with grid layout using the fixed FFmpeg xstack compositor
            return self._create_grid_layout_chunk_fixed(track_video_segments, chunk_path, chunk_duration)
            
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
                if self.preview_mode:
                    import shutil
                    shutil.move(str(final_path), str(self.output_path))
                    normalized_path = self.output_path
                else:
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
        from video_utils import build_ffmpeg_command
        
        # Use the standardized command builder
        return build_ffmpeg_command(
            inputs=inputs,
            output=output_path,
            filter_complex=filter_complex,
            preset='fast',
            crf=23,
            audio_bitrate='192k',
            use_gpu=True
        )

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

    # def _process_instrument_track_for_chunk_fixed(self, track, chunk_start_time, chunk_duration, chunk_idx, track_id):
    #     """
    #     FIXED: Process instrument track with proper path resolution and shorter filenames
    #     """
    #     try:
    #         # Get track info
    #         if isinstance(track.get('instrument'), dict):
    #             track_name = track['instrument'].get('name', 'unknown')
    #         else:
    #             track_name = track.get('instrument', f'track_{track_id}')
            
    #         notes = track.get('notes', [])
            
    #         # Filter notes for this chunk
    #         chunk_notes = [
    #             note for note in notes 
    #             if chunk_start_time <= note.get('time', 0) < chunk_start_time + chunk_duration
    #         ]
            
    #         if not chunk_notes:
    #             return None
            
    #         # FIXED: Use PathRegistry to find instrument video
    #         registry = PathRegistry.get_instance()
            
    #         # Try multiple strategies to find the video
    #         video_path = None
            
    #         # Strategy 1: Try with first note's MIDI value
    #         if chunk_notes:
    #             first_note_midi = chunk_notes[0].get('midi', 60)
    #             video_path = registry.get_instrument_path(track_name, str(first_note_midi))
            
    #         # Strategy 2: Try with default middle C (60)
    #         if not video_path:
    #             video_path = registry.get_instrument_path(track_name, "60")
            
    #         # Strategy 3: Try fallback approach - find any video for this instrument
    #         if not video_path:
    #             normalized_name = normalize_instrument_name(track_name)
    #             instrument_paths = registry.instrument_paths.get(normalized_name, {})
    #             if instrument_paths:
    #                 video_path = next(iter(instrument_paths.values()))
            
    #         if not video_path or not os.path.exists(video_path):
    #             logging.warning(f"No video found for instrument: {track_name}")
    #             return None
            
    #         # FIXED: Create note-triggered video with correct parameters matching function signature
    #         import uuid
    #         short_id = str(uuid.uuid4())[:8]
            
    #         # Convert chunk-relative note times for the function
    #         chunk_relative_notes = []
    #         for note in chunk_notes:
    #             relative_note = note.copy()
    #             relative_note['time'] = float(note.get('time', 0)) - chunk_start_time
    #             chunk_relative_notes.append(relative_note)
            
    #         triggered_video = self._create_note_triggered_video_sequence_fixed(
    #             video_path=video_path,
    #             notes=chunk_relative_notes,  # Use chunk-relative notes
    #             total_duration=chunk_duration,  # Use correct parameter name
    #             track_name=track_name,
    #             unique_id=short_id
    #         )
            
    #         if triggered_video and os.path.exists(triggered_video):
    #             return {
    #                 'video_path': triggered_video,
    #                 'track_id': track_id,  # Use original track ID for grid positioning
    #                 'track_name': track_name,
    #                 'notes': chunk_notes,
    #                 'type': 'instrument'
    #             }
    #         else:
    #             logging.warning(f"Failed to create triggered video for {track_name}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
    #         return None

    def _process_instrument_track_for_chunk_fixed(self, track, chunk_start_time, chunk_duration, chunk_idx, track_id):
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

            # Strategy 1: Try with first note's MIDI value via PathRegistry
            if chunk_notes:
                first_note_midi = chunk_notes[0].get('midi', 60)
                video_path = registry.get_instrument_path(track_name, str(first_note_midi))

            # Strategy 2: Try with default middle C (60)
            if not video_path:
                video_path = registry.get_instrument_path(track_name, "60")

            # Strategy 3: Fallback approach - find any video for this instrument in registry
            if not video_path:
                normalized_name = normalize_instrument_name(track_name)
                instrument_paths = registry.instrument_paths.get(normalized_name, {})
                if instrument_paths:
                    video_path = next(iter(instrument_paths.values()))
                    logging.info(f"Instrument fallback used from registry: {track_name} -> {video_path}")

            # Strategy 4: Robust filename search in uploads directory
            if not video_path:
                video_path = self._find_instrument_video_file(track_name)

            if not video_path or not os.path.exists(video_path):
                logging.warning(f"No video found for instrument: {track_name}")
                return None

            # Always use a unique_id for the note-triggered video
            import uuid
            short_id = str(uuid.uuid4())[:8]
            # Compute onset offset (non-destructive)
            onset_offset = self._get_onset_offset(video_path)

            # Build per-note pre-tuned audio map from the preprocessing cache.
            # Each unique MIDI note needed in this chunk gets a cached tuned video
            # whose audio stream replaces the raw asetrate-based pitch shift.
            note_audio_map = {}
            unique_midi_notes = {n.get('midi') for n in chunk_notes if n.get('midi') is not None}
            for midi_note in unique_midi_notes:
                tuned = self.get_optimized_tuned_video(track_name, midi_note)
                if tuned and os.path.exists(tuned):
                    note_audio_map[midi_note] = tuned
                    logging.debug(f"[PitchShift] {track_name} MIDI {midi_note} → {os.path.basename(tuned)}")
                else:
                    logging.debug(f"[PitchShift] {track_name} MIDI {midi_note} → asetrate fallback")

            triggered_video = self._create_note_triggered_video_sequence_fixed(
                video_path=video_path,
                notes=chunk_notes,
                total_duration=chunk_duration,
                track_name=track_name,
                unique_id=short_id,
                chunk_start_time=chunk_start_time,
                onset_offset=onset_offset,
                note_audio_map=note_audio_map if note_audio_map else None,
            )

            if not triggered_video:
                return None  # or fallback already handled

            if triggered_video and os.path.exists(triggered_video):
                return {
                    'video_path': triggered_video,
                    'track_id': str(track_id),  # Use original track ID (string) for grid positioning
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
    # #     """
    # #     FIXED: Use the EXACT same approach as drums (which works!)
    # #     """
    # #     try:
    # #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
            
    # #         if not notes or not os.path.exists(video_path):
    #     try:
    #         # Get track info
    #         if isinstance(track.get('instrument'), dict):
    #             track_name = track['instrument'].get('name', 'unknown')
    #         else:
    #             track_name = track.get('instrument', f'track_{track_id}')

    #         notes = track.get('notes', [])
    #         logging.info(f"[Instrument] Track '{track_name}' (id={track_id}) has {len(notes)} total notes.")

    #         # Filter notes for this chunk (overlapping the chunk)
    #         chunk_notes = [
    #             note for note in notes
    #             if note.get('time', 0) < chunk_start_time + chunk_duration and
    #                note.get('time', 0) + note.get('duration', 1) > chunk_start_time
    #         ]
    #         logging.info(f"[Instrument] Track '{track_name}' (id={track_id}) chunk {chunk_idx}: {len(chunk_notes)} notes in chunk.")
    #         if not chunk_notes:
    #             logging.warning(f"[Instrument] No notes for '{track_name}' (id={track_id}) in chunk {chunk_idx}.")
    #             return None
    #         # Adjust note times to be chunk-relative
    #         rel_notes = []
    #         for note in chunk_notes:
    #             note_copy = note.copy()
    #             note_copy['time'] = float(note_copy.get('time', 0)) - chunk_start_time
    #             rel_notes.append(note_copy)

    #         # Use PathRegistry to find instrument video
    #         registry = PathRegistry.get_instance()
    #         # Try multiple strategies to find the video
    #         video_path = None
    #         # Strategy 1: Try with first note's MIDI value
    #         if chunk_notes:
    #             first_note_midi = chunk_notes[0].get('midi', 60)
    #             video_path = registry.get_instrument_path(track_name, str(first_note_midi))
    #             logging.info(f"[Instrument] Lookup: {track_name} midi={first_note_midi} -> {video_path}")
    #         # Strategy 2: Try with default middle C (60)
    #         if not video_path:
    #             video_path = registry.get_instrument_path(track_name, "60")
    #             logging.info(f"[Instrument] Fallback midi=60: {track_name} -> {video_path}")
    #         # Strategy 3: Try fallback approach - find any video for this instrument
    #         if not video_path:
    #             normalized_name = normalize_instrument_name(track_name)
    #             instrument_paths = registry.instrument_paths.get(normalized_name, {})
    #             if instrument_paths:
    #                 video_path = next(iter(instrument_paths.values()))
    #                 logging.info(f"[Instrument] Fallback any: {track_name} -> {video_path}")
    #         if not video_path or not os.path.exists(video_path):
    #             logging.warning(f"[Instrument] No video found for: {track_name} (track_id={track_id}) in chunk {chunk_idx}. Tried: {video_path}")
    #             return None

    #         # Create note-triggered video with shorter filename
    #         import uuid
    #         short_id = str(uuid.uuid4())[:8]
    #         triggered_video = self._create_note_triggered_video_sequence_fixed(
    #             video_path=video_path,
    #             notes=rel_notes,
    #             total_duration=chunk_duration,
    #             track_name=track_name,
    #             unique_id=short_id
    #         )
    #         if triggered_video and os.path.exists(triggered_video):
    #             logging.info(f"[Instrument] Successfully created triggered video for {track_name} (id={track_id}) in chunk {chunk_idx}.")
    #             return {
    #                 'video_path': triggered_video,
    #                 'track_id': track_id,  # Use original track ID for grid positioning
    #                 'track_name': track_name,
    #                 'notes': chunk_notes,
    #                 'type': 'instrument'
    #             }
    #         else:
    #             logging.warning(f"[Instrument] Failed to create triggered video for {track_name} (id={track_id}) in chunk {chunk_idx}.")
    #             return None

    #     except Exception as e:
    #         logging.error(f"[Instrument] Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
    #         return None
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

    # def _create_note_triggered_video_sequence_fixed(self, video_path, notes, chunk_start_time, chunk_duration, track_name, unique_id):
    #     """
    #     FIXED: Creates a correctly timed, note-for-note triggered video for an instrument track.
    #     This now correctly uses relative note timing within the chunk and builds a valid FFmpeg filter graph.
    #     """
    #     try:
    #         output_path = self.temp_dir / f"{track_name}_{unique_id}.mp4"
    #         if not notes or not os.path.exists(video_path):
    #             return None
    #         if output_path.exists():
    #             output_path.unlink()

    #         filter_parts = []
    #         # Inputs are: [0:v]/[0:a] = instrument video, [1:v] = black bg, [2:a] = silent audio
    #         # Correctly refer to inputs by index.
    #         video_layers = ["[1:v]"]  # Start with the black background from input 1
    #         audio_segments = ["[2:a]"] # Start with the silent audio from input 2

    #         for i, note in enumerate(notes):
    #             note_start_abs = float(note.get('time', 0))
    #             # --- TIMING FIX: Calculate time relative to the chunk start ---
    #             relative_start = note_start_abs - chunk_start_time
                
    #             # Ensure the note is actually within the current chunk's timeframe
    #             if relative_start < 0 or relative_start >= chunk_duration:
    #                 continue

    #             duration = float(note.get('duration', 0.5))
    #             # Ensure note doesn't play past the end of the chunk
    #             duration = min(duration, chunk_duration - relative_start)
    #             if duration <= 0:
    #                 continue

    #             pitch = note.get('midi', 60)
    #             pitch_semitones = pitch - 60
    #             pitch_factor = 2 ** (pitch_semitones / 12.0)

    #             # Create a trimmed video segment from the source instrument video (input 0)
    #             filter_parts.append(f"[0:v]trim=0:{duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                
    #             # Create a corresponding pitched audio segment
    #             audio_trim_filter = f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS"
    #             if abs(pitch_factor - 1.0) > 0.01:
    #                 filter_parts.append(f"{audio_trim_filter},asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]")
    #             else:
    #                 filter_parts.append(f"{audio_trim_filter}[note_a{i}]")

    #             # Overlay the note video at the correct relative time
    #             prev_video_layer = video_layers[-1]
    #             filter_parts.append(f"{prev_video_layer}[note_v{i}]overlay=enable='between(t,{relative_start},{relative_start + duration})'[video_out{i}]")
    #             video_layers.append(f"[video_out{i}]")

    #             # Delay the note audio to match its start time and add it to the list for mixing
    #             delay_ms = int(relative_start * 1000)
    #             filter_parts.append(f"[note_a{i}]adelay={delay_ms}|{delay_ms}[delayed_a{i}]")
    #             audio_segments.append(f"[delayed_a{i}]")

    #         if len(audio_segments) <= 1: # Only silent audio was added
    #             logging.warning(f"No valid notes found in chunk for {track_name}, skipping video creation.")
    #             return None

    #         # Mix all the delayed audio segments together
    #         audio_inputs = ''.join(audio_segments)
    #         filter_parts.append(f"{audio_inputs}amix=inputs={len(audio_segments)}:duration=longest[final_audio]")
            
    #         final_video_layer = video_layers[-1]

    #         cmd = [
    #             'ffmpeg', '-y',
    #             '-i', str(video_path),  # Input 0
    #             '-f', 'lavfi', '-i', f'color=black:size=640x360:duration={chunk_duration}:rate=30', # Input 1
    #             '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100:duration={chunk_duration}', # Input 2
    #             '-filter_complex', ';'.join(filter_parts),
    #             '-map', f'{final_video_layer}',
    #             '-map', '[final_audio]',
    #             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
    #             '-c:a', 'aac', '-b:a', '192k',
    #             '-t', str(chunk_duration),
    #             '-r', '30',
    #             str(output_path)
    #         ]
            
    #         logging.info(f"🎵 Creating note-triggered video for {track_name} with {len(notes)} notes")
    #         result = subprocess.run(cmd, capture_output=True, text=True)
            
    #         if result.returncode == 0:
    #             logging.info(f"✅ Note-triggered video created: {output_path}")
    #             return str(output_path)
    #         else:
    #             logging.error(f"❌ Failed to create note-triggered video for {track_name}: {result.stderr}")
    #             return None
                
    #     except Exception as e:
    #         logging.error(f"Error in _create_note_triggered_video_sequence_fixed for {track_name}: {e}", exc_info=True)
    #         return None

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
            
    #         # FIXED: Use the same MIDI-triggered approach as drums
    #         filter_parts = []
            
    #         # Create silent base
    #         filter_parts.append(f"color=black:size=640x360:duration={total_duration}:rate=30[base_video]")
    #         filter_parts.append(f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}[base_audio]")
            
    #         # Create overlays for each MIDI note (like drums do)
    #         video_layers = ["[base_video]"]
    #         audio_segments = ["[base_audio]"]
            
    #         for i, note in enumerate(notes):
    #             start_time = float(note.get('time', 0))
    #             duration = float(note.get('duration', 0.5))
    #             pitch = note.get('midi', 60)
                
    #             # Convert to chunk-relative time
    #             if start_time >= total_duration:
    #                 continue
                
    #             # Limit duration to not exceed chunk boundary
    #             duration = min(duration, total_duration - start_time)
    #             if duration <= 0:
    #                 continue
                
    #             # FIXED: Enforce minimum duration to prevent FFmpeg precision errors
    #             MIN_DURATION = 0.1  # Minimum 0.1 seconds (100ms)
    #             if duration < MIN_DURATION:
    #                 duration = MIN_DURATION
    #                 logging.debug(f"Extended note duration to {MIN_DURATION}s for MIDI {pitch} at {start_time}s")
                
    #             # Ensure duration doesn't exceed chunk boundary after extension
    #             duration = min(duration, total_duration - start_time)
                
    #             # Calculate pitch adjustment
    #             pitch_semitones = pitch - 60
    #             pitch_factor = 2 ** (pitch_semitones / 12.0)
                
    #             # Create video segment for this note (like drums)
    #             filter_parts.append(f"[0:v]trim=0:{duration},setpts=PTS-STARTPTS,scale=640:360[note_v{i}]")
                
    #             # Create audio segment with pitch adjustment
    #             if abs(pitch_factor - 1.0) > 0.01:
    #                 filter_parts.append(
    #                     f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS,"
    #                     f"asetrate=44100*{pitch_factor},aresample=44100[note_a{i}]"
    #                 )
    #             else:
    #                 filter_parts.append(f"[0:a]atrim=0:{duration},asetpts=PTS-STARTPTS[note_a{i}]")
                
    #             # Overlay at exact note time (like drums)
    #             prev_video = video_layers[-1]
    #             filter_parts.append(f"{prev_video}[note_v{i}]overlay=enable='between(t,{start_time},{start_time + duration})'[video_out{i}]")
    #             video_layers.append(f"[video_out{i}]")
                
    #             # Add delayed audio (like drums)
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

    def _get_windows_font_path(self):
        """Return a usable font path for FFmpeg drawtext on Windows."""
        candidates = [
            r'C:/Windows/Fonts/arial.ttf',
            r'C:/Windows/Fonts/verdana.ttf',
            r'C:/Windows/Fonts/segoeui.ttf',
            r'C:/Windows/Fonts/calibri.ttf',
            r'C:/Windows/Fonts/tahoma.ttf',
        ]
        for p in candidates:
            if os.path.exists(p.replace('/', os.sep)):
                # Return path with forward slashes — no colon escaping needed
                # when the path is wrapped in single quotes in the filtergraph.
                return p
        return None  # Let FFmpeg use its built-in font

    def _get_color_grade_filter(self, grade):
        """Return FFmpeg filter string for a named color grade."""
        grades = {
            'warm':      'eq=saturation=1.2:gamma_r=1.1:gamma_b=0.88',
            'cool':      'eq=saturation=1.1:gamma_b=1.15:gamma_r=0.88',
            'vintage':   'colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131',
            'cyberpunk': 'eq=saturation=1.6:contrast=1.1,colorchannelmixer=1.15:0:0.15:0:0:1.15:0.15:0:0.25:0:1.0',
            'bw':        'hue=s=0',
            'vivid':     'eq=saturation=1.8:contrast=1.1:brightness=0.04',
        }
        return grades.get(grade, '')

    def _hex_to_ffmpeg_color(self, hex_color):
        """Convert #RRGGBB to 0xRRGGBB for FFmpeg."""
        h = hex_color.lstrip('#')
        if len(h) == 6:
            return f'0x{h.upper()}'
        return '0x000000'

    def _write_text_tempfile(self, text, prefix='ats_text_'):
        """Write text to a temp file and return its path (for drawtext textfile= option)."""
        fd, path = tempfile.mkstemp(prefix=prefix, suffix='.txt')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception:
            os.close(fd)
        return path

    def _escape_path_for_filter(self, path):
        """Escape a filesystem path for use in an FFmpeg filter option value.

        In FFmpeg filter strings, ':' is an option separator.  Single-quoted
        wrappers ('...') do NOT protect a Windows drive colon in this FFmpeg
        build — FFmpeg still splits on it.  The correct level-1 escape is a
        backslash before the colon: C\:/path/to/file.txt (no outer quotes).
        """
        p = path.replace('\\', '/')   # normalise to forward slashes
        p = p.replace(':', '\\:')     # escape drive-letter colon: C: → C\:
        return p

    def _apply_cell_style_filters(self, filter_parts, input_label, output_label,
                                   cell_w, cell_h, track_id, cell_segment, temp_files):
        """
        Build per-cell styling filters (scale, pad/bgColor, color grade, beat flash, label, border).
        Chains filters from input_label → output_label.
        temp_files: list to append any created temp file paths for cleanup.
        """
        cs = getattr(self, 'clip_styles', {})

        # Look up style: frontend keys use prefixes ('track-0', 'drum-drum_snare_drum')
        # while backend track_ids are bare ('0', 'drum_crash_cymbal'), so try all formats.
        candidates = [track_id, f'track-{track_id}', f'drum-{track_id}']
        style = next((cs[k] for k in candidates if k in cs), {})

        logging.info(
            f"[style] cell={track_id!r}  candidates={candidates}  "
            f"matched={'yes ('+next((k for k in candidates if k in cs), 'none')+')'}  "
            f"effects={[k for k,v in style.items() if v and k.endswith('Enabled')]}"
        )

        bg_color = style.get('bgColor', '#1a1a2e')
        border_width = int(style.get('borderWidth', 0))
        border_color = style.get('borderColor', '#7c3aed')
        color_grade = style.get('colorGrade', 'none')
        rounded_corners = bool(style.get('roundedCorners', False))
        corner_radius = int(style.get('cornerRadius', 12))
        label_enabled = bool(style.get('labelEnabled', False))
        label_text = style.get('labelText', '') or ''
        label_color = style.get('labelColor', '#ffffff')
        label_size = int(style.get('labelFontSize', 14))
        beat_flash_enabled = bool(style.get('beatFlashEnabled', False))
        beat_flash_intensity = float(style.get('beatFlashIntensity', 0.4))
        fade_enabled = bool(style.get('fadeEnabled', False))
        fade_duration = float(style.get('fadeDuration', 0.15))
        transparent_bg = bool(style.get('transparentBg', False))

        font_path = self._get_windows_font_path()
        current = input_label

        # ── 1. Scale to cell dimensions, letterbox ───────────────────────────
        # When transparentBg is on: use the global composition background so the
        # letterbox "padding" blends seamlessly with the canvas background.
        comp_bg_color = getattr(self, 'composition_style', {}).get('backgroundColor', '#0a0a0f')
        if transparent_bg:
            pad_color = self._hex_to_ffmpeg_color(comp_bg_color)
            logging.info(f"[style] cell={track_id!r} transparentBg=True → pad color={comp_bg_color}")
        else:
            pad_color = self._hex_to_ffmpeg_color(bg_color)

        # With zoom-to-fill (transparent_bg), video always fills the full cell —
        # borders/rounded corners always apply to the full cell bounds.
        content_x, content_y, content_w, content_h = 0, 0, cell_w, cell_h

        bg_ffmpeg = self._hex_to_ffmpeg_color(bg_color)
        next_label = f'v_pad_{output_label[1:-1]}'
        if transparent_bg:
            # Zoom-to-fill: scale so the video fills the entire cell in both dimensions,
            # then center-crop. This eliminates ALL black margins — both letterbox bars
            # added by FFmpeg and any black bars baked into the recording itself.
            filter_parts.append(
                f"{current}scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase,"
                f"crop={cell_w}:{cell_h}[{next_label}]"
            )
            logging.info(f"[style] cell={track_id!r} transparentBg zoom-to-fill → {cell_w}x{cell_h}")
        else:
            # Letterbox: scale to fit, pad remaining area with per-clip bg color
            filter_parts.append(
                f"{current}scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
                f"pad={cell_w}:{cell_h}:-1:-1:color={pad_color}[{next_label}]"
            )
        current = f'[{next_label}]'

        # ── 2. Color grade ───────────────────────────────────────────────────
        grade_filter = self._get_color_grade_filter(color_grade)
        if grade_filter:
            next_label = f'v_gr_{output_label[1:-1]}'
            filter_parts.append(f"{current}{grade_filter}[{next_label}]")
            current = f'[{next_label}]'

        # ── 3. Beat flash ────────────────────────────────────────────────────
        # Simple onset flash: bright burst at t=0 (chunk/clip start = note onset).
        # For multi-note chunks, add additional flashes based on notes in segment.
        if beat_flash_enabled:
            notes = cell_segment.get('notes', []) if cell_segment else []
            flash_windows = [(0.0, 0.1)]  # Always flash at clip start
            for note in notes[:20]:  # Cap to 20 notes for expression length
                t = float(note.get('chunk_time', note.get('time', 0)))
                if t > 0.05:  # Skip if too close to the start flash
                    flash_windows.append((round(t, 3), round(t + 0.1, 3)))
            enable_expr = '+'.join(f'between(t,{s},{e})' for s, e in flash_windows)
            next_label = f'v_fl_{output_label[1:-1]}'
            filter_parts.append(
                f"{current}eq=brightness={beat_flash_intensity:.2f}:enable='{enable_expr}'[{next_label}]"
            )
            current = f'[{next_label}]'

        # ── 4. Instrument label ──────────────────────────────────────────────
        if label_enabled:
            if label_text:
                display_text = label_text
            elif cell_segment:
                display_text = (
                    cell_segment.get('track_name')
                    or cell_segment.get('drum_name')
                    or cell_segment.get('instrument_name')
                    or track_id
                )
            else:
                display_text = track_id
            # Inline text= with single-quote wrapping and '\\'' apostrophe escaping.
            # This avoids Windows path issues with textfile= and correctly handles
            # apostrophes/colons in track names (unlike double-quote wrapping which
            # FFmpeg's filter_complex parser does not recognise as a quote character).
            escaped_label = display_text.replace('\r', '').replace('\n', ' ') \
                                        .replace("'", "'\\''")
            lc = self._hex_to_ffmpeg_color(label_color)
            next_label = f'v_lbl_{output_label[1:-1]}'
            filter_parts.append(
                f"{current}drawtext=text='{escaped_label}'"
                f":expansion=none"
                f":x=6:y=h-{label_size + 6}:fontsize={label_size}"
                f":fontcolor={lc}:alpha='1':box=1:boxcolor=0x000000@0.45:boxborderw=3[{next_label}]"
            )
            current = f'[{next_label}]'

        # ── 5. Clip fade-in (applied before border so border is always visible) ──
        if fade_enabled and fade_duration > 0:
            fd = max(fade_duration, 0.001)
            next_label = f'v_fade_{output_label[1:-1]}'
            filter_parts.append(
                f"{current}fade=t=in:st=0:d={fd:.3f}[{next_label}]"
            )
            current = f'[{next_label}]'

        # ── 6. Border ────────────────────────────────────────────────────────
        if border_width > 0:
            bc = self._hex_to_ffmpeg_color(border_color)
            next_label = f'v_brd_{output_label[1:-1]}'
            # When transparent bg is on, draw border around actual video content
            bx, by, bw, bh = (content_x, content_y, content_w, content_h) \
                              if transparent_bg else (0, 0, cell_w, cell_h)
            filter_parts.append(
                f"{current}drawbox=x={bx}:y={by}:w={bw}:h={bh}"
                f":color={bc}@1.0:t={border_width}[{next_label}]"
            )
            current = f'[{next_label}]'

        # ── 7. Rounded corners (paint corner-fill color over corners) ────────
        if rounded_corners and corner_radius > 0:
            # When transparent bg is on, round the actual video content corners;
            # corner fill uses the global bg so they blend into the canvas.
            if transparent_bg:
                rx, ry, rw, rh = content_x, content_y, content_w, content_h
                corner_fill = self._hex_to_ffmpeg_color(comp_bg_color)
            else:
                rx, ry, rw, rh = 0, 0, cell_w, cell_h
                corner_fill = self._hex_to_ffmpeg_color(bg_color)
            r = min(corner_radius, rw // 4, rh // 4)
            if r > 0:
                next_label = f'v_rnd_{output_label[1:-1]}'
                filter_parts.append(
                    f"{current}"
                    f"drawbox=x={rx}:y={ry}:w={r}:h={r}:color={corner_fill}@1:t=fill,"
                    f"drawbox=x={rx + rw - r}:y={ry}:w={r}:h={r}:color={corner_fill}@1:t=fill,"
                    f"drawbox=x={rx}:y={ry + rh - r}:w={r}:h={r}:color={corner_fill}@1:t=fill,"
                    f"drawbox=x={rx + rw - r}:y={ry + rh - r}:w={r}:h={r}:color={corner_fill}@1:t=fill"
                    f"[{next_label}]"
                )
                current = f'[{next_label}]'

        # Terminate chain at the output_label expected by the caller
        filter_parts.append(f"{current}null{output_label}")

    def _apply_global_style_filters(self, filter_parts, current_label,
                                     target_w, target_h, duration,
                                     audio_label, temp_files):
        """
        Apply global composition effects after xstack:
        title, tagline, watermark, waveform overlay, vignette, glitch.
        Returns (final_video_label, final_audio_label).
        """
        cs = getattr(self, 'composition_style', {})
        if not cs:
            logging.info("[style] No composition_style set — skipping global effects")
            return current_label, audio_label

        logging.info(
            f"[style] global effects enabled: "
            f"{[k for k,v in cs.items() if str(k).endswith('Enabled') and v]}"
        )

        def _esc(t):
            """Escape text for single-quoted FFmpeg drawtext text= value.

            FFmpeg's filter_complex parser only recognises single-quotes '...'
            for quoting. Double-quotes are NOT special — using "..." causes
            an apostrophe (e.g. in "Ain't") to start a single-quoted segment
            that swallows ':' option separators and ';' chain separators.

            To include a literal apostrophe inside a single-quoted filter
            value, use the '\\'' pattern: end the current quote, then use a
            backslash-escaped apostrophe (outside quotes), then open a new
            quote. This is the same backslash-escape mechanism used by
            _escape_path_for_filter for ':' in Windows drive letters.

            expansion=none (added to each filter) prevents % format strings.
            Newlines are stripped — drawtext is single-line only.
            """
            return t.replace('\r', '').replace('\n', ' ') \
                     .replace("'", "'\\''")

        def add_drawtext(text, x_expr, y_expr, size, color_hex, alpha_expr='1', enabled='1'):
            nonlocal current_label, filter_parts
            # Use inline text= with single-quote wrapping and '\\'' apostrophe
            # escaping. FFmpeg's filter_complex parser uses single-quotes for
            # quoting; double-quotes are NOT recognised — an apostrophe in the
            # text value would start an errant single-quoted segment that
            # swallows ';' chain separators, breaking all subsequent filters.
            # expansion=none prevents % format string expansion on user text.
            # No fontfile/font specified — FFmpeg uses its built-in fallback font.
            escaped = _esc(text)
            fc = self._hex_to_ffmpeg_color(color_hex)
            nxt = f'v_gt_{len(filter_parts)}'
            filter_parts.append(
                f"[{current_label}]drawtext=text='{escaped}'"
                f":expansion=none"
                f":x={x_expr}:y={y_expr}:fontsize={size}"
                f":fontcolor={fc}:alpha='{alpha_expr}':enable='{enabled}'[{nxt}]"
            )
            current_label = nxt

        # ── Title ────────────────────────────────────────────────────────────
        if cs.get('titleEnabled') and cs.get('titleText', '').strip():
            pos = cs.get('titlePosition', 'top-center')
            size = int(cs.get('titleFontSize', 56))
            color = cs.get('titleColor', '#ffffff')
            animated = bool(cs.get('titleAnimated', True))
            # Use comma-free expression: equivalent to min(t,1) for t>=0.
            # FFmpeg's filter_complex_script parser splits chains on ',' before
            # honouring single-quote quoting, so expressions with commas inside
            # alpha='...' break the filter graph.  This abs()-based form is safe.
            alpha = 't-(t-1+abs(t-1))/2' if animated else '1'
            if pos == 'top-center':
                y = str(max(20, size // 2))
            elif pos == 'bottom-center':
                y = f'h-{size * 2}'
            else:
                y = '(h-text_h)/2'
            add_drawtext(cs['titleText'], '(w-text_w)/2', y, size, color, alpha_expr=alpha)

        # ── Tagline ──────────────────────────────────────────────────────────
        if cs.get('taglineEnabled') and cs.get('taglineText', '').strip():
            size = int(cs.get('taglineFontSize', 24))
            color = cs.get('taglineColor', '#cccccc')
            y = f'h-{size * 2 + 10}'
            add_drawtext(cs['taglineText'], '(w-text_w)/2', y, size, color)

        # ── Watermark ────────────────────────────────────────────────────────
        if cs.get('watermarkEnabled') and cs.get('watermarkText', '').strip():
            size = int(cs.get('watermarkFontSize', 18))
            color = cs.get('watermarkColor', '#ffffff')
            opacity = float(cs.get('watermarkOpacity', 0.5))
            wpos = cs.get('watermarkPosition', 'bottom-right')
            padding = 16
            if wpos == 'bottom-right':
                x, y = f'w-text_w-{padding}', f'h-text_h-{padding}'
            elif wpos == 'bottom-left':
                x, y = str(padding), f'h-text_h-{padding}'
            elif wpos == 'top-right':
                x, y = f'w-text_w-{padding}', str(padding)
            else:
                x, y = str(padding), str(padding)
            add_drawtext(cs['watermarkText'], x, y, size, color, alpha_expr=str(round(opacity, 2)))

        # ── Waveform bar ─────────────────────────────────────────────────────
        if cs.get('waveformEnabled'):
            wh = int(cs.get('waveformHeight', 60))
            wc = self._hex_to_ffmpeg_color(cs.get('waveformColor', '#00ff88'))
            # Split audio: one copy for waveform vis, one continues as the output audio
            wave_audio = f'audio_wave_{len(filter_parts)}'
            audio_main = f'audio_main_{len(filter_parts)}'
            filter_parts.append(f"[{audio_label}]asplit=2[{audio_main}][{wave_audio}]")
            audio_label = audio_main
            # showwaves renders on a solid black background; key out black so the
            # waveform composites transparently over the video.
            wave_video = f'waveform_{len(filter_parts)}'
            filter_parts.append(
                f"[{wave_audio}]showwaves=s={target_w}x{wh}:mode=cline:rate=30"
                f":colors={wc},"
                f"colorkey=0x000000:0.1:0.0,"
                f"format=yuva420p[{wave_video}]"
            )
            nxt = f'v_wf_{len(filter_parts)}'
            filter_parts.append(
                f"[{current_label}][{wave_video}]overlay=x=0:y=h-{wh}:format=auto[{nxt}]"
            )
            current_label = nxt

        # ── Vignette ─────────────────────────────────────────────────────────
        if cs.get('vignetteEnabled'):
            strength = float(cs.get('vignetteStrength', 0.5))
            angle = round(strength * 3.14159, 3)
            nxt = f'v_vig_{len(filter_parts)}'
            filter_parts.append(f"[{current_label}]vignette=angle={angle}[{nxt}]")
            current_label = nxt

        # ── Glitch / VHS ─────────────────────────────────────────────────────
        if cs.get('glitchEnabled'):
            intensity_map = {'subtle': 8, 'medium': 20, 'heavy': 40}
            noise_level = intensity_map.get(cs.get('glitchIntensity', 'subtle'), 8)
            # Split into two separate filter_parts (joined by ';') rather than a
            # comma-chained pair, so the ',' chain separator never appears in the
            # filter-complex string and can't be misread as an option delimiter.
            noise_mid = f'v_gnoise_{len(filter_parts)}'
            filter_parts.append(f"[{current_label}]noise=alls={noise_level}:allf=t[{noise_mid}]")
            nxt = f'v_glitch_{len(filter_parts)}'
            filter_parts.append(f"[{noise_mid}]eq=saturation=0.85[{nxt}]")
            current_label = nxt

        return current_label, audio_label

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
            
            # NOTE: Do NOT shortcut single-segment; run through the style/solo pipeline
            # so per-clip and global styles are applied even for 1-track chunks.

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

            # Use render config resolution for cell calculations
            target_width, target_height = map(int, self.render_config['resolution'].split('x'))
            # Ensure even dimensions — libx264 requires width/height divisible by 2
            cell_width = (target_width // grid_cols) & ~1
            cell_height = (target_height // grid_rows) & ~1
            
            logging.info(f"   Target resolution: {target_width}x{target_height} ({'PREVIEW' if self.preview_mode else 'PRODUCTION'})")
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
            style_temp_files = []  # Temp files created during styling (cleaned up after ffmpeg)

            # Determine canvas background color from composition style
            cs = getattr(self, 'composition_style', {}) or {}
            bg_hex = cs.get('backgroundColor', '#0a0a0f')
            bg_ffmpeg = self._hex_to_ffmpeg_color(bg_hex)

            # Add a placeholder input for empty cells using background color
            cmd.extend(['-f', 'lavfi', '-i', f'color={bg_ffmpeg}:s={cell_width}x{cell_height}:r=30:d={duration}'])
            black_video_input_idx = input_idx
            input_idx += 1
            cmd.extend(['-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={duration}'])
            silent_audio_input_idx = input_idx
            input_idx += 1
            
            logging.info(f"🎛️ Building FFmpeg filter complex...")
            logging.info(f"   Added bg placeholder ({bg_hex}, input {black_video_input_idx}) and silent audio (input {silent_audio_input_idx})")

            # Process each cell in the grid
            cells_processed = 0
            cells_with_content = 0
            decode_args = self._get_ffmpeg_decode_args()
            for r in range(grid_rows):
                for c in range(grid_cols):
                    cells_processed += 1
                    cell_segment = grid_cells[r][c]
                    if cell_segment and os.path.exists(cell_segment['video_path']):
                        cells_with_content += 1
                        cmd.extend([*decode_args, '-i', cell_segment['video_path']])

                        # Apply per-cell style filters (scale, pad/bgColor, grade, flash, label, border)
                        track_id = cell_segment.get('track_id', f'{r}_{c}')
                        self._apply_cell_style_filters(
                            filter_parts, f"[{input_idx}:v]", f"[v{r}_{c}]",
                            cell_width, cell_height, track_id, cell_segment, style_temp_files
                        )
                        video_inputs_for_stack.append(f"[v{r}_{c}]")
                        
                        # --- VOLUME FIX START ---
                        vol_db = float(self._resolve_segment_volume(cell_segment))
                        velocity_val = (
                            cell_segment.get('velocity')
                            or cell_segment.get('midi_velocity')
                            or cell_segment.get('note_velocity')
                        )
                        vol_db += self._velocity_to_db(velocity_val)
                        vol_linear = 10 ** (vol_db / 20.0)
                        filter_parts.append(f"[{input_idx}:a]volume={vol_linear:.2f}[a{r}_{c}]")
                        audio_inputs_for_mix.append(f"[a{r}_{c}]")
                        # --- VOLUME FIX END ---

                        logging.info(f"      Cell ({r},{c}): {cell_segment.get('type', 'unknown')} - {Path(cell_segment['video_path']).name} - Vol: {vol_db}dB")
                        input_idx += 1
                    else:
                        # Use the black placeholder for empty cells
                        video_inputs_for_stack.append(f"[{black_video_input_idx}:v]")
                        logging.info(f"      Cell ({r},{c}): EMPTY (using placeholder)")
            
            logging.info(f"   Grid cells: {cells_with_content}/{cells_processed} contain actual content")

            # ── Solo-resolution optimisation ────────────────────────────────
            # When only one cell has content in this chunk, skip xstack and
            # render the single clip at the full canvas resolution so the active
            # instrument fills the screen instead of appearing as a tiny cell.
            if cells_with_content == 1:
                solo_w = target_width & ~1
                solo_h = target_height & ~1
                # The single video is always at input index 2
                # (index 0 = bg placeholder, 1 = silent audio, 2 = first video)
                filter_parts.clear()
                style_temp_files.clear()
                solo_segment = next(
                    (grid_cells[r][c] for r in range(grid_rows) for c in range(grid_cols)
                     if grid_cells[r][c] and os.path.exists(grid_cells[r][c].get('video_path', ''))),
                    None
                )
                solo_track_id = solo_segment.get('track_id', 'solo') if solo_segment else 'solo'
                self._apply_cell_style_filters(
                    filter_parts, '[2:v]', '[v_solo]',
                    solo_w, solo_h, solo_track_id, solo_segment, style_temp_files
                )
                if solo_segment:
                    vol_db = float(self._resolve_segment_volume(solo_segment))
                    velocity_val = (
                        solo_segment.get('velocity')
                        or solo_segment.get('midi_velocity')
                        or solo_segment.get('note_velocity')
                    )
                    vol_db += self._velocity_to_db(velocity_val)
                    vol_linear = 10 ** (vol_db / 20.0)
                    filter_parts.append(
                        f"[2:a]volume={vol_linear:.2f},"
                        f"aformat=sample_fmts=fltp:channel_layouts=stereo[audio_pre]"
                    )
                else:
                    filter_parts.append(
                        f"[{silent_audio_input_idx}:a]"
                        f"aformat=sample_fmts=fltp:channel_layouts=stereo[audio_pre]"
                    )
                final_video, final_audio = self._apply_global_style_filters(
                    filter_parts, 'v_solo', solo_w, solo_h,
                    duration, 'audio_pre', style_temp_files
                )
                final_video_map = f'[{final_video}]' if not final_video.startswith('[') else final_video
                final_audio_map = f'[{final_audio}]' if not final_audio.startswith('[') else final_audio

                logging.info(f"🎯 Solo mode: rendering at {solo_w}x{solo_h} (full canvas, no xstack)")
                encoding_args = self._get_encoding_settings()
                fc_str = ';'.join(filter_parts)
                # Use filter_complex_script on Windows to avoid the 32k cmd-line length limit.
                # (-/filter_complex is the non-deprecated form but has parsing issues in this
                # FFmpeg build — keep using -filter_complex_script until that is resolved.)
                fc_script_path = None
                try:
                    fd, fc_script_path = tempfile.mkstemp(prefix='ats_fc_', suffix='.txt')
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(fc_str)
                    cmd.extend(['-filter_complex_script', fc_script_path])
                except Exception:
                    cmd.extend(['-filter_complex', fc_str])
                cmd.extend([
                    '-map', final_video_map, '-map', final_audio_map,
                    *encoding_args,
                    '-c:a', 'aac', '-b:a', self.render_config['audio_bitrate'],
                    '-pix_fmt', 'yuv420p',
                    '-t', str(duration), '-r', '30', str(output_path)
                ])
                result = subprocess.run(cmd, capture_output=True, text=True)
                # Cleanup temp files
                for tf in style_temp_files:
                    try: os.unlink(tf)
                    except Exception: pass
                if fc_script_path:
                    try: os.unlink(fc_script_path)
                    except Exception: pass
                if result.returncode == 0:
                    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    logging.info(f"✅ Solo chunk created: {output_size:,} bytes")
                    return str(output_path)
                else:
                    logging.warning(f"⚠️ Solo FFmpeg failed for chunk, skipping: {result.stderr[-300:]}")
                    return None
            # ── End solo-resolution optimisation ────────────────────────────

            force_cpu_encode = cells_with_content > self.max_concurrent_streams
            if force_cpu_encode:
                logging.warning(
                    f"High stream pressure ({cells_with_content} active cells) exceeds ATS_MAX_CONCURRENT_STREAMS={self.max_concurrent_streams}. "
                    "Falling back to CPU encode for stability."
                )

            # Create the xstack and amix filters
            layout_string = "|".join([f"{c*cell_width}_{r*cell_height}" for r in range(grid_rows) for c in range(grid_cols)])
            filter_parts.append(f"{''.join(video_inputs_for_stack)}xstack=inputs={grid_rows*grid_cols}:layout={layout_string}[xstack_out]")
            # Mix only actual audio inputs to avoid normalization over empty streams
            audio_input_count = len(audio_inputs_for_mix)
            if audio_input_count == 0:
                audio_inputs_for_mix = [f"[{silent_audio_input_idx}:a]"]
                audio_input_count = 1
            filter_parts.append(
                f"{''.join(audio_inputs_for_mix)}"
                f"amix=inputs={audio_input_count}:duration=longest:normalize=0,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"alimiter=limit=0.8:attack=80:release=500[audio_pre]"
            )

            # Apply global composition effects (title, tagline, watermark, waveform, vignette, glitch)
            final_video, final_audio = self._apply_global_style_filters(
                filter_parts, 'xstack_out', target_width, target_height,
                duration, 'audio_pre', style_temp_files
            )
            final_video_map = final_video if final_video.startswith('[') else f'[{final_video}]'
            final_audio_map = final_audio if final_audio.startswith('[') else f'[{final_audio}]'

            logging.info(f"🎬 Final FFmpeg command construction:")
            logging.info(f"   Total inputs: {input_idx}")
            logging.info(f"   Filter complex parts: {len(filter_parts)}")
            logging.info(f"   Grid layout: {layout_string}")

            # Use filter_complex_script on Windows to avoid the 32k cmd-line length limit.
            # (-/filter_complex is the non-deprecated form but has parsing issues in this
            # FFmpeg build — keep using -filter_complex_script until that is resolved.)
            fc_str = ';'.join(filter_parts)
            fc_script_path = None
            try:
                fd, fc_script_path = tempfile.mkstemp(prefix='ats_fc_', suffix='.txt')
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(fc_str)
                cmd.extend(['-filter_complex_script', fc_script_path])
            except Exception:
                cmd.extend(['-filter_complex', fc_str])

            # Get encoding settings based on preview mode and GPU availability
            encoding_args = self._get_encoding_settings()
            if force_cpu_encode:
                encoding_args = ['-c:v', 'libx264', '-preset', self.render_config['preset'], '-crf', self.render_config['crf']]
            
            cmd.extend([
                '-map', final_video_map, '-map', final_audio_map,
                *encoding_args,
                '-c:a', 'aac', '-b:a', self.render_config['audio_bitrate'],
                '-pix_fmt', 'yuv420p',
                '-t', str(duration), '-r', '30', str(output_path)
            ])

            logging.info(f"🚀 Executing FFmpeg grid composition...")
            logging.info(f"   Filter complex length: {len(fc_str)} characters")
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Cleanup temp files
            for tf in style_temp_files:
                try: os.unlink(tf)
                except Exception: pass
            if fc_script_path:
                try: os.unlink(fc_script_path)
                except Exception: pass
            
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
                logging.info(f"   Filter complex dump:\n{fc_str}")
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
                # Align drum source to onset to capture attack
                onset_offset = self._get_onset_offset(drum_video_path)
                triggered_video = self._create_note_triggered_video_sequence_fixed(
                    video_path=drum_video_path,
                    notes=chunk_notes,
                    total_duration=chunk_duration,
                    track_name=drum_track_id,
                    unique_id=short_id,
                    onset_offset=onset_offset
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

    def _find_instrument_video_file(self, name_or_normalized, original_name: str | None = None):
        """Find the video file for an instrument.

        Accepts either a normalized instrument name or an original display name.
        The second param is optional to preserve older call sites.
        """
        # Derive both original and normalized tokens safely
        if original_name is None:
            original_name = str(name_or_normalized)
        normalized_name = normalize_instrument_name(str(name_or_normalized))

        # Build robust search patterns covering common variations
        safe_original_underscore = original_name.lower().replace(' ', '_')
        search_patterns = [
            f"*{normalized_name}*.mp4",
            f"*{safe_original_underscore}*.mp4",
            f"*{original_name.lower()}*.mp4",
            f"processed_*{normalized_name}*.mp4",
            f"processed_*{safe_original_underscore}*.mp4",
        ]

        # Look in uploads directory first
        for pattern in search_patterns:
            for video_file in self.uploads_dir.glob(pattern):
                # Exclude drum files from instrument lookup
                if 'drum' in video_file.name.lower():
                    continue
                logging.info(f"🎹 Instrument match: {original_name} → {video_file.name}")
                return str(video_file)

        # Also check processed videos directory
        for pattern in search_patterns:
            for video_file in self.processed_videos_dir.glob(pattern):
                if 'drum' in video_file.name.lower():
                    continue
                logging.info(f"🎹 Instrument match (processed): {original_name} → {video_file.name}")
                return str(video_file)

        # No match found
        logging.warning(f"No instrument video found for: original='{original_name}', normalized='{normalized_name}'")
        return None
        
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
