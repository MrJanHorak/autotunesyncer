import os
import re
import gc
import hashlib
import threading
import subprocess
import sys
import logging
import traceback
import os.path
import math
import shutil
import tempfile
import json
import time
import mmap
import weakref
import numpy as np
import torch
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from cachetools import LRUCache
from logging.handlers import RotatingFileHandler

# Optional audio analysis
try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    aubio = None
    AUBIO_AVAILABLE = False
    logging.warning("aubio not available, some audio analysis features disabled")

# Optional tqdm with graceful fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    class tqdm:  # noqa: N801
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return iter(self.iterable) if self.iterable else iter([])
        def update(self, n=1): pass
        def set_postfix(self, **kwargs): pass
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available, progress bars disabled")

from moviepy import (
    VideoFileClip,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips,
)

try:
    from utils import normalize_instrument_name, midi_to_note
except ImportError:
    def normalize_instrument_name(name):
        """Match frontend's normalizeInstrumentName."""
        return name.lower().replace(' ', '_')

    def midi_to_note(midi_num):
        """Convert MIDI note number to note name."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{notes[midi_num % 12]}{(midi_num // 12) - 1}"

from drum_utils import DRUM_NOTES, is_drum_kit

from processing_utils import encoder_queue, GPUManager

GRID_SLOT_PADDING_REFERENCE_SIDE = 1080
GRID_SLOT_PADDING_AT_REFERENCE = 8

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

try:
    from gpu_pipeline import GPUPipelineProcessor
except ImportError:
    GPUPipelineProcessor = None
    logging.warning("GPU pipeline processor not available")

from video_utils import run_ffmpeg_command, encode_video, validate_video
from path_registry import PathRegistry
from optimized_autotune_cache import OptimizedAutotuneCache as _RealOptimizedAutotuneCache

# ── Utility classes / helpers (see composer_utils.py) ─────────────────────────
from composer_utils import (
    GPUStreamManager,
    gpu_subprocess_run,
    ClipPool,
    ClipManager,
    VideoComposerConfig,
    get_system_metrics,
)

# Log to an absolute path in the backend/ directory (next to this file's parent)
_LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'video_processing.log'))

# Persistent disk cache for per-clip onset offsets so the expensive librosa
# onset-detection pass is skipped on subsequent renders of the same source clips.
_ONSET_CACHE_PATH = Path.home() / '.autotunesyncer' / 'onset_offsets.json'
_ONSET_CACHE_MAX_ENTRIES = 1000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Rotate at 10 MB, keep last 5 files — prevents unbounded log growth.
        RotatingFileHandler(_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'),
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

# Use the real OptimizedAutotuneCache from optimized_autotune_cache.py
OptimizedAutotuneCache = _RealOptimizedAutotuneCache


# ── Main compositor ────────────────────────────────────────────────────────────
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
            self.midi_data = midi_data
            
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
            self.background_media = midi_data.get('backgroundMedia', {}) or {}
            if self.composition_style:
                logging.info(f"Composition style loaded: {list(self.composition_style.keys())}")
            else:
                logging.info("Composition style: empty (no global effects)")
            if self.clip_styles:
                logging.info(f"Clip styles loaded for: {list(self.clip_styles.keys())}")
            else:
                logging.info("Clip styles: empty (no per-clip effects)")
            if self.background_media:
                logging.info(
                    f"Background media loaded: {self.background_media.get('kind')} "
                    f"{self.background_media.get('path')}"
                )

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
            self._setup_paths(processed_videos_dir, output_path)
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
            # Cross-run disk cache for onset offsets (loaded once, flushed on composition end)
            self._onset_disk_cache: dict = self._load_onset_disk_cache()
            self._onset_disk_cache_dirty: bool = False
            # Media duration cache — avoids duplicate ffprobe calls from parallel stem threads
            self._duration_cache: dict = {}
            self._duration_cache_lock = threading.RLock()
            # Global FFmpeg concurrency cap — shared by stem sub-batches AND video chunks.
            # Caps simultaneous filter-complex FFmpeg processes to prevent GPU memory exhaustion.
            # For GPU encoding: allow more streams (GPU can handle 32-64+ concurrent h264_nvenc).
            # For CPU encoding: limit to half the CPU count for stability.
            _cpu = os.cpu_count() or 4
            self._ffmpeg_semaphore = threading.Semaphore(max(4, _cpu // 2))
            # Default: 32 streams (suitable for GPU; will be limited by CPU fallback if needed)
            self.max_concurrent_streams = int(os.environ.get('ATS_MAX_CONCURRENT_STREAMS', '32'))
            self.ffmpeg_hwaccel = self._detect_ffmpeg_hwaccel()
            
            # CRITICAL: Log GPU status explicitly
            if self.ffmpeg_hwaccel == 'cuda':
                logging.info(f"✅ GPU ACCELERATION ENABLED: h264_nvenc (NVIDIA CUDA)")
            elif self.ffmpeg_hwaccel == 'videotoolbox':
                logging.info(f"✅ GPU ACCELERATION ENABLED: h264_videotoolbox (Apple)")
            else:
                logging.warning(f"⚠️ GPU NOT AVAILABLE: Falling back to CPU encoding (libx264) — composition will be VERY SLOW (~40-50x slower)")
                logging.warning(f"   Please check FFmpeg installation: run 'ffmpeg -hwaccels' to verify CUDA support")
            
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
                    logging.info(f"✅ CUDA initialized successfully: {torch.cuda.get_device_name(0)}")
                    
                    # Set optimal settings for video processing
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    logging.warning("⚠️ CUDA not available in PyTorch, using CPU processing")
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
        requested_dimensions = {}
        if isinstance(getattr(self, 'midi_data', None), dict):
            requested_dimensions = self.midi_data.get('renderDimensions', {}) or {}

        default_width = 640 if self.preview_mode else 1920
        default_height = 360 if self.preview_mode else 1080

        requested_width = requested_dimensions.get('width')
        requested_height = requested_dimensions.get('height')

        width = int(requested_width) if str(requested_width).isdigit() else default_width
        height = int(requested_height) if str(requested_height).isdigit() else default_height
        resolution = f'{width}x{height}'

        if self.preview_mode:
            return {
                'resolution': resolution,
                'preset': 'ultrafast',  # CPU: ultrafast, GPU: p1/p2
                'crf': '28',            # Lower quality
                'audio_bitrate': '128k',
                'video_bitrate': '1M',
                'scale_filter': f'scale={width}:{height}'
            }
        else:
            return {
                'resolution': resolution,
                'preset': 'fast',       # CPU: fast, GPU: p4
                'crf': '23',            # Keep chunk detail higher so enlarged tiles stay crisp after final encode
                'audio_bitrate': '192k',
                'video_bitrate': '5M',  # Used by non-CRF encoders; moderate bump helps larger tiles retain detail
                'scale_filter': f'scale={width}:{height}'
            }

    def _get_target_resolution(self):
        try:
            return map(int, self.render_config['resolution'].split('x'))
        except Exception:
            return (
                (640, 360)
                if self.preview_mode
                else (1920, 1080)
            )

    def _get_preview_stage_scale_factor(self):
        """Scale editor preview-authored pixel values to target render pixels."""
        try:
            preview_dimensions = self.midi_data.get('previewStageDimensions', {}) or {}
            preview_width = float(preview_dimensions.get('width') or 0)
            preview_height = float(preview_dimensions.get('height') or 0)
            target_width, target_height = self._get_target_resolution()

            # The interactive stage can expand well beyond the preset preview canvas
            # on desktop. Text styling is authored against the preset preview size,
            # so clamp the reference box to that canonical canvas to avoid exporting
            # titles and watermarks at an undersized scale.
            canonical_preview_width = max(1.0, float(target_width) / 3.0)
            canonical_preview_height = max(1.0, float(target_height) / 3.0)

            reference_width = preview_width if preview_width > 0 else canonical_preview_width
            reference_height = preview_height if preview_height > 0 else canonical_preview_height
            reference_width = min(reference_width, canonical_preview_width)
            reference_height = min(reference_height, canonical_preview_height)

            width_scale = float(target_width) / reference_width
            height_scale = float(target_height) / reference_height
            scale = min(width_scale, height_scale)

            if scale <= 0:
                return 1.0

            return max(0.1, min(10.0, scale))
        except Exception:
            return 1.0

    def _get_encoding_settings(self):
        """Get FFmpeg encoding arguments based on configuration and hardware"""
        config = self.render_config

        if self.ffmpeg_hwaccel == 'cuda':
            # NVENC settings with safer, more compatible options
            # Preset: p1 (fastest) to p7 (highest quality) — use p2-p4 for balanced speed/quality
            _prod_preset = os.environ.get('ATS_NVENC_PRESET', 'p4')
            _prod_bitrate = os.environ.get('ATS_NVENC_BITRATE', '')
            _prod_maxrate = os.environ.get('ATS_NVENC_MAXRATE', '')

            # Quality/CRF: For NVENC, map libx264 CRF (0-51) to NVENC CQ (0-51)
            # libx264 crf='26' -> NVENC cq=26 (lossy but reasonable)
            crf_val = int(config['crf']) if isinstance(config['crf'], str) else config['crf']
            crf_val = max(0, min(51, crf_val))  # Clamp to valid range

            args = [
                '-c:v', 'h264_nvenc',
                '-preset', 'p1' if self.preview_mode else _prod_preset,
                '-rc', 'vbr',  # Variable bitrate for better quality
                '-cq', str(crf_val),  # Quality level
            ]
            if not self.preview_mode:
                if _prod_bitrate:
                    args += ['-b:v', _prod_bitrate]
                if _prod_maxrate:
                    args += ['-maxrate', _prod_maxrate]
            return args
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
                # CUDA hwaccel is available in FFmpeg, but encoder might not work at runtime
                # Verify the encoder is actually functional before returning
                if self._verify_gpu_encoder_works():
                    return 'cuda'
                else:
                    logging.warning("❌ CUDA detected but h264_nvenc encoder verification FAILED")
                    return None
            if 'videotoolbox' in lower:
                return 'videotoolbox'
            return None
        except Exception as e:
            logging.warning(f"Failed to detect FFmpeg hwaccels: {e}")
            return None

    def _verify_gpu_encoder_works(self):
        """
        Quick test: can h264_nvenc actually encode?
        Creates a 1-frame test video to verify GPU encoder works before composition starts.
        """
        try:
            import tempfile
            import os
            
            # Create a temporary test output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                test_output = f.name
            
            try:
                # Quick test: 1 second of black video encoded with h264_nvenc
                test_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-f', 'lavfi', '-i', 'color=black:s=320x240:r=30:d=0.5',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p1',
                    '-t', '0.5',
                    test_output
                ]
                
                result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10
                )
                
                success = result.returncode == 0 and os.path.exists(test_output) and os.path.getsize(test_output) > 100
                
                if success:
                    logging.info("✅ h264_nvenc encoder verification PASSED")
                else:
                    logging.warning(f"❌ h264_nvenc encoder test failed: {result.stderr[-200:] if result.stderr else 'no output'}")
                
                return success
            finally:
                # Clean up test file
                if os.path.exists(test_output):
                    try:
                        os.unlink(test_output)
                    except Exception:
                        pass
        except subprocess.TimeoutExpired:
            logging.warning("❌ h264_nvenc encoder test timed out (GPU may be busy or driver issue)")
            return False
        except Exception as e:
            logging.warning(f"❌ h264_nvenc encoder test error: {e}")
            return False

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

    # ── Onset disk-cache helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_onset_disk_cache() -> dict:
        """Load persisted onset-offset map from disk; returns {} on any failure."""
        try:
            if _ONSET_CACHE_PATH.exists():
                with open(_ONSET_CACHE_PATH, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _flush_onset_disk_cache(self) -> None:
        """Atomically write the in-memory onset-offset map to disk, pruning to the
        most-recently-added *_ONSET_CACHE_MAX_ENTRIES* entries when over the cap.
        Silently no-ops if the map is unchanged or the write fails."""
        if not self._onset_disk_cache_dirty:
            return
        try:
            cache = self._onset_disk_cache
            if len(cache) > _ONSET_CACHE_MAX_ENTRIES:
                # Keep the last N entries (most recently added; dict preserves insertion order)
                entries = list(cache.items())[-_ONSET_CACHE_MAX_ENTRIES:]
                cache = dict(entries)
                self._onset_disk_cache = cache
            _ONSET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = str(_ONSET_CACHE_PATH) + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as fh:
                json.dump(cache, fh)
            os.replace(tmp_path, str(_ONSET_CACHE_PATH))
            self._onset_disk_cache_dirty = False
            logging.debug(f"[onset-cache] flushed {len(cache)} entries to disk")
        except Exception as exc:
            logging.warning(f"[onset-cache] flush failed: {exc}")

    @staticmethod
    def _onset_content_key(path: str) -> 'str | None':
        """Stable content fingerprint for a video file: sha256 of first+last 1 KB
        bytes concatenated with the file size.  Returns None on I/O error."""
        try:
            size = os.path.getsize(path)
            with open(path, 'rb') as fh:
                head = fh.read(1024)
                if size > 2048:
                    fh.seek(-1024, 2)
                    tail = fh.read(1024)
                else:
                    tail = b''
            digest = hashlib.sha256(head + tail + str(size).encode()).hexdigest()[:24]
            return digest
        except Exception:
            return None

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
        A cross-run disk cache (keyed by file content fingerprint) avoids re-running
        the expensive librosa pipeline when the same source file is used across renders.
        """
        try:
            if not video_path or not os.path.exists(video_path):
                return 0.0
            # Fast path: in-memory cache (pure read, GIL-safe for dict lookup)
            if video_path in self.onset_offset_cache:
                return self.onset_offset_cache[video_path]

            # Per-path lock: only one thread runs onset detection per clip path.
            # defaultdict(Lock) creation is GIL-protected under CPython.
            with self._onset_path_locks[video_path]:
                # Re-check: another thread may have populated while we waited.
                if video_path in self.onset_offset_cache:
                    return self.onset_offset_cache[video_path]

                # Check cross-run disk cache before invoking librosa
                content_key = self._onset_content_key(video_path)
                if content_key and content_key in self._onset_disk_cache:
                    cached_offset = self._onset_disk_cache[content_key]
                    self.onset_offset_cache[video_path] = cached_offset
                    logging.debug(f"[onset-cache] disk hit {content_key[:8]}… → {cached_offset:.3f}s for {Path(video_path).name}")
                    return cached_offset

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
                    if content_key:
                        self._onset_disk_cache[content_key] = offset
                        self._onset_disk_cache_dirty = True
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
                        if content_key:
                            self._onset_disk_cache[content_key] = offset
                            self._onset_disk_cache_dirty = True
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
        # Guard: if no track has notes, nothing to normalize
        any_notes = any(track.get('notes') for track in midi_data.get('tracks', []))
        if not any_notes:
            logging.warning("normalize_midi_timing: no notes in any track; skipping")
            return midi_data
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
            # No explicit video files supplied — this should only happen when VideoComposer
            # is used directly (tests, CLI) rather than through the Node.js wrapper.
            # Scanning the shared uploads dir can pick up videos from prior sessions, so
            # emit a prominent WARNING to make contamination visible in logs.
            logging.warning(
                "⚠️  explicit_video_files is empty — falling back to shared uploads scan. "
                "This is expected only for direct/test usage; in production the Node.js "
                "wrapper must always supply videoFiles in the composition payload."
            )
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
        style_track_id=None,
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

            source_width, source_height, _ = self._get_video_info(video_path)
            target_width = max(2, int(source_width or 640) & ~1)
            target_height = max(2, int(source_height or 360) & ~1)

            # Sanitize & normalize notes
            MIN_DUR = 0.10  # 100 ms min to avoid ffmpeg micro durations
            valid = []
            for n in notes or []:
                raw_start = float(n.get("time", 0.0))
                audio_dur = float(n.get("duration", 0.0))
                if audio_dur <= 0:
                    continue

                # Convert to chunk-relative
                rel_start = raw_start - chunk_start_time
                # If notes were already relative (e.g. small start while chunk_start_time>0),
                # allow negative tolerance then clamp.
                if rel_start < -0.001:
                    # Starts before this chunk; trim head
                    head_trim = -rel_start
                    audio_dur -= head_trim
                    rel_start = 0.0
                if rel_start >= total_duration:
                    continue

                # Clamp to chunk boundary
                if rel_start + audio_dur > total_duration:
                    audio_dur = total_duration - rel_start
                if audio_dur <= 0:
                    continue
                if audio_dur < MIN_DUR:
                    audio_dur = min(MIN_DUR, max(0.0, total_duration - rel_start))
                    if audio_dur <= 0:
                        continue

                midi_note = int(n.get("midi", 60))
                valid.append({
                    'start': round(rel_start, 3),
                    'audio_duration': round(audio_dur, 3),
                    'midi_note': midi_note,
                    'note_ref': n,
                })

            if not valid:
                # Fallback: simple loop (keeps something visible)
                return self._create_simple_loop(video_path, out_path, total_duration)

            valid.sort(key=lambda item: item['start'])
            visual_timing_notes = self._calculate_visual_durations(
                [
                    {
                        'time': item['start'],
                        'duration': item['audio_duration'],
                    }
                    for item in valid
                ],
                total_duration,
            )
            for item, visual_note in zip(valid, visual_timing_notes):
                visual_duration = round(
                    float(visual_note.get('visual_duration', item['audio_duration'])),
                    3,
                )
                item['video_duration'] = visual_duration
                item['note_ref']['visual_duration'] = visual_duration

            # Build the mapping from midi_note → FFmpeg input index for pre-tuned audio.
            # Inputs 0, 1, 2 are: source video, black frame, silent audio.
            # Pre-tuned video files (audio only) start at index 3.
            note_input_index: dict = {}  # midi_note -> ffmpeg input index
            extra_audio_inputs: list = []  # paths appended as additional -i args
            BASE_EXTRA_IDX = 3
            if note_audio_map:
                unique_cached = sorted(
                    {
                        item['midi_note']
                        for item in valid
                        if item['midi_note'] in note_audio_map
                    }
                )
                for mn in unique_cached:
                    note_input_index[mn] = BASE_EXTRA_IDX + len(extra_audio_inputs)
                    extra_audio_inputs.append(note_audio_map[mn])

            background_media = self._get_active_background_media()
            style_lookup_id = style_track_id if style_track_id is not None else track_name
            clip_style, _, matched_key = self._resolve_clip_style(style_lookup_id)
            preserve_idle_alpha = bool(background_media) or bool(clip_style.get('transparentBg'))
            out_suffix = '.mov' if preserve_idle_alpha else '.mp4'
            out_path = self.temp_dir / f"{track_name}_{unique_id}{out_suffix}"
            if out_path.exists():
                try:
                    out_path.unlink()
                except:
                    pass

            cs = getattr(self, 'composition_style', {}) or {}
            bg_hex = cs.get('backgroundColor', '#0a0a0f')
            if background_media:
                logging.info(
                    f"[style] note-trigger base for {track_name!r}: "
                    f"{background_media['kind']} {Path(background_media['path']).name}"
                )
            elif preserve_idle_alpha:
                logging.info(
                    f"[style] note-trigger base for {track_name!r}: transparent idle gaps "
                    f"({matched_key or style_lookup_id})"
                )
            elif (
                clip_style.get('bgColorEnabled')
                and clip_style.get('bgColor')
                and not clip_style.get('transparentBg')
            ):
                bg_hex = clip_style.get('bgColor')
                logging.info(
                    f"[style] note-trigger base for {track_name!r}: {bg_hex} "
                    f"({matched_key or style_lookup_id})"
                )
            bg_ffmpeg = self._hex_to_ffmpeg_color(bg_hex)

            # Build filter parts
            filter_parts = [
                # Base cell background and silent audio come from inputs 1 & 2
                (
                    f"[1:v]trim=0:{total_duration},format=rgba,colorchannelmixer=aa=0,setpts=PTS-STARTPTS[base_v]"
                    if preserve_idle_alpha
                    else f"[1:v]trim=0:{total_duration},setpts=PTS-STARTPTS[base_v]"
                ),
                f"[2:a]atrim=0:{total_duration},asetpts=PTS-STARTPTS[base_a]"
            ]

            video_chain = "[base_v]"
            audio_streams = ["[base_a]"]

            # Base onset offset
            onset_base = 0.0 if onset_offset is None else max(0.0, min(float(onset_offset), 5.0))
            source_duration = self._get_media_duration(video_path)

            for i, item in enumerate(valid):
                start = item['start']
                audio_dur = item['audio_duration']
                video_dur = item['video_duration']
                midi_note = item['midi_note']

                # Pitch factor relative to C4 (60)
                pitch_factor = 2 ** ((midi_note - 60) / 12.0)

                # Compute safe onset per note so trim doesn't overshoot
                safe_onset = onset_base
                if source_duration > 0.0:
                    max_start = max(0.0, source_duration - video_dur - 0.01)
                    safe_onset = min(onset_base, max_start)
                if safe_onset < 0.0:
                    safe_onset = 0.0
                logging.info(
                    f"[NoteTrigger] {track_name} note {i}: onset_base={onset_base:.3f}s, "
                    f"safe_onset={safe_onset:.3f}s, audio_dur={audio_dur:.3f}s, "
                    f"video_dur={video_dur:.3f}s"
                )

                # Video always comes from the original source (input 0)
                if preserve_idle_alpha:
                    filter_parts.append(
                        f"[0:v]trim=start={safe_onset}:duration={video_dur},setpts=PTS-STARTPTS,"
                        f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=increase,"
                        f"crop={target_width}:{target_height},setsar=1,format=rgba,"
                        f"tpad=stop_mode=clone:stop_duration={video_dur:.3f},"
                        f"trim=duration={video_dur:.3f},"
                        f"setpts=PTS-STARTPTS+{start:.3f}/TB[v{i}]"
                    )
                else:
                    filter_parts.append(
                        f"[0:v]trim=start={safe_onset}:duration={video_dur},setpts=PTS-STARTPTS,"
                        f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=increase,"
                        f"crop={target_width}:{target_height},setsar=1,"
                        f"setpts=PTS-STARTPTS+{start:.3f}/TB[v{i}]"
                    )

                # Audio: prefer pre-tuned file → fall back to asetrate+atempo
                if midi_note in note_input_index:
                    idx = note_input_index[midi_note]
                    filter_parts.append(
                        f"[{idx}:a]atrim=start={safe_onset}:duration={audio_dur},asetpts=PTS-STARTPTS[a{i}]"
                    )
                elif abs(pitch_factor - 1.0) > 0.01:
                    # asetrate shifts pitch but compresses/stretches duration by 1/pitch_factor.
                    # atempo chain compensates to restore the original duration.
                    atempo = self._build_atempo_chain(1.0 / pitch_factor)
                    filter_parts.append(
                        f"[0:a]atrim=start={safe_onset}:duration={audio_dur},asetpts=PTS-STARTPTS,"
                        f"asetrate=44100*{pitch_factor},aresample=44100,{atempo}[a{i}]"
                    )
                else:
                    filter_parts.append(
                        f"[0:a]atrim=start={safe_onset}:duration={audio_dur},asetpts=PTS-STARTPTS[a{i}]"
                    )

                # Overlay enable window
                end = start + video_dur
                filter_parts.append(
                    f"{video_chain}[v{i}]overlay="
                    f"eof_action=pass:repeatlast=0:format={'auto' if preserve_idle_alpha else 'yuv420'}:"
                    f"enable='between(t,{start:.3f},{end:.3f})'[ov{i}]"
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

            filter_parts.append(
                f"{video_chain}format={'argb' if preserve_idle_alpha else 'yuv420p'}[final_v]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
            ]
            if preserve_idle_alpha:
                cmd += [
                    "-f", "lavfi", "-i",
                    f"color=c=black@0.0:size={target_width}x{target_height}:rate=30:duration={total_duration}",
                ]
            else:
                cmd += [
                    "-f", "lavfi", "-i",
                    f"color={bg_ffmpeg}:size={target_width}x{target_height}:rate=30:duration={total_duration}",
                ]
            cmd += [
                "-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_duration}",
            ]
            # Add pre-tuned audio sources (one per unique cached midi_note)
            for tuned_path in extra_audio_inputs:
                cmd += ["-i", str(tuned_path)]

            tail_args = [
                "-map", "[final_v]", "-map", "[final_a]",
                "-t", f"{total_duration:.3f}",
                *( ["-c:v", "qtrle"] if preserve_idle_alpha else ["-c:v", "libx264", "-preset", "fast", "-crf", "23"] ),
                "-c:a", "aac", "-b:a", "192k",
                "-r", "30",
                "-pix_fmt", "argb" if preserve_idle_alpha else "yuv420p",
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

            def parse_grid_int(value, fallback=None, minimum=0):
                try:
                    parsed = int(round(float(value)))
                except (TypeError, ValueError):
                    return fallback
                return max(minimum, parsed)
            
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
            self.grid_stage_rows = 1
            self.grid_stage_cols = 1
            
            # Handle different grid arrangement formats
            if 'layout' in grid_arrangement:
                # New format with layout matrix
                layout = grid_arrangement['layout']
                rows = parse_grid_int(grid_arrangement.get('rows', len(layout)), max(1, len(layout)), 1)
                cols = parse_grid_int(
                    grid_arrangement.get('cols', len(layout[0]) if layout else 1),
                    max(1, len(layout[0]) if layout else 1),
                    1,
                )
                self.grid_stage_rows = rows
                self.grid_stage_cols = cols
                
                logging.info(f"Processing layout matrix: {rows}x{cols}")
                
                for row_idx, row in enumerate(layout):
                    for col_idx, cell in enumerate(row):
                        if isinstance(cell, dict) and 'instrument' in cell:
                            track_id = str(cell.get('track', f"{cell['instrument']}"))
                            self.grid_positions[track_id] = {
                                'row': row_idx,
                                'column': col_idx,
                                'w': parse_grid_int(cell.get('w', 1), 1, 1),
                                'h': parse_grid_int(cell.get('h', 1), 1, 1),
                            }
                            logging.info(f"Mapped track {track_id} ({cell['instrument']}) to position row={row_idx}, col={col_idx}")
            elif isinstance(grid_arrangement.get('items'), dict):
                items = grid_arrangement.get('items', {})
                self.grid_stage_rows = parse_grid_int(grid_arrangement.get('rows'), 1, 1)
                self.grid_stage_cols = parse_grid_int(grid_arrangement.get('columns'), 1, 1)

                logging.info(
                    f"Processing span-aware grid items: {self.grid_stage_rows}x{self.grid_stage_cols}"
                )

                for track_id, pos_data in items.items():
                    if not isinstance(pos_data, dict):
                        logging.error(f"Invalid v2 position data for {track_id}: {pos_data}")
                        continue

                    row = parse_grid_int(pos_data.get('y', pos_data.get('row')), None, 0)
                    column = parse_grid_int(pos_data.get('x', pos_data.get('column')), None, 0)
                    if row is None or column is None:
                        logging.error(f"Invalid v2 coordinates for {track_id}: {pos_data}")
                        continue

                    width = parse_grid_int(pos_data.get('w', 1), 1, 1)
                    height = parse_grid_int(pos_data.get('h', 1), 1, 1)
                    self.grid_positions[str(track_id)] = {
                        'row': row,
                        'column': column,
                        'w': width,
                        'h': height,
                    }
                    logging.info(
                        f"Mapped v2 track {track_id} to row={row}, col={column}, span={width}x{height}"
                    )
            else:
                # Original format with direct track ID mappings
                max_row_end = 1
                max_col_end = 1
                for track_id, pos_data in grid_arrangement.items():
                    if isinstance(pos_data, dict):
                        # Validate position data - check for required keys
                        required_keys = ['row', 'column']
                        if not all(k in pos_data for k in required_keys):
                            logging.error(f"Invalid position data for {track_id}: {pos_data}")
                            continue

                        row = parse_grid_int(pos_data.get('row'), None, 0)
                        column = parse_grid_int(pos_data.get('column'), None, 0)
                        if row is None or column is None:
                            logging.error(f"Invalid position data for {track_id}: {pos_data}")
                            continue

                        width = parse_grid_int(pos_data.get('w', 1), 1, 1)
                        height = parse_grid_int(pos_data.get('h', 1), 1, 1)
                        
                        # Store position based on track index
                        self.grid_positions[track_id] = {
                            'row': row,
                            'column': column,
                            'w': width,
                            'h': height,
                        }
                        max_row_end = max(max_row_end, row + height)
                        max_col_end = max(max_col_end, column + width)
                        logging.info(
                            f"Mapped track {track_id} to position row={row}, col={column}, span={width}x{height}"
                        )

                self.grid_stage_rows = max(1, max_row_end)
                self.grid_stage_cols = max(1, max_col_end)

            logging.info(
                f"Grid stage contract: {self.grid_stage_rows} rows x {self.grid_stage_cols} columns"
            )

            # Add this call to analyze MIDI timing data
            self._analyze_midi_timing()

        except Exception as e:
            logging.error(f"Error setting up track configuration: {e}")
            raise

    def _get_grid_stage_dimensions(self):
        grid_rows = int(getattr(self, 'grid_stage_rows', 0) or 0)
        grid_cols = int(getattr(self, 'grid_stage_cols', 0) or 0)

        if grid_rows > 0 and grid_cols > 0:
            return grid_rows, grid_cols

        max_row_end = max(
            (
                int(pos.get('row', 0)) + max(1, int(pos.get('h', 1) or 1))
                for pos in self.grid_positions.values()
            ),
            default=1,
        )
        max_col_end = max(
            (
                int(pos.get('column', 0)) + max(1, int(pos.get('w', 1) or 1))
                for pos in self.grid_positions.values()
            ),
            default=1,
        )
        return max(1, max_row_end), max(1, max_col_end)

    def _get_grid_slot_padding(self, target_width, target_height, slot_width, slot_height):
        base_padding = max(
            2,
            int(
                round(
                    (min(target_width, target_height) / GRID_SLOT_PADDING_REFERENCE_SIDE)
                    * GRID_SLOT_PADDING_AT_REFERENCE
                )
            ),
        )
        max_slot_padding = max(0, min(slot_width // 6, slot_height // 6))
        return min(base_padding, max_slot_padding)

    def _build_grid_render_slots(self, track_segments, target_width, target_height):
        def coerce_int(value, fallback, minimum=0):
            try:
                parsed = int(round(float(value)))
            except (TypeError, ValueError):
                return fallback
            return max(minimum, parsed)

        grid_rows, grid_cols = self._get_grid_stage_dimensions()
        unit_width = max(2, (target_width // grid_cols) & ~1)
        unit_height = max(2, (target_height // grid_rows) & ~1)
        slots = []
        occupied_cells = {}

        for segment in track_segments:
            track_id = segment.get('track_id')
            position = self.grid_positions.get(track_id, {})
            row = coerce_int(segment.get('grid_row', position.get('row')), None, 0)
            column = coerce_int(segment.get('grid_col', position.get('column')), None, 0)
            if row is None or column is None:
                logging.warning(
                    f"   ⚠️ No resolved grid position found for track_id: {track_id}. It will be excluded."
                )
                continue

            span_w = coerce_int(segment.get('grid_w', position.get('w', 1)), 1, 1)
            span_h = coerce_int(segment.get('grid_h', position.get('h', 1)), 1, 1)

            if row + span_h > grid_rows or column + span_w > grid_cols:
                logging.warning(
                    f"   ❌ Track {track_id} span ({row}, {column}, {span_w}, {span_h}) is out of bounds for grid {grid_rows}x{grid_cols}."
                )
                continue

            overlap_owner = None
            for r in range(row, row + span_h):
                for c in range(column, column + span_w):
                    owner = occupied_cells.get((r, c))
                    if owner and owner != track_id:
                        overlap_owner = owner
                        break
                if overlap_owner:
                    break

            if overlap_owner:
                logging.warning(
                    f"   ⚠️ Track {track_id} overlaps grid cells already assigned to {overlap_owner}. Later overlay order will win."
                )

            for r in range(row, row + span_h):
                for c in range(column, column + span_w):
                    occupied_cells[(r, c)] = track_id

            pixel_x = column * unit_width
            pixel_y = row * unit_height
            pixel_width = (
                target_width - pixel_x if column + span_w >= grid_cols else unit_width * span_w
            )
            pixel_height = (
                target_height - pixel_y if row + span_h >= grid_rows else unit_height * span_h
            )
            pixel_width = max(2, int(pixel_width) & ~1)
            pixel_height = max(2, int(pixel_height) & ~1)
            slot_padding = self._get_grid_slot_padding(
                target_width,
                target_height,
                pixel_width,
                pixel_height,
            )
            render_width = max(2, int(max(2, pixel_width - (slot_padding * 2))) & ~1)
            render_height = max(2, int(max(2, pixel_height - (slot_padding * 2))) & ~1)

            slots.append({
                'segment': segment,
                'track_id': track_id,
                'row': row,
                'column': column,
                'span_w': span_w,
                'span_h': span_h,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'pixel_width': pixel_width,
                'pixel_height': pixel_height,
                'slot_padding': slot_padding,
                'render_x': pixel_x + slot_padding,
                'render_y': pixel_y + slot_padding,
                'render_width': render_width,
                'render_height': render_height,
            })

        slots.sort(key=lambda slot: (slot['row'], slot['column'], str(slot.get('track_id', ''))))
        return {
            'grid_rows': grid_rows,
            'grid_cols': grid_cols,
            'unit_width': unit_width,
            'unit_height': unit_height,
            'slots': slots,
        }

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
        target_width, target_height = self._get_target_resolution()
        return self._render_background_only_chunk(
            chunk_path,
            self.CHUNK_DURATION,
            target_width,
            target_height,
        )
        
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

        if self.drum_tracks:
            # Merge all drum tracks into a single deduplicated stem task.
            # MIDI files with N drum tracks often carry the same channel-10 note
            # list in each entry, so building them as separate stems then amixing
            # inflates drum volume by N×.  Deduplicating by (midi, centisecond)
            # collapses identical hits while preserving genuinely distinct events.
            seen_note_keys: set = set()
            merged_notes: list = []
            for dt in self.drum_tracks:
                for n in dt.get('notes', []):
                    key = (n.get('midi', 0), round(n.get('time', 0) * 100))
                    if key not in seen_note_keys:
                        seen_note_keys.add(key)
                        merged_notes.append(n)
            if merged_notes:
                merged_drum = {**self.drum_tracks[0], 'notes': merged_notes}
                stem_path = audio_dir / "stem_drums_merged.wav"
                tasks.append(('drum', 0, merged_drum, None, stem_path))
                if len(self.drum_tracks) > 1:
                    logging.info(
                        f"🥁 Merged {len(self.drum_tracks)} drum tracks → "
                        f"{len(merged_notes)} unique hits for single stem "
                        f"(was {sum(len(dt.get('notes',[])) for dt in self.drum_tracks)} total)"
                    )

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

        # Use an MP4/M4A container for stable/accurate duration metadata.
        # Raw ADTS AAC can report wildly incorrect duration via ffprobe.
        mastered_path = audio_dir / "mastered_audio.m4a"
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
            logging.error(f"❌ Audio strip failed: {r.stderr[-1500:]}")
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
            pad_audio = (
                audio_dur is not None
                and total_duration is not None
                and audio_dur < total_duration - 0.05
            )
            cmd = [
                'ffmpeg', '-y',
                '-i', str(silent_video),
                '-i', str(audio_path),
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy',
            ]
            if pad_audio:
                silence_pad = max(0.0, total_duration - audio_dur)
                logging.info(
                    f"🔇 Padding mux audio with {silence_pad:.2f}s silence to match {total_duration:.2f}s"
                )
                cmd.extend([
                    '-af', f'apad=whole_dur={total_duration:.3f}',
                    '-c:a', 'aac',
                    '-b:a', self.render_config.get('audio_bitrate', '192k'),
                ])
            else:
                cmd.extend(['-c:a', 'copy'])
            cmd.extend([
                '-t', str(total_duration),
                str(output_path),
            ])
            r = subprocess.run(
                cmd,
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if r.returncode == 0 and os.path.exists(output_path):
                logging.info(f"✅ Mux complete: {os.path.getsize(output_path):,} bytes")
                return str(output_path)
            logging.error(f"❌ Mux failed: {r.stderr[-1500:]}")
            return None
        except Exception as e:
            logging.error(f"❌ _mux_video_audio: {e}")
            return None

    def _compress_final_output(self, input_path: str, output_path: str,
                                total_duration: 'float | None' = None) -> 'str | None':
        """
        Final re-encode pass: compress the muxed output with GPU acceleration
        when available, falling back to CPU (libx264) on encoder failure.

        When ``total_duration`` is provided AND any text overlay (intro card /
        title / tagline / watermark) is enabled, the overlay drawtext chain is
        applied IN THIS SAME PASS — eliminating a third re-encode (was previously:
        chunk-encode → mux copy → compress → overlay re-encode = 3 video encodes).

        Always writes the result to output_path (moves input_path if all encoders fail).
        """
        tmp = output_path + '.tmp_compress.mp4'
        # Build overlay chain ONCE so we can reuse it across encoder retries.
        overlay_chain = (
            self._build_overlay_filter_chain(total_duration)
            if total_duration is not None else None
        )
        fc_script_path = None
        if overlay_chain:
            try:
                fd, fc_script_path = tempfile.mkstemp(prefix='ats_finalize_', suffix='.txt')
                with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                    fh.write(overlay_chain)
            except Exception:
                fc_script_path = None  # fall back to inline -filter_complex below

        try:
            enc = self._get_encoding_settings()
            use_nvenc = '-c:v' in enc and enc[enc.index('-c:v') + 1] == 'h264_nvenc'

            def _build_cmd(use_gpu: bool) -> list:
                if use_gpu:
                    _final_preset = os.environ.get('ATS_NVENC_FINAL_PRESET', 'p5')
                    _final_bitrate = os.environ.get('ATS_NVENC_FINAL_BITRATE', '6M')
                    _final_maxrate = os.environ.get('ATS_NVENC_FINAL_MAXRATE', '12M')
                    video_enc = ['-c:v', 'h264_nvenc', '-preset', _final_preset,
                                 '-b:v', _final_bitrate, '-maxrate', _final_maxrate]
                else:
                    video_enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']
                cmd = ['ffmpeg', '-y', '-i', input_path]
                if overlay_chain:
                    if fc_script_path:
                        cmd += ['-filter_complex_script', fc_script_path]
                    else:
                        cmd += ['-filter_complex', overlay_chain]
                    cmd += ['-map', '[text_out]', '-map', '0:a?']
                cmd += [
                    *video_enc,
                    '-c:a', 'aac', '-b:a', '192k',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                ]
                if total_duration is not None:
                    cmd += ['-t', str(total_duration)]
                cmd += [tmp]
                return cmd

            label = 'Final compression + overlays' if overlay_chain else 'Final compression'
            logging.info(f'🗜️  {label} pass…')
            cmd = _build_cmd(use_nvenc)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

            if r.returncode != 0 and use_nvenc:
                logging.warning('⚠️  NVENC pass failed, retrying with CPU (libx264)…')
                if os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except Exception: pass
                r = subprocess.run(_build_cmd(False), capture_output=True, text=True,
                                   encoding='utf-8', errors='replace')

            if r.returncode == 0 and os.path.exists(tmp):
                orig_size = os.path.getsize(input_path)
                new_size = os.path.getsize(tmp)
                os.replace(tmp, output_path)
                delta_pct = 100 * (new_size - orig_size) / orig_size
                size_note = (f'{abs(delta_pct):.0f}% smaller'
                             if delta_pct < 0 else f'{delta_pct:.0f}% larger (quality pass)')
                logging.info(
                    f'✅ {label} complete: {orig_size:,} → {new_size:,} bytes ({size_note})'
                )
                return output_path
            logging.warning(f'⚠️  {label} pass failed, keeping original: {r.stderr[-1500:]}')

            # Preserve overlays/effects even if the combined compression pass fails.
            if overlay_chain:
                logging.warning('⚠️ Trying overlay-only fallback to preserve effects…')
                overlay_out = self._apply_text_overlays_to_video(input_path, output_path, total_duration)
                if overlay_out and os.path.exists(overlay_out):
                    logging.info('✅ Overlay-only fallback succeeded; effects preserved')
                    return overlay_out
        except Exception as e:
            logging.warning(f'⚠️  Compression pass error: {e}')
        finally:
            if os.path.exists(tmp):
                try: os.unlink(tmp)
                except Exception: pass
            if fc_script_path:
                try: os.unlink(fc_script_path)
                except Exception: pass

        # All encoders failed — move the uncompressed mux to the expected output path
        try:
            import shutil as _sh
            _sh.move(input_path, output_path)
            logging.info(f'↩️  Kept uncompressed mux at output path: {Path(output_path).name}')
        except Exception as mv_err:
            logging.error(f'❌ Could not move mux to output path: {mv_err}')
            return None
        return output_path

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
            logging.info("🗜️  Phase 4: final compression + overlays pass…")
            # Pass total_duration so overlays (intro card / title / tagline /
            # watermark) are applied in the SAME ffmpeg run as compression —
            # eliminates a third re-encode and avoids dropping back to libx264.
            result = self._compress_final_output(mux_path, str(self.output_path),
                                                  total_duration=total_duration)
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
                        # Overlays already applied inside _compress_final_output
                        # in the same ffmpeg run as compression — no third pass.
                        elapsed = time.time() - start_time
                        logging.info(f"🎉 Audio-first composition complete! {elapsed:.2f}s")
                        self._flush_onset_disk_cache()
                        return result
                    logging.warning("⚠️ Audio-first returned no output — falling back to legacy")
                except Exception as e:
                    logging.warning(f"⚠️ Audio-first failed ({e}) — falling back to legacy")

            # Legacy chunk-based pipeline (fallback / preview mode)
            if total_chunks > 2 and self.max_workers > 1:
                logging.info(f"Using parallel processing for {total_chunks} chunks")
                result = self.create_composition_with_parallel_processing()
            else:
                logging.info(f"Using sequential processing for {total_chunks} chunks")
                result = self._create_composition_sequential()

            if result and os.path.exists(str(result)):
                result = self._apply_text_overlays_inplace(str(result), total_duration)
            self._flush_onset_disk_cache()
            return result

        except Exception as e:
            logging.error(f"❌ Enhanced composition error: {e}")
            self._flush_onset_disk_cache()
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
            target_width, target_height = self._get_target_resolution()
            success = self._render_background_only_chunk(
                chunk_path,
                duration,
                target_width,
                target_height,
            )
            return str(chunk_path) if success and chunk_path.exists() else None
            
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
        culprit_note = None
        
        # Check all tracks (both regular and drum)
        all_tracks = self.regular_tracks + self.drum_tracks
        
        for track_idx, track in enumerate(all_tracks):
            track_name = track.get('instrument', {}).get('name', f'track_{track_idx}')
            for note_idx, note in enumerate(track.get('notes', [])):
                note_time = float(note.get('time', 0))
                note_duration = float(note.get('duration', 1))
                note_end = note_time + note_duration
                
                if note_end > max_end_time:
                    max_end_time = note_end
                    culprit_note = (track_name, note_idx, note_time, note_duration, note_end)
        
        # Add reasonable buffer (not hardcoded to specific song)
        buffer_time = min(3.0, max_end_time * 0.1)  # 10% buffer, max 3 seconds
        total_duration = max_end_time + buffer_time
        
        logging.info(f"📏 Duration calculation for ANY MIDI file:")
        logging.info(f"   Max note end time: {max_end_time:.2f}s")
        if culprit_note:
            track, idx, t, dur, end = culprit_note
            # Only flag truly abnormal single-note lengths. End-time position alone
            # is expected to be large in long songs and is not a corruption signal.
            if dur > 120.0:
                logging.warning(f"   ⚠️ SUSPICIOUSLY LONG NOTE: Track '{track}' note #{idx}")
                logging.warning(f"      Start: {t:.2f}s, Duration: {dur:.2f}s, Ends at: {end:.2f}s")
                logging.warning(f"      This may indicate MIDI timing corruption!")
            else:
                logging.info(f"   Latest note: Track '{track}' ends at {end:.2f}s")
        logging.info(f"   Dynamic buffer: {buffer_time:.2f}s") 
        logging.info(f"   Total duration: {total_duration:.2f}s")
        
        # Safety check: if duration is suspiciously long, cap it at a reasonable max
        # (most songs are < 10 minutes; flag anything > 20 minutes as potential corruption)
        MAX_REASONABLE_DURATION = 1200.0  # 20 minutes
        if total_duration > MAX_REASONABLE_DURATION:
            logging.error(f"❌ MIDI duration {total_duration:.2f}s exceeds max reasonable ({MAX_REASONABLE_DURATION}s)")
            if culprit_note:
                track, idx, t, dur, end = culprit_note
                logging.error(f"   Caused by: Track '{track}' note #{idx} ending at {end:.2f}s")
                logging.error(f"   This is likely a MIDI import/parsing error.")
                logging.error(f"   SOLUTION: Check the MIDI file — ensure it was uploaded correctly.")
            # For now, continue with the corrupted duration (user should fix their MIDI)
            # In future, could auto-cap or reject the composition
        
        return total_duration
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
                style_track_id=track_id,
            )

            if not triggered_video:
                return None  # or fallback already handled

            if triggered_video and os.path.exists(triggered_video):
                # Add chunk_time (chunk-relative) alongside absolute time so that
                # _apply_cell_style_filters can build correct FFmpeg enable expressions.
                chunk_notes_with_rel = []
                for note in chunk_notes:
                    note_copy = note.copy()
                    note_copy['chunk_time'] = float(note_copy.get('time', 0)) - chunk_start_time
                    chunk_notes_with_rel.append(note_copy)
                return {
                    'video_path': triggered_video,
                    'track_id': str(track_id),  # Use original track ID (string) for grid positioning
                    'track_name': track_name,
                    'notes': chunk_notes_with_rel,
                    'preserve_idle_alpha': triggered_video.lower().endswith('.mov'),
                    'type': 'instrument'
                }
            else:
                logging.warning(f"Failed to create triggered video for {track_name}")
                return None

        except Exception as e:
            logging.error(f"Error processing instrument track {track.get('instrument', 'unknown')}: {e}")
            return None
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
            unmapped_segments = []
            
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
                    segment['grid_w'] = position.get('w', 1)
                    segment['grid_h'] = position.get('h', 1)
                    positioned_segments.append(segment)
                    logging.info(
                        f"      ✅ Positioned at grid ({position.get('row')}, {position.get('column')}) "
                        f"span {position.get('w', 1)}x{position.get('h', 1)} using {strategy_used}"
                    )
                else:
                    unmapped_segments.append({
                        'track_id': track_id,
                        'track_name': track_name,
                        'segment_type': segment_type,
                        'video_path': video_path,
                    })
                    logging.error(
                        f"      ❌ No grid mapping found for {segment_type} "
                        f"track_id={track_id!r} track_name={track_name!r}"
                    )
            
            logging.info(f"📊 Grid positioning summary:")
            logging.info(f"   - Successfully positioned: {len(positioned_segments)} segments")

            if unmapped_segments:
                logging.error(
                    "❌ Grid layout aborted because one or more segments could not be mapped: %s",
                    [
                        f"{item['segment_type']}:{item['track_id']}:{Path(item['video_path']).name if item['video_path'] != 'missing' else 'MISSING'}"
                        for item in unmapped_segments
                    ],
                )
                return None
            
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
    _FONT_FILES = {
        'arial':     'arial.ttf',
        'verdana':   'verdana.ttf',
        'impact':    'impact.ttf',
        'courier':   'cour.ttf',
        'times':     'times.ttf',
        'georgia':   'georgia.ttf',
        'trebuchet': 'trebuc.ttf',
        'comic':     'comic.ttf',
    }

    _FONT_FILES_BOLD = {
        'default':   'segoeuib.ttf',
        'arial':     'arialbd.ttf',
        'verdana':   'verdanab.ttf',
        'impact':    'impact.ttf',
        'courier':   'courbd.ttf',
        'times':     'timesbd.ttf',
        'georgia':   'georgiab.ttf',
        'trebuchet': 'trebucbd.ttf',
        'comic':     'comicbd.ttf',
    }

    def _get_windows_font_path(self, font_key=None, bold=False):
        """Return the escaped font path for FFmpeg drawtext's fontfile= option.

        font_key: one of the values from FONT_OPTIONS (e.g. 'arial', 'impact').
        Returns an _escape_path_for_filter-escaped path string, or None to let
        FFmpeg fall back to its built-in font.
        """
        base = r'C:/Windows/Fonts/'
        candidates = []

        if bold:
            if font_key in (None, 'default'):
                candidates.append(base + self._FONT_FILES_BOLD['default'])
            else:
                bold_name = self._FONT_FILES_BOLD.get(font_key)
                if bold_name:
                    candidates.append(base + bold_name)

        if font_key and font_key != 'default':
            fname = self._FONT_FILES.get(font_key)
            if fname:
                candidates.append(base + fname)

        # Always append a safe fallback chain so some font is found
        candidates += [
            base + 'segoeuib.ttf',
            base + 'arial.ttf',
            base + 'arialbd.ttf',
            base + 'verdana.ttf',
            base + 'segoeui.ttf',
            base + 'calibri.ttf',
            base + 'tahoma.ttf',
        ]

        # Return only the requested font if it was explicitly requested;
        # only fall back to system fonts when no key is given.
        if font_key and font_key != 'default':
            # Try the requested font only — no silent substitution
            font_map = self._FONT_FILES_BOLD if bold else self._FONT_FILES
            p = base + (font_map.get(font_key) or '')
            if p and os.path.exists(p.replace('/', os.sep)):
                return self._escape_path_for_filter(p)
            return None  # font not found — caller uses no fontfile= (FFmpeg default)

        if font_key == 'default' and bold:
            p = base + self._FONT_FILES_BOLD['default']
            if os.path.exists(p.replace('/', os.sep)):
                return self._escape_path_for_filter(p)

        for p in candidates:
            if os.path.exists(p.replace('/', os.sep)):
                return self._escape_path_for_filter(p)
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
        """Convert #RRGGBB to 0xRRGGBB for FFmpeg. Returns black for None/empty."""
        if not hex_color:
            return '0x000000'
        h = hex_color.lstrip('#')
        if len(h) == 6:
            return f'0x{h.upper()}'
        return '0x000000'

    def _get_active_background_media(self):
        cs = getattr(self, 'composition_style', {}) or {}
        background_mode = cs.get('backgroundMode', 'color')
        if background_mode not in ('image', 'video'):
            return None

        background_media = getattr(self, 'background_media', {}) or {}
        media_path = str(background_media.get('path') or '').strip()
        if not media_path or not os.path.exists(media_path):
            if media_path:
                logging.warning(
                    f"Background media missing on disk, falling back to color: {media_path}"
                )
            return None

        media_kind = background_media.get('kind') or background_mode
        if media_kind not in ('image', 'video'):
            mime_type = str(background_media.get('mimeType') or '')
            media_kind = 'video' if mime_type.startswith('video/') else 'image'

        return {
            'path': media_path,
            'kind': media_kind,
            'mime_type': background_media.get('mimeType'),
            'original_name': background_media.get('originalName'),
        }

    def _append_canvas_background_inputs(
        self,
        cmd,
        filter_parts,
        input_idx,
        width,
        height,
        duration,
        output_label='grid_base',
    ):
        cs = getattr(self, 'composition_style', {}) or {}
        bg_hex = cs.get('backgroundColor', '#0a0a0f')
        bg_ffmpeg = self._hex_to_ffmpeg_color(bg_hex)
        background_media = self._get_active_background_media()

        if background_media:
            if background_media['kind'] == 'image':
                cmd.extend(['-loop', '1', '-t', str(duration), '-i', background_media['path']])
            else:
                cmd.extend(['-stream_loop', '-1', '-i', background_media['path']])

            canvas_video_input_idx = input_idx
            input_idx += 1
            filter_parts.append(
                f"[{canvas_video_input_idx}:v]fps=30,format=yuv420p,setsar=1,"
                f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},trim=duration={duration:.3f},setpts=PTS-STARTPTS[{output_label}]"
            )
            background_description = (
                f"{background_media['kind']} {Path(background_media['path']).name}"
            )
        else:
            cmd.extend(['-f', 'lavfi', '-i', f'color={bg_ffmpeg}:s={width}x{height}:r=30:d={duration}'])
            canvas_video_input_idx = input_idx
            input_idx += 1
            filter_parts.append(f"[{canvas_video_input_idx}:v]null[{output_label}]")
            background_description = f"color {bg_hex}"

        cmd.extend(['-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={duration}'])
        silent_audio_input_idx = input_idx
        input_idx += 1

        return input_idx, canvas_video_input_idx, silent_audio_input_idx, background_description

    def _render_background_only_chunk(self, output_path, duration, width=1920, height=1080):
        background_media = self._get_active_background_media()
        cs = getattr(self, 'composition_style', {}) or {}
        bg_hex = cs.get('backgroundColor', '#0a0a0f')
        bg_ffmpeg = self._hex_to_ffmpeg_color(bg_hex)

        if background_media:
            cmd = ['ffmpeg', '-y']
            if background_media['kind'] == 'image':
                cmd.extend(['-loop', '1', '-t', str(duration), '-i', background_media['path']])
            else:
                cmd.extend(['-stream_loop', '-1', '-i', background_media['path']])
            cmd.extend([
                '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-map', '0:v', '-map', '1:a',
                '-vf',
                (
                    f'fps=30,format=yuv420p,setsar=1,'
                    f'scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=increase,'
                    f'crop={width}:{height}'
                ),
                '-t', str(duration),
                '-c:v', 'h264_nvenc', '-preset', 'p4',
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path),
            ])
        else:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'color={bg_ffmpeg}:s={width}x{height}:r=30',
                '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-t', str(duration),
                '-c:v', 'h264_nvenc', '-preset', 'p4',
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path),
            ]

        try:
            gpu_subprocess_run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating background chunk: {e}")
            return False

    def _get_video_info(self, path):
        """Use ffprobe to get (width, height, duration). Returns None on failure."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height:format=duration',
                 '-of', 'csv=p=0', path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
                w = h = dur = None
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            w, h = int(parts[0]), int(parts[1])
                        except ValueError:
                            pass
                    elif len(parts) == 1:
                        try:
                            dur = float(parts[0])
                        except ValueError:
                            pass
                return w, h, dur
        except Exception as e:
            logging.warning(f"ffprobe failed for {path}: {e}")
        return None, None, None

    def _preprocess_extend_clip(self, clip_path, chunk_duration, bg_hex):
        """
        Ensure clip_path runs for the full chunk_duration by appending background-
        colored frames. Browser-recorded clips often have N/A duration metadata, so
        we ALWAYS attempt extension (not just when duration is known to be short).
        Returns (result_path, is_temp).
        """
        w, h, dur = self._get_video_info(clip_path)

        # Skip extension only when we have a confirmed duration that already fills the chunk
        if dur is not None and dur >= chunk_duration - 0.05:
            return clip_path, False

        # Browser-recorded clips often report N/A duration — always extend those too.
        # Use the clip's actual dimensions for the bg source so concat works without rescaling.
        if not w or not h:
            w, h = 1280, 720  # safe fallback for browser clips

        bg_ffmpeg = self._hex_to_ffmpeg_color(bg_hex)

        fd, ext_path = tempfile.mkstemp(prefix='ats_ext_', suffix='.mp4')
        os.close(fd)

        filter_complex = (
            # Normalise input clip to CFR 30fps, yuv420p, known SAR
            f'[0:v]fps=30,format=yuv420p,setsar=1[clip];'
            # Background padding source at clip's own dimensions (avoids concat size mismatch)
            f'[1:v]trim=duration={chunk_duration:.3f},setpts=PTS-STARTPTS[bg];'
            # Append bg after the clip; -t will trim the total to exactly chunk_duration
            f'[clip][bg]concat=n=2:v=1:a=0[out]'
        )

        cmd = [
            'ffmpeg', '-y',
            '-i', clip_path,
            '-f', 'lavfi', '-i', f'color=c={bg_ffmpeg}:s={w}x{h}:r=30:d={chunk_duration}',
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-t', str(chunk_duration),
            ext_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=60)
        if result.returncode == 0 and os.path.exists(ext_path) and os.path.getsize(ext_path) > 0:
            dur_str = f"{dur:.2f}s" if dur is not None else "N/A"
            logging.info(
                f"   ✅ Extended {Path(clip_path).name}: {dur_str} → {chunk_duration:.2f}s "
                f"(bg {bg_hex})"
            )
            return ext_path, True

        logging.warning(
            f"   ⚠️ Failed to extend {Path(clip_path).name}: "
            f"{result.stderr[-300:] if result.stderr else 'no stderr'}"
        )
        try:
            os.unlink(ext_path)
        except Exception:
            pass
        return clip_path, False

    def _resolve_clip_style(self, track_id):
        """Resolve frontend clip-style keys for a backend track id."""
        clip_styles = getattr(self, 'clip_styles', {}) or {}
        track_key = str(track_id)
        candidates = [track_key, f'track-{track_key}', f'drum-{track_key}']
        matched_key = next(
            (candidate for candidate in candidates if candidate in clip_styles),
            None,
        )
        resolved_style = {
            'roundedCorners': True,
            'cornerRadius': 12,
        }
        resolved_style.update(clip_styles.get(matched_key, {}) or {})
        return resolved_style, candidates, matched_key

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
        """Escape a filesystem path for use in an FFmpeg filter option value
        (e.g. drawtext's fontfile=).

        FFmpeg parses filter graphs in two passes:
          1. The graph parser tokenises filter chains and unescapes ONE level
             of backslashes inside option values.
          2. The filter (drawtext) parser then sees the unescaped value.

        For a Windows drive colon to survive both passes intact, the source
        string must contain a DOUBLE-backslash before the colon (``C\\:/path``).
        After pass 1 it becomes ``C\:`` (escaped colon), which pass 2 then
        reads as the literal value ``C:``.

        A single backslash (``C\:/path``) is silently consumed by pass 1,
        leaving ``C:`` which pass 2 then mis-parses as ``option:value`` and
        fails with ``No option name near '/Windows/Fonts/...''``.
        """
        p = path.replace('\\', '/')      # normalise to forward slashes
        p = p.replace(':', '\\\\:')      # escape drive-letter colon: C: → C\\:
        return p

    def _apply_cell_style_filters(self, filter_parts, input_label, output_label,
                                   cell_w, cell_h, track_id, cell_segment, temp_files,
                                   chunk_duration=None, beat_sync_stats=None):
        """
        Build per-cell styling filters (scale, pad/bgColor, color grade, beat flash, label, border).
        Chains filters from input_label → output_label.
        temp_files: list to append any created temp file paths for cleanup.
        chunk_duration: if provided, short clips are overlaid on a bg-color source so they
                        never go black after the clip ends (overlay eof_action=pass).
        """
        # Look up style: frontend keys use prefixes ('track-0', 'drum-drum_snare_drum')
        # while backend track_ids are bare ('0', 'drum_crash_cymbal'), so try all formats.
        style, candidates, matched_key = self._resolve_clip_style(track_id)

        logging.info(
            f"[style] cell={track_id!r}  candidates={candidates}  "
            f"matched={matched_key or 'none'}  "
            f"effects={[k for k,v in style.items() if v and k.endswith('Enabled')]}  "
            f"roundedCorners={style.get('roundedCorners', True)}  "
            f"beatFlashColor={style.get('beatFlashColor', 'N/A')}"
        )

        bg_color_enabled = bool(style.get('bgColorEnabled', False))
        bg_color = style.get('bgColor') or None   # None = use composition background
        border_width = int(style.get('borderWidth', 0))
        border_color = style.get('borderColor', '#7c3aed')
        color_grade = style.get('colorGrade', 'none')
        rounded_corners = bool(style.get('roundedCorners', True))
        corner_radius = int(style.get('cornerRadius', 12))
        label_enabled = bool(style.get('labelEnabled', False))
        label_text = style.get('labelText', '') or ''
        label_color = style.get('labelColor', '#ffffff')
        label_size = int(style.get('labelFontSize', 14))
        label_font = style.get('labelFont', 'default')
        beat_flash_enabled = bool(style.get('beatFlashEnabled', False))
        beat_flash_intensity = float(style.get('beatFlashIntensity', 0.4))
        fade_enabled = bool(style.get('fadeEnabled', False))
        fade_duration = float(style.get('fadeDuration', 0.15))
        transparent_bg = bool(style.get('transparentBg', False))
        preserve_idle_alpha = bool(cell_segment and cell_segment.get('preserve_idle_alpha'))

        font_path = self._get_windows_font_path(label_font)
        current = input_label

        # ── 1. Scale to cell dimensions, zoom-to-fill ───────────────────────
        # Always zoom-to-fill: scale up so the video covers the full cell in
        # both dimensions, then center-crop to the exact cell size.
        # This matches preprocessing (also zoom-to-fill) and works for any
        # cell aspect ratio without letterbox bars or composition-bg bleed.
        comp_bg_color = getattr(self, 'composition_style', {}).get('backgroundColor', '#0a0a0f')

        # Video always fills the full cell — borders/rounded corners apply to
        # full cell bounds regardless of the transparentBg flag.
        content_x, content_y, content_w, content_h = 0, 0, cell_w, cell_h

        next_label = f'v_pad_{output_label[1:-1]}'
        filter_parts.append(
            f"{current}scale={cell_w}:{cell_h}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={cell_w}:{cell_h},setsar=1{',format=rgba' if preserve_idle_alpha else ''}[{next_label}]"
        )
        logging.info(f"[style] cell={track_id!r} zoom-to-fill → {cell_w}x{cell_h}")
        current = f'[{next_label}]'

        notes = cell_segment.get('notes', []) if cell_segment else []
        active_note_windows = []
        for note in notes:
            t = float(note.get('chunk_time', note.get('time', 0)))
            dur = float(note.get('visual_duration', note.get('duration', 0.3)))
            if t >= 0 and dur > 0:
                active_note_windows.append((round(t, 3), round(t + dur, 3)))

        active_note_windows.sort(key=lambda window: window[0])
        merged_active_note_windows = []
        for start, end in active_note_windows:
            if (
                merged_active_note_windows
                and start <= merged_active_note_windows[-1][1] + 0.01
            ):
                merged_active_note_windows[-1] = (
                    merged_active_note_windows[-1][0],
                    max(merged_active_note_windows[-1][1], end),
                )
            else:
                merged_active_note_windows.append((start, end))

        if len(merged_active_note_windows) > 60:
            merged_active_note_windows = merged_active_note_windows[:60]

        active_note_enable_expr = (
            '+'.join(
                f'between(t,{start},{end})'
                for start, end in merged_active_note_windows
            )
            if merged_active_note_windows
            else None
        )

        # ── 2. Color grade (note-active windows only) ────────────────────────
        # Apply colorGrade only when notes are playing so gap/background frames
        # show the composition background without per-clip colour tinting.
        grade_filter = self._get_color_grade_filter(color_grade)
        if grade_filter and active_note_enable_expr:
            # Split multi-filter chains (e.g. 'cyberpunk' = 'eq=...,colorchannelmixer=...')
            _grade_sub_filters = [f.strip() for f in grade_filter.split(',')]
            for _j, _sub in enumerate(_grade_sub_filters):
                _gl = f'v_gr_{_j}_{output_label[1:-1]}'
                filter_parts.append(
                    f"{current}{_sub}:enable='{active_note_enable_expr}'[{_gl}]"
                )
                current = f'[{_gl}]'
        # else: no active note windows — skip colorGrade (would only affect background)

        # ── 3. Beat flash (colored overlay) ─────────────────────────────────
        # ── 3a. Beat-sync track-cell modulation (export parity) ───────────
        # Applies subtle pulse to individual grid cells when beatSync targets
        # include track-cells. The effect is gated to note-active windows and
        # uses conservative amplitudes for CPU/GPU stability.
        gstyle = getattr(self, 'composition_style', {}) or {}
        beat_sync_enabled = bool(gstyle.get('beatSyncEnabled'))
        beat_targets = gstyle.get('beatSyncTargets')
        if isinstance(beat_targets, (list, tuple, set)):
            beat_targets_set = {str(v).strip().lower() for v in beat_targets if str(v).strip()}
        elif isinstance(beat_targets, str) and beat_targets.strip():
            beat_targets_set = {s.strip().lower() for s in beat_targets.split(',') if s.strip()}
        else:
            beat_targets_set = set()

        if beat_sync_enabled and 'track-cells' in beat_targets_set:
            if isinstance(beat_sync_stats, dict):
                beat_sync_stats['eligible_cells'] = beat_sync_stats.get('eligible_cells', 0) + 1
            beat_sensitivity = (gstyle.get('beatSyncSensitivity') or 'medium').strip().lower()
            beat_mode = (gstyle.get('beatPulseMode') or 'scale').strip().lower()
            beat_interval_map = {
                'low': 0.90,
                'medium': 0.65,
                'high': 0.45,
            }
            beat_amp_map = {
                'low': 0.65,
                'medium': 1.0,
                'high': 1.35,
            }
            beat_mode_amp_map = {
                'scale': 0.95,
                'glow': 1.2,
                'shake-lite': 0.9,
            }

            windows = []
            velocity_samples = []
            for note in notes:
                t = float(note.get('chunk_time', note.get('time', 0)))
                dur = max(
                    0.02,
                    float(note.get('visual_duration', note.get('duration', 0.25))),
                )
                if t >= 0 and dur > 0:
                    windows.append((round(t, 3), round(t + dur, 3)))
                v = note.get('velocity')
                if v is None:
                    v = note.get('midi_velocity')
                if v is None:
                    v = note.get('note_velocity')
                try:
                    fv = float(v)
                    velocity_samples.append((fv / 127.0) if fv > 1.0 else fv)
                except Exception:
                    pass

            windows.sort(key=lambda w: w[0])
            merged = []
            for s, e in windows:
                if merged and s <= merged[-1][1] + 0.03:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            if len(merged) > 80:
                merged = merged[:80]

            if merged:
                if isinstance(beat_sync_stats, dict):
                    beat_sync_stats['modulated_cells'] = beat_sync_stats.get('modulated_cells', 0) + 1
                    beat_sync_stats['modulated_windows'] = beat_sync_stats.get('modulated_windows', 0) + len(merged)
                enable_expr = '+'.join(f'between(t,{s},{e})' for s, e in merged)
                avg_vel = 0.72
                if velocity_samples:
                    avg_vel = min(1.0, max(0.0, sum(velocity_samples) / len(velocity_samples)))
                velocity_factor = 0.85 + 0.4 * avg_vel

                beat_interval = beat_interval_map.get(beat_sensitivity, 0.65)
                pulse = (
                    beat_amp_map.get(beat_sensitivity, 1.0)
                    * beat_mode_amp_map.get(beat_mode, 0.95)
                    * velocity_factor
                )
                wave_expr = f"(0.5+0.5*sin(6.28318*t/{beat_interval:.3f}))"
                bright_amp = min(0.09, 0.038 * pulse)
                sat_amp = min(0.22, 0.11 * pulse)
                contrast_amp = min(0.16, 0.08 * pulse)

                next_label = f'v_bs_{output_label[1:-1]}'
                filter_parts.append(
                    f"{current}eq="
                    f"brightness='{bright_amp:.4f}*({wave_expr}-0.5)':"
                    f"saturation='1+{sat_amp:.4f}*({wave_expr}-0.5)':"
                    f"contrast='1+{contrast_amp:.4f}*({wave_expr}-0.5)':"
                    f"enable='{enable_expr}'[{next_label}]"
                )
                current = f'[{next_label}]'

        # ── 3b. Beat flash (colored overlay) ───────────────────────────────
        # Onset flash: colored burst at each note onset.
        # Uses drawbox with the user's chosen beatFlashColor at the specified
        # intensity (opacity).  Adjacent flashes (<60ms apart) are merged so
        # dense drum chunks aren't truncated.  Cap at 60 merged windows.
        if beat_flash_enabled:
            beat_flash_color = style.get('beatFlashColor', '#ffffff')
            FLASH_DUR = 0.10
            MERGE_GAP = 0.06
            MAX_WINDOWS = 60
            raw = []
            for note in notes:
                t = float(note.get('chunk_time', note.get('time', 0)))
                if t >= 0:  # valid chunk-relative onset
                    raw.append((round(t, 3), round(t + FLASH_DUR, 3)))
            raw.sort(key=lambda w: w[0])
            merged: list = []
            for s, e in raw:
                if merged and s - merged[-1][1] < MERGE_GAP:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            if len(merged) > MAX_WINDOWS:
                logging.debug(f"[style] beat-flash: {len(merged)} → {MAX_WINDOWS} windows (truncated)")
                merged = merged[:MAX_WINDOWS]
            if merged:
                enable_expr = '+'.join(f'between(t,{s},{e})' for s, e in merged)
                # Apply colored flash via semi-transparent drawbox overlay
                flash_ffmpeg_color = self._hex_to_ffmpeg_color(beat_flash_color)
                opacity = min(max(beat_flash_intensity, 0.0), 1.0)
                next_label = f'v_fl_{output_label[1:-1]}'
                filter_parts.append(
                    f"{current}drawbox=x=0:y=0:w=iw:h=ih"
                    f":color={flash_ffmpeg_color}@{opacity:.2f}:t=fill"
                    f":enable='{enable_expr}'[{next_label}]"
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
                                        .replace('\u0027', '\u2019')
            lc = self._hex_to_ffmpeg_color(label_color)
            next_label = f'v_lbl_{output_label[1:-1]}'
            fontfile_part = f":fontfile={font_path}" if font_path else ''
            filter_parts.append(
                f"{current}drawtext=text='{escaped_label}'"
                f":expansion=none"
                f":x=6:y=h-{label_size + 6}:fontsize={label_size}"
                f":fontcolor={lc}:alpha='1':box=1:boxcolor=0x000000@0.45:boxborderw=3"
                f"{fontfile_part}[{next_label}]"
            )
            current = f'[{next_label}]'

        # ── 5. Clip fade (note-triggered brightness boost) ───────────────────
        # "Fade-in/out on note trigger" effect:
        #   1. Single fade=t=in at chunk start (smooth opening)
        #   2. Subtle brightness BOOST during note-active windows so clips
        #      "pop" when notes play. Clips stay at normal brightness between
        #      notes — never dimmed/dark.
        #
        # This avoids the old approach of dimming between notes which made
        # clips with sparse notes appear too dark.
        if fade_enabled and fade_duration > 0:
            fd = max(fade_duration, 0.05)
            # Build note-active windows (onset → onset+duration)
            MAX_WINDOWS = 50
            windows = []
            for n in notes:
                t = float(n.get('chunk_time', n.get('time', 0)))
                dur = float(n.get('visual_duration', n.get('duration', 0.3)))
                windows.append((round(max(0.0, t), 3), round(t + dur, 3)))
            windows.sort(key=lambda w: w[0])
            # Merge overlapping windows
            merged_w: list = []
            for s, e in windows:
                if merged_w and s <= merged_w[-1][1] + 0.01:
                    merged_w[-1] = (merged_w[-1][0], max(merged_w[-1][1], e))
                else:
                    merged_w.append((s, e))
            if len(merged_w) > MAX_WINDOWS:
                step = len(merged_w) / MAX_WINDOWS
                merged_w = [merged_w[int(i * step)] for i in range(MAX_WINDOWS)]

            next_label = f'v_fade_{output_label[1:-1]}'
            if merged_w:
                active_expr = '+'.join(f'between(t,{s},{e})' for s, e in merged_w)
                # Brightness boost during notes (+0.12) — visible pop without
                # over-exposing.  Clip is at normal brightness between notes.
                filter_parts.append(
                    f"{current}fade=t=in:st=0:d={fd:.3f},"
                    f"eq=brightness=0.12:enable='{active_expr}'[{next_label}]"
                )
            else:
                # No notes — just fade in at chunk start (legacy behavior)
                filter_parts.append(
                    f"{current}fade=t=in:st=0:d={fd:.3f}[{next_label}]"
                )
            current = f'[{next_label}]'

        # ── 6. Border ────────────────────────────────────────────────────────
        if border_width > 0 and active_note_enable_expr:
            bc = self._hex_to_ffmpeg_color(border_color)
            next_label = f'v_brd_{output_label[1:-1]}'
            # When transparent bg is on, draw border around actual video content
            bx, by, bw, bh = (content_x, content_y, content_w, content_h) \
                              if transparent_bg else (0, 0, cell_w, cell_h)
            filter_parts.append(
                f"{current}drawbox=x={bx}:y={by}:w={bw}:h={bh}"
                f":color={bc}@1.0:t={border_width}:enable='{active_note_enable_expr}'[{next_label}]"
            )
            current = f'[{next_label}]'

        # ── 7. Rounded corners (staircase drawbox approximation of curve) ───────
        if rounded_corners and corner_radius > 0:
            # Video now always fills the full cell (zoom-to-fill), so rounded
            # corners always apply to the video itself. Fill corners with the
            # composition background so they blend seamlessly into the canvas.
            rx, ry, rw, rh = content_x, content_y, content_w, content_h
            corner_bg_hex = (
                bg_color
                if bg_color_enabled and bg_color and not transparent_bg
                else comp_bg_color
            )
            corner_fill = 'black' if preserve_idle_alpha else self._hex_to_ffmpeg_color(corner_bg_hex)
            corner_fill_alpha = '0.0' if preserve_idle_alpha else '1'
            corner_fill_replace = ':replace=1' if preserve_idle_alpha else ''
            r = min(corner_radius, rw // 4, rh // 4)
            if r > 0:
                next_label = f'v_rnd_{output_label[1:-1]}'
                # Staircase approximation: divide radius into N horizontal strips.
                # For each strip at vertical offset y0-y1, compute the x-width of
                # the corner cutout using the circle equation x = r - sqrt(r²-(r-y)²).
                # More steps → smoother curve (8 is a good balance of quality/speed).
                n_steps = min(r, 8)
                boxes = []
                for step in range(n_steps):
                    y0 = int(step * r / n_steps)
                    y1 = int((step + 1) * r / n_steps)
                    hs = y1 - y0
                    if hs <= 0:
                        continue
                    y_mid = (y0 + y1) / 2.0
                    inner_sq = max(0.0, float(r) ** 2 - (float(r) - y_mid) ** 2)
                    xw = min(rw // 2, int(float(r) - math.sqrt(inner_sq)) + 1)
                    if xw <= 0:
                        continue
                    # top-left
                    boxes.append(f"drawbox=x={rx}:y={ry+y0}:w={xw}:h={hs}:color={corner_fill}@{corner_fill_alpha}:t=fill{corner_fill_replace}")
                    # top-right
                    boxes.append(f"drawbox=x={rx+rw-xw}:y={ry+y0}:w={xw}:h={hs}:color={corner_fill}@{corner_fill_alpha}:t=fill{corner_fill_replace}")
                    # bottom-left  (mirror: rows ry+rh-y1 to ry+rh-y0)
                    boxes.append(f"drawbox=x={rx}:y={ry+rh-y1}:w={xw}:h={hs}:color={corner_fill}@{corner_fill_alpha}:t=fill{corner_fill_replace}")
                    # bottom-right
                    boxes.append(f"drawbox=x={rx+rw-xw}:y={ry+rh-y1}:w={xw}:h={hs}:color={corner_fill}@{corner_fill_alpha}:t=fill{corner_fill_replace}")
                if boxes:
                    filter_parts.append(
                        f"{current}" + ','.join(boxes) + f"[{next_label}]"
                    )
                    current = f'[{next_label}]'

        # Each clip is pre-extended to chunk_duration via _preprocess_extend_clip before
        # reaching the main filter_complex, so streams never terminate early and xstack
        # never fills any cell with black.
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

            FFmpeg's filter_complex parser uses single-quotes '...' for quoting.
            An ASCII apostrophe (U+0027) inside a single-quoted segment would
            prematurely close the quote, and backslash-escaping outside quotes
            (the '\\'' pattern) is silently dropped in this FFmpeg build.

            Solution: replace ASCII APOSTROPHE (U+0027) with Unicode RIGHT SINGLE
            QUOTATION MARK (U+2019). The two characters are visually identical in
            every common font, but U+2019 is a multi-byte UTF-8 sequence that the
            FFmpeg filter parser never treats as a quote delimiter.

            expansion=none (added to each filter) prevents % format strings.
            Newlines are stripped — drawtext is single-line only.
            """
            return t.replace('\r', '').replace('\n', ' ') \
                     .replace('\u0027', '\u2019')
        def add_drawtext(text, x_expr, y_expr, size, color_hex, alpha_expr='1', enabled='1', font_key=None):
            nonlocal current_label, filter_parts
            # Use inline text= with single-quote wrapping. ASCII apostrophes are
            # replaced with U+2019 (RIGHT SINGLE QUOTATION MARK) to avoid breaking
            # the filter_complex_script single-quote parser.
            # expansion=none prevents % format string expansion on user text.
            escaped = _esc(text)
            fc = self._hex_to_ffmpeg_color(color_hex)
            fp = self._get_windows_font_path(font_key) if font_key and font_key != 'default' else None
            fontfile_part = f":fontfile={fp}" if fp else ''
            nxt = f'v_gt_{len(filter_parts)}'
            filter_parts.append(
                f"[{current_label}]drawtext=text='{escaped}'"
                f":expansion=none"
                f":x={x_expr}:y={y_expr}:fontsize={size}"
                f":fontcolor={fc}:alpha='{alpha_expr}':enable='{enabled}'"
                f"{fontfile_part}[{nxt}]"
            )
            current_label = nxt

        # Title, tagline, and watermark are intentionally NOT applied per-chunk.
        # They are applied once to the full composed video in _apply_text_overlays_to_video
        # so time expressions (fade-in/out alpha) reference global video time, not per-chunk
        # time (which resets to 0 at every chunk boundary).

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

    def _build_overlay_filter_chain(self, total_duration: float) -> 'str | None':
        """Build the filter_complex chain for intro card / title / tagline /
        watermark text overlays.  Returns the chain string (ending in
        '[text_out]') or None if no overlays are enabled.

        Extracted from _apply_text_overlays_to_video so the same chain can be
        reused by the combined compress+overlay finalization pass.
        """
        cs = getattr(self, 'composition_style', {}) or {}
        preview_stage_scale = self._get_preview_stage_scale_factor()

        def scale_preview_px(value, minimum=1):
            try:
                scaled = float(value or 0) * preview_stage_scale
            except Exception:
                scaled = float(minimum)
            return max(float(minimum), scaled)

        def wrap_overlay_text(text, font_size, max_width_ratio):
            source = str(text or '').strip()
            if not source:
                return []

            existing_lines = [line.strip() for line in source.splitlines() if line.strip()]
            if len(existing_lines) > 1:
                return existing_lines[:4]

            words = source.split()
            if len(words) <= 1:
                return [source]

            target_width, _ = self._get_target_resolution()
            max_width_px = max(120.0, float(target_width) * float(max_width_ratio))
            estimated_char_width = max(1.0, float(font_size) * 0.74)
            max_chars = max(3, int(max_width_px / estimated_char_width))

            lines = []
            current_line = words[0]
            for word in words[1:]:
                candidate = f"{current_line} {word}"
                if len(candidate) <= max_chars:
                    current_line = candidate
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
            return lines[:4]

        title_text    = (cs.get('titleText') or '').strip()
        title_subtext = (cs.get('titleSubtitleText') or '').strip()
        intro_title_text = (cs.get('introCardText') or title_text or '').strip()
        intro_title_subtext = (cs.get('introCardSubtext') or title_subtext or '').strip()
        tagline_text  = (cs.get('taglineText') or '').strip()
        title_font    = cs.get('titleFont') or cs.get('introCardFont', 'default')
        title_color   = cs.get('titleColor') or cs.get('introCardTextColor', '#ffffff')
        title_subcolor = cs.get('titleSubtitleColor', '#d8d8e6')
        title_subsize = int(round(scale_preview_px(cs.get('titleSubtitleFontSize', 24), 10)))
        title_glow_enabled = bool(cs.get('titleGlowEnabled', False))
        title_glow_color = cs.get('titleGlowColor', '#ffffff')
        title_glow_size = scale_preview_px(cs.get('titleGlowSize', 8), 0)
        title_shadow_enabled = bool(cs.get('titleShadowEnabled', True))
        title_shadow_size = scale_preview_px(cs.get('titleShadowSize', 2), 0)
        title_shadow_color = cs.get('titleShadowColor', '#000000')
        title_bg      = cs.get('titleBackgroundColor') or cs.get('introCardBg', '#000000')
        title_bg_opacity = float(cs.get('titleBackgroundOpacity', 0.82) or 0.82)
        title_has_bg  = bool(cs.get('titleBackgroundEnabled'))
        title_bg_mode = cs.get('titleBackgroundMode', 'card')
        has_title     = bool(cs.get('titleEnabled') and title_text)
        has_tagline   = bool(cs.get('taglineEnabled') and tagline_text)
        has_watermark = bool(cs.get('watermarkEnabled') and cs.get('watermarkText', '').strip())
        has_intro     = bool(cs.get('titleEnabled') and cs.get('introCardEnabled') and intro_title_text)

        # Keep title visible after intro by default.
        # If users explicitly hide it after intro, respect that.
        repeat_title_after_intro = bool(
            cs.get('titleShowAfterIntro') or cs.get('titleRepeatAfterIntro')
        )
        hide_title_after_intro = bool(cs.get('titleHideAfterIntro'))
        if has_intro and hide_title_after_intro:
            has_title = False
        
        # Check if ANY effects are enabled (text overlays + visual effects)
        has_any_text = has_title or has_tagline or has_watermark or has_intro
        has_any_visual = (
            bool(cs.get('transitionEnabled')) or
            bool(cs.get('outroEffectEnabled')) or
            bool(cs.get('waveformEnabled')) or
            bool(cs.get('vignetteEnabled')) or
            bool(cs.get('glitchEnabled')) or
            bool(cs.get('beatSyncEnabled'))
        )
        if not (has_any_text or has_any_visual):
            return None

        filter_parts: list = []
        current_label = '0:v'

        transition_enabled = bool(cs.get('transitionEnabled'))
        transition_preset = (cs.get('transitionPreset') or 'none').strip().lower()
        transition_duration = max(0.2, float(cs.get('transitionDuration', 0.6) or 0.6))
        transition_strength = (cs.get('transitionStrength') or 'medium').strip().lower()
        transition_on = (cs.get('transitionOn') or 'start').strip().lower()
        if transition_on in {'sections', 'interval'}:
            transition_on = 'section'
        elif transition_on == 'auto':
            transition_on = 'phrase'
        elif transition_on == 'manual-marker':
            transition_on = 'start'
        transition_section_interval = max(2.0, float(cs.get('transitionSectionInterval', 8) or 8.0))
        transition_auto_cadence = cs.get('transitionAutoCadenceSeconds')
        transition_auto_reason = str(cs.get('transitionAutoReason') or '').strip()
        transition_auto_source = 'manual'
        transition_auto_note_density = None
        transition_auto_note_count = 0

        if transition_on == 'phrase':
            try:
                client_cadence = float(transition_auto_cadence)
            except Exception:
                client_cadence = None

            if client_cadence is not None and 2.0 <= client_cadence <= 20.0:
                transition_section_interval = client_cadence
                transition_auto_source = 'frontend'
                transition_auto_reason = transition_auto_reason or 'Provided by UI auto timing'
            else:
                tracks = []
                try:
                    tracks = list((getattr(self, 'midi_data', {}) or {}).get('tracks', []) or [])
                except Exception:
                    tracks = []
                total_notes = 0
                for trk in tracks:
                    notes = trk.get('notes') if isinstance(trk, dict) else None
                    if isinstance(notes, list):
                        total_notes += len(notes)
                safe_duration = max(1.0, float(total_duration or 0.0))
                note_density = total_notes / safe_duration
                transition_auto_note_density = note_density
                transition_auto_note_count = total_notes
                transition_auto_source = 'backend'
                if note_density >= 12:
                    transition_section_interval = 2.5
                    transition_auto_reason = 'Very dense arrangement detected'
                elif note_density >= 8:
                    transition_section_interval = 3.5
                    transition_auto_reason = 'Dense arrangement detected'
                elif note_density >= 4:
                    transition_section_interval = 5.0
                    transition_auto_reason = 'Balanced arrangement detected'
                elif note_density >= 2:
                    transition_section_interval = 6.5
                    transition_auto_reason = 'Light arrangement detected'
                else:
                    transition_section_interval = 8.0
                    transition_auto_reason = 'Sparse arrangement detected'

        transition_repeat = transition_on in {'section', 'phrase'}
        # Repeating transitions (section/phrase) should affect composited text layers too,
        # so they are applied after text overlays.
        transition_apply_post_text = transition_repeat or bool(cs.get('transitionApplyAfterText'))
        transition_strength_map = {
            'low': 0.55,
            'medium': 0.8,
            'high': 1.0,
        }
        transition_strength_factor = transition_strength_map.get(transition_strength, 0.8)
        transition_fill_color = self._hex_to_ffmpeg_color(
            cs.get('backgroundColor', '#0a0a0f')
        )

        beat_sync_enabled = bool(cs.get('beatSyncEnabled'))
        beat_sync_sensitivity = (cs.get('beatSyncSensitivity') or 'medium').strip().lower()
        beat_sync_mode = (cs.get('beatPulseMode') or 'scale').strip().lower()
        beat_targets_raw = cs.get('beatSyncTargets')
        if isinstance(beat_targets_raw, (list, tuple, set)):
            beat_sync_targets = {
                str(v).strip().lower() for v in beat_targets_raw if str(v).strip()
            }
        elif isinstance(beat_targets_raw, str) and beat_targets_raw.strip():
            beat_sync_targets = {
                s.strip().lower() for s in beat_targets_raw.split(',') if s.strip()
            }
        else:
            beat_sync_targets = set()

        beat_interval_map = {
            'low': 0.90,
            'medium': 0.65,
            'high': 0.45,
        }
        beat_base_amp_map = {
            'low': 0.07,
            'medium': 0.11,
            'high': 0.15,
        }
        beat_mode_amp_factor = {
            'scale': 0.85,
            'glow': 1.0,
            'shake-lite': 0.75,
        }
        beat_shake_px_map = {
            'low': 1.1,
            'medium': 1.8,
            'high': 2.5,
        }
        beat_sync_interval = beat_interval_map.get(beat_sync_sensitivity, 0.65)
        beat_sync_amp = (
            beat_base_amp_map.get(beat_sync_sensitivity, 0.11)
            * beat_mode_amp_factor.get(beat_sync_mode, 0.85)
        )
        beat_sync_wave_expr = f"(0.5+0.5*sin(6.28318*t/{beat_sync_interval:.3f}))"
        beat_sync_mod_expr = (
            f"(1-{beat_sync_amp:.3f}+{beat_sync_amp:.3f}*{beat_sync_wave_expr})"
        )
        beat_shake_px = beat_shake_px_map.get(beat_sync_sensitivity, 1.8)

        if beat_sync_enabled:
            logging.info(
                "🥁 Beat sync export: "
                f"mode={beat_sync_mode}, "
                f"sensitivity={beat_sync_sensitivity}, "
                f"targets={','.join(sorted(beat_sync_targets)) or 'none'}, "
                f"interval={beat_sync_interval:.2f}s"
            )

        outro_enabled = bool(cs.get('outroEffectEnabled'))
        outro_preset = (cs.get('outroEffectPreset') or 'fade-black').strip().lower()
        outro_duration = max(0.4, float(cs.get('outroEffectDuration', 1.2) or 1.2))
        outro_strength = (cs.get('outroEffectStrength') or 'medium').strip().lower()
        outro_strength_factor = transition_strength_map.get(outro_strength, 0.8)

        intro_duration_for_timing = 0.0
        if has_intro:
            intro_duration_for_timing = max(1.0, float(cs.get('introCardDuration', 3) or 3))

        # By default, opening transition starts after intro card so intro remains readable.
        # Set compositionStyle.transitionApplyToIntroCard=true to opt into transitioning intro too.
        transition_apply_to_intro = bool(cs.get('transitionApplyToIntroCard'))
        transition_anchor = 0.0
        if (
            transition_enabled
            and transition_on == 'start'
            and has_intro
            and not transition_apply_to_intro
        ):
            transition_anchor = intro_duration_for_timing

        if transition_repeat:
            transition_time_expr = (
                f"mod(max(t-{transition_anchor:.3f},0),{transition_section_interval:.3f})"
            )
            transition_progress_expr = (
                f"min(max(({transition_time_expr})/{transition_duration:.3f},0),1)"
            )
            transition_active_expr = (
                f"gte(t,{transition_anchor:.3f})*lt(({transition_time_expr}),{transition_duration:.3f})"
            )
        else:
            transition_time_expr = f"max(t-{transition_anchor:.3f},0)"
            transition_progress_expr = (
                f"min(max((t-{transition_anchor:.3f})/{transition_duration:.3f},0),1)"
            )
            transition_active_expr = (
                f"between(t,{transition_anchor:.3f},{transition_anchor + transition_duration:.3f})"
            )

        logging.info(
            "[style] overlay-plan: "
            f"intro={has_intro}, intro_dur={intro_duration_for_timing:.2f}, "
            f"title_enabled_effective={has_title}, title_repeat_after_intro={repeat_title_after_intro}, "
            f"transition_enabled={transition_enabled and transition_preset != 'none'}, "
            f"transition_on={transition_on}, transition_anchor={transition_anchor:.2f}, "
            f"transition_post_text={transition_apply_post_text}, "
            f"transition_apply_to_intro={transition_apply_to_intro}, "
            f"outro_enabled={outro_enabled}"
        )

        def _esc(t: str) -> str:
            return (t or '').replace('\r', '').replace('\n', ' ') \
                            .replace('\\', '\\\\') \
                            .replace(':', '\\:') \
                            .replace('%', '%%') \
                            .replace('\u0027', '\u2019')

        def add_drawtext(text, x_expr, y_expr, size, color_hex,
                         alpha_expr='1', enabled='1', font_key=None,
                         bold=False,
                         shadow_size=0, shadow_color_hex=None,
                         border_size=0, border_color_hex=None):
            nonlocal current_label
            escaped = _esc(text)
            fc = self._hex_to_ffmpeg_color(color_hex)
            fp = self._get_windows_font_path(font_key, bold=bold)
            fontfile_part = f':fontfile={fp}' if fp else ''
            shadow_part = ''
            shadow_size = float(shadow_size or 0)
            if shadow_size > 0:
                shadow_px = max(1, int(round(shadow_size)))
                sc = self._hex_to_ffmpeg_color(shadow_color_hex or '#000000')
                shadow_part = f":shadowx={shadow_px}:shadowy={shadow_px}:shadowcolor={sc}"
            border_part = ''
            border_size = float(border_size or 0)
            if border_size > 0:
                border_px = max(1, int(round(border_size)))
                bc = self._hex_to_ffmpeg_color(border_color_hex or color_hex)
                border_part = f":borderw={border_px}:bordercolor={bc}"
            nxt = f'v_to_{len(filter_parts)}'
            filter_parts.append(
                f"[{current_label}]drawtext=text='{escaped}'"
                f":expansion=none"
                f":x='{x_expr}':y='{y_expr}':fontsize={size}"
                f":fontcolor={fc}:alpha='{alpha_expr}':enable='{enabled}'"
                f"{shadow_part}{border_part}{fontfile_part}[{nxt}]"
            )
            current_label = nxt

        # ── Opening transition (preview/export parity) ───────────────────────
        if transition_enabled and transition_preset != 'none' and not transition_apply_post_text:
            applied_transition = transition_preset
            if transition_preset == 'crossfade':
                if transition_repeat:
                    # ffmpeg fade is one-shot; emulate repeating crossfade with subtle brightness settle.
                    nxt = f'v_to_{len(filter_parts)}'
                    filter_parts.append(
                        f"[{current_label}]eq=brightness='-{0.22 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':enable='{transition_active_expr}'[{nxt}]"
                    )
                    current_label = nxt
                else:
                    nxt = f'v_to_{len(filter_parts)}'
                    filter_parts.append(
                        f"[{current_label}]fade=t=in:st=0:d={transition_duration:.3f}[{nxt}]"
                    )
                    current_label = nxt
            elif transition_preset == 'dip-black':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=brightness='-{0.72 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
            elif transition_preset == 'dip-white':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=brightness='{0.72 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
            elif transition_preset == 'glitch-cut':
                glitch_d = min(0.35, transition_duration)
                glitch_active_expr = (
                    f"lt(({transition_time_expr}),{glitch_d:.3f})"
                )
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]noise=alls={int(14 + 24 * transition_strength_factor)}:allf=t:enable='{glitch_active_expr}'[{nxt}]"
                )
                current_label = nxt
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=saturation='{1.0 + 0.5 * transition_strength_factor:.3f}':contrast='{1.0 + 0.35 * transition_strength_factor:.3f}':enable='{glitch_active_expr}'[{nxt}]"
                )
                current_label = nxt
            elif transition_preset in {'slide-left', 'push-left'}:
                applied_transition = 'push-left'
                # Start slightly right-shifted, then settle to centered frame.
                slide_span = f"iw*{0.08 * transition_strength_factor:.3f}"
                crop_w = f"iw-{slide_span}"
                padded = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]pad=w=iw+{slide_span}:h=ih:x={slide_span}:y=0:color={transition_fill_color}[{padded}]"
                )
                current_label = padded
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]crop=w='{crop_w}':h=ih:x='{slide_span}*{transition_progress_expr}':y=0[{nxt}]"
                )
                current_label = nxt
            elif transition_preset in {'slide-right', 'push-right'}:
                applied_transition = 'push-right'
                # Start slightly left-shifted, then settle to centered frame.
                slide_span = f"iw*{0.08 * transition_strength_factor:.3f}"
                crop_w = f"iw-{slide_span}"
                padded = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]pad=w=iw+{slide_span}:h=ih:x=0:y=0:color={transition_fill_color}[{padded}]"
                )
                current_label = padded
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]crop=w='{crop_w}':h=ih:x='{slide_span}*(1-{transition_progress_expr})':y=0[{nxt}]"
                )
                current_label = nxt
            elif transition_preset in {'zoom-in', 'zoom'}:
                applied_transition = 'zoom'
                # Dimension-safe "zoom-like" opening effect.
                # Previous implementation used per-frame scale/crop with dynamic dimensions,
                # which can cause encoder failures in the final overlay pass.
                # Keep geometry stable and emulate zoom energy via contrast/brightness settle.
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=contrast='{1.0 + 0.22 * transition_strength_factor:.3f}-(0.22*{transition_strength_factor:.3f}*{transition_progress_expr})':"
                    f"brightness='{-0.06 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':"
                    f"enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
            else:
                # Unknown preset fallback.
                applied_transition = 'crossfade(fallback)'
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]fade=t=in:st=0:d={transition_duration:.3f}[{nxt}]"
                )
                current_label = nxt

            transition_log_suffix = ''
            if transition_on == 'phrase':
                transition_log_suffix = (
                    f", auto_source={transition_auto_source}, "
                    f"auto_reason={transition_auto_reason or 'n/a'}"
                )
                if transition_auto_note_density is not None:
                    transition_log_suffix += (
                        f", note_density={transition_auto_note_density:.3f}, "
                        f"total_notes={transition_auto_note_count}"
                    )

            logging.info(
                f"🎞️ Opening transition overlay: preset={applied_transition}, "
                f"duration={transition_duration:.2f}s, strength={transition_strength}, "
                f"timing={transition_on}, interval={transition_section_interval:.2f}s, "
                f"anchor={transition_anchor:.2f}s"
                f"{transition_log_suffix}"
            )

        if beat_sync_enabled and 'overlays' in beat_sync_targets:
            overlay_boost = min(0.065, 0.026 + beat_sync_amp * 0.23)
            nxt = f'v_to_{len(filter_parts)}'
            filter_parts.append(
                f"[{current_label}]eq=brightness='{overlay_boost:.4f}*({beat_sync_wave_expr}-0.5)'[{nxt}]"
            )
            current_label = nxt

        # ── Intro card (solid background fill + title + subtitle at video start) ──
        if has_intro:
            intro_dur  = max(1.0, float(cs.get('introCardDuration', 3)))
            bg_col     = self._hex_to_ffmpeg_color(title_bg)
            txt_col    = title_color
            animated   = bool(cs.get('titleAnimated', cs.get('introCardAnimated', True)))
            ic_font    = title_font
            d          = f'{intro_dur:.3f}'
            fade_out   = f'{intro_dur - 0.3:.3f}'
            alpha_expr = (
                f'min(t/0.5,1)*gt(t,0)*lt(t,{fade_out})'
                f'+max(0,1-(t-{fade_out})/0.3)*lt(t,{d})'
                if animated else '1'
            )
            nxt = f'v_to_{len(filter_parts)}'
            filter_parts.append(
                f"[{current_label}]drawbox=x=0:y=0:w=iw:h=ih"
                f":color={bg_col}@{min(title_bg_opacity + 0.13, 0.95):.3f}:t=fill:enable='lt(t,{d})'[{nxt}]"
            )
            current_label = nxt
            ic_title = intro_title_text or 'AutoTune Composition'
            ic_sub   = intro_title_subtext
            if ic_title:
                intro_size = int(round(scale_preview_px(72, 18)))
                intro_lines = wrap_overlay_text(ic_title, intro_size, 0.72)
                intro_step = max(14, int(round(intro_size * 0.96)))
                intro_shift = ((len(intro_lines) - 1) * intro_step) / 2
                for index, line in enumerate(intro_lines):
                    line_y = '(h-text_h)/2'
                    if intro_shift:
                        line_y = f"({line_y})-{intro_shift:.1f}+{index * intro_step}"
                    add_drawtext(line, '(w-text_w)/2', line_y,
                                 intro_size, txt_col,
                                 alpha_expr=alpha_expr, enabled=f'lt(t,{d})',
                                 font_key=ic_font, bold=True)
            if ic_sub:
                add_drawtext(ic_sub, '(w-text_w)/2', 'h*0.62',
                             int(round(scale_preview_px(36, 12))), txt_col,
                             alpha_expr=alpha_expr, enabled=f'lt(t,{d})',
                             font_key=ic_font, bold=True)
            logging.info(f'🎬 Intro card overlay: {intro_dur}s, bg={bg_col}, '
                         f'title={bool(ic_title)}, subtitle={bool(ic_sub)}')

        # ── Title (fade-in at start, fade-out) ────────────────────────────────
        if has_title:
            pos  = cs.get('titlePosition', 'top-center')
            size = int(round(scale_preview_px(cs.get('titleFontSize', 56), 12)))
            color = title_color
            animated = bool(cs.get('titleAnimated', True))
            # Prevent a second "fly-in" title by default when intro is enabled.
            if has_intro and not repeat_title_after_intro:
                animated = False
            title_anim_preset = (cs.get('titleAnimationPreset') or 'fade').strip().lower()
            title_anim_delay = max(0.0, float(cs.get('titleAnimDelay', 0) or 0.0))
            title_anim_duration = max(0.3, float(cs.get('titleAnimDuration', 0.7) or 0.7))
            title_anim_intensity = (cs.get('titleAnimIntensity') or 'medium').strip().lower()
            title_anim_direction = (cs.get('titleAnimDirection') or 'left').strip().lower()

            intensity_factor_map = {
                'low': 0.7,
                'medium': 1.0,
                'high': 1.35,
            }
            intensity_factor = intensity_factor_map.get(title_anim_intensity, 1.0)
            direction_sign = 1 if title_anim_direction == 'right' else -1
            motion_y = max(8.0 * preview_stage_scale, 38.0 * intensity_factor * preview_stage_scale)
            motion_x = max(10.0 * preview_stage_scale, 46.0 * intensity_factor * preview_stage_scale) * direction_sign
            bounce_overshoot = max(4.0, motion_y * 0.22)

            title_duration = float(cs.get('titleDuration', 0) or 0)
            start_at = (max(0.0, float(cs.get('introCardDuration', 3))) if has_intro else 0.0) + title_anim_delay
            enabled = '1'
            if title_duration > 0:
                fin = 0.6 if animated else 0.0
                fout = 0.8 if animated else 0.0
                T_end = min(start_at + title_duration + fout, total_duration)
                if animated:
                    fade_in_end = start_at + fin
                    hold_end = max(fade_in_end, min(start_at + title_duration, T_end - fout))
                    ai = f"if(lt(t,{start_at:.3f}),0,if(lt(t,{fade_in_end:.3f}),(t-{start_at:.3f})/{max(fin,0.001):.3f},1))"
                    ao = f"if(lt(t,{hold_end:.3f}),1,max(0,1-(t-{hold_end:.3f})/{max(fout,0.001):.3f}))"
                    alpha = f"({ai})*({ao})"
                else:
                    alpha = '1'
                enabled = f"between(t,{start_at:.3f},{T_end:.3f})"
            elif animated:
                fin  = 2.0
                fout = 2.0
                hold = min(max(total_duration * 0.15, 3.0), 8.0)
                T_end = min(start_at + fin + hold + fout, total_duration)
                if T_end > total_duration:
                    hold = max(0.0, total_duration - fin - fout)
                    T_end = total_duration
                F  = f'{(start_at + fin):.3f}'
                Te = f'{T_end:.3f}'
                Fo = f'{fout:.3f}'
                ai = f"(t+{F}-abs(t-{F}))/(2*{F})"
                pu = f"(({Te}-t)+abs({Te}-t))/2"
                ao = f"(({pu})+{Fo}-abs(({pu})-{Fo}))/(2*{Fo})"
                alpha   = f"({ai})*({ao})"
                enabled = f"between(t,{start_at:.3f},{Te})"
            else:
                alpha, enabled = '1', '1'
            if title_has_bg and title_bg_mode == 'fullscreen':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]drawbox=x=0:y=0:w=iw:h=ih"
                    f":color={self._hex_to_ffmpeg_color(title_bg)}@{title_bg_opacity:.3f}:t=fill:enable='{enabled}'[{nxt}]"
                )
                current_label = nxt
                y = '(h-text_h)/2'
                tag_y = 'h*0.58'
            elif title_has_bg:
                box_color = self._hex_to_ffmpeg_color(title_bg)
                if pos == 'bottom-center':
                    box_x, box_y, box_w, box_h = 'w*0.16', 'h*0.72', 'w*0.68', 'h*0.18'
                    title_y = 'h*0.765'
                    tag_y = 'h*0.835'
                elif pos == 'center':
                    box_x, box_y, box_w, box_h = 'w*0.16', 'h*0.39', 'w*0.68', 'h*0.22'
                    title_y = 'h*0.455'
                    tag_y = 'h*0.53'
                else:
                    box_x, box_y, box_w, box_h = 'w*0.16', 'h*0.04', 'w*0.68', 'h*0.18'
                    title_y = 'h*0.095'
                    tag_y = 'h*0.16'
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}"
                    f":color={box_color}@{title_bg_opacity:.3f}:t=fill:enable='{enabled}'[{nxt}]"
                )
                current_label = nxt
                y = title_y
            else:
                y = (str(max(20, size // 2)) if pos == 'top-center'
                     else f'h-{size * 2}'   if pos == 'bottom-center'
                     else '(h-text_h)/2')
            title_x_expr = '(w-text_w)/2'
            title_y_expr = y
            subtitle_x_expr = '(w-text_w)/2'

            if animated:
                entry_end = start_at + title_anim_duration
                # p goes from ~1 to 0 during entry; drawtext x/y are quoted so comma-based expressions are valid.
                p = (
                    f"if(lt(t,{start_at:.3f}),1,"
                    f"if(lt(t,{entry_end:.3f}),({entry_end:.3f}-t)/{max(title_anim_duration, 0.001):.3f},0))"
                )
                if title_anim_preset == 'scroll-up':
                    title_y_expr = f"{y}+({motion_y:.2f}*({p}))"
                elif title_anim_preset == 'scroll-left':
                    title_x_expr = f"(w-text_w)/2+({motion_x:.2f}*({p}))"
                    subtitle_x_expr = title_x_expr
                elif title_anim_preset == 'bounce':
                    # Starts lower and briefly overshoots upward before settling.
                    title_y_expr = (
                        f"{y}+({motion_y:.2f}*({p}))"
                        f"-({bounce_overshoot:.2f}*sin((1-({p}))*3.14159))"
                    )
                elif title_anim_preset == 'spin-soft':
                    # drawtext cannot rotate directly; emulate with directional drift.
                    title_x_expr = f"(w-text_w)/2+({motion_x:.2f}*({p}))"
                    subtitle_x_expr = title_x_expr
                    title_y_expr = f"{y}+({motion_y * 0.28:.2f}*({p}))"
                elif title_anim_preset == 'blur-focus':
                    # drawtext cannot blur directly; keep fade alpha behavior and tiny settle motion.
                    title_y_expr = f"{y}+({motion_y * 0.18:.2f}*({p}))"

                if title_anim_preset == 'typewriter':
                    # Approximate typewriter with stepped alpha reveal over entry duration.
                    title_chars = max(1, len((title_text or '').strip()))
                    title_step = max(0.02, title_anim_duration / float(title_chars))
                    title_progress = (
                        f"if(lt(t,{start_at:.3f}),0,"
                        f"if(lt(t,{entry_end:.3f}),"
                        f"min(1,max(0,floor((t-{start_at:.3f})/{title_step:.4f}+1)/{title_chars})),1))"
                    )
                    alpha = f"({alpha})*({title_progress})"

            title_render_alpha = alpha
            if beat_sync_enabled and 'title' in beat_sync_targets:
                title_render_alpha = f"({alpha})*({beat_sync_mod_expr})"
                if beat_sync_mode == 'shake-lite':
                    title_x_expr = (
                        f"({title_x_expr})+({beat_shake_px:.2f}*sin(6.28318*t/{beat_sync_interval:.3f}))"
                    )
                    subtitle_x_expr = title_x_expr

            title_max_width_ratio = 1.0 if title_bg_mode == 'fullscreen' else (0.70 if title_has_bg else 0.85)
            title_lines = wrap_overlay_text(title_text, size, title_max_width_ratio)
            title_line_step = max(12, int(round(size * 0.94)))
            title_anchor_shift = ((len(title_lines) - 1) * title_line_step) / 2
            for index, line in enumerate(title_lines):
                line_y_expr = title_y_expr
                if title_anchor_shift:
                    line_y_expr = f"({title_y_expr})-{title_anchor_shift:.1f}+{index * title_line_step}"
                add_drawtext(line, title_x_expr, line_y_expr, size, color,
                             alpha_expr=title_render_alpha, enabled=enabled,
                             font_key=title_font, bold=True,
                             shadow_size=title_shadow_size if title_shadow_enabled else 0,
                             shadow_color_hex=title_shadow_color,
                             border_size=(title_glow_size * 0.14) if title_glow_enabled else 0,
                             border_color_hex=title_glow_color)
            if title_subtext:
                title_block_extra = max(0, (len(title_lines) - 1) * title_line_step)
                subtitle_base_y = tag_y if title_has_bg else (
                    'h*0.58' if pos == 'center' else f'{y}+{max(title_subsize + int(round(scale_preview_px(10, 2))), size // 2)}'
                )
                subtitle_y = subtitle_base_y if title_block_extra == 0 else f'({subtitle_base_y})+{title_block_extra}'
                subtitle_y_expr = subtitle_y
                if animated and title_anim_preset in {'scroll-up', 'bounce', 'spin-soft', 'blur-focus'}:
                    entry_end = start_at + title_anim_duration
                    p = (
                        f"if(lt(t,{start_at:.3f}),1,"
                        f"if(lt(t,{entry_end:.3f}),({entry_end:.3f}-t)/{max(title_anim_duration, 0.001):.3f},0))"
                    )
                    if title_anim_preset == 'scroll-up':
                        subtitle_y_expr = f"{subtitle_y}+({motion_y:.2f}*({p}))"
                    elif title_anim_preset == 'bounce':
                        subtitle_y_expr = (
                            f"{subtitle_y}+({motion_y:.2f}*({p}))"
                            f"-({bounce_overshoot:.2f}*sin((1-({p}))*3.14159))"
                        )
                    elif title_anim_preset == 'spin-soft':
                        subtitle_y_expr = f"{subtitle_y}+({motion_y * 0.28:.2f}*({p}))"
                    elif title_anim_preset == 'blur-focus':
                        subtitle_y_expr = f"{subtitle_y}+({motion_y * 0.18:.2f}*({p}))"
                subtitle_alpha = title_render_alpha
                if animated and title_anim_preset == 'typewriter':
                    sub_chars = max(1, len((title_subtext or '').strip()))
                    sub_step = max(0.02, title_anim_duration / float(sub_chars))
                    sub_progress = (
                        f"if(lt(t,{start_at:.3f}),0,"
                        f"if(lt(t,{entry_end:.3f}),"
                        f"min(1,max(0,floor((t-{start_at:.3f})/{sub_step:.4f}+1)/{sub_chars})),1))"
                    )
                    subtitle_alpha = f"({title_render_alpha})*({sub_progress})"
                add_drawtext(title_subtext, subtitle_x_expr, subtitle_y_expr,
                             int(max(16, title_subsize)),
                             title_subcolor,
                             alpha_expr=subtitle_alpha, enabled=enabled,
                             font_key=title_font, bold=True,
                             shadow_size=(title_shadow_size * 0.9) if title_shadow_enabled else 0,
                             shadow_color_hex=title_shadow_color,
                             border_size=(title_glow_size * 0.1) if title_glow_enabled else 0,
                             border_color_hex=title_glow_color)

        # ── Tagline ───────────────────────────────────────────────────────────
        if has_tagline:
            size  = int(round(scale_preview_px(cs.get('taglineFontSize', 24), 10)))
            color = cs.get('taglineColor', '#cccccc')
            tagline_position = cs.get('taglinePosition', 'bottom-center')
            tagline_alignment = cs.get('taglineAlignment', 'center')
            tagline_shape = cs.get('taglineShape', 'rounded')
            tagline_shadow_enabled = bool(cs.get('taglineShadowEnabled', True))
            tagline_shadow_size = scale_preview_px(cs.get('taglineShadowSize', 2), 0)
            tagline_shadow_color = cs.get('taglineShadowColor', '#000000')
            tagline_width_pct = min(100.0, max(20.0, float(cs.get('taglineWidth', 72) or 72)))
            tagline_vertical_offset = min(
                160.0 * preview_stage_scale,
                max(
                    -160.0 * preview_stage_scale,
                    float(cs.get('taglineVerticalOffset', 0) or 0.0) * preview_stage_scale,
                ),
            )
            tagline_fade_in = max(0.0, float(cs.get('taglineFadeInDuration', 0.5) or 0.0))
            tagline_fade_out = max(0.0, float(cs.get('taglineFadeOutDuration', 0.5) or 0.0))
            tagline_bg_enabled = bool(cs.get('taglineBackgroundEnabled'))
            tagline_bg = self._hex_to_ffmpeg_color(cs.get('taglineBackgroundColor', '#0c1220'))
            tagline_bg_opacity = float(cs.get('taglineBackgroundOpacity', 0.72) or 0.72)
            tagline_accent = self._hex_to_ffmpeg_color(cs.get('taglineAccentColor', '#ff4db8'))
            tagline_start = max(0.0, float(cs.get('introCardDuration', 3))) if has_intro else 0.0
            tagline_fade_out_start = max(tagline_start + tagline_fade_in, total_duration - tagline_fade_out)
            alpha_in = '1'
            if tagline_fade_in > 0:
                alpha_in = (
                    f"if(lt(t,{tagline_start:.3f}),0,"
                    f"if(lt(t,{tagline_start + tagline_fade_in:.3f}),(t-{tagline_start:.3f})/{max(tagline_fade_in, 0.001):.3f},1))"
                )
            alpha_out = '1'
            if tagline_fade_out > 0:
                alpha_out = (
                    f"if(lt(t,{tagline_fade_out_start:.3f}),1,"
                    f"max(0,1-(t-{tagline_fade_out_start:.3f})/{max(tagline_fade_out, 0.001):.3f}))"
                )
            tagline_alpha = f"({alpha_in})*({alpha_out})"
            container_w = f'w*{tagline_width_pct / 100.0:.4f}'
            if tagline_position == 'bottom-left':
                container_x = '14'
            elif tagline_position == 'bottom-right':
                container_x = f'w-{container_w}-14'
            else:
                container_x = f'(w-{container_w})/2'
            top_padding = max(8, size // 3)
            text_inset = 18
            if tagline_bg_enabled:
                box_height_px = size * 2 + 14
                if tagline_shape == 'pill':
                    box_height_px = size * 2 + 20
                    top_padding = max(10, size // 3)
                    text_inset = 22
                elif tagline_shape == 'square':
                    top_padding = max(6, size // 4)
                    text_inset = 14
                elif tagline_shape == 'outline':
                    text_inset = 20
                elif tagline_shape == 'accent-left':
                    text_inset = 22

                box_y = f'h-{size * 2 + 28 + tagline_vertical_offset:.3f}'
                box_h = f'{box_height_px:.3f}'
                text_y = f'{box_y}+{top_padding}'
            else:
                box_y = None
                box_h = None
                text_y = f'h-{size + 18 + tagline_vertical_offset:.3f}'
            if tagline_alignment == 'left':
                text_x = f'{container_x}+{text_inset}'
            elif tagline_alignment == 'right':
                text_x = f'{container_x}+{container_w}-text_w-{text_inset}'
            else:
                text_x = f'{container_x}+({container_w}-text_w)/2'

            tagline_draw_alpha = tagline_alpha
            if beat_sync_enabled and 'tagline' in beat_sync_targets:
                tagline_draw_alpha = f"({tagline_alpha})*({beat_sync_mod_expr})"
                if beat_sync_mode == 'shake-lite':
                    text_x = (
                        f"({text_x})+({beat_shake_px:.2f}*sin(6.28318*t/{beat_sync_interval:.3f}))"
                    )

            if tagline_bg_enabled:
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]drawbox=x={container_x}:y={box_y}:w={container_w}:h={box_h}"
                    f":color={tagline_bg}@{tagline_bg_opacity:.3f}:t=fill:enable='gte(t,{tagline_start:.3f})'[{nxt}]"
                )
                current_label = nxt
                nxt = f'v_to_{len(filter_parts)}'
                if tagline_shape == 'accent-left':
                    filter_parts.append(
                        f"[{current_label}]drawbox=x={container_x}:y={box_y}:w=4:h={box_h}"
                        f":color={tagline_accent}@1.0:t=fill:enable='gte(t,{tagline_start:.3f})'[{nxt}]"
                    )
                elif tagline_shape == 'outline':
                    filter_parts.append(
                        f"[{current_label}]drawbox=x={container_x}:y={box_y}:w={container_w}:h={box_h}"
                        f":color={tagline_accent}@0.9:t=2:enable='gte(t,{tagline_start:.3f})'[{nxt}]"
                    )
                elif tagline_shape != 'pill':
                    filter_parts.append(
                        f"[{current_label}]drawbox=x={container_x}:y={box_y}:w={container_w}:h=4"
                        f":color={tagline_accent}@1.0:t=fill:enable='gte(t,{tagline_start:.3f})'[{nxt}]"
                    )
                else:
                    filter_parts.append(
                        f"[{current_label}]drawbox=x={container_x}:y={box_y}:w={container_w}:h={box_h}"
                        f":color={tagline_accent}@0.55:t=1:enable='gte(t,{tagline_start:.3f})'[{nxt}]"
                    )
                current_label = nxt
            add_drawtext(tagline_text, text_x, text_y,
                         size, color, alpha_expr=tagline_draw_alpha,
                         enabled=f'between(t,{tagline_start:.3f},{total_duration:.3f})',
                         shadow_size=tagline_shadow_size if tagline_shadow_enabled else 0,
                         shadow_color_hex=tagline_shadow_color,
                         font_key=cs.get('taglineFont', 'default'))

        # ── Watermark ─────────────────────────────────────────────────────────
        if has_watermark:
            size    = int(round(scale_preview_px(cs.get('watermarkFontSize', 18), 8)))
            color   = cs.get('watermarkColor', '#ffffff')
            opacity = float(cs.get('watermarkOpacity', 0.5))
            wpos    = cs.get('watermarkPosition', 'bottom-right')
            pad     = int(round(scale_preview_px(16, 6)))
            x, y = {
                'bottom-right': (f'w-text_w-{pad}', f'h-text_h-{pad}'),
                'bottom-left':  (str(pad),           f'h-text_h-{pad}'),
                'top-right':    (f'w-text_w-{pad}',  str(pad)),
            }.get(wpos, (str(pad), str(pad)))
            add_drawtext(cs['watermarkText'], x, y, size, color,
                         alpha_expr=str(round(opacity, 2)),
                         font_key=cs.get('watermarkFont', 'default'),
                         bold=True)

        # ── Post-text transition pass (for section/phrase cadence) ─────────
        if transition_enabled and transition_preset != 'none' and transition_apply_post_text:
            applied_transition = None
            if transition_preset in {'zoom-in', 'zoom', 'crossfade'}:
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=contrast='{1.0 + 0.34 * transition_strength_factor:.3f}-(0.34*{transition_strength_factor:.3f}*{transition_progress_expr})':"
                    f"brightness='{-0.11 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':"
                    f"saturation='{1.0 + 0.22 * transition_strength_factor:.3f}-(0.22*{transition_strength_factor:.3f}*{transition_progress_expr})':"
                    f"enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'zoom'
            elif transition_preset == 'dip-black':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=brightness='-{0.82 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'dip-black'
            elif transition_preset == 'dip-white':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=brightness='{0.82 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'dip-white'
            elif transition_preset == 'glitch-cut':
                glitch_d = min(0.35, transition_duration)
                glitch_active_expr = (
                    f"lt(({transition_time_expr}),{glitch_d:.3f})"
                )
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]noise=alls={int(16 + 26 * transition_strength_factor)}:allf=t:enable='{glitch_active_expr}'[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'glitch-cut'
            elif transition_preset in {'slide-left', 'push-left'}:
                slide_span = f"iw*{0.08 * transition_strength_factor:.3f}"
                crop_w = f"iw-{slide_span}"
                padded = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]pad=w=iw+{slide_span}:h=ih:x={slide_span}:y=0:color={transition_fill_color}[{padded}]"
                )
                current_label = padded
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]crop=w='{crop_w}':h=ih:x='{slide_span}*{transition_progress_expr}':y=0[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'push-left'
            elif transition_preset in {'slide-right', 'push-right'}:
                slide_span = f"iw*{0.08 * transition_strength_factor:.3f}"
                crop_w = f"iw-{slide_span}"
                padded = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]pad=w=iw+{slide_span}:h=ih:x=0:y=0:color={transition_fill_color}[{padded}]"
                )
                current_label = padded
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]crop=w='{crop_w}':h=ih:x='{slide_span}*(1-{transition_progress_expr})':y=0[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'push-right'
            else:
                logging.warning(
                    "[style] Unsupported post-text transition preset '%s', falling back to zoom",
                    transition_preset,
                )
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=contrast='{1.0 + 0.34 * transition_strength_factor:.3f}-(0.34*{transition_strength_factor:.3f}*{transition_progress_expr})':"
                    f"brightness='{-0.11 * transition_strength_factor:.3f}*(1-{transition_progress_expr})':"
                    f"saturation='{1.0 + 0.22 * transition_strength_factor:.3f}-(0.22*{transition_strength_factor:.3f}*{transition_progress_expr})':"
                    f"enable='{transition_active_expr}'[{nxt}]"
                )
                current_label = nxt
                applied_transition = 'zoom(fallback)'

            logging.info(
                f"🎞️ Post-text transition overlay: preset={applied_transition}, "
                f"duration={transition_duration:.2f}s, strength={transition_strength}, "
                f"timing={transition_on}, interval={transition_section_interval:.2f}s, "
                f"anchor={transition_anchor:.2f}s"
            )

        # ── Ending effect / outro ────────────────────────────────────────────
        if outro_enabled:
            outro_start = max(0.0, total_duration - outro_duration)
            if outro_preset == 'fade-black':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]fade=t=out:st={outro_start:.3f}:d={outro_duration:.3f}[{nxt}]"
                )
                current_label = nxt
            elif outro_preset == 'fade-white':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]fade=t=out:st={outro_start:.3f}:d={outro_duration:.3f}:color=white[{nxt}]"
                )
                current_label = nxt
            elif outro_preset == 'glitch-out':
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]noise=alls={int(12 + 22 * outro_strength_factor)}:allf=t:enable='between(t,{outro_start:.3f},{total_duration:.3f})'[{nxt}]"
                )
                current_label = nxt
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]eq=saturation='{1.0 + 0.42 * outro_strength_factor:.3f}':contrast='{1.0 + 0.30 * outro_strength_factor:.3f}':enable='between(t,{outro_start:.3f},{total_duration:.3f})'[{nxt}]"
                )
                current_label = nxt
            elif outro_preset == 'zoom-out':
                zoom_factor = 0.14 * outro_strength_factor
                zoom_progress = (
                    f"min(max((t-{outro_start:.3f})/{max(outro_duration, 0.001):.3f},0),1)"
                )
                zoom_expr = f"(1-{zoom_factor:.3f}*({zoom_progress}))"
                scaled = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]scale=w='iw*{zoom_expr}':h='ih*{zoom_expr}':eval=frame[{scaled}]"
                )
                current_label = scaled
                nxt = f'v_to_{len(filter_parts)}'
                filter_parts.append(
                    f"[{current_label}]pad=w=iw:h=ih:x='(iw-in_w)/2':y='(ih-in_h)/2':color={transition_fill_color}[{nxt}]"
                )
                current_label = nxt

            logging.info(
                f"🏁 Outro effect overlay: preset={outro_preset}, "
                f"duration={outro_duration:.2f}s, strength={outro_strength}"
            )

        filter_parts.append(f"[{current_label}]null[text_out]")
        return ';'.join(filter_parts)

    def _apply_text_overlays_to_video(self, input_path: str, output_path: str,
                                       total_duration: float) -> 'str | None':
        """Post-processing pass: apply intro card, title, tagline, and watermark
        drawtext overlays to the full composed video in one FFmpeg pass.

        Must run AFTER chunk concatenation so alpha/enable expressions reference
        global video time (not per-chunk time which resets to 0 at every boundary).

        Uses GPU encoding when available so the overlay pass doesn't undo the
        prior compression's quality settings (was previously hardcoded to libx264
        CRF 18, which on its own bloated outputs ~3-4x).
        Returns output_path on success, None on failure.
        """
        fc_str = self._build_overlay_filter_chain(total_duration)
        if fc_str is None:
            import shutil as _sh
            _sh.copy2(input_path, output_path)
            return output_path

        fc_script_path = None
        try:
            fd, fc_script_path = tempfile.mkstemp(prefix='ats_to_', suffix='.txt')
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(fc_str)
            fc_arg = ['-filter_complex_script', fc_script_path]
        except Exception:
            fc_arg = ['-filter_complex', fc_str]

        # Match encoder settings when possible, but retry on CPU if NVENC fails.
        enc = self._get_encoding_settings()
        use_nvenc = ('-c:v' in enc and enc[enc.index('-c:v') + 1] == 'h264_nvenc')

        def _overlay_cmd(use_gpu: bool) -> list:
            if use_gpu:
                video_enc = enc
            else:
                video_enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23']
            return [
                'ffmpeg', '-y', '-i', input_path,
                *fc_arg,
                '-map', '[text_out]', '-map', '0:a?',
                *video_enc,
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-c:a', 'copy',
                '-t', str(total_duration),
                output_path,
            ]

        result = subprocess.run(_overlay_cmd(use_nvenc), capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        if result.returncode != 0 and use_nvenc:
            logging.warning('⚠️ Overlay pass NVENC failed, retrying with CPU (libx264)…')
            result = subprocess.run(_overlay_cmd(False), capture_output=True, text=True,
                                    encoding='utf-8', errors='replace')
        if fc_script_path:
            try: os.unlink(fc_script_path)
            except Exception: pass

        if result.returncode == 0 and os.path.exists(output_path):
            logging.info(
                f"✅ Text overlays applied to full video: {os.path.getsize(output_path):,} bytes"
            )
            return output_path
        logging.error(f"❌ Text overlay pass failed: {result.stderr[-1500:]}")
        return None

    def _apply_text_overlays_inplace(self, path: str, total_duration: float) -> str:
        """Apply intro card and text overlays to a video file, replacing it in-place.
        Returns the (possibly unchanged) path."""
        cs = getattr(self, 'composition_style', {}) or {}
        has_any = any([
            # Text overlays (intro card, title, tagline, watermark)
            cs.get('introCardEnabled'),
            cs.get('titleEnabled')     and cs.get('titleText',     '').strip(),
            cs.get('taglineEnabled')   and cs.get('taglineText',   '').strip(),
            cs.get('watermarkEnabled') and cs.get('watermarkText', '').strip(),
            # Visual effects (transitions, outro, waveform, vignette, glitch, beat sync)
            cs.get('transitionEnabled'),
            cs.get('outroEffectEnabled'),
            cs.get('waveformEnabled'),
            cs.get('vignetteEnabled'),
            cs.get('glitchEnabled'),
            cs.get('beatSyncEnabled'),
        ])
        if not has_any:
            return path
        tmp = path + '.text_tmp.mp4'
        out = self._apply_text_overlays_to_video(path, tmp, total_duration)
        if out and os.path.exists(out):
            import shutil as _sh
            _sh.move(out, path)
            logging.info(f"✅ Text overlays applied in-place to {Path(path).name}")
        else:
            logging.warning("⚠️ Text overlay pass failed — returning original video unchanged")
        return path


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
            
            if not input_map:
                logging.error(f"❌ No valid video inputs found")
                return None
            if len(input_map) == 1:
                logging.info(
                    "🧱 One valid input detected; keeping grid compositor active so "
                    "empty cells and composition styles still render"
                )
            
            logging.info(f"📐 Calculating grid dimensions...")

            # Use render config resolution for stage-aware tile calculations
            target_width, target_height = map(int, self.render_config['resolution'].split('x'))
            layout_plan = self._build_grid_render_slots(track_segments, target_width, target_height)
            grid_rows = layout_plan['grid_rows']
            grid_cols = layout_plan['grid_cols']
            cell_width = layout_plan['unit_width']
            cell_height = layout_plan['unit_height']
            render_slots = layout_plan['slots']
            
            logging.info(f"   Target resolution: {target_width}x{target_height} ({'PREVIEW' if self.preview_mode else 'PRODUCTION'})")
            logging.info(f"   Base cell dimensions: {cell_width}x{cell_height} pixels")
            logging.info(f"📊 Grid placement summary: {len(render_slots)}/{len(track_segments)} segments placed")

            if not render_slots:
                logging.error("❌ No valid render slots resolved for grid layout.")
                return None

            # Build the FFmpeg command from the populated grid
            cmd = ['ffmpeg', '-y']
            filter_parts = []
            audio_inputs_for_mix = []
            input_idx = 0
            style_temp_files = []  # Temp files created during styling (cleaned up after ffmpeg)
            beat_sync_stats = {
                'eligible_cells': 0,
                'modulated_cells': 0,
                'modulated_windows': 0,
            }

            # Add a full-canvas background input so tiles can be overlaid at arbitrary spans.
            (
                input_idx,
                canvas_video_input_idx,
                silent_audio_input_idx,
                background_description,
            ) = self._append_canvas_background_inputs(
                cmd,
                filter_parts,
                input_idx,
                target_width,
                target_height,
                duration,
            )
            cs = getattr(self, 'composition_style', {}) or {}
            bg_hex = cs.get('backgroundColor', '#0a0a0f')
            
            logging.info(f"🎛️ Building FFmpeg filter complex...")
            logging.info(
                f"   Added canvas background ({background_description}, input {canvas_video_input_idx}) "
                f"and silent audio (input {silent_audio_input_idx})"
            )

            # ── Pre-process: extend short clips to chunk_duration ────────────
            # Even with overlay composition, pre-extending clips preserves clip-specific
            # idle backgrounds and keeps note-triggered cells visually stable across the
            # full chunk duration.
            preprocess_temp_files = []
            extended_clips = {}  # original_path -> extended_path

            for slot in render_slots:
                seg = slot['segment']
                orig = seg.get('video_path', '')
                if not orig or not os.path.exists(orig) or orig in extended_clips:
                    continue
                if seg.get('preserve_idle_alpha'):
                    extended_clips[orig] = orig
                    logging.info(
                        f"[style] cell={seg.get('track_id', 'unknown')!r} keeping alpha note-trigger clip without gap-fill extension"
                    )
                    continue
                track_id = seg.get('track_id', f"{slot['row']}_{slot['column']}")
                clip_style, _, matched_key = self._resolve_clip_style(track_id)
                clip_bg_hex = bg_hex
                if (
                    clip_style.get('bgColorEnabled')
                    and clip_style.get('bgColor')
                    and not clip_style.get('transparentBg')
                ):
                    clip_bg_hex = clip_style.get('bgColor')
                    logging.info(
                        f"[style] cell={track_id!r} extending idle frames with {clip_bg_hex} "
                        f"({matched_key or 'direct track id'})"
                    )
                ext, is_temp = self._preprocess_extend_clip(orig, duration, clip_bg_hex)
                extended_clips[orig] = ext
                if is_temp:
                    preprocess_temp_files.append(ext)

            if preprocess_temp_files:
                logging.info(f"   📼 Pre-processed {len(preprocess_temp_files)} short clip(s) → extended to {duration:.2f}s")
            # ── End pre-process ──────────────────────────────────────────────

            # Process each resolved tile slot in the grid
            cells_processed = len(render_slots)
            cells_with_content = 0
            overlay_slots = []
            decode_args = self._get_ffmpeg_decode_args()
            for slot_index, slot in enumerate(render_slots):
                cell_segment = slot['segment']
                if cell_segment and os.path.exists(cell_segment['video_path']):
                    cells_with_content += 1
                    # Use extended clip if available (avoids idle cells falling back to global bg too early)
                    video_path = extended_clips.get(cell_segment['video_path'], cell_segment['video_path'])
                    per_input_decode_args = [] if cell_segment.get('preserve_idle_alpha') else decode_args
                    cmd.extend([*per_input_decode_args, '-i', video_path])

                    track_id = cell_segment.get('track_id', f"{slot['row']}_{slot['column']}")
                    video_output_label = f"[v_{slot_index}]"
                    self._apply_cell_style_filters(
                        filter_parts,
                        f"[{input_idx}:v]",
                        video_output_label,
                        slot['render_width'],
                        slot['render_height'],
                        track_id,
                        cell_segment,
                        style_temp_files,
                        chunk_duration=duration,
                        beat_sync_stats=beat_sync_stats,
                    )
                    overlay_slots.append({
                        'label': video_output_label,
                        'x': slot['render_x'],
                        'y': slot['render_y'],
                        'track_id': track_id,
                        'input_idx': input_idx,
                        'segment': cell_segment,
                    })

                    # --- VOLUME FIX START ---
                    vol_db = float(self._resolve_segment_volume(cell_segment))
                    velocity_val = (
                        cell_segment.get('velocity')
                        or cell_segment.get('midi_velocity')
                        or cell_segment.get('note_velocity')
                    )
                    vol_db += self._velocity_to_db(velocity_val)
                    vol_linear = 10 ** (vol_db / 20.0)
                    filter_parts.append(f"[{input_idx}:a]volume={vol_linear:.2f}[a_{slot_index}]")
                    audio_inputs_for_mix.append(f"[a_{slot_index}]")
                    # --- VOLUME FIX END ---

                    logging.info(
                        f"      Cell ({slot['row']},{slot['column']}) span {slot['span_w']}x{slot['span_h']} "
                        f"→ {slot['render_width']}x{slot['render_height']} @ ({slot['render_x']},{slot['render_y']}) "
                        f"(slot {slot['pixel_width']}x{slot['pixel_height']}, inset {slot['slot_padding']}px): "
                        f"{cell_segment.get('type', 'unknown')} - {Path(cell_segment['video_path']).name} - Vol: {vol_db}dB"
                    )
                    input_idx += 1
                else:
                    logging.warning(
                        f"      Cell ({slot['row']},{slot['column']}) has no usable video input and will remain background only"
                    )
            
            logging.info(f"   Grid cells: {cells_with_content}/{cells_processed} contain actual content")
            gstyle = getattr(self, 'composition_style', {}) or {}
            beat_targets = gstyle.get('beatSyncTargets')
            if isinstance(beat_targets, (list, tuple, set)):
                beat_targets_set = {str(v).strip().lower() for v in beat_targets if str(v).strip()}
            elif isinstance(beat_targets, str) and beat_targets.strip():
                beat_targets_set = {s.strip().lower() for s in beat_targets.split(',') if s.strip()}
            else:
                beat_targets_set = set()
            if bool(gstyle.get('beatSyncEnabled')) and 'track-cells' in beat_targets_set:
                logging.info(
                    "🥁 Beat sync cells summary: "
                    f"eligible={beat_sync_stats.get('eligible_cells', 0)}, "
                    f"modulated={beat_sync_stats.get('modulated_cells', 0)}, "
                    f"windows={beat_sync_stats.get('modulated_windows', 0)}"
                )

            # ── Optional solo-resolution optimisation ───────────────────────
            # Disabled by default because the frontend preview always preserves
            # the grid footprint, even when only one cell is active.
            solo_mode_enabled = str(os.getenv('ATS_ENABLE_SOLO_MODE', '')).strip().lower() in {
                '1', 'true', 'yes', 'on'
            }
            if solo_mode_enabled and cells_with_content == 1:
                solo_w = target_width & ~1
                solo_h = target_height & ~1
                filter_parts.clear()
                style_temp_files.clear()
                solo_slot = overlay_slots[0] if overlay_slots else None
                solo_segment = solo_slot.get('segment') if solo_slot else None
                solo_track_id = solo_segment.get('track_id', 'solo') if solo_segment else 'solo'
                solo_input_idx = solo_slot.get('input_idx', 2) if solo_slot else 2
                self._apply_cell_style_filters(
                    filter_parts, f'[{solo_input_idx}:v]', '[v_solo]',
                    solo_w, solo_h, solo_track_id, solo_segment, style_temp_files,
                    chunk_duration=duration,
                    beat_sync_stats={},
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
                        f"[{solo_input_idx}:a]volume={vol_linear:.2f},"
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
                # Cleanup temp files (style + preprocess)
                for tf in style_temp_files + preprocess_temp_files:
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
            elif cells_with_content == 1:
                logging.info("🧱 Single active cell detected; preserving grid layout because solo mode is disabled")
            # ── End solo-resolution optimisation ────────────────────────────

            force_cpu_encode = cells_with_content > self.max_concurrent_streams
            if force_cpu_encode:
                logging.warning(
                    f"⚠️ HIGH STREAM PRESSURE: {cells_with_content} active cells > {self.max_concurrent_streams} max allowed. "
                    f"Falling back to CPU encode for stability. "
                    f"To use GPU for all streams, increase ATS_MAX_CONCURRENT_STREAMS env var. "
                    f"(e.g., export ATS_MAX_CONCURRENT_STREAMS=64 or set in .env)"
                )

            current_video_label = 'grid_base'
            for slot_index, overlay_slot in enumerate(overlay_slots):
                next_video_label = f'grid_ov_{slot_index}'
                filter_parts.append(
                    f"[{current_video_label}]{overlay_slot['label']}overlay="
                    f"x={overlay_slot['x']}:y={overlay_slot['y']}:eof_action=pass:format=auto"
                    f"[{next_video_label}]"
                )
                current_video_label = next_video_label

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
                filter_parts, current_video_label, target_width, target_height,
                duration, 'audio_pre', style_temp_files
            )
            final_video_map = final_video if final_video.startswith('[') else f'[{final_video}]'
            final_audio_map = final_audio if final_audio.startswith('[') else f'[{final_audio}]'

            logging.info(f"🎬 Final FFmpeg command construction:")
            logging.info(f"   Total inputs: {input_idx}")
            logging.info(f"   Filter complex parts: {len(filter_parts)}")
            logging.info(f"   Overlay slots: {len(overlay_slots)}")

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
            logging.info(f"   Using codec: {encoding_args[1] if len(encoding_args) > 1 else 'unknown'}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for GPU encoder failure and retry with CPU fallback
            if (result.returncode != 0 and 
                'h264_nvenc' in ' '.join(encoding_args) and 
                any(err in result.stderr.lower() for err in ['unknown', 'error', 'not found', 'failed'])):
                logging.warning(f"⚠️ GPU encoder (h264_nvenc) failed, attempting CPU fallback...")
                logging.warning(f"   Error: {result.stderr[:300]}")
                
                # Rebuild command with CPU encoder
                cpu_encoding_args = ['-c:v', 'libx264', '-preset', self.render_config['preset'], '-crf', self.render_config['crf']]
                cmd_cpu = cmd[:-len(encoding_args) - 5]  # Remove old encoding args
                cmd_cpu.extend([
                    '-map', final_video_map, '-map', final_audio_map,
                    *cpu_encoding_args,
                    '-c:a', 'aac', '-b:a', self.render_config['audio_bitrate'],
                    '-pix_fmt', 'yuv420p',
                    '-t', str(duration), '-r', '30', str(output_path)
                ])
                
                logging.info(f"🔄 Retrying with CPU encoder (libx264)...")
                result = subprocess.run(cmd_cpu, capture_output=True, text=True)
            
            # Cleanup temp files (style + preprocess)
            for tf in style_temp_files + preprocess_temp_files:
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
                    onset_offset=onset_offset,
                    style_track_id=drum_track_id,
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
                        'preserve_idle_alpha': triggered_video.lower().endswith('.mov'),
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
        """Find the recorded video file for a specific drum sound.

        Checks sources in priority order:
          1. PathRegistry  — populated at startup from the session's uploaded files.
          2. explicit_video_files dict — the videoFiles payload from the frontend.
          3. Filesystem glob of self.uploads_dir (fallback for legacy layouts).

        The drum_name argument comes from DRUM_NOTES, e.g. "Bass Drum", "Hi-Hat Closed".
        Both PathRegistry and the explicit_video_files dict use a lower-snake_case key
        (hyphens and spaces replaced with underscores), optionally prefixed with "drum_".
        """
        # Canonical form used by PathRegistry: lower-snake, no leading drum_ prefix.
        # Both spaces AND hyphens must be normalised to underscore — the registry
        # registration path does this (video_composer.py:1077) but get_drum_path's
        # normalization only replaces spaces, leaving hyphens intact.
        canonical = drum_name.lower().replace(' ', '_').replace('-', '_')

        # 1. PathRegistry (fastest path — already populated at session start)
        registry = PathRegistry.get_instance()
        path = registry.get_drum_path(canonical)
        if path and os.path.exists(path):
            logging.debug(f"[drum] PathRegistry hit: {drum_name} → {path}")
            return path

        # 2. explicit_video_files dict (key may be prefixed with "drum_")
        for key_variant in (f'drum_{canonical}', canonical):
            info = self.explicit_video_files.get(key_variant)
            if isinstance(info, dict):
                p = info.get('path', '')
            elif info:
                p = str(info)
            else:
                # Also try hyphen variant (Node.js may send drum_hi-hat_closed)
                hyphen_key = key_variant.replace('_', '-', 1) if key_variant.startswith('drum_') else None
                info2 = self.explicit_video_files.get(hyphen_key) if hyphen_key else None
                if isinstance(info2, dict):
                    p = info2.get('path', '')
                elif info2:
                    p = str(info2)
                else:
                    continue
            if p and os.path.exists(p):
                logging.debug(f"[drum] explicit_video_files hit: {key_variant} → {p}")
                return p

        # 3. Glob fallback (legacy layout — searches self.uploads_dir)
        search_patterns = [
            f"*{drum_name.lower()}*",
            f"*{canonical}*",
            f"*{drum_name.replace(' ', '').lower()}*",
            f"*drum*{canonical}*",
            f"*{drum_name.split()[0].lower()}*",
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
                    'notes': chunk_notes,
                    'preserve_idle_alpha': triggered_video.lower().endswith('.mov')
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
        """Compatibility wrapper: route legacy callers into the fixed grid compositor."""
        logging.info("[compat] _create_grid_layout_chunk -> _create_grid_layout_chunk_fixed")
        return self._create_grid_layout_chunk_fixed(track_segments, output_path, duration)

    def _create_ffmpeg_grid_layout(self, track_segments, output_path, duration, grid_rows, grid_cols):
        """Compatibility wrapper: route legacy callers into the fixed FFmpeg grid compositor."""
        logging.info(
            "[compat] _create_ffmpeg_grid_layout -> _create_ffmpeg_grid_layout_fixed "
            "(legacy grid_rows=%s, grid_cols=%s ignored)",
            grid_rows,
            grid_cols,
        )
        return self._create_ffmpeg_grid_layout_fixed(track_segments, output_path, duration)

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
            target_width, target_height = self._get_target_resolution()
            bg_hex = (getattr(self, 'composition_style', {}) or {}).get(
                'backgroundColor',
                '#0a0a0f',
            )
            bg_ffmpeg = self._hex_to_ffmpeg_color(bg_hex)
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'color={bg_ffmpeg}:size={target_width}x{target_height}:duration={duration}:rate=30',
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
        """Compatibility wrapper for legacy callers using the removed MIDI sync helper."""
        logging.info(
            "[compat] create_midi_synchronized_composition -> VideoComposer.create_composition"
        )

        compat_midi_data = dict(midi_data or {})
        compat_midi_data['videoFiles'] = video_paths or {}

        uploads_dir = getattr(self, 'uploads_dir', None)
        if uploads_dir and 'uploadsDir' not in compat_midi_data:
            compat_midi_data['uploadsDir'] = str(uploads_dir)

        compat_composer = type(self)(
            compat_midi_data,
            str(getattr(self, 'processed_videos_dir', self.temp_dir)),
            output_path,
            preview_mode=getattr(self, 'preview_mode', False),
        )
        return compat_composer.create_composition()

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
