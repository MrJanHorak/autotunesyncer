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
import aubio
import syncio
from pstats import SortKey
from tqdm import tqdm
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

from utils import normalize_instrument_name, midi_to_note
from drum_utils import (
    DRUM_NOTES,
    # process_drum_track,
    # get_drum_groups,
    # get_drum_name,
    is_drum_kit
)

from processing_utils import encoder_queue, GPUManager
from get_system_metrics import get_system_metrics
from video_utils import run_ffmpeg_command, encode_video, validate_video
from cuda_compositing import CudaVideoProcessor
import traceback

from gpu_pipeline import GPUPipelineProcessor
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

@contextmanager
def timing_block(name):
    """Context manager to measure execution time"""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logging.info(f"Operation [{name}] took {duration:.3f} seconds")

class GPUStreamManager:
    def __init__(self, num_streams=4):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.stream_idx = 0
        self.lock = threading.Lock()

    def get_stream(self):
        with self.lock:
            stream = self.streams[self.stream_idx]
            self.stream_idx = (self.stream_idx + 1) % len(self.streams)
            return stream

    def synchronize_all(self):
        for stream in self.streams:
            torch.cuda.synchronize(stream.device)

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
            'drums': 0.2,
            'instruments': 1.5
        }

class VideoComposer:

    FRAME_RATE = 30
    CHUNK_DURATION = 4
    OVERLAP_DURATION = 1
    CROSSFADE_DURATION = 0.5
    MIN_VIDEO_DURATION = 1.0
    DURATION = 1.0
    VOLUME_MULTIPLIERS = {
        'drums': 0.2,
        'instruments': 1.5
    }

    def __init__(self, processed_videos_dir, midi_data, output_path):
        """Initialize VideoComposer with proper path handling"""
        try:
            logging.info("=== VideoComposer Initialization ===")
            logging.info(f"Received MIDI data structure: {list(midi_data.keys())}")
            logging.info(f"Grid arrangement from MIDI: {midi_data.get('gridArrangement')}")

            self.processed_videos_dir = Path(processed_videos_dir)
            # Fix uploads path - go up two levels to /backend/uploads
            self.uploads_dir = Path(processed_videos_dir).parent.parent.parent / "uploads"
            logging.info(f"Setting uploads directory: {self.uploads_dir}")
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
                    "-preset", "p4",
                    "-surfaces", "16",
                    "-tune", "hq",
                    "-rc", "vbr_hq",
                    "-cq", "20",
                    "-b:v", "5M",
                    "-maxrate", "10M",
                    "-bufsize", "10M",
                    "-profile:v", "high"
                ]
            }
            self.chunk_size = max(1, min(16, os.cpu_count()))  # Smaller chunk size
            self.max_workers = min(2, os.cpu_count())  # Limit workers
            self.use_gpu = True
            self.lock = RLock()  # Add class-level lock
            self.clip_pool = ClipPool(max_size=8)  # Add clip pool
            self.chunk_cache_lock = RLock()  # Add dedicated cache lock
            self.max_cache_size = 1024 * 1024 * 100
            self.active_readers = set()  # Add reader tracking
            # Initialize path registry
            self.path_registry = PathRegistry()
            
            # After copying files, register them
            self._register_video_paths()
            self.video_cache = LRUCache(maxsize=64)  # Increase cache size
            self.audio_cache = LRUCache(maxsize=64) 
            self.autotune_cache = LRUCache(maxsize=64) # Increase cache size
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
        """Register paths for all videos"""
        # Get singleton instance
        registry = PathRegistry.get_instance()
        
        # Set registry file if needed
        registry_file = self.processed_videos_dir / "path_registry.json"
        
        # Register from processed videos directory
        registry.register_from_directory(self.processed_videos_dir)
        registry.save_registry(str(registry_file))
        
        # Log registration stats
        logging.info(f"Registered {len(registry.drum_paths)} drum videos")
        logging.info(f"Registered {sum(len(notes) for notes in registry.instrument_paths.values())} instrument notes")

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
                    
                    self.drum_tracks.append(normalized_track)
                else:
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


    def _normalize_track(self, track):
        """Convert track data to standard format"""
        if isinstance(track, int):
            return {
                'notes': [],
                'instrument': {'name': f'track_{track}'},
                'isDrum': False
            }
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
                    clip.reader.close()
                    self.active_readers.remove(clip.reader)
                else:
                    clip.reader.close()
            if hasattr(clip, 'close'):
                clip.close()
        except Exception as e:
            logging.warning(f"Error closing clip {id(clip)}: {e}")

    def _setup_track_configuration(self):
        try:
            grid_arrangement = self.midi_data.get('gridArrangement', {})
            logging.info(f"Grid arrangement received: {grid_arrangement}")
            
            if not grid_arrangement:
                raise ValueError("No grid arrangement provided")

            # Store positions and validate
            self.grid_positions = {}
            for track_id, pos_data in grid_arrangement.items():
                # Validate position data
                if not all(k in pos_data for k in ['row', 'column', 'position']):
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
        if result.returncode != 0:
            logging.error(f"Encoding failed: {result.stderr}")
            raise Exception(f"Encoding failed: {result.stderr}")
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

    # def calculate_chunk_lengths(self):
    #     """Calculate chunk lengths for composition"""
    #     try:
    #         # Get all track notes in one list
    #         all_notes = []
            
    #         # Add regular track notes
    #         for track in self.midi_data['tracks']:
    #             if isinstance(track, dict) and 'notes' in track:
    #                 all_notes.extend(track['notes'])
            
    #         if not all_notes:
    #             raise ValueError("No notes found in any tracks")
            
    #         # Find last note end time
    #         last_note_time = 0
    #         for note in all_notes:
    #             if isinstance(note, dict):
    #                 note_end = float(note['time']) + float(note['duration'])
    #                 last_note_time = max(last_note_time, note_end)
            
    #         # Calculate chunks based on exact duration needed
    #         full_chunks = math.floor(last_note_time / self.CHUNK_DURATION)
    #         final_chunk = last_note_time % self.CHUNK_DURATION
            
    #         # Only include final chunk if there's actual content
    #         if final_chunk < 0.1:  # If less than 0.1s remaining, ignore final chunk
    #             final_chunk = 0
                
    #         logging.info(f"Total duration: {last_note_time:.2f}s")
    #         logging.info(f"Full chunks: {full_chunks}")
    #         logging.info(f"Final chunk: {final_chunk:.2f}s")
            
    #         return full_chunks, final_chunk
                
    #     except Exception as e:
    #         logging.error(f"Error calculating chunks: {str(e)}")
    #         return 0, 0

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
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error creating placeholder chunk: {e}")
            return False
        
    # def has_valid_notes(self, track):
    #     """Check if track has any valid notes"""
    #     notes = track.get('notes', [])
    #     return len(notes) > 0

    

    def get_track_layout(self):
        """Get grid dimensions from frontend arrangement"""
        try:
            grid_arrangement = self.midi_data.get('gridArrangement', {})
            
            if not grid_arrangement:
                return self._calculate_default_layout()

            # Get dimensions from stored positions
            max_row = max(pos['row'] for pos in self.grid_positions.values())
            max_col = max(pos['column'] for pos in self.grid_positions.values())
            rows = max_row + 1
            cols = max_col + 1

            # Log final layout for debugging
            logging.info("\nGrid Visual Layout:")
            for row in range(rows):
                row_str = ""
                for col in range(cols):
                    track_id = next((k for k, v in self.grid_positions.items() 
                                if v['row'] == row and v['column'] == col), "empty")
                    row_str += f"[{track_id:^20}] "
                logging.info(row_str)
                
            return (rows, cols)
                
        except Exception as e:
            logging.error(f"Layout error: {str(e)}")
            return (1, 1)
        
    def get_note_volume(self, velocity, is_drum=False, midi_note=None):
        # Existing normalization logic
        normalized_velocity = velocity if velocity <= 2.0 else float(velocity) / 127.0
        
        # Dynamic multipliers based on context
        context_multipliers = {
            'drums': self._calculate_drum_multiplier(midi_note),
            'instruments': self._calculate_instrument_multiplier(midi_note),
            'bass': self._calculate_bass_multiplier(midi_note)
        }
        
        # Advanced frequency-dependent adjustment
        multiplier_key = 'bass' if midi_note and midi_note < 48 else ('drums' if is_drum else 'instruments')
        
        # Use velocity curve instead of linear mapping
        volume = 0.4 + (1 - math.exp(-2 * normalized_velocity)) * 2.6 * context_multipliers[multiplier_key]
        
        return min(volume, 3.0)
        
    # def get_note_volume(self, velocity, is_drum=False, midi_note=None):
    #     # Check if velocity is already normalized (less than 2.0)
    #     if velocity <= 2.0:
    #         normalized_velocity = velocity
    #     else:
    #         normalized_velocity = float(velocity) / 127.0
        
    #     # Set base multipliers with bass boost
    #     multipliers = {
    #         'drums': 0.6,       # Drums at 60%
    #         'instruments': 1.5,  # Instruments at 150%
    #         'bass': 2.5         # Bass instruments at 250%
    #     }
        
    #     # Apply frequency-dependent boost for bass notes
    #     is_bass_note = midi_note and midi_note < 48  # Notes below C3
    #     multiplier_key = 'bass' if is_bass_note else ('drums' if is_drum else 'instruments')
        
    #     # Calculate volume with better minimum
    #     base_volume = normalized_velocity * 1.8  # Higher overall boost
    #     volume = max(0.4, base_volume * multipliers[multiplier_key])
        
    #     # Cap at reasonable maximum 
    #     volume = min(volume, 3.0)
        
    #     return volume
    
    def _add_drum_clips(self, drum_dir, drum_key, notes, grid, row, col, active_clips, start_time):
        """Add drum clips to the grid"""
        try:
            drum_file = drum_dir / f"{drum_key}.mp4"
            if not drum_file.exists():
                logging.warning(f"Drum file not found: {drum_file}")
                return
            
            for note in notes:
                try:
                    clip = VideoFileClip(str(drum_file))
                    active_clips.append(clip)
                    
                    # Apply volume based on note velocity
                    velocity = float(note.get('velocity', 100))
                    volume = self.get_note_volume(velocity, is_drum=True)
                    # clip = clip.VolumeX(volume)
                    
                    # Set timing
                    time = float(note['time']) - start_time
                    duration = min(float(note['duration']), clip.duration)
                    clip = clip.subclipped(0, duration).with_start(time)
                    
                    if isinstance(grid[row][col], ColorClip):
                        grid[row][col] = clip
                    else:
                        existing = grid[row][col]
                        grid[row][col] = CompositeVideoClip([existing, clip])
                    
                    logging.info(f"Added {drum_key} at [{row}][{col}] t={time}")
                except Exception as e:
                    logging.error(f"Error processing drum clip: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error adding drum clips: {e}")

    # def _process_drum_chunk_gpu(self, track_idx, chunk_notes, grid_config, start_time, rows, cols):
    #     """Process drum chunk directly into grid_config using path registry"""
    #     drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
    #     if not drum_dir.exists():
    #         return

    #     for note in chunk_notes:
    #         try:
    #             midi_note = int(note['midi'])
    #             drum_name = DRUM_NOTES.get(midi_note)
    #             if not drum_name:
    #                 continue

    #             drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"

    #             if drum_key not in self.grid_positions:
    #                 logging.warning(f"No grid position for {drum_key}, skipping")
    #                 continue
                
    #             pos_data = self.grid_positions[drum_key]
    #             row, col = int(pos_data['row']), int(pos_data['column'])
                
    #             # Get timing info
    #             time = float(note['time']) - start_time
    #             audio_duration = float(note.get('duration', 0.5))

    #             video_duration = max(audio_duration, self.MIN_VIDEO_DURATION)
                
    #             # Find video path through registry
    #             video_path = self._find_drum_video(drum_key)
    #             if not video_path:
    #                 continue
                    
    #             if grid_config[row][col].get('empty', True):
    #                 grid_config[row][col] = {
    #                     'path': video_path,
    #                     'start_time': 0,
    #                     'audio_duration': audio_duration,  # Original duration for audio
    #                     'video_duration': video_duration,  # Extended duration for video
    #                     'offset': time,
    #                     'empty': False
    #                 }
    #             else:
    #                 # If cell already has content, create a list of clips
    #                 if 'clips' not in grid_config[row][col]:
    #                     grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                        
    #                 grid_config[row][col]['clips'].append({
    #                     'path': video_path,
    #                     'start_time': 0,
    #                     'audio_duration': audio_duration,
    #                     'video_duration': video_duration,
    #                     'offset': time,
    #                 })
                    
    #             logging.info(f"Added {drum_key} at [{row}][{col}] t={time}, audio_dur={audio_duration:.2f}, video_dur={video_duration:.2f}")
                    
    #         except Exception as e:
    #             logging.error(f"Error processing drum note: {e}")
    #             continue\
    

    def _process_drum_chunk_gpu(self, track_idx, chunk_notes, grid_config, start_time, rows, cols):
        """Process drum chunk directly into grid_config using path registry"""
        drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
        if not drum_dir.exists():
            return

        for note in chunk_notes:
            try:
                midi_note = int(note['midi'])
                drum_name = DRUM_NOTES.get(midi_note)
                if not drum_name:
                    continue

                drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"

                if drum_key not in self.grid_positions:
                    logging.warning(f"No grid position for {drum_key}, skipping")
                    continue
                
                pos_data = self.grid_positions[drum_key]
                row, col = int(pos_data['row']), int(pos_data['column'])
                
                # Get timing info
                time = float(note['time']) - start_time
                audio_duration = float(note.get('duration', 0.5))

                video_duration = max(audio_duration, self.MIN_VIDEO_DURATION)
                
                # Find video path through registry
                video_path = self._find_drum_video(drum_key)
                if not video_path:
                    continue

                # Check cache
                autotune_params = (video_path, midi_note)
                cached_video = self._get_cached_video(video_path, autotune_params)
                if cached_video:
                    video_path = cached_video
                else:
                    # Autotune the audio
                    autotuned_audio_path = self._autotune_audio(video_path, midi_note)
                    if autotuned_audio_path:
                        video_path = autotuned_audio_path
                    self._cache_video(video_path, autotuned_audio_path, autotune_params)
                    
                if grid_config[row][col].get('empty', True):
                    grid_config[row][col] = {
                        'path': video_path,
                        'start_time': 0,
                        'audio_duration': audio_duration,  # Original duration for audio
                        'video_duration': video_duration,  # Extended duration for video
                        'offset': time,
                        'empty': False
                    }
                else:
                    # If cell already has content, create a list of clips
                    if 'clips' not in grid_config[row][col]:
                        grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                        
                    grid_config[row][col]['clips'].append({
                        'path': video_path,
                        'start_time': 0,
                        'audio_duration': audio_duration,
                        'video_duration': video_duration,
                        'offset': time,
                    })
                    
                logging.info(f"Added {drum_key} at [{row}][{col}] t={time}, audio_dur={audio_duration:.2f}, video_dur={video_duration:.2f}")
                    
            except Exception as e:
                logging.error(f"Error processing drum note: {e}")
                continue
    
    def _calculate_optimal_chunk_sizes(self):
        """Calculate optimal chunk sizes based on available resources"""
        # Get system resources
        metrics = get_system_metrics()
        available_memory = 100 - metrics['memory_percent']
        
        # Adjust chunk size based on available memory
        if available_memory < 20:  # Low memory
            return [1] * self.full_chunks  # Process one chunk at a time
        elif available_memory < 50:  # Medium memory
            return [2] * (self.full_chunks // 2) + ([1] if self.full_chunks % 2 else [])
        else:  # High memory
            return [4] * (self.full_chunks // 4) + [self.full_chunks % 4] if self.full_chunks % 4 else []

    def _classify_chunks_by_complexity(self):
        """Divide chunks into simple and complex based on note density"""
        simple_chunks = []
        complex_chunks = []
        
        for chunk_idx in range(self.full_chunks + (1 if self.final_duration > 0 else 0)):
            # Calculate chunk boundaries
            start_time = chunk_idx * self.CHUNK_DURATION
            end_time = start_time + (self.final_duration if chunk_idx == self.full_chunks else self.CHUNK_DURATION)
            
            # Count notes in this chunk
            note_count = 0
            for track in self.midi_data['tracks']:
                if isinstance(track, dict) and 'notes' in track:
                    chunk_notes = [
                        note for note in track.get('notes', [])
                        if start_time <= float(note['time']) < end_time
                    ]
                    note_count += len(chunk_notes)
            
            # Classify based on note density
            if note_count > 30:  # High complexity threshold
                complex_chunks.append(chunk_idx)
            else:
                simple_chunks.append(chunk_idx)
        
        return simple_chunks, complex_chunks
    
    def _get_cached_video(self, video_path, autotune_params=None):
        """Get cached video with autotune parameters in key"""
        cache_key = (str(video_path), str(autotune_params))  # Include autotune params
        if cache_key in self.video_cache:
            logging.info(f"Video cache hit: {video_path} with {autotune_params}")
            return self.video_cache[cache_key]
        return None

    def _cache_video(self, video_path, cached_path, autotune_params=None):
        """Cache video with autotune parameters in key"""
        cache_key = (str(video_path), str(autotune_params))  # Include autotune params
        self.video_cache[cache_key] = cached_path
        logging.info(f"Video cache miss: {video_path} with {autotune_params}")

    def _get_cached_audio(self, audio_path, autotune_params=None):
        """Get cached audio with autotune parameters in key"""
        cache_key = (str(audio_path), str(autotune_params))  # Include autotune params
        if cache_key in self.audio_cache:
            logging.info(f"Audio cache hit: {audio_path} with {autotune_params}")
            return self.audio_cache[cache_key]
        return None

    def _cache_audio(self, audio_path, cached_path, autotune_params=None):
        """Cache audio with autotune parameters in key"""
        cache_key = (str(audio_path), str(autotune_params))  # Include autotune params
        self.audio_cache[cache_key] = cached_path
        logging.info(f"Audio cache miss: {audio_path} with {autotune_params}")

    def _get_optimal_encoding_params(self, frame_complexity):
        """Adjust encoding parameters based on content complexity"""
        # Default parameters
        params = {
            'codec': 'h264_nvenc',
            'preset': 'p4',  # Medium preset
            'bitrate': '6M',
            'maxrate': '9M',
            'bufsize': '12M'
        }
        
        # Adjust quality based on complexity
        if frame_complexity > 0.7:  # High complexity
            params.update({
                'preset': 'p2',      # Higher quality preset
                'bitrate': '8M',
                'maxrate': '12M',
                'bufsize': '16M'
            })
        elif frame_complexity < 0.4:  # Low complexity
            params.update({
                'preset': 'p6',      # Faster preset
                'bitrate': '4M',
                'maxrate': '6M',
                'bufsize': '8M'
            })
            
        return params
    
    def process_chunks_parallel(self):
        """Process video chunks in parallel using thread pool executor"""
        # Use a Queue for thread-safe result collection
        from queue import Queue
        result_queue = Queue()
        
        # Define max_workers based on CPU count (missing variable)
        max_workers = min(os.cpu_count() or 4, 4)  # Limit to 4 or CPU count
        
        # Calculate which chunks need complex vs. simple processing
        # (These variables were missing in your code)
        all_chunks = list(range(self.total_chunks))
        
        # Determine complexity for each chunk
        complex_chunks = []
        simple_chunks = []
        
        for chunk_idx in all_chunks:
            # Check if chunk has complex elements (instruments, effects, etc.)
            if self._is_complex_chunk(chunk_idx):
                complex_chunks.append(chunk_idx)
            else:
                simple_chunks.append(chunk_idx)
        
        logging.info(f"Processing {len(complex_chunks)} complex chunks and {len(simple_chunks)} simple chunks")
        
        # Define the method to process a chunk and put result in queue
        def _process_chunk_with_queue(chunk_idx, result_queue):
            try:
                # Process the chunk using existing method
                result = self._process_chunk(chunk_idx)
                if result:
                    result_queue.put(result)
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                traceback.print_exc()
        
        # Process chunks using thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process complex chunks first (they take longer)
            for chunk_idx in complex_chunks:
                executor.submit(_process_chunk_with_queue, chunk_idx, result_queue)
                
            # Small delay to prevent thread contention
            time.sleep(0.05)
            
            # Then process simple chunks
            for chunk_idx in simple_chunks:
                executor.submit(_process_chunk_with_queue, chunk_idx, result_queue)
        
        # Collect results after all threads complete
        chunk_files = []
        while not result_queue.empty():
            result = result_queue.get()
            if result:
                chunk_files.append(result)
        
        return sorted(chunk_files)

    def _is_complex_chunk(self, chunk_idx):
        """Determine if a chunk requires complex processing"""
        # Fix: Use CHUNK_DURATION instead of chunk_duration
        start_time = chunk_idx * self.CHUNK_DURATION
        end_time = start_time + self.CHUNK_DURATION
        
        # Simple implementation that doesn't depend on _track_has_notes_in_timeframe
        # Count notes in this time range across all tracks
        note_count = 0
        for track in self.midi_data['tracks']:
            if 'notes' in track:
                chunk_notes = [
                    note for note in track['notes']
                    if start_time <= float(note['time']) < end_time
                ]
                note_count += len(chunk_notes)
        
        # Complex if more than 20 notes
        return note_count > 20

    # def process_chunks_parallel(self):
    #     """Process chunks in parallel with adaptive workload distribution"""
    #     chunk_files = []
    #     total_chunks = self.full_chunks + (1 if self.final_duration > 0 else 0)
        
    #     # Get optimized grouping of chunks
    #     simple_chunks, complex_chunks = self._classify_chunks_by_complexity()
        
    #     logging.info(f"Classified chunks: {len(simple_chunks)} simple, {len(complex_chunks)} complex")
        
    #     # Determine optimal worker count based on system resources
    #     metrics = get_system_metrics()
    #     max_workers = min(os.cpu_count(), 4)  # Default max
        
    #     # Adjust workers based on memory pressure
    #     if metrics['memory_percent'] > 80:
    #         max_workers = 2  # Reduce workers when low memory
        
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         # Process complex chunks first (they need more time)
    #         complex_futures = {executor.submit(self._process_chunk, idx): idx for idx in complex_chunks}
            
    #         # Process result as they complete
    #         for future in as_completed(complex_futures):
    #             chunk_idx = complex_futures[future]
    #             try:
    #                 result = future.result()
    #                 if result:
    #                     chunk_files.append(result)
    #                     logging.info(f"Complex chunk {chunk_idx} completed")
    #                 else:
    #                     logging.error(f"Complex chunk {chunk_idx} failed")
    #             except Exception as e:
    #                 logging.error(f"Error processing complex chunk {chunk_idx}: {e}")
            
    #         # Then process simple chunks
    #         simple_futures = {executor.submit(self._process_chunk, idx): idx for idx in simple_chunks}
            
    #         for future in as_completed(simple_futures):
    #             chunk_idx = simple_futures[future]
    #             try:
    #                 result = future.result()
    #                 if result:
    #                     chunk_files.append(result)
    #                     logging.info(f"Simple chunk {chunk_idx} completed")
    #                 else:
    #                     logging.error(f"Simple chunk {chunk_idx} failed")
    #             except Exception as e:
    #                 logging.error(f"Error processing simple chunk {chunk_idx}: {e}")
        
    #     # Sort by chunk index to maintain order
    #     chunk_files.sort()
    #     return chunk_files
    
    def _calculate_drum_multiplier(self, midi_note=None):
        """Calculate volume multiplier for drum sounds based on pitch"""
        # Base multiplier for drums
        base_multiplier = 0.6
        
        # Adjust based on drum type if available
        if midi_note:
            # Kick drums (lower notes) need more emphasis
            if midi_note < 40:  # Bass drum/kick range
                return base_multiplier * 1.2
            # Hi-hats and cymbals (higher notes) need less volume
            elif midi_note > 50:  # Cymbals/hi-hat range
                return base_multiplier * 0.8
        
        return base_multiplier

    def _calculate_instrument_multiplier(self, midi_note=None):
        """Calculate volume multiplier for regular instrument sounds"""
        # Base multiplier for instruments
        base_multiplier = 1.5
        
        # Adjust based on pitch range
        if midi_note:
            # Slight boost for middle range notes for better clarity
            if 60 <= midi_note <= 72:  # Middle octave range
                return base_multiplier * 1.1
        
        return base_multiplier

    def _calculate_bass_multiplier(self, midi_note=None):
        """Calculate volume multiplier for bass sounds"""
        # Bass notes need significant boosting
        base_multiplier = 2.5
        
        # Further boost deeper bass notes
        if midi_note and midi_note < 40:  # Very deep bass
            return base_multiplier * 1.2
        
        return base_multiplier
    
    

    # def _process_instrument_chunk_gpu(self, track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols):
    #     # Get instrument name for lookup
    #     instrument_name = normalize_instrument_name(track['instrument']['name'])
        
    #     # Try multiple possible ID formats
    #     possible_keys = [
    #         f"track-{track_idx}",            # Format with prefix
    #         f"track-{instrument_name}",      # Name with prefix
    #         f"{track_idx}",                  # Bare index (what frontend actually uses)
    #         str(track_idx),                  # Ensure string comparison
    #         f"{instrument_name}"             # Bare name
    #     ]
        
    #     # Find the first key that exists in grid_positions
    #     track_key = next((key for key in possible_keys if key in self.grid_positions), None)
        
    #     if track_key:
    #         pos_data = self.grid_positions[track_key]
    #         row, col = int(pos_data['row']), int(pos_data['column'])
    #         logging.info(f"Found grid position for {instrument_name}: row={row}, col={col}")
    #     else:
    #         # Fallback to default
    #         row = (track_idx % (rows-1)) + 1
    #         col = min(track_idx // (rows-1), cols-1)
    #         logging.warning(f"No grid position for {instrument_name}, using default: row={row}, col={col}")
        
    #     # Group notes by time to handle chords
    #     time_groups = {}
    #     for note in chunk_notes:
    #         time_pos = float(note['time'])
    #         if time_pos not in time_groups:
    #             time_groups[time_pos] = []
    #         time_groups[time_pos].append(note)
        
    #     # Sort time positions for look-ahead
    #     sorted_times = sorted(time_groups.keys())
        
    #     # Process each group of notes
    #     for i, time_pos in enumerate(sorted_times):
    #         notes = time_groups[time_pos]
    #         try:
    #             # Get instrument info
    #             instrument_name = track.get('instrument', {}).get('name', 'piano')
    #             midi_notes = [int(note['midi']) for note in notes]
                
    #             # Find video path for the instrument note
    #             video_path = self._find_instrument_video(instrument_name, midi_notes[0])
    #             if not video_path:
    #                 continue
                    
    #             # Calculate timing
    #             time = time_pos - start_time
    #             audio_duration = min(float(notes[0].get('duration', 0.5)), end_time - time_pos)
                
    #             # Look ahead to next note on same instrument to avoid overlaps
    #             next_note_time = float('inf')
    #             if i < len(sorted_times) - 1:
    #                 next_note_time = sorted_times[i+1]
                
    #             # Available time until next note or end of chunk
    #             available_time = min(next_note_time - time_pos, end_time - time_pos)
                
    #             # Use minimum duration only if we have enough space
    #             video_duration = min(max(audio_duration, self.MIN_VIDEO_DURATION), available_time)
                
    #             # Add to grid config
    #             if grid_config[row][col].get('empty', True):
    #                 grid_config[row][col] = {
    #                     'path': video_path,
    #                     'start_time': 0,
    #                     'audio_duration': audio_duration,  # Original duration for audio
    #                     'video_duration': video_duration,  # Extended when safe
    #                     'offset': time,
    #                     'empty': False
    #                 }
    #             else:
    #                 # If cell already has content, create a list of clips
    #                 if 'clips' not in grid_config[row][col]:
    #                     grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                        
    #                 grid_config[row][col]['clips'].append({
    #                     'path': video_path,
    #                     'start_time': 0,
    #                     'audio_duration': audio_duration,
    #                     'video_duration': video_duration,
    #                     'offset': time,
    #                 })
                    
    #             logging.info(f"Added {instrument_name} note {midi_notes[0]} at [{row}][{col}] t={time}, audio_dur={audio_duration:.2f}, video_dur={video_duration:.2f}")
                    
    #         except Exception as e:
    #             logging.error(f"Error processing instrument note: {e}")
    #             continue

    # def _autotune_audio(self, video_path, midi_note):
    #     """Autotunes audio from video to match midi note"""
    #     cache_key = (video_path, midi_note)
    #     if cache_key in self.autotune_cache:
    #         return self.autotune_cache[cache_key]
        
    #     try:
    #         # Extract audio from video
    #         audio_path = os.path.join(self.temp_dir, f"audio_{midi_note}.wav")
    #         cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', audio_path]
    #         subprocess.run(cmd, check=True, capture_output=True)

    #         # Autotune audio
    #         autotuned_audio_path = os.path.join(self.temp_dir, f"autotuned_{midi_note}.wav")
    #         autotune_script = os.path.join(os.path.dirname(__file__), 'autotune.py')  # Adjust path as needed
    #         cmd = ['python', autotune_script, audio_path, autotuned_audio_path, str(midi_note)]
    #         subprocess.run(cmd, check=True, capture_output=True)

    #         self.autotune_cache[cache_key] = autotuned_audio_path
    #         return autotuned_audio_path
    #     except Exception as e:
    #         logging.error(f"Autotune error: {e}")
    #         return None

    def _analyze_pitch(self, video_path):
        """Analyze pitch of audio in video file"""
        try:
            # Extract audio from video
            audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
                audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Analyze pitch using aubio
            s = aubio.source(audio_path, rate=44100, channels=1)
            pitch_o = aubio.pitch("yin", 44100, 1024, s.samplerate)
            pitch_o.set_unit("midi")
            pitch_o.set_tolerance(0.8)

            pitches = []
            while True:
                samples, read = s()
                pitch = pitch_o(samples)[0]
                pitches.append(pitch)
                if read < 1024:
                    break

            # Calculate average pitch
            avg_pitch = sum(pitches) / len(pitches)
            return avg_pitch

        except Exception as e:
            logging.error(f"Pitch analysis error: {e}")
            return None

    def _autotune_audio(self, video_path, midi_note):
        """Autotunes audio from video to match midi note"""
        cache_key = (str(video_path), midi_note)
        if cache_key in self.autotune_cache:
            return self.autotune_cache[cache_key]
        
        try:
            # Extract audio from video
            audio_path = os.path.join(self.temp_dir, f"audio_{midi_note}.wav")
            cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', audio_path]
            subprocess.run(cmd, check=True, capture_output=True)

            # Get current pitch of the audio
            current_pitch = self._analyze_pitch(audio_path)
            if current_pitch is None:
                logging.warning(f"Could not determine pitch for {audio_path}, skipping autotune")
                return None

            # Calculate required pitch shift
            target_pitch = midi_note
            pitch_shift = target_pitch - current_pitch
            
            logging.info(f"Pitch analysis - Current: {current_pitch}, Target: {target_pitch}, Shift: {pitch_shift}")
            
            # Limit pitch shift to reasonable range to avoid extreme artifacts
            if abs(pitch_shift) > 12:  # More than an octave
                logging.warning(f"Limiting extreme pitch shift from {pitch_shift} to {math.copysign(12, pitch_shift)}")
                pitch_shift = math.copysign(12, pitch_shift)  # Keep sign but limit magnitude
            
            # Autotune audio
            autotuned_audio_path = os.path.join(self.temp_dir, f"autotuned_{midi_note}.wav")
            autotune_script = os.path.join(os.path.dirname(__file__), 'autotune.py')
            cmd = ['python', autotune_script, audio_path, autotuned_audio_path, str(midi_note)]
            subprocess.run(cmd, check=True, capture_output=True)

            self.autotune_cache[cache_key] = autotuned_audio_path
            logging.info(f"Autotune cache miss: {video_path}, {midi_note}")
            return autotuned_audio_path
        except Exception as e:
            logging.error(f"Autotune error: {e}")
            return None

    # def _process_instrument_chunk_gpu(self, track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols):
    #     """Process instrument chunk with proper track ID mapping"""
    #     # Get instrument name for lookup
    #     instrument_name = normalize_instrument_name(track['instrument']['name'])
        
    #     # Extract the actual track ID from the track data
    #     track_id = track.get('id', str(track_idx))  # Look for ID in track data
        
    #     # Log the track information for debugging
    #     logging.info(f"Processing instrument: {instrument_name} (ID: {track_id}, Index: {track_idx})")
        
    #     # Try multiple possible ID formats to find grid position
    #     # possible_keys = [
    #     #     track_id,                    # Actual track ID from MIDI data
    #     #     str(track_id),               # Ensure string comparison
    #     #     f"track-{track_id}",         # Format with ID prefix
    #     #     f"{track_idx}",              # Array index as fallback
    #     #     str(track_idx),              # String version of index
    #     #     f"track-{track_idx}",        # Index with prefix
    #     #     f"track-{instrument_name}",  # Name with prefix
    #     #     f"{instrument_name}"         # Bare name
    #     # ]
    #     possible_keys = [
    #         track_id,                    # Actual track ID from MIDI data
    #         str(track_id),               # Ensure string comparison
    #         f"{track_idx}",              # Array index as fallback
    #         str(track_idx),              # String version of index
    #         f"{instrument_name}"         # Bare name
    #     ]
        
    #     # Find the first key that exists in grid_positions
    #     track_key = next((key for key in possible_keys if key in self.grid_positions), None)
        
    #     if track_key:
    #         pos_data = self.grid_positions[track_key]
    #         row, col = int(pos_data['row']), int(pos_data['column'])
    #         logging.info(f"Found grid position for {instrument_name}/{track_id}: row={row}, col={col}, using key={track_key}")
    #     else:
    #         # FIXED: Better fallback position calculation
    #         row = min((track_idx % rows), rows-1)  # Ensure row is within bounds
    #         col = min((track_idx // rows), cols-1)  # Ensure column is within bounds
    #         logging.warning(f"No grid position for {instrument_name}/{track_id}, tried keys: {possible_keys}. Using default: row={row}, col={col}")

    # def _process_instrument_chunk_gpu(self, track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols):
    #     """Process instrument chunk with proper track ID mapping"""
    #     # Get instrument name for lookup
    #     instrument_name = normalize_instrument_name(track['instrument']['name'])
        
    #     # Extract the actual track ID from the track data
    #     track_id = track.get('id', str(track_idx))  # Look for ID in track data
        
    #     # Log the track information for debugging
    #     logging.info(f"Processing instrument: {instrument_name} (ID: {track_id}, Index: {track_idx})")
        
    #     # Use the track ID directly to find grid position
    #     if track_id in self.grid_positions:
    #         pos_data = self.grid_positions[track_id]
    #         row, col = int(pos_data['row']), int(pos_data['column'])
    #         logging.info(f"Found grid position for {instrument_name} (ID: {track_id}): row={row}, col={col}")
    #     else:
    #         # Fallback to default
    #         row = (track_idx % (rows-1)) + 1
    #         col = min(track_idx // (rows-1), cols-1)
    #         logging.warning(f"No grid position for {instrument_name} (ID: {track_id}), using default: row={row}, col={col}")
        
    #     # Rest of your existing code for processing instrument notes
    #     registry = PathRegistry.get_instance()
        
    #     # Track instrumentation stats
    #     note_count = 0
    #     notes_placed = 0
        
    #     # Find suitable grid cell based on track index or grid arrangement
    #     for note in chunk_notes:
    #         # Calculate timing
    #         note_time = float(note['time']) - start_time
    #         note_duration = float(note['duration'])  
            
    #         # Adjust timing if note starts before this chunk
    #         if note_time < 0:
    #             note_duration += note_time  # Reduce duration by overlap
    #             note_time = 0

    #             # Skip if resulting duration would be too small
    #             if note_duration <= 0.05:  # Minimum practical duration
    #                 continue
                
    #         # Skip if note is completely outside this chunk
    #         if note_time < 0 or note_time >= (end_time - start_time):
    #             continue
                
    #         # Find the note path
    #         note_path = registry.get_instrument_path(instrument_name, note['midi'])
    #         if not note_path:
    #             continue

    #         # Autotune the audio
    #         autotuned_audio_path = self._autotune_audio(note_path, note['midi'])
    #         if autotuned_audio_path:
    #             note_path = autotuned_audio_path
            
    #         # Place the note in the grid
    #         cell = grid_config[row][col]
    #         if cell.get('empty', True):
    #             # First note in this cell
    #             # grid_config[row][col] = {
    #             #     'path': note_path,
    #             #     'start_time': note_time,
    #             #     'duration': note_duration,
    #             #     'volume': float(note.get('velocity', 0.8)) / 127,
    #             #     'empty': False
    #             # }
    #             grid_config[row][col] = {
    #                 'path': note_path,
    #                 'start_time': 0,
    #                 'offset': note_time,  # Use offset instead of start_time
    #                 'audio_duration': note_duration,
    #                 'video_duration': note_duration,
    #                 'duration': note_duration,
    #                 'volume': float(note.get('velocity', 0.8)) / 127,
    #                 'empty': False
    #             }
    #         else:
    #             # Add to existing cell
    #             if 'clips' not in grid_config[row][col]:
    #                 grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                    
    #             # grid_config[row][col]['clips'].append({
    #             #     'path': note_path,
    #             #     'start_time': note_time,
    #             #     'duration': note_duration,
    #             #     'volume': float(note.get('velocity', 0.8)) / 127
    #             # })
    #             grid_config[row][col] = {
    #                 'path': note_path,
    #                 'start_time': 0,
    #                 'offset': note_time,  # Use offset instead of start_time
    #                 'audio_duration': note_duration,
    #                 'video_duration': note_duration,
    #                 'duration': note_duration,
    #                 'volume': float(note.get('velocity', 0.8)) / 127,
    #                 'empty': False
    #             }
            
    #         notes_placed += 1
            
    #     note_count = len(chunk_notes)
    #     logging.debug(f"Processed {notes_placed}/{note_count} {instrument_name} notes in chunk {start_time:.1f}s - {end_time:.1f}s")

    def _process_instrument_chunk_gpu(self, track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols):
        """Process instrument chunk with proper track ID mapping"""
        # Get instrument name for lookup
        instrument_name = normalize_instrument_name(track['instrument']['name'])
        
        # Extract the actual track ID from the track data
        track_id = track.get('id', str(track_idx))  # Look for ID in track data
        
        # Log the track information for debugging
        logging.info(f"Processing instrument: {instrument_name} (ID: {track_id}, Index: {track_idx})")
        
        # Use the track ID directly to find grid position
        if track_id in self.grid_positions:
            pos_data = self.grid_positions[track_id]
            row, col = int(pos_data['row']), int(pos_data['column'])
            logging.info(f"Found grid position for {instrument_name} (ID: {track_id}): row={row}, col={col}")
        else:
            # Fallback to default
            row = (track_idx % (rows-1)) + 1
            col = min(track_idx // (rows-1), cols-1)
            logging.warning(f"No grid position for {instrument_name} (ID: {track_id}), using default: row={row}, col={col}")
        
        # Rest of your existing code for processing instrument notes
        registry = PathRegistry.get_instance()
        
        # Track instrumentation stats
        note_count = 0
        notes_placed = 0
        
        # Find suitable grid cell based on track index or grid arrangement
        for note in chunk_notes:
            # Calculate timing
            note_time = float(note['time']) - start_time
            note_duration = float(note['duration'])  
            
            # Adjust timing if note starts before this chunk
            if note_time < 0:
                note_duration += note_time  # Reduce duration by overlap
                note_time = 0

                # Skip if resulting duration would be too small
                if note_duration <= 0.05:  # Minimum practical duration
                    continue
                
            # Skip if note is completely outside this chunk
            if note_time < 0 or note_time >= (end_time - start_time):
                continue
                
            # Find the note path
            note_path = registry.get_instrument_path(instrument_name, note['midi'])
            if not note_path:
                continue

            # Autotune the audio
            autotune_params = (note_path, note['midi'])
            cached_video = self._get_cached_video(note_path, autotune_params)
            if cached_video:
                note_path = cached_video
            else:
                autotuned_audio_path = self._autotune_audio(note_path, note['midi'])
                if autotuned_audio_path:
                    note_path = autotuned_audio_path
                self._cache_video(note_path, autotuned_audio_path, autotune_params)
            
            # Place the note in the grid
            cell = grid_config[row][col]
            if cell.get('empty', True):
                # First note in this cell
                # grid_config[row][col] = {
                #     'path': note_path,
                #     'start_time': note_time,
                #     'duration': note_duration,
                #     'volume': float(note.get('velocity', 0.8)) / 127,
                #     'empty': False
                # }
                grid_config[row][col] = {
                    'path': note_path,
                    'start_time': 0,
                    'offset': note_time,  # Use offset instead of start_time
                    'audio_duration': note_duration,
                    'video_duration': note_duration,
                    'duration': note_duration,
                    'volume': float(note.get('velocity', 0.8)) / 127,
                    'empty': False
                }
            else:
                # Add to existing cell
                if 'clips' not in grid_config[row][col]:
                    grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                    
                # grid_config[row][col]['clips'].append({
                #     'path': note_path,
                #     'start_time': note_time,
                #     'duration': note_duration,
                #     'volume': float(note.get('velocity', 0.8)) / 127
                # })
                grid_config[row][col] = {
                    'path': note_path,
                    'start_time': 0,
                    'offset': note_time,  # Use offset instead of start_time
                    'audio_duration': note_duration,
                    'video_duration': note_duration,
                    'duration': note_duration,
                    'volume': float(note.get('velocity', 0.8)) / 127,
                    'empty': False
                }
            
            notes_placed += 1
            
        note_count = len(chunk_notes)
        logging.debug(f"Processed {notes_placed}/{note_count} {instrument_name} notes in chunk {start_time:.1f}s - {end_time:.1f}s")


    def _find_drum_video(self, drum_key):
        """Find drum video in track directory"""
        try:
            # Iterate through drum tracks to find the correct index
            for track_idx, track in enumerate(self.drum_tracks):
                # Construct the drum directory path
                drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                if not drum_dir.exists():
                    continue  # Try next drum track

                # Construct the drum file path
                drum_file = drum_dir / f"{drum_key.replace('drum_', '')}.mp4"
                if not drum_file.exists():
                    continue  # Try next drum track

                # If we found the file, return its path
                return str(drum_file)

            # If no drum file was found, log a warning and return None
            logging.warning(f"Drum file not found for {drum_key} in any drum track")
            return None

        except Exception as e:
            logging.error(f"Error finding drum video: {e}")
            return None

    def _find_instrument_video(self, instrument, midi_note):
        """Find video for instrument note using registry"""
        path = PathRegistry.get_instance().get_instrument_path(instrument, midi_note)
        if path:
            logging.info(f"Found instrument video for {instrument}, note {midi_note}: {path}")
            return path
        
        logging.warning(f"No video found for instrument: {instrument}, note: {midi_note}")
        return None

    def _get_drum_key_for_note(self, midi_note):
        """Get drum name for MIDI note"""
        from drum_utils import DRUM_NOTES
        return DRUM_NOTES.get(midi_note)
    


    # def _process_chunk(self, chunk_idx):
    #     """Process chunk and collect operations for GPU processing"""
    #     stream = self.gpu_stream_manager.get_stream()
    #     with torch.cuda.stream(stream):
    #         cache_key = f"chunk_{chunk_idx}"
    #         if cache_key in self.chunk_cache:
    #             return self.chunk_cache[cache_key]
            
    #         try:
    #             # Quick cache check
    #             cache_key = f"chunk_{chunk_idx}"
    #             with self.chunk_cache_locks[chunk_idx]:
    #                 if cache_key in self.chunk_cache:
    #                     return self.chunk_cache[cache_key]

    #             # Calculate timing
    #             start_time = chunk_idx * self.CHUNK_DURATION
    #             end_time = start_time + (
    #                 self.final_duration if chunk_idx == self.full_chunks
    #                 else self.CHUNK_DURATION
    #             )
    #             chunk_duration = end_time - start_time

    #             # Get grid layout

    def has_valid_notes(self, track):
        """Check if a track has valid notes"""
        if not isinstance(track, dict) or 'notes' not in track:
            return False
        
        notes = track['notes']
        if not isinstance(notes, list) or len(notes) == 0:
            return False
        
        return True

    def _process_chunk(self, chunk_idx):
        """Process chunk using pure GPU pipeline with batched audio processing"""
        try:
            logging.info(f"Starting processing of chunk {chunk_idx}")
            # Quick cache check with minimal lock time
            cache_key = f"chunk_{chunk_idx}"
            with self.chunk_cache_lock:
                if cache_key in self.chunk_cache:
                    logging.info(f"Chunk {chunk_idx} found in cache")
                    return self.chunk_cache[cache_key]

            # Calculate timing
            start_time = chunk_idx * self.CHUNK_DURATION 
            end_time = start_time + (
                self.final_duration if chunk_idx == self.full_chunks
                else self.CHUNK_DURATION
            )
            chunk_duration = end_time - start_time
            logging.info(f"Chunk {chunk_idx} timing: start={start_time}, end={end_time}, duration={chunk_duration}")

            # Get grid layout
            rows, cols = self.get_track_layout()
            logging.info(f"Chunk {chunk_idx} grid layout: rows={rows}, cols={cols}")
            
            # Create grid configuration directly for GPU pipeline
            grid_config = [
                [{'empty': True} for _ in range(cols)] 
                for _ in range(rows)
            ]
            
            # For batched audio processing - collect all audio operations
            audio_operations = []
            all_notes = []
            
            # Process tracks using information from the path registry
            for track_idx, track in enumerate(self.midi_data['tracks']):
                # Find notes in this time chunk with proper overlap handling
                chunk_notes = [
                    note for note in track.get('notes', [])
                    if (start_time - self.OVERLAP_DURATION) <= float(note['time']) < end_time
                    or (float(note['time']) < start_time and float(note['time']) + float(note['duration']) > start_time)
                ]

                # Collect all notes for complexity analysis
                all_notes.extend(chunk_notes)
                logging.info(f"Track {track_idx} has {len(chunk_notes)} notes in chunk {chunk_idx}")
                
                if not chunk_notes:
                    continue
                    
                # Collect video and audio operations for the grid
                if is_drum_kit(track.get('instrument', {})):
                    # ADD MISSING CODE HERE
                    self._process_drum_chunk_gpu(track_idx, chunk_notes, grid_config, start_time, rows, cols)
                    drum_ops = self._collect_drum_operations(
                        track_idx, chunk_notes, start_time, rows, cols
                    )
                    # Add to grid config and collect audio operations
                    for op in drum_ops:  # FIXED: Using drum_ops instead of instrument_ops
                        # Update grid config
                        row, col = op['position']
                        if grid_config[row][col].get('empty', True):
                            grid_config[row][col] = {
                                'path': op['video_path'],
                                'start_time': 0,
                                'audio_duration': op.get('audio_duration', op['duration']),
                                'video_duration': op.get('video_duration', op['duration']),
                                'duration': op['duration'],  # Keep for backwards compatibility
                                'offset': op['offset'],
                                'empty': False
                            }
                        else:
                            # If cell has content, create a clips list
                            if 'clips' not in grid_config[row][col]:
                                grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                            grid_config[row][col]['clips'].append({
                                'path': op['video_path'],
                                'start_time': 0,
                                'audio_duration': op.get('audio_duration', op['duration']),
                                'video_duration': op.get('video_duration', op['duration']),
                                'duration': op['duration'],  # Keep for backwards compatibility
                                'offset': op['offset']
                            })
                        # Add to audio operations
                        audio_operations.append(op)
                else:
                    self._process_instrument_chunk_gpu(track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols)
                    instrument_ops = self._collect_instrument_operations(
                        track_idx, track, chunk_notes, start_time, end_time, rows, cols
                    )
                    # Add to grid config and collect audio operations
                    for op in instrument_ops:
                        # Update grid config
                        row, col = op['position']
                        if grid_config[row][col].get('empty', True):
                            grid_config[row][col] = {
                                'path': op['video_path'],
                                'start_time': 0,
                                'audio_duration': op.get('audio_duration', op['duration']),
                                'video_duration': op.get('video_duration', op['duration']),
                                'duration': op['duration'],  # Keep for backward compatibility
                                'offset': op['offset'],
                                'empty': False
                            }
                        else:
                            # If cell has content, create a clips list
                            if 'clips' not in grid_config[row][col]:
                                grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
                            grid_config[row][col]['clips'].append({
                                'path': op['video_path'],
                                'start_time': 0,
                                'audio_duration': op.get('audio_duration', op['duration']),
                                'video_duration': op.get('video_duration', op['duration']),
                                'duration': op['duration'],  # Keep for backward compatibility
                                'offset': op['offset']
                            })
                        # Add to audio operations
                        audio_operations.append(op)

            # NEW: Analyze frame complexity based on collected notes
            frame_complexity = self._estimate_frame_complexity(all_notes)
            
            # NEW: Get optimal encoding parameters based on complexity
            encoding_params = self._get_optimal_encoding_params(frame_complexity)

            # Use GPU pipeline directly
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            pipeline = GPUPipelineProcessor(composer=self)

            # Check if the pipeline supports encoding parameters
            supports_encoding_params = hasattr(pipeline, 'set_encoding_params')
            
            # Configure pipeline with optimal parameters if supported
            if supports_encoding_params:
                pipeline.set_encoding_params(encoding_params)
        
            
            # First batch process all audio operations
            mixed_audio = None
            if audio_operations:
                audio_output = self.temp_dir / f"mixed_audio_{chunk_idx}.aac"
                mixed_audio = pipeline.batch_process_audio(
                    audio_operations, 
                    str(audio_output)
                )
            
            # Then process the video with the pre-mixed audio
            if supports_encoding_params:
                pipeline.process_chunk_pure_gpu(
                    grid_config=grid_config,
                    output_path=str(chunk_path),
                    fps=self.FRAME_RATE,
                    duration=chunk_duration,
                    audio_path=mixed_audio,
                    encoding_params=encoding_params
                )
            else:
                # Fallback to calling without encoding_params
                pipeline.process_chunk_pure_gpu(
                    grid_config=grid_config,
                    output_path=str(chunk_path),
                    fps=self.FRAME_RATE,
                    duration=chunk_duration,
                    audio_path=mixed_audio
                )

            if chunk_path.exists():
                result = str(chunk_path)
                with self.chunk_cache_lock:  # Minimal lock time
                    self.chunk_cache[cache_key] = result
                logging.info(f"Chunk {chunk_idx} processed successfully, cached, and returning")
                return result
            else:
                logging.warning(f"Chunk {chunk_idx} processing failed, chunk file not found")
                
        except Exception as e:
            logging.error(f"GPU chunk processing error: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None    


    # def _process_chunk(self, chunk_idx):
    #     """Process chunk using pure GPU pipeline with batched audio processing"""
    #     try:
    #         logging.info(f"Starting processing for chunk {chunk_idx}")  # Add logging
    #         # Quick cache check with minimal lock time
    #         cache_key = f"chunk_{chunk_idx}"
    #         with self.chunk_cache_lock:
    #             if cache_key in self.chunk_cache:
    #                 logging.info(f"Chunk {chunk_idx} found in cache")  # Add logging
    #                 return self.chunk_cache[cache_key]

    #         # Calculate timing
    #         start_time = chunk_idx * self.CHUNK_DURATION
    #         end_time = start_time + (
    #             self.final_duration if chunk_idx == self.full_chunks
    #             else self.CHUNK_DURATION
    #         )
    #         chunk_duration = end_time - start_time

    #         # Get grid layout
    #         rows, cols = self.get_track_layout()
            
    #         # Create grid configuration directly for GPU pipeline
    #         grid_config = [
    #             [{'empty': True} for _ in range(cols)] 
    #             for _ in range(rows)
    #         ]
            
    #         # For batched audio processing - collect all audio operations
    #         audio_operations = []
    #         all_notes = []
            
    #         # Process tracks using information from the path registry
    #         for track_idx, track in enumerate(self.midi_data['tracks']):
    #             # Find notes in this time chunk with proper overlap handling
    #             chunk_notes = [
    #                 note for note in track.get('notes', [])
    #                 if start_time <= float(note['time']) < end_time
    #             ]
                
    #             # Process instrument tracks
    #             if not is_drum_kit(track) and self.has_valid_notes(track):
    #                 self._process_instrument_chunk_gpu(track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols)
                
    #             # Process drum tracks
    #             elif is_drum_kit(track):
    #                 self._process_drum_chunk_gpu(track_idx, chunk_notes, grid_config, start_time, rows, cols)

    #         # Create output path
    #         chunk_path = self.processed_videos_dir / f"chunk_{chunk_idx}.mp4"
            
    #         # Process chunk using pure GPU pipeline
    #         gpu_pipeline = GPUPipelineProcessor(composer=self)
    #         gpu_pipeline.process_chunk_pure_gpu(grid_config, str(chunk_path), duration=chunk_duration)
            
    #         if chunk_path.exists():
    #             result = str(chunk_path)
    #             with self.chunk_cache_lock:
    #                 self.chunk_cache[cache_key] = result
    #             logging.info(f"Chunk {chunk_idx} processed successfully and cached")  # Add logging
    #             return result
    #         else:
    #             logging.error(f"Chunk {chunk_idx} processing failed: output file not found")  # Add logging
    #             return None

    #     except Exception as e:
    #         logging.error(f"GPU chunk processing error: {e}")
    #         logging.error(f"Traceback: {traceback.format_exc()}")
    #         return None    #             rows, cols = self.get_track_layout()

    #             # Create grid configuration
    #             grid_config = [[{'empty': True} for _ in range(cols)] for _ in range(rows)]

    #             # Collect audio operations
    #             audio_operations = []

    #             # Process tracks
    #             for track_idx, track in enumerate(self.midi_data['tracks']):
    #                 # Find notes in this chunk
    #                 chunk_notes = []
    #                 for note in track.get('notes', []):
    #                     note_start = float(note['time'])
    #                     note_end = note_start + float(note['duration'])

    #                     # Check if the note overlaps with the chunk
    #                     if (start_time <= note_end and note_start < end_time):
    #                         # Clip the note if necessary
    #                         clipped_start = max(note_start, start_time)
    #                         clipped_end = min(note_end, end_time)

    #                         # Create a new clipped note
    #                         clipped_note = note.copy()
    #                         clipped_note['time'] = str(clipped_start)
    #                         clipped_note['duration'] = str(clipped_end - clipped_start)
    #                         chunk_notes.append(clipped_note)

    #                 if not chunk_notes:
    #                     continue

    #                 # Process drum or instrument track
    #                 if is_drum_kit(track.get('instrument', {})):
    #                     ops = self._collect_drum_operations(track_idx, chunk_notes, start_time, rows, cols)
    #                 else:
    #                     ops = self._collect_instrument_operations(track_idx, track, chunk_notes, start_time, end_time, rows, cols)

    #                 # Apply operations to grid and collect audio
    #                 for op in ops:
    #                     row, col = op['position']
    #                     if grid_config[row][col].get('empty', True):
    #                         grid_config[row][col] = {
    #                             'path': op['video_path'],
    #                             'start_time': 0,
    #                             'audio_duration': op.get('audio_duration', op['duration']),
    #                             'video_duration': op.get('video_duration', op['duration']),
    #                             'duration': op['duration'],
    #                             'offset': op['offset'],
    #                             'empty': False
    #                         }
    #                     else:
    #                         if 'clips' not in grid_config[row][col]:
    #                             grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
    #                         grid_config[row][col]['clips'].append({
    #                             'path': op['video_path'],
    #                             'start_time': 0,
    #                             'audio_duration': op.get('audio_duration', op['duration']),
    #                             'video_duration': op.get('video_duration', op['duration']),
    #                             'duration': op['duration'],
    #                             'offset': op['offset']
    #                         })
    #                     audio_operations.append(op)

    #             # Process audio
    #             mixed_audio = None
    #             if audio_operations:
    #                 audio_output = self.temp_dir / f"mixed_audio_{chunk_idx}.aac"
    #                 pipeline = GPUPipelineProcessor()
    #                 mixed_audio = pipeline.batch_process_audio(audio_operations, str(audio_output))

    #             # Process video
    #             chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
    #             pipeline = GPUPipelineProcessor()
    #             pipeline.process_chunk_pure_gpu(
    #                 grid_config=grid_config,
    #                 output_path=str(chunk_path),
    #                 fps=self.FRAME_RATE,
    #                 duration=chunk_duration,
    #                 audio_path=mixed_audio
    #             )

    #             # Cache result
    #             if chunk_path.exists():
    #                 result = str(chunk_path)
    #                 with self.chunk_cache_locks[chunk_idx]:
    #                     self.chunk_cache[cache_key] = result
    #                 return result

    #         except Exception as e:
    #             logging.error(f"GPU chunk processing error: {e}")
    #             logging.error(f"Traceback: {traceback.format_exc()}")
    #             return None


    # def _process_chunk(self, chunk_idx):
    #     """Process chunk using pure GPU pipeline with batched audio processing"""
    #     try:                    
    #         # Quick cache check with minimal lock time
    #         cache_key = f"chunk_{chunk_idx}"
    #         with self.chunk_cache_lock:
    #             if cache_key in self.chunk_cache:
    #                 return self.chunk_cache[cache_key]

    #         # Calculate timing
    #         start_time = chunk_idx * self.CHUNK_DURATION 
    #         end_time = start_time + (
    #             self.final_duration if chunk_idx == self.full_chunks
    #             else self.CHUNK_DURATION
    #         )
    #         chunk_duration = end_time - start_time

    #         # Get grid layout
    #         rows, cols = self.get_track_layout()
            
    #         # Create grid configuration directly for GPU pipeline
    #         grid_config = [
    #             [{'empty': True} for _ in range(cols)] 
    #             for _ in range(rows)
    #         ]
            
    #         # For batched audio processing - collect all audio operations
    #         audio_operations = []
    #         all_notes = []
            
    #         # Process tracks using information from the path registry
    #         for track_idx, track in enumerate(self.midi_data['tracks']):
    #             # Find notes in this time chunk with proper overlap handling
    #             chunk_notes = [
    #                 note for note in track.get('notes', [])
    #                 if (start_time - self.OVERLAP_DURATION) <= float(note['time']) < end_time
    #                 or (float(note['time']) < start_time and float(note['time']) + float(note['duration']) > start_time)
    #             ]

    #             # Collect all notes for complexity analysis
    #             all_notes.extend(chunk_notes)
                
    #             if not chunk_notes:
    #                 continue
                    
    #             # Collect video and audio operations for the grid
    #             if is_drum_kit(track.get('instrument', {})):
    #                 # ADD MISSING CODE HERE
    #                 self._process_drum_chunk_gpu(track_idx, chunk_notes, grid_config, start_time, rows, cols)
    #                 drum_ops = self._collect_drum_operations(
    #                     track_idx, chunk_notes, start_time, rows, cols
    #                 )
    #                 # Add to grid config and collect audio operations
    #                 for op in drum_ops:  # FIXED: Using drum_ops instead of instrument_ops
    #                     # Update grid config
    #                     row, col = op['position']
    #                     if grid_config[row][col].get('empty', True):
    #                         grid_config[row][col] = {
    #                             'path': op['video_path'],
    #                             'start_time': 0,
    #                             'audio_duration': op.get('audio_duration', op['duration']),
    #                             'video_duration': op.get('video_duration', op['duration']),
    #                             'duration': op['duration'],  # Keep for backwards compatibility
    #                             'offset': op['offset'],
    #                             'empty': False
    #                         }
    #                     else:
    #                         # If cell has content, create a clips list
    #                         if 'clips' not in grid_config[row][col]:
    #                             grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
    #                         grid_config[row][col]['clips'].append({
    #                             'path': op['video_path'],
    #                             'start_time': 0,
    #                             'audio_duration': op.get('audio_duration', op['duration']),
    #                             'video_duration': op.get('video_duration', op['duration']),
    #                             'duration': op['duration'],  # Keep for backwards compatibility
    #                             'offset': op['offset']
    #                         })
    #                     # Add to audio operations
    #                     audio_operations.append(op)
    #             else:
    #                 self._process_instrument_chunk_gpu(track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols)
    #                 instrument_ops = self._collect_instrument_operations(
    #                     track_idx, track, chunk_notes, start_time, end_time, rows, cols
    #                 )
    #                 # Add to grid config and collect audio operations
    #                 for op in instrument_ops:
    #                     # Update grid config
    #                     row, col = op['position']
    #                     if grid_config[row][col].get('empty', True):
    #                         grid_config[row][col] = {
    #                             'path': op['video_path'],
    #                             'start_time': 0,
    #                             'audio_duration': op.get('audio_duration', op['duration']),
    #                             'video_duration': op.get('video_duration', op['duration']),
    #                             'duration': op['duration'],  # Keep for backward compatibility
    #                             'offset': op['offset'],
    #                             'empty': False
    #                         }
    #                     else:
    #                         # If cell has content, create a clips list
    #                         if 'clips' not in grid_config[row][col]:
    #                             grid_config[row][col]['clips'] = [dict(grid_config[row][col])]
    #                         grid_config[row][col]['clips'].append({
    #                             'path': op['video_path'],
    #                             'start_time': 0,
    #                             'audio_duration': op.get('audio_duration', op['duration']),
    #                             'video_duration': op.get('video_duration', op['duration']),
    #                             'duration': op['duration'],  # Keep for backward compatibility
    #                             'offset': op['offset']
    #                         })
    #                     # Add to audio operations
    #                     audio_operations.append(op)

    #         # NEW: Analyze frame complexity based on collected notes
    #         frame_complexity = self._estimate_frame_complexity(all_notes)
            
    #         # NEW: Get optimal encoding parameters based on complexity
    #         encoding_params = self._get_optimal_encoding_params(frame_complexity)

    #         # Use GPU pipeline directly
    #         chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
    #         pipeline = GPUPipelineProcessor(composer=self)

    #         # Check if the pipeline supports encoding parameters
    #         supports_encoding_params = hasattr(pipeline, 'set_encoding_params')
            
    #         # Configure pipeline with optimal parameters if supported
    #         if supports_encoding_params:
    #             pipeline.set_encoding_params(encoding_params)
        
            
    #         # First batch process all audio operations
    #         mixed_audio = None
    #         if audio_operations:
    #             audio_output = self.temp_dir / f"mixed_audio_{chunk_idx}.aac"
    #             mixed_audio = pipeline.batch_process_audio(
    #                 audio_operations, 
    #                 str(audio_output)
    #             )
            
    #         # Then process the video with the pre-mixed audio
    #         if supports_encoding_params:
    #             pipeline.process_chunk_pure_gpu(
    #                 grid_config=grid_config,
    #                 output_path=str(chunk_path),
    #                 fps=self.FRAME_RATE,
    #                 duration=chunk_duration,
    #                 audio_path=mixed_audio,
    #                 encoding_params=encoding_params
    #             )
    #         else:
    #             # Fallback to calling without encoding_params
    #             pipeline.process_chunk_pure_gpu(
    #                 grid_config=grid_config,
    #                 output_path=str(chunk_path),
    #                 fps=self.FRAME_RATE,
    #                 duration=chunk_duration,
    #                 audio_path=mixed_audio
    #             )

    #         if chunk_path.exists():
    #             result = str(chunk_path)
    #             with self.chunk_cache_lock:  # Minimal lock time
    #                 self.chunk_cache[cache_key] = result
    #             return result
                
    #     except Exception as e:
    #         logging.error(f"GPU chunk processing error: {e}")
    #         logging.error(f"Traceback: {traceback.format_exc()}")
    #         return None
        
    def _collect_drum_operations(self, track_idx, chunk_notes, start_time, rows, cols):
        """Collect drum operations with proper cross-chunk audio continuity"""
        operations = []
        
        for note in chunk_notes:
            try:
                midi_note = int(note['midi'])
                drum_name = DRUM_NOTES.get(midi_note)
                if not drum_name:
                    continue
                    
                drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                if drum_key not in self.grid_positions:
                    continue
                    
                pos_data = self.grid_positions[drum_key]
                row, col = int(pos_data['row']), int(pos_data['column'])
                
                # Get original timing information 
                original_time = float(note['time'])
                original_duration = float(note.get('duration', 0.5))
                note_end_time = original_time + original_duration
                
                # For notes that started in previous chunks
                if original_time < start_time:
                    time_offset = 0  # Always start at beginning of chunk
                    sample_offset = start_time - original_time  # How far into sample to start
                    
                    # Handle edge cases with minimum duration
                    if sample_offset >= original_duration:
                        # Use a very short tail for late continuations - prevents negative durations
                        audio_duration = 0.2
                        sample_offset = min(sample_offset, original_duration - 0.05)
                    else:
                        # Play the remaining portion of the sound
                        audio_duration = original_duration - sample_offset
                    
                    is_continuation = True
                else:
                    time_offset = original_time - start_time
                    audio_duration = original_duration
                    sample_offset = 0  # Start from beginning of sample
                    is_continuation = False
                    
                # For notes that extend into next chunk
                chunk_end = start_time + self.CHUNK_DURATION
                crosses_chunk = False
                if note_end_time > chunk_end:
                    crosses_chunk = True
                    fade_out_duration = 0.2  # Short crossfade
                else:
                    fade_out_duration = 0.05  # Normal fade
                    
                # Ensure we have valid durations
                audio_duration = max(0.1, audio_duration)  # Minimum reasonable duration
                
                # Find video path
                video_path = self._find_drum_video(drum_key)
                if not video_path:
                    continue

                # Add operation with enhanced parameters
                operations.append({
                    'video_path': video_path,
                    'offset': time_offset,
                    'audio_duration': audio_duration,
                    'sample_offset': sample_offset,
                    'video_duration': min(max(audio_duration, self.MIN_VIDEO_DURATION), 
                                        self.CHUNK_DURATION - time_offset),
                    'position': (row, col),
                    'volume': self.get_note_volume(float(note.get('velocity', 100)), is_drum=True, midi_note=midi_note),
                    'fade_in': 0.05 if not is_continuation else 0.0, 
                    'fade_out': fade_out_duration,
                    'is_continuation': is_continuation,
                    'duration': audio_duration,
                    'crosses_chunk': crosses_chunk
                })            
                    
            except Exception as e:
                logging.error(f"Error collecting drum operation: {e}")
                    
        return operations
    
    def _extract_audio_with_offset(self, video_path, output_path, duration, sample_offset=0, volume=1.0):
        """Extract audio from video with support for starting at an offset into the sample"""
        try:
            # Ensure positive duration to prevent FFmpeg errors
            if duration <= 0.1:
                logging.warning(f"Audio extraction with very short duration ({duration:.3f}s), using minimum")
                duration = 0.1
                
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),  # Ensure path is string
                '-vn',  # No video
                '-ss', f"{sample_offset:.3f}",  # Start at offset
                '-af', f"volume={volume:.1f}",
                '-acodec', 'pcm_s16le',
                '-t', f"{duration:.3f}",
                str(output_path)  # Ensure path is string
            ]
            
            logging.debug(f"Audio extraction command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Audio extraction error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            return False
        
    def _estimate_frame_complexity(self, notes):
        """Estimate visual complexity based on note density and characteristics"""
        # Count total notes
        note_count = len(notes)
        
        if note_count == 0:
            return 0.3  # Default low complexity for empty frames
        
        # Analyze note velocities and durations
        avg_velocity = sum(float(note.get('velocity', 100)) for note in notes) / max(1, note_count)
        
        # Calculate density (notes per second)
        time_span = max(float(note.get('time', 0)) + float(note.get('duration', 0.5)) 
                    for note in notes) - min(float(note.get('time', 0)) for note in notes)
        density = note_count / max(1.0, time_span)
        
        # Combine factors - higher velocity and density mean more complex frames
        complexity = min(1.0, (0.4 * (avg_velocity / 127)) + (0.6 * min(1.0, density / 10)))
        
        logging.info(f"Frame complexity: {complexity:.2f} (notes={note_count}, density={density:.1f}/sec)")
        return complexity

    def _collect_instrument_operations(self, track_idx, track, chunk_notes, start_time, end_time, rows, cols):
        """Collect instrument operations for batch processing"""
        operations = []
        instrument_name = normalize_instrument_name(track['instrument']['name'])
        
        # Get the actual track ID from the track data or MIDI data if available
        track_id = track.get('id', str(track_idx))  # Look for ID in track data
        
        # Find position in grid - try the actual ID first, then fallbacks
        possible_keys = [
            track_id,                    # Actual track ID from MIDI data
            f"{track_id}",               # String version
            f"track-{track_id}",         # With prefix
            f"track-{instrument_name}",  
            f"{track_idx}",              # Array index as fallback
            str(track_idx),
            instrument_name
        ]
        track_key = next((key for key in possible_keys if key in self.grid_positions), None)
        if track_key:
            logging.info(f"Found track {track_id}/{instrument_name} with key {track_key} in grid positions")
        else:
            logging.warning(f"No grid position for track {track_id}/{instrument_name}, tried keys: {possible_keys}")
        
        if track_key:
            pos_data = self.grid_positions[track_key]
            row, col = int(pos_data['row']), int(pos_data['column'])
        else:
            row = (track_idx % (rows-1)) + 1
            col = min(track_idx // (rows-1), cols-1)
            logging.warning(f"No grid position for {instrument_name}, using default: row={row}, col={col}")
        
        # Group notes by time to handle chords
        time_groups = {}
        for note in chunk_notes:
            time_pos = float(note['time'])
            if time_pos not in time_groups:
                time_groups[time_pos] = []
            time_groups[time_pos].append(note)
        
        # Sort time positions for look-ahead
        sorted_times = sorted(time_groups.keys())
        
        # Process each group of notes
        for i, time_pos in enumerate(sorted_times):
            notes = time_groups[time_pos]
            try:
                midi_notes = [int(note['midi']) for note in notes]
                
                # Find video path
                video_path = self._find_instrument_video(instrument_name, midi_notes[0])
                if not video_path:
                    continue
                    
                # Calculate timing
                time_offset = time_pos - start_time
                audio_duration = min(float(notes[0].get('duration', 0.5)), end_time - time_pos)

                # IMPORTANT: Add this check for negative durations
                if audio_duration <= 0:
                    logging.warning(f"Skipping instrument note with negative duration: {instrument_name}, " 
                                f"midi={midi_notes[0]}, time={time_pos:.2f}")
                    continue
                
                # Look ahead to next note
                next_note_time = float('inf')
                if i < len(sorted_times) - 1:
                    next_note_time = sorted_times[i+1]
                
                # Available time until next note or end of chunk
                available_time = min(next_note_time - time_pos, end_time - time_pos)
                
                # Use minimum duration only if we have enough space
                video_duration = min(max(audio_duration, self.MIN_VIDEO_DURATION), available_time)
                
                velocity = float(notes[0].get('velocity', 100))
                volume = self.get_note_volume(velocity, is_drum=False)
                
                operations.append({
                    'video_path': video_path,
                    'offset': time_offset,
                    'audio_duration': audio_duration,
                    'video_duration': video_duration,
                    'duration': audio_duration,
                    'position': (row, col),
                    'velocity': velocity,
                    'volume': volume  # Add volume parameter here
                })
                
            except Exception as e:
                logging.error(f"Error collecting instrument operation: {e}")
                    
        return operations


    # def create_composition(self):
    #     """Simplified composition approach with explicit timing"""
    #     try:
    #         # Calculate chunk lengths and store as instance attributes
    #         self.full_chunks, self.final_duration = self.calculate_chunk_lengths()
    #         logging.info(f"Set full_chunks={self.full_chunks}, final_duration={self.final_duration}")
            
    #         import time
    #         total_start = time.time()
            
    #         print(f"=== Starting Video Composition ({time.strftime('%H:%M:%S')}) ===")
            
    #         # 1. Process chunks sequentially - no locks needed
    #         chunk_files = []
    #         full_chunks, final_duration = self.calculate_chunk_lengths()
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
            
    #         print(f"Processing {total_chunks} chunks sequentially")
            
    #         for chunk_idx in range(total_chunks):
    #             chunk_start = time.time()
    #             print(f"\nProcessing chunk {chunk_idx+1}/{total_chunks} (Time: {time.strftime('%H:%M:%S')})")
                
    #             # Process single chunk
    #             chunk_path = self._process_chunk(chunk_idx)
    #             if chunk_path:
    #                 chunk_files.append(chunk_path)
    #                 print(f"Chunk {chunk_idx+1} completed in {time.time() - chunk_start:.1f}s")
    #             else:
    #                 print(f"Chunk {chunk_idx+1} failed")
                    
    #         # 2. Combine chunks using direct ffmpeg concatenation instead of MoviePy
    #         if chunk_files:
    #             print(f"\n=== Combining {len(chunk_files)} chunks ===")
    #             combine_start = time.time()
                
    #             # Create concat file
    #             concat_file = self.temp_dir / "concat.txt"
    #             with open(concat_file, 'w') as f:
    #                 for chunk in sorted(chunk_files):
    #                     f.write(f"file '{chunk}'\n")
                
    #             ffmpeg_cmd = [
    #                 'ffmpeg', '-y',
    #                 '-hwaccel', 'cuda',
    #                 '-hwaccel_device', '0',
    #                 '-f', 'concat',
    #                 '-safe', '0',
    #                 '-i', str(concat_file),
    #                 # Copy streams directly instead of re-encoding when possible
    #                 '-c:v', 'copy',
    #                 '-c:a', 'copy',
    #                 # Add sync options to fix audio sync
    #                 '-vsync', 'cfr',
    #                 '-async', '1',
    #                 str(self.output_path)
    #             ]
                
    #             print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
    #             result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
    #             if result.returncode == 0:
    #                 print(f"Combination successful in {time.time() - combine_start:.1f}s")
    #                 return str(self.output_path)
    #             else:
    #                 print(f"FFmpeg error: {result.stderr}")
                    
    #                 # Fallback to CPU if GPU concat fails
    #                 print("Falling back to CPU for final concat...")
    #                 ffmpeg_cmd = [
    #                     'ffmpeg', '-y',
    #                     '-f', 'concat',
    #                     '-safe', '0',
    #                     '-i', str(concat_file),
    #                     '-c:v', 'libx264',
    #                     '-preset', 'medium',
    #                     '-crf', '23',
    #                     '-pix_fmt', 'yuv420p',
    #                     '-c:a', 'aac',
    #                     '-b:a', '192k',
    #                     str(self.output_path)
    #                 ]
                    
    #                 result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    #                 if result.returncode == 0:
    #                     print(f"CPU combination successful in {time.time() - combine_start:.1f}s")
    #                     return str(self.output_path)
    #                 else:
    #                     print(f"CPU FFmpeg error: {result.stderr}")
    #                     return None
            
    #         print(f"Total time: {time.time() - total_start:.1f}s")
    #         return None
            
    #     except Exception as e:
    #         import traceback
    #         print(f"Composition error: {str(e)}")
    #         traceback.print_exc()
    #         return None

    def create_composition(self):
        """Optimized composition approach with parallel chunk processing"""
        try:
            # Calculate chunk lengths and store as instance attributes
            self.full_chunks, self.final_duration = self.calculate_chunk_lengths()
            logging.info(f"Set full_chunks={self.full_chunks}, final_duration={self.final_duration}")

            # Add this line to set total_chunks attribute
            self.total_chunks = self.full_chunks + (1 if self.final_duration > 0 else 0)

            import time
            total_start = time.time()
            
            print(f"=== Starting Video Composition ({time.strftime('%H:%M:%S')}) ===")
            
            # Use parallel chunk processing instead of sequential
            print(f"Processing chunks in parallel with adaptive resource allocation")
            chunk_files = self.process_chunks_parallel()
            self._verify_and_fix_chunks()
            
            # Rest of method remains the same (combining chunks)
            if chunk_files:
                print(f"\n=== Combining {len(chunk_files)} chunks ===")
                combine_start = time.time()
                
                # Create concat file
                concat_file = self.temp_dir / "concat.txt"
                with open(concat_file, 'w') as f:
                    for chunk in sorted(chunk_files, key=lambda x: int(re.search(r'chunk_(\d+)\.mp4$', x).group(1))):
                        f.write(f"file '{chunk}'\n")
                
                # The rest of your ffmpeg code stays the same
                # ffmpeg_cmd = [
                #     'ffmpeg', '-y',
                #     '-hwaccel', 'cuda',
                #     '-hwaccel_device', '0',
                #     '-f', 'concat',
                #     '-safe', '0',
                #     '-i', str(concat_file),
                #     # Copy streams directly instead of re-encoding when possible
                #     '-c:v', 'copy',
                #     '-c:a', 'copy',
                #     # Add sync options to fix audio sync
                #     '-vsync', 'cfr',
                #     '-async', '1',
                #     str(self.output_path)
                # ]
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', '0',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    # Re-encode with strict sync parameters instead of copy
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-vsync', 'cfr',  # Constant frame rate
                    '-af', 'aresample=async=1000',  # Audio resampling for better sync
                    '-avoid_negative_ts', '1',
                    '-max_muxing_queue_size', '9999',
                    str(self.output_path)
                ]
                
                # Existing ffmpeg execution code stays the same
                print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Combination successful in {time.time() - combine_start:.1f}s")
                    return str(self.output_path)
                else:
                    # Existing fallback code stays the same
                    print(f"FFmpeg error: {result.stderr}")
                    # Fallback to CPU code remains unchanged...
            
            print(f"Total time: {time.time() - total_start:.1f}s")
            return None
            
        except Exception as e:
            import traceback
            print(f"Composition error: {str(e)}")
            traceback.print_exc()
            return None

        

def compose_from_processor_output(processor_result, output_path):
    try:
        # Create a Profile object
        profiler = cProfile.Profile()
        
        # Start profiling
        profiler.enable()

        base_dir = processor_result['processed_videos_dir']
        logging.info(f"Using base directory: {base_dir}")
        logging.info("=== Processing Grid Arrangement ===")
        logging.info(f"Grid arrangement in processor result: {processor_result.get('tracks', {}).get('gridArrangement')}")

        logging.info("=== Processing Grid Arrangement ===")
        logging.info(f"Processor result tracks: {processor_result['tracks']}")
        logging.info(f"Grid arrangement in tracks: {processor_result['tracks'].get('gridArrangement')}")
        
        # Store validated tracks
        validated_tracks = {}
        
        if 'tracks' in processor_result['processed_files']:
            for instrument, data in processor_result['processed_files']['tracks'].items():
                try:
                    track_idx = data.get('track_idx', 0)
                    
                    # Keep full instrument name, just remove track index if present
                    instrument_parts = instrument.split('_')
                    if instrument_parts[-1].isdigit():
                        instrument_name = '_'.join(instrument_parts[:-1])
                    else:
                        instrument_name = instrument
                    
                    # Construct track path preserving full instrument name
                    track_path = os.path.join(
                        base_dir, 
                        f"track_{track_idx}_{instrument_name}"
                    )
                    
                    logging.info(f"Checking track {track_idx}: {instrument_name}")
                    logging.info(f"Track path: {track_path}")
                    
                    if os.path.exists(track_path):
                        validated_tracks[instrument] = {
                            'base_path': track_path,
                            'instrument_name': instrument_name,  # Store full name
                            'notes': data.get('notes', {})
                        }
                        
                except Exception as e:
                    logging.error(f"Error processing track {instrument}: {str(e)}")
                    continue

        composer = VideoComposer(base_dir, processor_result['tracks'], output_path)
        composer.tracks = validated_tracks
        result = composer.create_composition()

           # Stop profiling
        profiler.disable()
        
        # Use StringIO instead of direct file writes
        import io
        from pstats import SortKey
        
        # Capture stats output to string
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(20)
        
        # Write the captured string data to file
        profile_path = Path(base_dir) / 'composition_profile.log'
        with open(profile_path, 'w') as f:
            f.write(s.getvalue())
        
        # Log a simple reference message
        logging.info("\n=== Performance Profile ===")
        logging.info(f"Full profile data written to {profile_path}")

        return result
    except Exception as e:
        logging.error(f"Error in video composition: {str(e)}")
        raise
 