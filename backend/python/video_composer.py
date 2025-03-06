import os
import gc
import subprocess
import sys
import logging
import traceback
import os.path
from pathlib import Path
import math
import shutil
import tempfile
import cProfile
import pstats
import threading
from threading import RLock
import asyncio
from pstats import SortKey
from tqdm import tqdm
import time
from contextlib import contextmanager

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
    def __init__(self):
        self.streams = []
        self._init_streams()
    
    def _init_streams(self):
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(4)]

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
class VideoComposer:

    FRAME_RATE = 30
    CHUNK_DURATION = 4
    OVERLAP_DURATION = 1
    CROSSFADE_DURATION = 0.5
    MIN_VIDEO_DURATION = 1.0
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
            self.gpu_manager = GPUManager()
            self.output_path = output_path
            self.midi_data = midi_data
            self._setup_paths(processed_videos_dir, output_path)
            self.midi_data = midi_data
            self._process_midi_data(midi_data)
            self._setup_track_configuration()
            self.clip_manager = ClipManager()
            self.chunk_cache = {}
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

    def _process_midi_data(self, midi_data):
        """Process MIDI data and organize tracks"""
        if not isinstance(midi_data, dict):
            raise ValueError(f"Expected dict for midi_data, got {type(midi_data)}")
            
        if 'tracks' not in midi_data:
            raise ValueError("Missing 'tracks' in midi_data")

        tracks = midi_data['tracks']
        self.tracks = {}
        self.drum_tracks = []
        self.regular_tracks = []
        
        logging.info("\n=== Processing Tracks ===")
        
        # List all preprocessed files
        upload_files = list(self.uploads_dir.glob('processed_*.mp4'))
        logging.info(f"\nPreprocessed files in ({self.uploads_dir}):")
        for file in upload_files:
            logging.info(f"Found preprocessed file: {file.name}")

        for idx, track in enumerate(tracks if isinstance(tracks, list) else tracks.values()):
            track_id = str(idx)
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
                logging.info(f"\nProcessing drum track {track_id}")
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
                            shutil.copy2(str(file), str(dest_file))
                            logging.info(f"Copied preprocessed drum file: {file.name} -> {dest_file}")
                            break
                    else:
                        logging.warning(f"No preprocessed file found for {normalized_name}")
                
                self.drum_tracks.append(normalized_track)
                
            else:
                self.regular_tracks.append(normalized_track)
                self.tracks[track_id] = normalized_track

        logging.info(f"\nProcessed {len(self.tracks)} regular tracks and {len(self.drum_tracks)} drum tracks")


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

    def calculate_chunk_lengths(self):
        """Calculate chunk lengths for composition"""
        try:
            # Get all track notes in one list
            all_notes = []
            
            # Add regular track notes
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
        """Process chunks in parallel with adaptive workload distribution"""
        chunk_files = []
        total_chunks = self.full_chunks + (1 if self.final_duration > 0 else 0)
        
        # Get optimized grouping of chunks
        simple_chunks, complex_chunks = self._classify_chunks_by_complexity()
        
        logging.info(f"Classified chunks: {len(simple_chunks)} simple, {len(complex_chunks)} complex")
        
        # Determine optimal worker count based on system resources
        metrics = get_system_metrics()
        max_workers = min(os.cpu_count(), 4)  # Default max
        
        # Adjust workers based on memory pressure
        if metrics['memory_percent'] > 80:
            max_workers = 2  # Reduce workers when low memory
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process complex chunks first (they need more time)
            complex_futures = {executor.submit(self._process_chunk, idx): idx for idx in complex_chunks}
            
            # Process result as they complete
            for future in as_completed(complex_futures):
                chunk_idx = complex_futures[future]
                try:
                    result = future.result()
                    if result:
                        chunk_files.append(result)
                        logging.info(f"Complex chunk {chunk_idx} completed")
                    else:
                        logging.error(f"Complex chunk {chunk_idx} failed")
                except Exception as e:
                    logging.error(f"Error processing complex chunk {chunk_idx}: {e}")
            
            # Then process simple chunks
            simple_futures = {executor.submit(self._process_chunk, idx): idx for idx in simple_chunks}
            
            for future in as_completed(simple_futures):
                chunk_idx = simple_futures[future]
                try:
                    result = future.result()
                    if result:
                        chunk_files.append(result)
                        logging.info(f"Simple chunk {chunk_idx} completed")
                    else:
                        logging.error(f"Simple chunk {chunk_idx} failed")
                except Exception as e:
                    logging.error(f"Error processing simple chunk {chunk_idx}: {e}")
        
        # Sort by chunk index to maintain order
        chunk_files.sort()
        return chunk_files
    
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
    
    

    def _process_instrument_chunk_gpu(self, track_idx, track, chunk_notes, grid_config, start_time, end_time, rows, cols):
        # Get instrument name for lookup
        instrument_name = normalize_instrument_name(track['instrument']['name'])
        
        # Try multiple possible ID formats
        possible_keys = [
            f"track-{track_idx}",            # Format with prefix
            f"track-{instrument_name}",      # Name with prefix
            f"{track_idx}",                  # Bare index (what frontend actually uses)
            str(track_idx),                  # Ensure string comparison
            f"{instrument_name}"             # Bare name
        ]
        
        # Find the first key that exists in grid_positions
        track_key = next((key for key in possible_keys if key in self.grid_positions), None)
        
        if track_key:
            pos_data = self.grid_positions[track_key]
            row, col = int(pos_data['row']), int(pos_data['column'])
            logging.info(f"Found grid position for {instrument_name}: row={row}, col={col}")
        else:
            # Fallback to default
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
                # Get instrument info
                instrument_name = track.get('instrument', {}).get('name', 'piano')
                midi_notes = [int(note['midi']) for note in notes]
                
                # Find video path for the instrument note
                video_path = self._find_instrument_video(instrument_name, midi_notes[0])
                if not video_path:
                    continue
                    
                # Calculate timing
                time = time_pos - start_time
                audio_duration = min(float(notes[0].get('duration', 0.5)), end_time - time_pos)
                
                # Look ahead to next note on same instrument to avoid overlaps
                next_note_time = float('inf')
                if i < len(sorted_times) - 1:
                    next_note_time = sorted_times[i+1]
                
                # Available time until next note or end of chunk
                available_time = min(next_note_time - time_pos, end_time - time_pos)
                
                # Use minimum duration only if we have enough space
                video_duration = min(max(audio_duration, self.MIN_VIDEO_DURATION), available_time)
                
                # Add to grid config
                if grid_config[row][col].get('empty', True):
                    grid_config[row][col] = {
                        'path': video_path,
                        'start_time': 0,
                        'audio_duration': audio_duration,  # Original duration for audio
                        'video_duration': video_duration,  # Extended when safe
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
                    
                logging.info(f"Added {instrument_name} note {midi_notes[0]} at [{row}][{col}] t={time}, audio_dur={audio_duration:.2f}, video_dur={video_duration:.2f}")
                    
            except Exception as e:
                logging.error(f"Error processing instrument note: {e}")
                continue

    def _find_drum_video(self, drum_key):
        """Find video for drum key using registry"""
        path = PathRegistry.get_instance().get_drum_path(drum_key)
        if path:
            logging.info(f"Found drum video: {path}")
            return path
        
        logging.warning(f"No video found for drum: {drum_key}")
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


    def _process_chunk(self, chunk_idx):
        """Process chunk using pure GPU pipeline with batched audio processing"""
        try:                    
            # Quick cache check with minimal lock time
            cache_key = f"chunk_{chunk_idx}"
            with self.chunk_cache_lock:
                if cache_key in self.chunk_cache:
                    return self.chunk_cache[cache_key]

            # Calculate timing
            start_time = chunk_idx * self.CHUNK_DURATION 
            end_time = start_time + (
                self.final_duration if chunk_idx == self.full_chunks
                else self.CHUNK_DURATION
            )
            chunk_duration = end_time - start_time

            # Get grid layout
            rows, cols = self.get_track_layout()
            
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
                
                if not chunk_notes:
                    continue
                    
                # Collect video and audio operations for the grid
                if is_drum_kit(track.get('instrument', {})):
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
            pipeline = GPUPipelineProcessor()

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
                return result
                
        except Exception as e:
            logging.error(f"GPU chunk processing error: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
        
    def _collect_drum_operations(self, track_idx, chunk_notes, start_time, rows, cols):
        """Collect drum operations for batch processing"""
        operations = []
        
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
                time_offset = float(note['time']) - start_time
                audio_duration = float(note.get('duration', 0.5))
                
                # Use minimum video duration
                video_duration = max(audio_duration, self.MIN_VIDEO_DURATION)
                
                # Find video path
                video_path = self._find_drum_video(drum_key)
                if not video_path:
                    continue

                # Calculate volume from velocity using your existing method
                velocity = float(note.get('velocity', 100))
                volume = self.get_note_volume(velocity, is_drum=True)
                
                # Add operation with volume parameter
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
                logging.error(f"Error collecting drum operation: {e}")
                    
        return operations
    
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
        
        # Find position in grid
        possible_keys = [
            f"track-{track_idx}", f"track-{instrument_name}", 
            f"{track_idx}", str(track_idx), f"{instrument_name}"
        ]
        track_key = next((key for key in possible_keys if key in self.grid_positions), None)
        
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
            
            import time
            total_start = time.time()
            
            print(f"=== Starting Video Composition ({time.strftime('%H:%M:%S')}) ===")
            
            # Use parallel chunk processing instead of sequential
            print(f"Processing chunks in parallel with adaptive resource allocation")
            chunk_files = self.process_chunks_parallel()
            
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
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', '0',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    # Copy streams directly instead of re-encoding when possible
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    # Add sync options to fix audio sync
                    '-vsync', 'cfr',
                    '-async', '1',
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
        
        # Create stats object and sort by cumulative time
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        
        # Save profiling results to a log file
        profile_log = Path(base_dir) / 'composition_profile.log'
        with open(profile_log, 'w') as log:
            stats.stream = log
            stats.print_stats()
            
        # Also log top 20 time-consuming functions
        logging.info("\n=== Performance Profile ===")
        logging.info("Top 20 time-consuming operations:")
        stats.sort_stats(SortKey.TIME).print_stats(20)

        return result

    except Exception as e:
        logging.error(f"Error in video composition: {str(e)}")
        raise
 