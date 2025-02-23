import os
import gc
import logging
import traceback
import os.path
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import shutil
import subprocess
import time
import cProfile
import pstats
from pstats import SortKey

# Third-party imports
import numpy as np
import cv2
from moviepy import (
    VideoFileClip,
    clips_array,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import weakref
import threading
from contextlib import contextmanager

from utils import normalize_instrument_name, midi_to_note
from drum_utils import (
    DRUM_NOTES,
    process_drum_track,
    get_drum_groups,
    get_drum_name,
    is_drum_kit
)

import threading
from queue import Queue

class EncoderQueue:
    def __init__(self, max_concurrent=2):
        self.queue = Queue()
        self.semaphore = threading.Semaphore(max_concurrent)
        
    def encode(self, ffmpeg_command):
        with self.semaphore:
            logging.info(f"EncoderQueue: Running command: {' '.join(ffmpeg_command)}")
            try:
                gc.collect()  # Force garbage collection
                return subprocess.run(ffmpeg_command, capture_output=True, text=True)
            except Exception as e:
                logging.error(f"EncoderQueue: Error executing command: {str(e)}")
                raise

encoder_queue = EncoderQueue(max_concurrent=6) 

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_composition.log', mode='w')
    ]
)
class VideoComposer:

    FRAME_RATE = 30
    CHUNK_DURATION = 4
    OVERLAP_DURATION = 0.3
    CROSSFADE_DURATION = 0.25
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
            self.output_path = output_path
            self.midi_data = midi_data
            self._setup_paths(processed_videos_dir, output_path)
            self.midi_data = midi_data
            self._process_midi_data(midi_data)
            self._setup_track_configuration()
            self.clip_manager = ClipManager()
            self.chuck_cache = {}
            self.max_cached_chunks = 4
            self.encoder_params = {
                'codec': 'h264_nvenc',
                'preset': 'p7',
                'ffmpeg_params': [
                    "-vsync", "cfr",
                    "-c:v", "h264_nvenc",
                    "-preset", "p7",
                    "-tune", "hq",
                    "-rc", "vbr_hq",
                    "-cq", "20",
                    "-b:v", "5M",
                    "-maxrate", "10M",
                    "-bufsize", "10M",
                    "-profile:v", "high",
                    "-gpu_select", "0"
                ]
            }
             # Log track information
            logging.info(f"Regular tracks: {len(self.tracks)}")
            logging.info(f"Drum tracks: {len(self.drum_tracks)}")
            for track in self.drum_tracks:
                logging.info(f"Drum track found: {track.get('instrument', {}).get('name')}")
                
        except Exception as e:
            logging.error(f"VideoComposer init error: {str(e)}")
            raise
  
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
        
   
    def _calculate_default_layout(self):
        """Calculate default grid layout"""
        rows = math.ceil(len(self.tracks) / 4)
        cols = min(4, len(self.tracks))
        return rows, cols

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
            

    def preprocess_video(self, video_path, output_path, target_size=None):
        """Convert uploaded video once to optimal format and size"""
        try:
            if target_size is None:
                rows, cols = self._calculate_default_layout()
                target_w = 1920//cols
                target_h = 1080//rows
            else:
                target_w, target_h = map(int, target_size.split('x'))

            # Fix the scale and pad filter syntax
            scale_filter = (
                f'scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,'
                f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black'
            )

            convert_cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', scale_filter,
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            logging.info(f"Running FFmpeg command: {' '.join(convert_cmd)}")
            result = encoder_queue.encode(convert_cmd)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg preprocessing failed: {result.stderr}")
                
            # Verify output file
            if not Path(output_path).exists() or Path(output_path).stat().st_size < 1024:
                raise Exception(f"Output file invalid or too small: {output_path}")
                
            return result

        except Exception as e:
            logging.error(f"Video preprocessing error: {str(e)}")
            raise

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
        
    def has_valid_notes(self, track):
        """Check if track has any valid notes"""
        notes = track.get('notes', [])
        return len(notes) > 0

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
        
    def get_note_volume(self, velocity, is_drum=False):
        """Calculate volume from MIDI velocity with better scaling"""
        # Normalize velocity (0-1)
        normalized_velocity = float(velocity) / 127.0
        
        # Set base multipliers
        multipliers = {
            'drums': 0.4,      # Drums at 40% 
            'instruments': 1.2  # Instruments at full volume
        }
        
        # Calculate volume with better minimum
        base_volume = normalized_velocity * 1.5  # Boost overall volume
        volume = max(0.3, base_volume * multipliers['drums' if is_drum else 'instruments'])
        
        logging.info(f"Volume calculation: velocity={velocity}, normalized={normalized_velocity:.2f}, final={volume:.2f}")
        return volume
    
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

    def _process_drum_chunk(self, track_idx, chunk_notes, grid, active_clips, start_time):
        drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
        if not drum_dir.exists():
            return

        drum_notes = {}
        for note in chunk_notes:
            midi_note = int(note['midi'])
            drum_name = DRUM_NOTES.get(midi_note)
            if drum_name:
                drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                if drum_key not in drum_notes:
                    drum_notes[drum_key] = []
                drum_notes[drum_key].append(note)

        for drum_key, notes in drum_notes.items():
            position_key = drum_key
            if position_key in self.grid_positions:
                pos_data = self.grid_positions[position_key]
                row = int(pos_data['row'])
                col = int(pos_data['column'])
                self._add_drum_clips(drum_dir, drum_key, notes, grid, row, col, active_clips, start_time)

    def _process_instrument_chunk(self, track_idx, track, chunk_notes, grid, active_clips, start_time, end_time):
        instrument_name = normalize_instrument_name(track['instrument']['name'])
        track_id = str(track_idx)
        
        # Get position from grid positions like drums do
        if track_id not in self.grid_positions:
            return
            
        pos_data = self.grid_positions[track_id]
        row = int(pos_data['row'])
        col = int(pos_data['column'])
        
        notes_dir = self.processed_videos_dir / f"{instrument_name}_notes"
        if not notes_dir.exists():
            return
            
        # Process each note and add directly to grid like drums
        for note in chunk_notes:
            midi_note = int(float(note['midi']))
            note_file = notes_dir / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
            
            if note_file.exists():
                try:
                    clip = VideoFileClip(str(note_file))
                    active_clips.append(clip)
                    
                    time = float(note['time']) - start_time
                    duration = min(float(note['duration']), clip.duration)
                    clip = (clip
                        .subclipped(0, duration)
                        .with_start(time))
                    
                    # Add to grid exactly like drums do
                    if isinstance(grid[row][col], ColorClip):
                        grid[row][col] = clip
                    else:
                        existing = grid[row][col]
                        grid[row][col] = CompositeVideoClip([existing, clip])
                        
                    logging.info(f"Added {instrument_name} note {midi_note} at [{row}][{col}] t={time}")
                    
                except Exception as e:
                    logging.error(f"Error creating clip: {str(e)}")
                    continue
    

    def _create_note_clip(self, note_file, note, start_time):
        """Creates a video clip for a single note with optimized resource management"""
        if not note_file.exists():
            logging.warning(f"Note file not found: {note_file}")
            return None
            
        try:
            with self.clip_manager.managed_clip(VideoFileClip(str(note_file))) as clip:
                # Apply volume based on note velocity
                velocity = float(note.get('velocity', 100))
                volume = self.get_note_volume(velocity, is_drum=False)
                
                # Set timing - calculate before subclipping to reduce memory operations
                time = float(note['time']) - start_time
                duration = min(float(note['duration']), clip.duration)
                
                # Single subclip operation instead of chaining
                processed_clip = (clip
                    .subclipped(0, duration)
                    .with_start(time))
                
                logging.info(f"Created clip for note at time {time:.2f}, duration {duration:.2f}")
                return processed_clip
            
        except Exception as e:
            logging.error(f"Error creating note clip: {e}")
            gc.collect()  # Force garbage collection after error
            return None
        
    
    def _chunk_generator(self, total_chunks, batch_size=3):
        """Generate batches of chunk indices"""
        for i in range(0, total_chunks, batch_size):
            yield range(i, min(i + batch_size, total_chunks))

    def _process_chunk(self, chunk_idx):
        try:
            start_time = chunk_idx * self.CHUNK_DURATION
            end_time = start_time + (
                self.final_duration if chunk_idx == self.full_chunks 
                else self.CHUNK_DURATION
            )
            chunk_duration = end_time - start_time
            
            logging.info(f"\nProcessing chunk {chunk_idx}")
            logging.info(f"Time range: {start_time:.2f} to {end_time:.2f}")
            logging.info(f"Chunk duration: {chunk_duration}")
            
            # Initialize grid with explicit durations
            rows, cols = self.get_track_layout()
            grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                            color=(0,0,0),
                            duration=chunk_duration)  # Set explicit duration
                    for _ in range(cols)] 
                    for _ in range(rows)]
            chunk_clips = []
            
            # Process tracks within chunk
            for track_idx, track in enumerate(self.midi_data['tracks']):
                chunk_notes = [
                    note for note in track.get('notes', [])
                    if start_time <= float(note['time']) < end_time
                ]
                
                if chunk_notes:
                    logging.info(f"Found {len(chunk_notes)} notes for track {track_idx}")
                    if is_drum_kit(track.get('instrument', {})):
                        self._process_drum_chunk(track_idx, chunk_notes, grid, chunk_clips, start_time)
                    else:
                        self._process_instrument_chunk(
                            track_idx, track, chunk_notes, grid, chunk_clips,
                            start_time, end_time
                        )

            # Create chunk composition with explicit duration
            chunk = clips_array(grid).with_duration(chunk_duration)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
            logging.info(f"Writing chunk {chunk_idx} with duration {chunk_duration}")
            chunk.write_videofile(
                str(chunk_path),
                fps=self.FRAME_RATE,
                codec='h264_nvenc',
                preset='p7', 
                ffmpeg_params=["-vsync", "cfr", "-c:v", "h264_nvenc", 
                            "-b:v", "5M", "-maxrate", "8M"]
            )

            # Cleanup chunk clips
            for clip in chunk_clips:
                try:
                    clip.close()
                except:
                    pass

            return str(chunk_path)
            
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_idx}: {str(e)}")
            return None
        
    def _cleanup_old_chunks(self):
        """Clean up old chunks to manage memory"""
        gc.collect()  # Force garbage collection
        if hasattr(self, 'chunk_cache'):
            while len(self.chunk_cache) > self.max_cached_chunks:
                oldest = next(iter(self.chunk_cache))
                del self.chunk_cache[oldest]
        
    def create_composition(self):
        """Create final video composition by processing chunks in parallel"""
        profiler = cProfile.Profile()
        profiler.enable()

        active_clips = []
        chunk_files = []
        
        try:
            self.full_chunks, self.final_duration = self.calculate_chunk_lengths()
            total_chunks = self.full_chunks + (1 if self.final_duration > 0 else 0)
            
            logging.info(f"Creating composition with {total_chunks} chunks")
            logging.info(f"Grid positions: {self.grid_positions}")
            
            # Process chunks
            chunk_files = []
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 6)) as executor:
                futures = [executor.submit(self._process_chunk, i) for i in range(total_chunks)]
                
                for future in as_completed(futures):
                    try:
                        chunk_path = future.result()
                        if chunk_path:
                            chunk_files.append(chunk_path)
                    except Exception as e:
                        logging.error(f"Chunk processing failed: {e}")

            if chunk_files:
                # Sort chunks by index
                chunk_files.sort(key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
                return self._combine_chunks(chunk_files)

            logging.error("No chunks were created")
            return None

        except Exception as e:
            logging.error(f"Composition error: {e}")
            traceback.print_exc()
            return None

        finally:
            # Cleanup any remaining clips
            for clip in active_clips:
                try:
                    clip.close()
                except:
                    pass
               
    def _combine_chunks(self, chunk_files):
        """Combine chunks with precise timing"""
        try:
            clips = []
            first_clip = VideoFileClip(chunk_files[0])
            target_fps = first_clip.fps
            
            total_duration = 0
            for i, chunk_file in enumerate(chunk_files):
                clip = VideoFileClip(chunk_file)
                
                # Trim to exact content duration
                if clip.duration > self.CHUNK_DURATION:
                    clip = clip.subclipped(0, self.CHUNK_DURATION)
                    
                # Only add crossfade between chunks, not at start
                if i > 0:
                    clip = clip.with_start(total_duration - self.CROSSFADE_DURATION)
                else:
                    clip = clip.with_start(0)
                    
                total_duration = clip.end
                clips.append(clip)

            final = concatenate_videoclips(
                clips,
                method="compose",
                padding=-self.CROSSFADE_DURATION
            )
            
            # Write with precise duration
            final.write_videofile(
                str(self.output_path),
                fps=target_fps,
                codec='h264_nvenc',
                preset='p7',
                ffmpeg_params=[
                    "-vsync", "cfr",
                    "-c:v", "h264_nvenc",
                    "-b:v", "5M",
                    "-maxrate", "8M",
                    "-bufsize", "8M",
                    "-rc", "vbr",
                    "-qmin", "0",
                    "-qmax", "51",
                    "-profile:v", "high",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"
                ]
            )
            return self.output_path
            
        except Exception as e:
            logging.error(f"Concatenation error: {e}")
            return None
            
        finally:
            # Cleanup all clips including final
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            try:
                if 'final' in locals():
                    final.close()
            except:
                pass
    

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
 