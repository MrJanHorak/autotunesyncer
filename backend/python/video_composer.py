import os
import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import shutil
import subprocess

# Third-party imports
import numpy as np
import cv2 # type: ignore
from moviepy.editor import ( # type: ignore
    VideoFileClip,
    clips_array,
    CompositeVideoClip,
    ColorClip,
    concatenate_videoclips
)

from utils import normalize_instrument_name, midi_to_note
from drum_utils import DRUM_NOTES, process_drum_track, get_drum_groups, get_drum_name, is_drum_kit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_composition.log', mode='w')
    ]
)

class VideoComposer:
    # Class-level constants
    FRAME_RATE = 30
    CHUNK_DURATION = 10
    OVERLAP_DURATION = 0.3
    CROSSFADE_DURATION = 0.25
    VOLUME_MULTIPLIERS = {
        'drums': 0.2,
        'instruments': 1.5
    }

    def __init__(self, processed_videos_dir, midi_data, output_path):
        """Initialize VideoComposer with proper path handling"""
        try:
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
        if not isinstance(midi_data, dict):
            raise ValueError(f"Expected dict for midi_data, got {type(midi_data)}")
            
        if 'tracks' not in midi_data:
            raise ValueError("Missing 'tracks' in midi_data")

        tracks = midi_data['tracks']
        self.tracks = {}
        self.drum_tracks = []
        self.regular_tracks = []
        
        # First, process and copy drum videos
        logging.info("\n=== Processing Drum Videos ===")
        
        # List all upload files once
        upload_files = list(self.uploads_dir.glob('*.mp4'))
        logging.info(f"\nUploads directory ({self.uploads_dir}) contents:")
        for file in upload_files:
            logging.info(f"Found uploaded file: {file.name}")

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
                logging.info(f"\nProcessing drum track {track_id}: {normalized_track.get('instrument', {}).get('name')}")
                
                # Create drum directory
                drum_dir = self.processed_videos_dir / f"track_{idx}_drums"
                drum_dir.mkdir(exist_ok=True)
                logging.info(f"Created drum directory: {drum_dir}")
                
                # Get unique drum types needed from notes
                needed_drums = set()
                for note in normalized_track.get('notes', []):
                    midi_note = note.get('midi')
                    drum_name = DRUM_NOTES.get(midi_note)
                    if drum_name:
                        needed_drums.add((midi_note, drum_name))
                        logging.info(f"Found drum note: {midi_note} -> {drum_name}")

                # Search for needed drum files
                for midi_note, drum_name in needed_drums:
                    normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
                    source_file = None
                    
                    # Search uploads for matching drum file
                    for file in upload_files:
                        if normalized_name in file.name.lower():
                            source_file = file
                            logging.info(f"Found matching drum file: {file.name}")
                            break

                    if source_file:
                        dest_file = drum_dir / f"{normalized_name}.mp4"
                        if not dest_file.exists():
                            try:
                                # Extract audio first
                                temp_audio = self.temp_dir / f"temp_{normalized_name}.wav"
                                cmd = [
                                    'ffmpeg', '-y',
                                    '-i', str(source_file),
                                    '-vn', '-acodec', 'pcm_s16le',
                                    '-ar', '44100', '-ac', '1',
                                    str(temp_audio)
                                ]
                                subprocess.run(cmd, check=True)
                                
                                # Combine video with processed audio
                                cmd = [
                                    'ffmpeg', '-y',
                                    '-i', str(source_file),
                                    '-i', str(temp_audio),
                                    '-map', '0:v',
                                    '-map', '1:a',
                                    '-c:v', 'h264',
                                    '-c:a', 'aac',
                                    '-movflags', '+faststart',
                                    str(dest_file)
                                ]
                                subprocess.run(cmd, check=True)
                                
                                # Cleanup temp file
                                temp_audio.unlink()
                                
                                logging.info(f"Processed and copied drum file: {source_file.name} -> {dest_file.name}")
                            except Exception as e:
                                logging.error(f"Error processing drum file {normalized_name}: {e}")
                                continue
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
        

    def _setup_track_configuration(self):
        """Setup track-specific configuration and grid layout"""
        try:
            valid_tracks = []
            self.grid_positions = {}  # Initialize grid positions
            total_slots = 0
            
            # First handle regular tracks
            for track_idx, track in enumerate(self.tracks.values()):
                if track.get('notes'):
                    valid_tracks.append((track_idx, track))
                    self.grid_positions[f"track_{track_idx}"] = total_slots
                    total_slots += 1
            
            # Handle drum tracks
            logging.info("\n=== DRUM TRACK CONFIGURATION ===")
            for track_idx, track in enumerate(self.drum_tracks):
                logging.info(f"\nDrum Track {track_idx}:")
                
                if track.get('notes'):
                    logging.info("Processing drum notes:")
                    drum_notes = {}
                    
                    for note in track.get('notes', []):
                        if isinstance(note, dict):
                            midi_note = note.get('midi')
                        else:
                            midi_note = int(note)
                        
                        # Use DRUM_NOTES mapping to get drum name
                        drum_name = DRUM_NOTES.get(midi_note)
                        
                        if drum_name:
                            logging.info(f"MIDI note {midi_note} maps to drum: {drum_name}")
                            position_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                            
                            if position_key not in drum_notes:
                                drum_notes[position_key] = []
                                # Assign grid position
                                self.grid_positions[position_key] = total_slots
                                total_slots += 1
                                
                            drum_notes[position_key].append(note)
            
            return total_slots  # Return total slots but ensure grid_positions is populated
                
        except Exception as e:
            logging.error(f"Layout error: {str(e)}")
            raise
    def validate_track_data(self, track):
        """Validate single track data structure"""
        if not isinstance(track, dict):
            return False
            
        required_fields = ['notes', 'video']
        return all(field in track for field in required_fields)

    def get_drum_position_key(self, midi_note, drum_name):
        """Helper to generate consistent drum position keys"""
        return f"drum_{midi_note}_{drum_name.lower()}"

    def get_chunk_notes(self, track, start_time, end_time, include_overlap=True):
        """Get notes within chunk timeframe including overlaps"""
        try:
            notes = []
            for note in track.get('notes', []):
                if not self.validate_midi_note(note, track.get('index', -1)):
                    continue
                note_start = float(note['time'])
                note_end = note_start + float(note.get('duration', 0))
                
                # Include notes that:
                # 1. Start within chunk
                # 2. End within chunk
                # 3. Span across chunk boundary
                if (note_start >= start_time and note_start < end_time) or \
                (note_end > start_time and note_end <= end_time) or \
                (note_start <= start_time and note_end >= end_time):
                    
                    # Calculate adjusted start time relative to chunk
                    if note_start < start_time:
                        note = note.copy()
                        time_diff = start_time - note_start
                        note['time'] = start_time
                        note['duration'] = float(note['duration']) - time_diff
                    
                    notes.append(note)
                    
            return notes
        except Exception as e:
            logging.error(f"Error getting chunk notes: {e}")
            return []

    def get_track_duration(self, track):
        """Get duration for any track type"""
        try:
            if isinstance(track, int):
                logging.warning(f"Received integer {track} instead of track dictionary")
                return 0
                
            notes = track.get('notes', [])
            if not notes:
                return 0
                
            # Calculate end times using direct key access
            end_times = [float(note['time'] + note['duration']) 
                        for note in notes]
            return max(end_times) if end_times else 0
                
        except Exception as e:
            logging.error(f"Error calculating track duration: {str(e)}")
            return 0

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
            
            # Calculate chunks
            full_chunks = int(last_note_time // self.CHUNK_DURATION)
            final_chunk = last_note_time % self.CHUNK_DURATION
            
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
        try:
            total_slots = 0
            self.grid_positions = {}

            # Filter tracks with notes first
            valid_tracks = [(idx, track) for idx, track in enumerate(self.midi_data['tracks'])
                            if self.has_valid_notes(track)]
            
            logging.info("\nGrid Layout Planning:")
            logging.info(f"\nFound {len(valid_tracks)} tracks with notes:")
            
            # Process tracks to determine grid positions
            for track_idx, track in valid_tracks:
                if is_drum_kit(track.get('instrument', {})):
                    # For drum tracks, check actual files in drum directory
                    drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                    if drum_dir.exists():
                        # Look for actual drum videos that were copied during processing
                        for drum_file in drum_dir.glob('*.mp4'):
                            drum_group = drum_file.stem.replace('drum_', '')  # Remove 'drum_' prefix if present
                            position_key = f"drum_{drum_group}"
                            self.grid_positions[position_key] = total_slots
                            logging.info(f"Position {total_slots}: Drum {drum_group}")
                            total_slots += 1
                else:
                    # Regular instrument track
                    position_key = f"track_{track_idx}"
                    self.grid_positions[position_key] = total_slots
                    instrument_name = track.get('instrument', {}).get('name', f'track_{track_idx}')
                    logging.info(f"Position {total_slots}: {instrument_name}")
                    total_slots += 1

            # Calculate grid dimensions
            cols = min(4, math.ceil(math.sqrt(total_slots)))
            rows = math.ceil(total_slots / cols)
            
            # Log grid layout
            logging.info("\nGrid Visual Layout:")
            for row in range(rows):
                row_str = ""
                for col in range(cols):
                    pos = row * cols + col
                    instrument = next((k for k, v in self.grid_positions.items() if v == pos), "empty")
                    row_str += f"[{instrument:^20}] "
                logging.info(row_str)
                
            logging.info(f"\nGrid dimensions: {rows}x{cols} ({total_slots} slots)")
            return (rows, cols)
                
        except Exception as e:
            logging.error(f"Layout error: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return (1, 1)
        
    def validate_midi_note(self, note, track_idx):
        """Validate MIDI note timing and duration"""
        try:
            midi_note = note.get('midi')
            start_time = float(note.get('time', 0))
            duration = float(note.get('duration', 0))
            
            logging.info(f"Validating MIDI note - Track: {track_idx}, Note: {midi_note}")
            logging.info(f"  Start Time: {start_time}")
            logging.info(f"  Duration: {duration}")
            logging.info(f"  Raw note data: {note}")
            
            if duration <= 0:
                logging.error(f"Invalid note duration for track {track_idx}, note {midi_note}: {duration}")
                return False
                
            if start_time < 0:
                logging.error(f"Invalid note start time for track {track_idx}, note {midi_note}: {start_time}")
                return False
                
            return True
                    
        except Exception as e:
            logging.error(f"Note validation error for track {track_idx}: {str(e)}")
            logging.error(f"Note data: {note}")
            return False
    
    def get_note_volume(self, velocity, is_drum=False):
        """Calculate volume from MIDI velocity with better scaling"""
        # Normalize velocity (0-1)
        normalized_velocity = float(velocity) / 127.0
        
        # Set base multipliers
        multipliers = {
            'drums': 0.4,      # Drums at 40% 
            'instruments': 1.0  # Instruments at full volume
        }
        
        # Calculate volume with better minimum
        base_volume = normalized_velocity * 1.5  # Boost overall volume
        volume = max(0.3, base_volume * multipliers['drums' if is_drum else 'instruments'])
        
        logging.info(f"Volume calculation: velocity={velocity}, normalized={normalized_velocity:.2f}, final={volume:.2f}")
        return volume
    
    def process_chunk(self, tracks, start_time, end_time, chunk_idx):
        try:
            rows, cols = self.get_track_layout()
            grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                            color=(0,0,0), 
                            duration=end_time - start_time) 
                    for _ in range(cols)] 
                    for _ in range(rows)]

            # Process regular tracks first
            for track_idx, track in enumerate(tracks):
                if not is_drum_kit(track.get('instrument', {})):
                    position_key = f"track_{track_idx}"
                    if position_key in self.grid_positions:
                        pos = self.grid_positions[position_key]
                        row, col = pos // cols, pos % cols
                        logging.info(f"Processing regular track {track_idx} at position [{row}][{col}]")

                        if row < rows and col < cols:
                            instrument_dir = self.processed_videos_dir / f"track_{track_idx}_{normalize_instrument_name(track['instrument']['name'])}"
                            if instrument_dir.exists():
                                clips_for_track = []
                                for note in self.get_chunk_notes(track, start_time, end_time):
                                    try:
                                        midi_note = int(note['midi'])
                                        note_file = instrument_dir / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                                        
                                        if note_file.exists():
                                            clip = VideoFileClip(str(note_file))
                                            time = float(note['time']) - start_time
                                            duration = float(note.get('duration', clip.duration))
                                            clip = clip.set_start(time).set_duration(duration)
                                            clips_for_track.append(clip)
                                    except Exception as e:
                                        logging.error(f"Error processing regular note: {e}")
                                        continue
                                
                                if clips_for_track:
                                    composite = CompositeVideoClip(clips_for_track)
                                    grid[row][col] = composite.set_duration(end_time - start_time)

            # Process drum tracks with duration handling
            for track in tracks:
                if is_drum_kit(track.get('instrument', {})):
                    chunk_notes = self.get_chunk_notes(track, start_time, end_time)
                    
                    # Group notes by drum type
                    drum_notes = {}
                    for note in chunk_notes:
                        midi_note = int(note['midi'])
                        drum_name = DRUM_NOTES.get(midi_note)
                        if drum_name:
                            normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
                            if normalized_name not in drum_notes:
                                drum_notes[normalized_name] = []
                            drum_notes[normalized_name].append(note)
                    
                    # Process each drum type
                    for drum_key, notes in drum_notes.items():
                        if drum_key in self.grid_positions:
                            pos = self.grid_positions[drum_key]
                            row = pos // cols
                            col = pos % cols
                            
                            if row < rows and col < cols:
                                drum_file = self.processed_videos_dir / f"track_5_drums" / f"{drum_key}.mp4"
                                if drum_file.exists():
                                    try:
                                        # Use cv2 to get video duration
                                        cap = cv2.VideoCapture(str(drum_file))
                                        fps = cap.get(cv2.CAP_PROP_FPS)
                                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                        duration = frame_count / fps if fps > 0 else 0.5  # Default to 0.5s if can't get fps
                                        cap.release()
                                        
                                        for note in notes:
                                            clip = VideoFileClip(str(drum_file), has_mask=True)
                                            time = float(note['time']) - start_time
                                            clip = clip.set_duration(duration).set_start(time)
                                            
                                            if isinstance(grid[row][col], ColorClip):
                                                grid[row][col] = clip
                                            else:
                                                grid[row][col] = CompositeVideoClip([grid[row][col], clip])
                                            
                                    except Exception as e:
                                        logging.error(f"Error processing drum clip: {e}")
                                        continue

            # Create and save chunk
            chunk = clips_array(grid)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            chunk.write_videofile(str(chunk_path), fps=30, codec='h264_nvenc')
            return str(chunk_path)

        except Exception as e:
            logging.error(f"Error in process_chunk: {str(e)}")
            return None
    
    def process_chunk(self, tracks, start_time, end_time, chunk_idx):
        """Process chunk with memory optimization"""
        try:
            active_clips = []  # Track active clips for cleanup
            rows, cols = self.get_track_layout()
            
            # Pre-allocate grid with empty clips
            grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                            color=(0,0,0), 
                            duration=end_time - start_time) 
                    for _ in range(cols)] 
                    for _ in range(rows)]

            for track_idx, track in enumerate(tracks):
                chunk_notes = self.get_chunk_notes(track, start_time, end_time)
                
                # Process in smaller batches
                BATCH_SIZE = 5
                for i in range(0, len(chunk_notes), BATCH_SIZE):
                    batch = chunk_notes[i:i + BATCH_SIZE]
                    
                    for note in batch:
                        try:
                            clip = self.get_note_clip(track, note)
                            if clip:
                                active_clips.append(clip)
                                # Process immediately
                                time = float(note['time']) - start_time
                                clip = clip.set_start(time)
                                row = track_idx // cols
                                col = track_idx % cols
                                
                                if isinstance(grid[row][col], ColorClip):
                                    grid[row][col] = clip
                                else:
                                    existing = grid[row][col]
                                    grid[row][col] = CompositeVideoClip([existing, clip])
                        
                        finally:
                            # Cleanup after each note
                            for clip in active_clips:
                                try:
                                    clip.close()
                                except:
                                    pass
                            active_clips.clear()
                            gc.collect()

            # Create chunk with optimized parameters
            chunk = clips_array(grid)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
            chunk.write_videofile(
                str(chunk_path),
                fps=30,
                codec='h264_nvenc',
                audio_codec='aac',
                preset='fast',  # Faster encoding
                ffmpeg_params=[
                    "-vsync", "1",
                    "-async", "1",
                    "-b:v", "5M",
                    "-tile-columns", "2",
                    "-threads", "8",
                    "-row-mt", "1"
                ]
            )
            
            return str(chunk_path)
            
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
            return None
            
        finally:
            # Final cleanup
            for clip in active_clips:
                try:
                    clip.close()
                except:
                    pass
            gc.collect()
        

    def create_composition(self):
            try:
                full_chunks, final_duration = self.calculate_chunk_lengths()
                total_chunks = full_chunks + (1 if final_duration > 0 else 0)
                rows, cols = self.get_track_layout()
                
                chunk_files = []
                active_clips = []
                for chunk_idx in range(total_chunks):
                    start_time = chunk_idx * 10
                    end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                    
                    # Debug grid creation
                    logging.info(f"\nProcessing Chunk {chunk_idx}")
                    logging.info(f"Time Range: {start_time}-{end_time}")
                    
                    grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                                    color=(0,0,0), 
                                    duration=end_time - start_time) 
                            for _ in range(cols)] 
                            for _ in range(rows)]
                    
                    # Process each track
                    for track_idx, track in enumerate(self.midi_data['tracks']):
                        chunk_notes = [
                            note for note in track.get('notes', [])
                            if start_time <= float(note['time']) < end_time
                        ]
                        
                        logging.info(f"\nTrack {track_idx}: {track.get('instrument', {}).get('name', 'unknown')}")
                        logging.info(f"Notes in chunk: {len(chunk_notes)}")

                        if not chunk_notes:
                            continue
                        
                        if is_drum_kit(track.get('instrument', {})):
                            drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                            if drum_dir.exists():
                                drum_notes = {}
                                # Group notes by drum type
                                for note in chunk_notes:
                                    midi_note = int(note['midi'])
                                    drum_name = DRUM_NOTES.get(midi_note)
                                    if drum_name:
                                        # Normalize drum name for file lookup
                                        normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
                                        if normalized_name not in drum_notes:
                                            drum_notes[normalized_name] = []
                                        drum_notes[normalized_name].append(note)
                                        logging.info(f"Added note {midi_note} to drum group {normalized_name}")

                                # Process each drum type
                                for drum_key, notes in drum_notes.items():
                                    position_key = drum_key  # Use normalized name directly as position key
                                    logging.info(f"Processing {drum_key} at position {position_key}")
                                    if position_key in self.grid_positions:
                                        position = self.grid_positions[position_key]
                                        row = position // cols
                                        col = position % cols
                                        
                                        # Use normalized drum filename
                                        drum_file = drum_dir / f"{drum_key}.mp4"
                                        if drum_file.exists():
                                            for note in notes:
                                                try:
                                                    clip = VideoFileClip(str(drum_file))
                                                    active_clips.append(clip)
                                                    
                                                    # Keep existing volume calculation
                                                    velocity = float(note.get('velocity', 100))
                                                    volume = self.get_note_volume(velocity, is_drum=True)
                                                    clip = clip.volumex(volume)
                                                    
                                                    # Keep existing timing logic
                                                    time = float(note['time']) - start_time
                                                    duration = min(float(note['duration']), clip.duration)
                                                    clip = clip.subclip(0, duration).set_start(time)
                                                    
                                                    if isinstance(grid[row][col], ColorClip):
                                                        grid[row][col] = clip
                                                    else:
                                                        existing = grid[row][col]
                                                        grid[row][col] = CompositeVideoClip([existing, clip])
                                                    
                                                    logging.info(f"Added {drum_key} at [{row}][{col}] t={time}")
                                                except Exception as e:
                                                    logging.error(f"Error processing drum clip: {e}")
                                                    continue
                    
                        else:
                            position_key = f"track_{track_idx}"
                            if position_key in self.grid_positions:
                                position = self.grid_positions[position_key]
                                row = position // cols
                                col = position % cols
                                
                                clips_for_instrument = []
                                for note in chunk_notes:
                                    midi_note = int(float(note['midi']))
                                    note_file = self.processed_videos_dir / f"track_{track_idx}_{normalize_instrument_name(track['instrument']['name'])}" / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                                    
                                    logging.info(f"Loading note: {note_file}")
                                    
                                    if note_file.exists():
                                        clip = VideoFileClip(str(note_file))
                                        active_clips.append(clip)
                                        
                                        # Get and normalize MIDI velocity
                                        velocity = float(note.get('velocity', 100))
                                        volume = self.get_note_volume(velocity, is_drum=False)  # Ensure minimum audible volume
                                        
                                        # Apply volume to clip
                                        clip = clip.volumex(volume)
                                        
                                        # Log volume level for debugging
                                        logging.info(f"Note {note.get('midi')}: velocity={velocity:.2f}, volume={volume:.2f}")
                                        
                                        time = float(note['time']) - start_time
                                        duration = min(float(note['duration']), clip.duration)
                                        clip = clip.subclip(0, duration).set_start(time)
                                        clips_for_instrument.append(clip)
                                
                                if clips_for_instrument:
                                    composite = CompositeVideoClip(clips_for_instrument)
                                    grid[row][col] = composite.set_duration(end_time - start_time)
                                    logging.info(f"Added instrument at [{row}][{col}]")
                    
                    # Create chunk
                    chunk = clips_array(grid)
                    chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
                    chunk.write_videofile(
                        str(chunk_path),
                        fps=30,
                        codec='h264_nvenc',  # NVIDIA GPU encoder
                        audio_codec='aac',
                        preset='medium',
                        ffmpeg_params=[
                            "-vsync", "1",
                            "-async", "1",
                            "-b:v", "5M",
                            "-maxrate", "10M",
                            "-bufsize", "10M",
                            "-rc", "vbr",
                            "-tune", "hq"
                        ]
                    )
                    chunk_files.append(str(chunk_path))
                    
                    # Aggressive cleanup after each chunk
                    for clip in active_clips:
                        try:
                            clip.close()
                        except:
                            pass
                    active_clips.clear()
                    
                    # Force garbage collection
                    gc.collect()
                
                # Combine chunks
                if chunk_files:
                    logging.info(f"Attempting to concatenate {len(chunk_files)} chunks")
                    logging.info(f"Chunk files: {chunk_files}")
                    return self._combine_chunks(chunk_files)
            
                    
            except Exception as e:
                logging.error(f"Composition error: {str(e)}")
                return None
            
    def _combine_chunks(self, chunk_files):
        """Combine chunks with precise timing"""
        try:
            clips = []
            first_clip = VideoFileClip(chunk_files[0])
            target_fps = first_clip.fps
            frame_duration = 1.0 / target_fps
            
            for i, chunk_file in enumerate(chunk_files):
                clip = VideoFileClip(chunk_file)
                
                # Ensure exact frame boundary alignment
                if i > 0:
                    trim_start = self.OVERLAP_DURATION
                    frame_count = int(round(trim_start * target_fps))
                    precise_trim = frame_count * frame_duration
                    clip = clip.subclip(precise_trim)
                    
                    # Ensure audio sync
                    clip = clip.set_start(i * 10.0 - self.CROSSFADE_DURATION)
                
                clips.append(clip)
                
            # Use precise frame-based concatenation
            final = concatenate_videoclips(
                clips,
                method="compose",
                padding=-self.CROSSFADE_DURATION
            )
            
            # Write with strict timing parameters
            final.write_videofile(
                str(self.output_path),
                fps=target_fps,
                codec='h264_nvenc',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=[
                    "-vsync", "cfr",     # Constant frame rate
                    "-b:v", "5M",        # Video bitrate
                    "-movflags", "+faststart"  # Web playback optimization
                ]
            )
            
            return self.output_path
            
        except Exception as e:
            logging.error(f"Concatenation error: {e}")
            return None
            
        finally:
            # Cleanup
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            # Remove temp files
            for f in chunk_files:
                try:
                    os.unlink(f)
                except:
                    pass

def calculate_chunk_lengths(self, midi_data):
        try:
            # Find last note end time
            last_note_time = 0
            for track in midi_data['tracks']:
                for note in track.get('notes', []):
                    note_end = float(note['time']) + float(note['duration'])
                    last_note_time = max(last_note_time, note_end)
            
            # Calculate chunks
            CHUNK_SIZE = 10  # seconds
            full_chunks = int(last_note_time // CHUNK_SIZE)
            final_chunk_duration = last_note_time % CHUNK_SIZE
            
            logging.info(f"Total duration: {last_note_time}")
            logging.info(f"Full chunks: {full_chunks}")
            logging.info(f"Final chunk duration: {final_chunk_duration}")
            
            return full_chunks, final_chunk_duration
            
        except Exception as e:
            logging.error(f"Error calculating chunks: {str(e)}")
            return None, None
            

def compose_from_processor_output(processor_result, output_path):
    try:
        base_dir = processor_result['processed_videos_dir']
        logging.info(f"Using base directory: {base_dir}")
        
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
        return composer.create_composition()

    except Exception as e:
        logging.error(f"Error in video composition: {str(e)}")
        raise