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

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from utils import normalize_instrument_name, midi_to_note
from drum_utils import (
    DRUM_NOTES,
    process_drum_track,
    get_drum_groups,
    get_drum_name,
    is_drum_kit
)

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
    CHUNK_DURATION = 4
    OVERLAP_DURATION = 0.3
    CROSSFADE_DURATION = 0.25
    VOLUME_MULTIPLIERS = {
        'drums': 0.2,
        'instruments': 1.5
    }

    # def __init__(self, processed_videos_dir, midi_data, output_path):
    #     """Initialize VideoComposer with proper path handling"""
    #     try:
    #         logging.info("=== VideoComposer Initialization ===")
    #         logging.info(f"Received MIDI data structure: {list(midi_data.keys())}")
    #         logging.info(f"Grid arrangement from MIDI: {midi_data.get('gridArrangement')}")

    #         self.processed_videos_dir = Path(processed_videos_dir)
    #         # Fix uploads path - go up two levels to /backend/uploads
    #         self.uploads_dir = Path(processed_videos_dir).parent.parent.parent / "uploads"
    #         logging.info(f"Setting uploads directory: {self.uploads_dir}")
    #         # Verify uploads directory exists
    #         if not self.uploads_dir.exists():
    #             raise ValueError(f"Uploads directory not found: {self.uploads_dir}")
    #         self.output_path = output_path
    #         self.midi_data = midi_data
    #         self._setup_paths(processed_videos_dir, output_path)
    #         self.midi_data = midi_data
    #         self._process_midi_data(midi_data)
    #         self._setup_track_configuration()
    #          # Log track information
    #         logging.info(f"Regular tracks: {len(self.tracks)}")
    #         logging.info(f"Drum tracks: {len(self.drum_tracks)}")
    #         for track in self.drum_tracks:
    #             logging.info(f"Drum track found: {track.get('instrument', {}).get('name')}")
                
    #     except Exception as e:
    #         logging.error(f"VideoComposer init error: {str(e)}")
    #         raise
    def __init__(self, processed_videos_dir, midi_data, output_path, use_av1=False):
        """Initialize VideoComposer with proper path handling"""
        try:
            logging.info("=== VideoComposer Initialization ===")
            logging.info(f"Received MIDI data structure: {list(midi_data.keys())}")
            logging.info(f"Grid arrangement from MIDI: {midi_data.get('gridArrangement')}")

            self.processed_videos_dir = Path(processed_videos_dir)
            self.uploads_dir = Path(processed_videos_dir).parent.parent.parent / "uploads"
            logging.info(f"Setting uploads directory: {self.uploads_dir}")
            if not self.uploads_dir.exists():
                raise ValueError(f"Uploads directory not found: {self.uploads_dir}")
            self.output_path = output_path
            self.use_av1 = use_av1
            self.midi_data = midi_data
            self._setup_paths(processed_videos_dir, output_path)
            self._process_midi_data(midi_data)
            self._setup_track_configuration()
            logging.info(f"Regular tracks: {len(self.tracks)}")
            logging.info(f"Drum tracks: {len(self.drum_tracks)}")
            for track in self.drum_tracks:
                logging.info(f"Drum track found: {track.get('instrument', {}).get('name')}")
                
        except Exception as e:
            logging.error(f"VideoComposer init error: {str(e)}")
            raise
    
    def _process_instrument_track(self, track_idx, track, video_file):
        """Process instrument track notes"""
        try:
            instrument_name = normalize_instrument_name(track['instrument']['name'])
            output_dir = self.processed_videos_dir / f"track_{track_idx}_{instrument_name}_notes"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for note in track['notes']:
                midi_note = int(note['midi'])
                note_file = output_dir / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                
                if not note_file.exists():
                    self._process_note(note, video_file, note_file)
                    
        except Exception as e:
            logging.error(f"Error processing instrument track {track_idx}: {str(e)}")

    def process_track_videos(self, midi_data, video_file):
        """Process all tracks from MIDI data"""
        try:
            for track_idx, track in enumerate(midi_data.get('tracks', [])):
                if not isinstance(track, dict):
                    logging.warning(f"Invalid track format at index {track_idx}")
                    continue
                    
                if not track.get('notes'):
                    continue

                if not is_drum_kit(track.get('instrument', {})):
                    self._process_instrument_track(track_idx, track, video_file)
                else:
                    self._process_drum_track(track_idx, track, video_file)
                    
        except Exception as e:
            logging.error(f"Error processing track videos: {str(e)}")
            raise

    def _process_drum_track(self, track_idx, track, video_file):
        """Process drum track separately"""
        try:
            if not isinstance(track, dict):
                logging.error(f"Invalid drum track format: {track}")
                return

            drum_groups = get_drum_groups(track.get('notes', []))
            for drum_key, notes in drum_groups.items():
                if not notes:
                    continue
                    
                output_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{drum_key}.mp4"
                if not output_file.exists():
                    self._process_drum_notes(notes, video_file, output_file)
                    
        except Exception as e:
            logging.error(f"Error processing drum track {track_idx}: {str(e)}")

    def _process_drum_notes(self, notes, video_file, output_file):
        """Process drum notes to create video clip"""
        try:
            clip = VideoFileClip(str(video_file))
            subclips = []
            
            for note in notes:
                start_time = float(note['time'])
                duration = float(note.get('duration', 0.25))  # Default duration
                subclip = clip.subclip(start_time, start_time + duration)
                subclips.append(subclip)
            
            if subclips:
                final_clip = concatenate_videoclips(subclips)
                final_clip.write_videofile(str(output_file))
                
            clip.close()
            
        except Exception as e:
            logging.error(f"Error processing drum notes: {str(e)}")

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
        
   
    def _calculate_default_layout(self):
        """Calculate default grid layout"""
        rows = math.ceil(len(self.tracks) / 4)
        cols = min(4, len(self.tracks))
        return rows, cols

    def _setup_track_configuration(self):
        """Setup track-specific configuration and grid layout"""
        logging.info("Received MIDI data: %s", self.midi_data)
        logging.info("Grid arrangement: %s", self.midi_data.get('gridArrangement'))
        try:
            grid_arrangement = self.midi_data.get('gridArrangement', {})
            logging.info(f"Extracted grid arrangement: {grid_arrangement}")
            self.grid_positions = {}
            
            if grid_arrangement:
                # Convert the frontend grid positions to video positions
                for track_id, pos_data in grid_arrangement.items():
                    # Remove 'drum-' or 'track-' prefix
                    # clean_id = track_id.replace('drum-', '').replace('track-', '')
                    # self.grid_positions[clean_id] = {
                    #     'row': pos_data['row'],
                    #     'column': pos_data['column'],
                    #     'position': pos_data['position']
                    # }
                    self.grid_positions[track_id] = {
                        'row': pos_data['row'],
                        'column': pos_data['column'],
                        'position': pos_data['position']
                    }
                    logging.info(f"Stored position for {track_id}: {self.grid_positions[track_id]}")
                    
                # Calculate dimensions from arrangement
                max_row = max(pos['row'] for pos in grid_arrangement.values())
                max_col = max(pos['column'] for pos in grid_arrangement.values())
                return max_row + 1, max_col + 1
                
            return self._calculate_default_layout()
        
        except Exception as e:
            logging.error(f"Error getting chunk notes: {e}")
            return self._calculate_default_layout()

    def _add_video_at_position(self, video_path, track_id, timestamp, grid, rows, cols):
        try:
            # Look up position data from grid arrangement
            logging.info(f"Adding video for {track_id}")
            
            if track_id in self.grid_positions:
                pos_data = self.grid_positions[track_id]
                row = pos_data['row']
                col = pos_data['column']
                logging.info(f"Found grid position for {track_id}: row={row}, col={col}")
            else:
                # Fallback calculation if position not found
                logging.warning(f"No grid position found for {track_id}")
                position = len(self.grid_positions)
                row = position // cols
                col = position % cols
                
            if row < rows and col < cols:
                logging.info(f"Adding video at grid position [{row}][{col}]")
                clip = VideoFileClip(str(video_path))
                
                if isinstance(grid[row][col], ColorClip):
                    grid[row][col] = clip
                else:
                    # Composite with existing clip
                    existing = grid[row][col]
                    grid[row][col] = CompositeVideoClip([existing, clip])
                return True
                
        except Exception as e:
            logging.error(f"Error adding video: {str(e)}")
            return False

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
            # Get grid arrangement from MIDI data
            grid_arrangement = self.midi_data.get('gridArrangement', {})
            logging.info(f"Using grid arrangement from frontend: {grid_arrangement}")
            
            if not grid_arrangement:
                logging.warning("No grid arrangement provided, falling back to default layout")
                return self._calculate_default_layout()

            # Find maximum dimensions from grid arrangement
            max_row = 0
            max_col = 0
            for pos_data in grid_arrangement.values():
                max_row = max(max_row, pos_data['row'])
                max_col = max(max_col, pos_data['column'])
            
            rows = max_row + 1
            cols = max_col + 1
            
            # Store grid positions for use in composition
            self.grid_positions = {}
            
            # Process tracks to map them to their grid positions
            for track_idx, track in enumerate(self.midi_data['tracks']):
                if not self.has_valid_notes(track):
                    continue
                    
                if is_drum_kit(track.get('instrument', {})):
                    # For drum tracks, map each drum type to its position
                    drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                    if drum_dir.exists():
                        for drum_file in drum_dir.glob('*.mp4'):
                            drum_key = drum_file.stem  # Keep full name including 'drum_' prefix
                            if drum_key in grid_arrangement:
                                pos_data = grid_arrangement[drum_key]
                                self.grid_positions[drum_key] = pos_data
                                logging.info(f"Mapped {drum_key} to position {pos_data}")
                else:
                    # For regular instrument tracks, use track index as key
                    track_key = str(track_idx)
                    if track_key in grid_arrangement:
                        pos_data = grid_arrangement[track_key]
                        self.grid_positions[f"track_{track_key}"] = pos_data
                        instrument_name = track.get('instrument', {}).get('name', f'track_{track_idx}')
                        logging.info(f"Mapped {instrument_name} to position {pos_data}")

            # Log final layout
            logging.info("\nGrid Visual Layout:")
            for row in range(rows):
                row_str = ""
                for col in range(cols):
                    # Find track at this position
                    track_id = next((k for k, v in self.grid_positions.items() 
                                if v['row'] == row and v['column'] == col), "empty")
                    row_str += f"[{track_id:^20}] "
                logging.info(row_str)
                
            logging.info(f"\nGrid dimensions: {rows}x{cols}")
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
            grid = [[ColorClip(...) for _ in range(cols)] for _ in range(rows)]
            
            # Handle regular tracks
            for track_idx, track in enumerate(tracks):
                if is_drum_kit(track.get('instrument', {})):
                    continue
                    
                track_id = f"track_{track_idx}"
                pos = self.grid_positions.get(track_id, track_idx)
                row = pos // cols
                col = pos % cols
                
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
                    for note in track.get('notes', []):
                        drum_name = DRUM_NOTES.get(note['midi'])
                        if drum_name:
                            drum_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                            pos = self.grid_positions.get(drum_key, 0)
                            row = pos // cols
                            col = pos % cols
                    
                    # Process each drum type
                    for drum_key, notes in drum_notes.items():
                        # Get position from arrangement or fall back to default
                        drum_id = f"{track_id}_{drum_key}"
                        if drum_id in arrangement:
                            pos = arrangement[drum_id]['position']
                        else:
                            pos = self.grid_positions.get(drum_key, 0)
                            
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
            #     ffmpeg_params=[
            #         "-vsync", "1",
            #         "-async", "1",
            #         "-b:v", "5M",
            #         "-tile-columns", "2",
            #         "-threads", "8",
            #         "-row-mt", "1"
            #     ]
            # )
                 ffmpeg_params=[
                        # Enables GPU-accelerated decoding if supported
                        '-hwaccel', 'cuda',
                        '-hwaccel_output_format', 'cuda',

                        # Helps keep video/frame alignment in sync
                        '-vsync', '1',
                        '-async', '1',

                        # Adjust to make full use of GPU hardware
                        '-b:v', '5M',
                        '-maxrate', '10M',
                        '-bufsize', '10M',
                        '-rc', 'vbr',
                        '-tune', 'hq',
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
                    clip = clip.volumex(volume)
                    
                    # Set timing
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
        clips_for_instrument = []
        for note in chunk_notes:
            midi_note = int(float(note['midi']))
            note_file = self.processed_videos_dir / f"track_{track_idx}_{normalize_instrument_name(track['instrument']['name'])}" / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
            
            if note_file.exists():
                clip = self._create_note_clip(note_file, note, start_time)
                if clip:
                    active_clips.append(clip)
                    clips_for_instrument.append(clip)

        if clips_for_instrument:
            position_key = f"track_{track_idx}"
            if position_key in self.grid_positions:
                pos_data = self.grid_positions[position_key]
                row = int(pos_data['row'])
                col = int(pos_data['column'])
                composite = CompositeVideoClip(clips_for_instrument)
                grid[row][col] = composite.set_duration(end_time - start_time)
                logging.info(f"Added instrument at [{row}][{col}]")

    def _create_note_clip(self, note_file, note, start_time):
        """Creates a video clip for a single note"""
        try:
            if not note_file.exists():
                logging.warning(f"Note file not found: {note_file}")
                return None
                
            clip = VideoFileClip(str(note_file))
            
            # Apply volume based on note velocity
            velocity = float(note.get('velocity', 100))
            volume = self.get_note_volume(velocity, is_drum=False)
            clip = clip.volumex(volume)
            
            # Set timing
            time = float(note['time']) - start_time
            duration = min(float(note['duration']), clip.duration)
            clip = clip.subclip(0, duration).set_start(time)
            
            logging.info(f"Created clip for note at time {time:.2f}, duration {duration:.2f}")
            return clip
            
        except Exception as e:
            logging.error(f"Error creating note clip: {e}")
            return None

    def create_composition(self):
        try:
            full_chunks, final_duration = self.calculate_chunk_lengths()
            total_chunks = full_chunks + (1 if final_duration > 0 else 0)
            rows, cols = self.get_track_layout()
            
            chunk_files = []
            self.chunk_clips = {}  # Initialize as instance variable

            def process_chunk(chunk_idx):
                try:
                    active_clips = []
                    start_time = chunk_idx * self.CHUNK_DURATION
                    end_time = start_time + (final_duration if chunk_idx == full_chunks else self.CHUNK_DURATION)
                    
                    logging.info(f"\nProcessing Chunk {chunk_idx}")
                    logging.info(f"Time Range: {start_time}-{end_time}")
                    
                    grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                                    color=(0,0,0), 
                                    duration=end_time - start_time) 
                            for _ in range(cols)] 
                            for _ in range(rows)]

                    # Process tracks within the chunk
                    for track_idx, track in enumerate(self.midi_data['tracks']):
                        if not track.get('notes'):
                            continue
                            
                        chunk_notes = [
                            note for note in track.get('notes', [])
                            if start_time <= float(note['time']) < end_time
                        ]

                        if not chunk_notes:
                            continue

                        if is_drum_kit(track.get('instrument', {})):
                            self._process_drum_chunk(track_idx, chunk_notes, grid, active_clips, start_time)
                        else:
                            self._process_instrument_chunk(track_idx, track, chunk_notes, grid, active_clips, start_time, end_time)

                    # Create chunk after processing ALL tracks
                    chunk = clips_array(grid)
                    chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
                    
                    # Configure encoding parameters
                    ffmpeg_params = [
                        # Input hardware acceleration
                        "-hwaccel", "cuda",
                        "-hwaccel_output_format", "cuda",
                        
                        # Video encoding settings
                        "-c:v", "h264_nvenc",
                        "-preset", "fast",  # Valid NVENC presets are: p1-p7 (p1 is fastest)
                        "-tune", "hq",
                        "-b:v", "5M",
                        "-maxrate", "10M",
                        "-bufsize", "10M",
                        "-rc", "vbr",
                        "-profile:v", "high",
                        "-pix_fmt", "yuv420p",
                        
                        # Audio settings
                        "-c:a", "aac",
                        "-b:a", "192k",
                        
                        # Container settings
                        "-movflags", "faststart"
                    ]

                    chunk.write_videofile(
                        str(chunk_path),
                        fps=self.FRAME_RATE,
                        codec="h264_nvenc",
                        audio_codec="aac",
                        ffmpeg_params=ffmpeg_params,
                        logger=None
                    )
                    
                    return str(chunk_path)

                except Exception as e:
                    logging.error(f"Error processing chunk: {str(e)}")
                    return None
                finally:
                    # Clean up clips
                    for clip in active_clips:
                        try:
                            clip.close()
                        except:
                            pass
                    gc.collect()

            # Process chunks in parallel
            with ProcessPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(process_chunk, chunk_idx): chunk_idx 
                    for chunk_idx in range(total_chunks)
                }
                
                # Wait for all chunks to complete
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_path = future.result()
                        if chunk_path:
                            chunk_files.append(chunk_path)
                            logging.info(f"Chunk {chunk_idx} completed: {chunk_path}")
                    except Exception as e:
                        logging.error(f"Chunk {chunk_idx} failed: {str(e)}")

            # Sort and combine chunks
            if chunk_files:
                chunk_files.sort(key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
                return self._combine_chunks(chunk_files)
            
            return None

        except Exception as e:
            logging.error(f"Composition error: {str(e)}")
            return None  
    # def create_composition(self):
    #     try:
    #         full_chunks, final_duration = self.calculate_chunk_lengths()
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
    #         rows, cols = self.get_track_layout()
            
    #         chunk_files = []
    #         self.chunk_clips = {}  # Initialize as instance variable

    #         def process_chunk(chunk_idx):
    #             try:
    #                 active_clips = []
    #                 start_time = chunk_idx * 10
    #                 end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                    
    #                 logging.info(f"\nProcessing Chunk {chunk_idx}")
    #                 logging.info(f"Time Range: {start_time}-{end_time}")
                    
    #                 grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                                 color=(0,0,0), 
    #                                 duration=end_time - start_time) 
    #                         for _ in range(cols)] 
    #                         for _ in range(rows)]

    #                 # Process tracks within the chunk
    #                 for track_idx, track in enumerate(self.midi_data['tracks']):
    #                     if not track.get('notes'):
    #                         continue
                            
    #                     chunk_notes = [
    #                         note for note in track.get('notes', [])
    #                         if start_time <= float(note['time']) < end_time
    #                     ]

    #                     if not chunk_notes:
    #                         continue

    #                     if is_drum_kit(track.get('instrument', {})):
    #                         self._process_drum_chunk(track_idx, chunk_notes, grid, active_clips, start_time)
    #                     else:
    #                         self._process_instrument_chunk(track_idx, track, chunk_notes, grid, active_clips, start_time, end_time)

    #                  # Create chunk after processing ALL tracks
    #                 chunk = clips_array(grid)
    #                 chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
                    
    #                 # Configure encoding parameters
    #                 ffmpeg_params = [
                        
    #                     # Video encoding settings
    #                     "-c:v", "h264_nvenc",
    #                     "-preset", "fast",  # Valid NVENC presets are: p1-p7 (p1 is fastest)
    #                     "-tune", "hq",
    #                     "-b:v", "5M",
    #                     "-maxrate", "10M",
    #                     "-bufsize", "10M",
    #                     "-rc", "vbr",
    #                     "-profile:v", "high",
    #                     "-pix_fmt", "yuv420p",
                        
    #                     # Audio settings
    #                     "-c:a", "aac",
    #                     "-b:a", "192k",
                        
    #                     # Container settings
    #                     "-movflags", "faststart"
    #                 ]

    #                 chunk.write_videofile(
    #                     str(chunk_path),
    #                     fps=self.FRAME_RATE,
    #                     codec="h264_nvenc",
    #                     audio_codec="aac",
    #                     ffmpeg_params=ffmpeg_params,
    #                     logger=None
    #                 )
                    
    #                 return str(chunk_path)

    #             except Exception as e:
    #                 logging.error(f"Error processing chunk: {str(e)}")
    #                 return None
    #             finally:
    #                 # Clean up clips
    #                 for clip in active_clips:
    #                     try:
    #                         clip.close()
    #                     except:
    #                         pass
    #                 gc.collect()

    #         # Process chunks in parallel
    #         with ThreadPoolExecutor() as executor:
    #             future_to_chunk = {
    #                 executor.submit(process_chunk, chunk_idx): chunk_idx 
    #                 for chunk_idx in range(total_chunks)
    #             }
                
    #             # Wait for all chunks to complete
    #             for future in as_completed(future_to_chunk):
    #                 chunk_idx = future_to_chunk[future]
    #                 try:
    #                     chunk_path = future.result()
    #                     if chunk_path:
    #                         chunk_files.append(chunk_path)
    #                         logging.info(f"Chunk {chunk_idx} completed: {chunk_path}")
    #                 except Exception as e:
    #                     logging.error(f"Chunk {chunk_idx} failed: {str(e)}")

    #         # Sort and combine chunks
    #         if chunk_files:
    #             chunk_files.sort(key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
    #             return self._combine_chunks(chunk_files)
            
    #         return None

    #     except Exception as e:
    #         logging.error(f"Composition error: {str(e)}")
    #         return None
        
    # def create_composition(self):
    #     try:
    #         full_chunks, final_duration = self.calculate_chunk_lengths()
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
    #         rows, cols = self.get_track_layout()
            
    #         chunk_files = []
    #         self.chunk_clips = {}  # Initialize as instance variable

    #         def process_chunk(chunk_idx):
    #             try:
    #                 active_clips = []
    #                 start_time = chunk_idx * 10
    #                 end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                    
    #                 logging.info(f"\nProcessing Chunk {chunk_idx}")
    #                 logging.info(f"Time Range: {start_time}-{end_time}")
                    
    #                 grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                                 color=(0,0,0), 
    #                                 duration=end_time - start_time) 
    #                         for _ in range(cols)] 
    #                         for _ in range(rows)]

    #                 # Process tracks within the chunk
    #                 for track_idx, track in enumerate(self.midi_data['tracks']):
    #                     if not track.get('notes'):
    #                         continue
                            
    #                     chunk_notes = [
    #                         note for note in track.get('notes', [])
    #                         if start_time <= float(note['time']) < end_time
    #                     ]

    #                     if not chunk_notes:
    #                         continue

    #                     if is_drum_kit(track.get('instrument', {})):
    #                         self._process_drum_chunk(track_idx, chunk_notes, grid, active_clips, start_time)
    #                     else:
    #                         self._process_instrument_chunk(track_idx, track, chunk_notes, grid, active_clips, start_time, end_time)

    #                 # Create chunk after processing ALL tracks
    #                 chunk = clips_array(grid)
    #                 chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
    #                 chunk.write_videofile(
    #                     str(chunk_path),
    #                     fps=30,
    #                     codec='h264_nvenc',
    #                     audio_codec='aac',
    #                     preset='fast',
    #                     ffmpeg_params=[
    #                         "-vsync", "1",
    #                         "-async", "1",
    #                         "-b:v", "5M",
    #                         "-maxrate", "10M",
    #                         "-bufsize", "10M",
    #                         "-rc", "vbr",
    #                         "-tune", "hq"
    #                     ]
    #                 )
                    
    #                 return str(chunk_path)

    #             except Exception as e:
    #                 logging.error(f"Error processing chunk: {str(e)}")
    #                 return None
    #             finally:
    #                 # Clean up clips
    #                 for clip in active_clips:
    #                     try:
    #                         clip.close()
    #                     except:
    #                         pass

    #         # Process chunks in parallel
    #         with ThreadPoolExecutor() as executor:
    #             future_to_chunk = {
    #                 executor.submit(process_chunk, chunk_idx): chunk_idx 
    #                 for chunk_idx in range(total_chunks)
    #             }
                
    #             # Wait for all chunks to complete
    #             for future in as_completed(future_to_chunk):
    #                 chunk_idx = future_to_chunk[future]
    #                 try:
    #                     chunk_path = future.result()
    #                     if chunk_path:
    #                         chunk_files.append(chunk_path)
    #                         logging.info(f"Chunk {chunk_idx} completed: {chunk_path}")
    #                 except Exception as e:
    #                     logging.error(f"Chunk {chunk_idx} failed: {str(e)}")

    #         # Sort and combine chunks
    #         if chunk_files:
    #             chunk_files.sort(key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
    #             return self._combine_chunks(chunk_files)
            
    #         return None

    #     except Exception as e:
    #         logging.error(f"Composition error: {str(e)}")
    #         return None

    # def create_composition(self):
    #     try:
    #         full_chunks, final_duration = self.calculate_chunk_lengths()
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
    #         rows, cols = self.get_track_layout()
            
    #         chunk_files = []
    #         self.chunk_clips = {}  # Initialize as instance variable

    #         def process_chunk(chunk_idx):
    #             try:
    #                 active_clips = []
    #                 start_time = chunk_idx * 10
    #                 end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                    
    #                 logging.info(f"\nProcessing Chunk {chunk_idx}")
    #                 logging.info(f"Time Range: {start_time}-{end_time}")
                    
    #                 grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                                 color=(0,0,0), 
    #                                 duration=end_time - start_time) 
    #                         for _ in range(cols)] 
    #                         for _ in range(rows)]

    #                 # Process tracks within the chunk
    #                 for track_idx, track in enumerate(self.midi_data['tracks']):
    #                     if not track.get('notes'):
    #                         continue
                            
    #                     chunk_notes = [
    #                         note for note in track.get('notes', [])
    #                         if start_time <= float(note['time']) < end_time
    #                     ]

    #                     if not chunk_notes:
    #                         continue

    #                     if is_drum_kit(track.get('instrument', {})):
    #                         self._process_drum_chunk(track_idx, chunk_notes, grid, active_clips, start_time)
    #                     else:
    #                         self._process_instrument_chunk(track_idx, track, chunk_notes, grid, active_clips, start_time, end_time)

    #                 # Create chunk after processing ALL tracks
    #                 chunk = clips_array(grid)
    #                 chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
    #                 chunk.write_videofile(
    #                     str(chunk_path),
    #                     fps=30,
    #                     codec='h264_nvenc',
    #                     audio_codec='aac',
    #                     preset='medium',
    #                     ffmpeg_params=["-vsync", "1", "-async", "1"]
    #                 )
                    
    #                 return str(chunk_path)

    #             except Exception as e:
    #                 logging.error(f"Error processing chunk: {str(e)}")
    #                 return None
    #             finally:
    #                 # Clean up clips
    #                 for clip in active_clips:
    #                     try:
    #                         clip.close()
    #                     except:
    #                         pass

    #         # Process chunks in parallel
    #         with ThreadPoolExecutor() as executor:
    #             future_to_chunk = {
    #                 executor.submit(process_chunk, chunk_idx): chunk_idx 
    #                 for chunk_idx in range(total_chunks)
    #             }
                
    #             # Wait for all chunks to complete
    #             for future in as_completed(future_to_chunk):
    #                 chunk_idx = future_to_chunk[future]
    #                 try:
    #                     chunk_path = future.result()
    #                     if chunk_path:
    #                         chunk_files.append(chunk_path)
    #                         logging.info(f"Chunk {chunk_idx} completed: {chunk_path}")
    #                 except Exception as e:
    #                     logging.error(f"Chunk {chunk_idx} failed: {str(e)}")

    #         # Sort and combine chunks
    #         if chunk_files:
    #             chunk_files.sort(key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
    #             return self._combine_chunks(chunk_files)
            
    #         return None

    #     except Exception as e:
    #         logging.error(f"Composition error: {str(e)}")
    #         return None

                
    def _combine_chunks(self, chunk_files):
        """Combine chunks with precise timing"""
        try:
            # Validate grid arrangement
            if 'gridArrangement' in self.midi_data:
                arrangement = self.midi_data['gridArrangement']
                if not all('position' in pos for pos in arrangement.values()):
                    logging.warning("Invalid grid arrangement, using default layout")
                    self.midi_data['gridArrangement'] = {}
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
                preset='fast',
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
            

# def compose_from_processor_output(processor_result, output_path):
#     try:
#         base_dir = processor_result['processed_videos_dir']
#         logging.info(f"Using base directory: {base_dir}")
#         logging.info("=== Processing Grid Arrangement ===")
#         logging.info(f"Grid arrangement in processor result: {processor_result.get('tracks', {}).get('gridArrangement')}")

#         logging.info("=== Processing Grid Arrangement ===")
#         logging.info(f"Processor result tracks: {processor_result['tracks']}")
#         logging.info(f"Grid arrangement in tracks: {processor_result['tracks'].get('gridArrangement')}")
        
#         # Store validated tracks
#         validated_tracks = {}
        
#         if 'tracks' in processor_result['processed_files']:
#             for instrument, data in processor_result['processed_files']['tracks'].items():
#                 try:
#                     track_idx = data.get('track_idx', 0)
                    
#                     # Keep full instrument name, just remove track index if present
#                     instrument_parts = instrument.split('_')
#                     if instrument_parts[-1].isdigit():
#                         instrument_name = '_'.join(instrument_parts[:-1])
#                     else:
#                         instrument_name = instrument
                    
#                     # Construct track path preserving full instrument name
#                     track_path = os.path.join(
#                         base_dir, 
#                         f"track_{track_idx}_{instrument_name}"
#                     )
                    
#                     logging.info(f"Checking track {track_idx}: {instrument_name}")
#                     logging.info(f"Track path: {track_path}")
                    
#                     if os.path.exists(track_path):
#                         validated_tracks[instrument] = {
#                             'base_path': track_path,
#                             'instrument_name': instrument_name,  # Store full name
#                             'notes': data.get('notes', {})
#                         }
                        
#                 except Exception as e:
#                     logging.error(f"Error processing track {instrument}: {str(e)}")
#                     continue

#         composer = VideoComposer(base_dir, processor_result['tracks'], output_path)
#         composer.tracks = validated_tracks
#         return composer.create_composition()

#     except Exception as e:
#         logging.error(f"Error in video composition: {str(e)}")
#         raise

def compose_from_processor_output(processor_result, output_path, use_av1=False):
    try:
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

        composer = VideoComposer(base_dir, processor_result['tracks'], output_path, use_av1=use_av1)
        composer.tracks = validated_tracks
        return composer.create_composition()

    except Exception as e:
        logging.error(f"Error in video composition: {str(e)}")
        raise