import os
import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import shutil

# Third-party imports
import numpy as np
import cv2
from moviepy.editor import (
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

        # tracks = midi_data['tracks']
        # self.tracks = {}
        # self.drum_tracks = []
        # self.regular_tracks = []
        
        # # First, process and copy drum videos
        # logging.info("\n=== Processing Drum Videos ===")
        # for idx, track in enumerate(tracks if isinstance(tracks, list) else tracks.values()):
        #     track_id = str(idx)
        #     normalized_track = self._normalize_track(track)
            
        #     # Check if it's a drum track
        #     is_drum = (
        #         normalized_track.get('isDrum') or 
        #         normalized_track.get('instrument', {}).get('isDrum') or
        #         normalized_track.get('channel') == 9 or
        #         any(name in normalized_track.get('instrument', {}).get('name', '').lower() 
        #             for name in ['drum', 'percussion', 'kit'])
        #     )
            
        #     if is_drum:
        #         logging.info(f"\nProcessing drum track {track_id}: {normalized_track.get('instrument', {}).get('name')}")
                
        #         # Create drum directory
        #         drum_dir = self.processed_videos_dir / f"track_{idx}_drums"
        #         drum_dir.mkdir(exist_ok=True)
        #         logging.info(f"Created drum directory: {drum_dir}")
                
        #         # Get unique drum types needed from notes
        #         needed_drums = set()
        #         for note in normalized_track.get('notes', []):
        #             midi_note = note.get('midi')
        #             drum_name = DRUM_NOTES.get(midi_note)
        #             if drum_name:
        #                 needed_drums.add((midi_note, drum_name))
        #                 logging.info(f"Found drum note: {midi_note} -> {drum_name}")
                
        #         logging.info(f"\nUploads directory contents:")
        #         for file in self.uploads_dir.glob('*.mp4'):
        #             logging.info(f"Found file: {file}")
        #         logging.info(f"Looking for drum files in uploads: {self.uploads_dir}")

        #         # Search both directories for drum files
        #         for midi_note, drum_name in needed_drums:
        #             normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
        #             source_file = None
                    
        #             # First check uploads directory
        #             logging.info(f"Looking for drum file pattern in uploads: *{normalized_name}.mp4")
        #             for file in self.uploads_dir.glob(f"*{normalized_name}.mp4"):
        #                 source_file = file
        #                 logging.info(f"Found drum file in uploads: {file}")
        #                 break
                        
        #             # If not found, check processed directory
        #             if not source_file:
        #                 logging.info(f"Looking for drum file pattern in processed: *{normalized_name}.mp4")
        #                 for file in self.processed_videos_dir.glob(f"*{normalized_name}.mp4"):
        #                     source_file = file
        #                     logging.info(f"Found drum file in processed: {file}")
        #                     break

        #             if source_file:
        #                 dest_file = drum_dir / f"{normalized_name}.mp4"
        #                 if not dest_file.exists():
        #                     shutil.copy2(str(source_file), str(dest_file))
        #                     logging.info(f"Copied drum file: {source_file} -> {dest_file}")
        #             else:
        #                 logging.error(f"Missing drum file for {drum_name} ({normalized_name})")
                
        #         self.drum_tracks.append(normalized_track)
        #     else:
        #         self.regular_tracks.append(normalized_track)
        #         self.tracks[track_id] = normalized_track

        # logging.info(f"\nProcessed {len(self.tracks)} regular tracks and {len(self.drum_tracks)} drum tracks")
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
                            shutil.copy2(str(source_file), str(dest_file))
                            logging.info(f"Copied drum file: {source_file.name} -> {dest_file.name}")
                    else:
                        logging.error(f"Missing drum file for {drum_name}")
                        logging.error(f"Looked for pattern: {normalized_name} in files: {[f.name for f in upload_files]}")
                
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
        
    
    # def _setup_track_configuration(self):
    #     """Setup track-specific configuration and grid layout"""
    #     try:
    #         # Count valid tracks including drums
    #         valid_tracks = []
    #         for track_idx, track in enumerate(self.tracks.values()):
    #             if track.get('notes'):
    #                 valid_tracks.append((track_idx, track))
                    
    #         # Add debug logging for drum tracks
    #         logging.info("\n=== DRUM TRACK CONFIGURATION ===")
    #         for track_idx, track in enumerate(self.drum_tracks):
    #             logging.info(f"\nDrum Track {track_idx}:")
    #             logging.info(f"Track data: {track}")
                
    #             if track.get('notes'):
    #                 logging.info("Processing drum notes:")
    #                 drum_notes = {}
    #                 for note in track.get('notes', []):
    #                     logging.info(f"Note type: {type(note)}")
    #                     logging.info(f"Raw note data: {note}")
                        
    #                     # Handle both dict and int midi note formats safely
    #                     if isinstance(note, dict):
    #                         midi_note = note.get('midi')
    #                         logging.info(f"Dict note - midi value: {midi_note}")
    #                     else:
    #                         midi_note = int(note)  # Ensure it's an integer
    #                         logging.info(f"Direct note value: {midi_note}")
                            
    #                     drum_group = get_drum_groups(midi_note)
    #                     logging.info(f"Mapped to drum group: {drum_group}")
                        
    #                     if drum_group:
    #                         if drum_group not in drum_notes:
    #                             drum_notes[drum_group] = []
    #                         drum_notes[drum_group].append(note)
                    
    #                 # Log drum groups found
    #                 logging.info(f"Found drum groups: {list(drum_notes.keys())}")
                    
    #                 # Add each drum group as separate track
    #                 for drum_group in drum_notes:
    #                     valid_tracks.append((f"drum_{drum_group}", track))

    #         logging.info(f"\nTotal valid tracks (including drums): {len(valid_tracks)}")
            
    #         # Setup grid positions
    #         total_slots = 0
    #         self.grid_positions = {}
            
    #         for track_id, track in valid_tracks:
    #             if isinstance(track_id, str) and track_id.startswith('drum_'):
    #                 # Handle drum tracks
    #                 logging.info(f"Position {total_slots}: {track_id}")
    #             else:
    #                 # Handle regular tracks
    #                 instrument_name = track.get('instrument', {}).get('name', f'track_{track_id}')
    #                 logging.info(f"Position {total_slots}: {instrument_name} (track_{track_id})")
                
    #             self.grid_positions[track_id] = total_slots
    #             total_slots += 1

    #     except Exception as e:
    #         logging.error(f"Layout error: {str(e)}")
    #         raise

    def _setup_track_configuration(self):
        """Setup track-specific configuration and grid layout"""
        try:
            valid_tracks = []
            
            # First handle regular tracks
            for track_idx, track in enumerate(self.tracks.values()):
                if track.get('notes'):
                    valid_tracks.append((track_idx, track))
            
            # Handle drum tracks
            logging.info("\n=== DRUM TRACK CONFIGURATION ===")
            for track_idx, track in enumerate(self.drum_tracks):
                logging.info(f"\nDrum Track {track_idx}:")
                
                if track.get('notes'):
                    logging.info("Processing drum notes:")
                    drum_notes = {}
                    
                    for note in track.get('notes', []):
                        logging.info(f"Note type: {type(note)}")
                        logging.info(f"Raw note data: {note}")
                        
                        # Get MIDI note number
                        if isinstance(note, dict):
                            midi_note = note.get('midi')
                        else:
                            midi_note = int(note)
                        
                        # Use DRUM_NOTES mapping to get drum name
                        from drum_utils import DRUM_NOTES
                        drum_name = DRUM_NOTES.get(midi_note)
                        
                        if drum_name:
                            logging.info(f"MIDI note {midi_note} maps to drum: {drum_name}")
                            position_key = f"drum_{drum_name.lower().replace(' ', '_')}"
                            
                            if position_key not in drum_notes:
                                drum_notes[position_key] = []
                            drum_notes[position_key].append(note)
                            valid_tracks.append((position_key, track))
                        else:
                            logging.warning(f"No drum mapping found for MIDI note: {midi_note}")
            
            # Setup grid positions
            total_slots = 0
            self.grid_positions = {}
            
            for track_id, track in valid_tracks:
                self.grid_positions[str(track_id)] = total_slots
                total_slots += 1
                
            logging.info(f"\nGrid positions: {self.grid_positions}")
            
            return len(valid_tracks)
            
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

    # def get_track_duration(self, track):
    #     """Get duration for any track type"""
    #     try:
    #         notes = track.get('notes', [])
    #         if not notes:
    #             return 0
    #         return max(float(note['time']) + float(note['duration']) 
    #                 for note in notes)
    #     except Exception as e:
    #         logging.error(f"Error calculating track duration: {e}")
    #         return 0

    def get_track_duration(self, track):
        """Get duration for any track type"""
        try:
            notes = track.get('notes', [])
            if not notes:
                return 0
                
            # For drum tracks, consider all drum hits
            if track.get('isDrum'):
                all_times = [float(note['time'] + note.get('duration', 0.5)) 
                            for note in notes]
                return max(all_times) if all_times else 0
                
            # For regular tracks
            end_times = [float(note['time'] + note.get('duration', 0)) 
                        for note in notes]
            return max(end_times) if end_times else 0
                
        except Exception as e:
            logging.error(f"Error calculating track duration: {str(e)}")
            return 0
        
    

    
    # def calculate_chunk_lengths(self, midi_data):
    #     try:
    #         # Find last note end time
    #         last_note_time = 0
    #         for track in midi_data['tracks']:
    #             for note in track.get('notes', []):
    #                 note_end = float(note['time']) + float(note['duration'])
    #                 last_note_time = max(last_note_time, note_end)
            
    #         # Calculate chunks
    #         CHUNK_SIZE = 10  # seconds
    #         full_chunks = int(last_note_time // CHUNK_SIZE)
    #         final_chunk_duration = last_note_time % CHUNK_SIZE
            
    #         logging.info(f"Total duration: {last_note_time}")
    #         logging.info(f"Full chunks: {full_chunks}")
    #         logging.info(f"Final chunk duration: {final_chunk_duration}")
            
    #         return full_chunks, final_chunk_duration
            
    #     except Exception as e:
    #         logging.error(f"Error calculating chunks: {str(e)}")
    #         return None, None

    # def calculate_chunk_lengths(self, midi_data):
    #     try:
    #         if not self.tracks:
    #             raise ValueError("No tracks available")
                
    #         durations = []
    #         for track_id, track in self.tracks.items():
    #             duration = self.get_track_duration(track)
    #             if duration > 0:
    #                 durations.append(duration)
    #                 logging.info(f"Track {track_id} duration: {duration}")
                    
    #         if not durations:
    #             raise ValueError("No valid durations found")
                
    #         total_duration = max(durations)
    #         chunk_size = 30  # 30 second chunks
            
    #         chunks = []
    #         current_time = 0
    #         while current_time < total_duration:
    #             end_time = min(current_time + chunk_size, total_duration)
    #             chunks.append((current_time, end_time))
    #             current_time += chunk_size
                
    #         logging.info(f"Created {len(chunks)} chunks for total duration {total_duration}")
    #         return chunks
            
    #     except Exception as e:
    #         logging.error(f"Error in calculate_chunk_lengths: {str(e)}")
    #         return [(0, 30)]  # Return default chunk as fallback

    def calculate_chunk_lengths(self):
        """Calculate chunk lengths for composition"""
        try:
            # Include both regular and drum tracks
            all_tracks = list(self.tracks.values()) + self.drum_tracks
            durations = []
            
            for track in all_tracks:
                duration = self.get_track_duration(track)
                logging.info(f"Track duration: {duration} ({track.get('instrument', {}).get('name')})")
                if duration > 0:
                    durations.append(duration)
            
            if not durations:
                raise ValueError("No valid track durations found")
                
            total_duration = max(durations)
            full_chunks = int(total_duration // self.CHUNK_DURATION)
            final_chunk = total_duration % self.CHUNK_DURATION
            
            logging.info(f"Calculated chunks: {full_chunks} full + {final_chunk:.2f}s remaining")
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
            
            # First pass - Count all tracks including drums
            for track_idx, track in valid_tracks:
                if is_drum_kit(track.get('instrument', {})):
                    # Get all drum groups used in this track
                    drum_groups = set()
                    for note in track.get('notes', []):
                        drum_group = get_drum_groups(note.get('midi'))
                        if drum_group:
                            drum_groups.add(drum_group)
                    
                    # Now process each drum group
                    for drum_group in drum_groups:
                        position_key = f"drum_{drum_group}"
                        self.grid_positions[position_key] = total_slots
                        logging.info(f"Position {total_slots}: Drum {drum_group} (track_{track_idx})")
                        total_slots += 1
                else:
                    # Regular instrument track processing remains the same
                    self.grid_positions[f"track_{track_idx}"] = total_slots
                    instrument_name = normalize_instrument_name(track.get('instrument', {}).get('name', 'unknown'))
                    logging.info(f"Position {total_slots}: {instrument_name} (track_{track_idx})")
                    total_slots += 1
            
            cols = min(4, math.ceil(math.sqrt(total_slots)))
            rows = math.ceil(total_slots / cols)
            
            # Visual grid representation
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
        
    # def process_chunk(self, tracks, start_time, end_time, chunk_idx):
    #     """Process chunk with overlapping notes"""
    #     try:
            
    #         # Extend chunk time range to include overlap
    #         actual_start = start_time - (self.OVERLAP_DURATION if chunk_idx > 0 else 0)
    #         actual_end = end_time + self.OVERLAP_DURATION
            
    #         # Get grid dimensions
    #         rows, cols = self.get_track_layout()
    #         grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                         color=(0,0,0), 
    #                         duration=actual_end - actual_start) 
    #                 for _ in range(cols)] 
    #                 for _ in range(rows)]

    #         # Get notes including overlap periods
    #         for track_idx, track in enumerate(self.midi_data['tracks']):
    #             chunk_notes = self.get_chunk_notes(
    #                 track, 
    #                 actual_start, 
    #                 actual_end, 
    #                 include_overlap=True
    #             )
                
    #             # Process notes with adjusted timing
    #             for note in chunk_notes:
    #                 # Adjust note timing relative to chunk start
    #                 note_time = float(note['time']) - actual_start
    #                 note_duration = float(note.get('duration', 0))
    #                 # Ensure notes at chunk boundaries aren't cut off
    #                 if chunk_idx > 0 and note_time < self.OVERLAP_DURATION:
    #                     note_time = max(0, note_time)

    #         # Get grid dimensions
    #         rows = math.ceil(len(tracks) / 2)
    #         cols = 2
    #         grid = [[None for _ in range(cols)] for _ in range(rows)]
            
    #         for track_idx, track in enumerate(tracks):
    #             row = track_idx // 2
    #             col = track_idx % 2
                
    #             # Get notes including overlaps
    #             notes = self.get_chunk_notes(track, start_time, end_time, include_overlap=True)
    #             clips_for_instrument = []
                
    #             for note in notes:
    #                 # Adjust timing relative to chunk start
    #                 note_start = float(note['time']) - start_time
    #                 note_duration = float(note.get('duration', 0))
                    
    #                 # Load and position clip
    #                 note_clip = self.get_note_clip(track, note)
    #                 if note_clip:
    #                     positioned_clip = note_clip.set_start(note_start)
    #                     positioned_clip = positioned_clip.set_duration(note_duration)
    #                     clips_for_instrument.append(positioned_clip)
                
    #             if clips_for_instrument:
    #                 composite = CompositeVideoClip(clips_for_instrument)
    #                 grid[row][col] = composite.set_duration(end_time - start_time)
                    
    #         # Create chunk with crossfade
    #         chunk = clips_array(grid)
    #         chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
    #         # Add small crossfade buffer
    #         if chunk_idx > 0:
    #             crossfade_duration = 0.1  # 100ms crossfade
    #             chunk = chunk.crossfadein(crossfade_duration)
                
    #         chunk.write_videofile(
    #             str(chunk_path),
    #             fps=30,
    #             codec='h264_nvenc',
    #             audio_codec='aac',
    #             preset='medium',
    #             ffmpeg_params=[
    #                 "-vsync", "1",
    #                 "-async", "1", 
    #                 "-b:v", "5M"
    #             ]
    #         )

    #          # Add crossfade
    #         if chunk_idx > 0:
    #             chunk = chunk.crossfadein(self.CROSSFADE_DURATION)
            
    #         return str(chunk_path)
            
    #     except Exception as e:
    #         logging.error(f"Error processing chunk: {e}")
    #         return None

    # def process_chunk(self, tracks, start_time, end_time, chunk_idx):
    #     """Process chunk with memory optimization"""
    #     try:
    #         active_clips = []  # Track active clips for cleanup
    #         rows, cols = self.get_track_layout()
            
    #         # Pre-allocate grid with empty clips
    #         grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                         color=(0,0,0), 
    #                         duration=end_time - start_time) 
    #                 for _ in range(cols)] 
    #                 for _ in range(rows)]

    #         for track_idx, track in enumerate(tracks):
    #             chunk_notes = self.get_chunk_notes(track, start_time, end_time)
                
    #             # Process in smaller batches
    #             BATCH_SIZE = 5
    #             for i in range(0, len(chunk_notes), BATCH_SIZE):
    #                 batch = chunk_notes[i:i + BATCH_SIZE]
                    
    #                 for note in batch:
    #                     try:
    #                         clip = self.get_note_clip(track, note)
    #                         if clip:
    #                             active_clips.append(clip)
    #                             # Process immediately
    #                             time = float(note['time']) - start_time
    #                             clip = clip.set_start(time)
    #                             row = track_idx // cols
    #                             col = track_idx % cols
                                
    #                             if isinstance(grid[row][col], ColorClip):
    #                                 grid[row][col] = clip
    #                             else:
    #                                 existing = grid[row][col]
    #                                 grid[row][col] = CompositeVideoClip([existing, clip])
                        
    #                     finally:
    #                         # Cleanup after each note
    #                         for clip in active_clips:
    #                             try:
    #                                 clip.close()
    #                             except:
    #                                 pass
    #                         active_clips.clear()
    #                         gc.collect()

    #         # Create chunk with optimized parameters
    #         chunk = clips_array(grid)
    #         chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
    #         chunk.write_videofile(
    #             str(chunk_path),
    #             fps=30,
    #             codec='h264_nvenc',
    #             audio_codec='aac',
    #             preset='fast',  # Faster encoding
    #             ffmpeg_params=[
    #                 "-vsync", "1",
    #                 "-async", "1",
    #                 "-b:v", "5M",
    #                 "-tile-columns", "2",
    #                 "-threads", "8",
    #                 "-row-mt", "1"
    #             ]
    #         )
            
    #         return str(chunk_path)
            
    #     except Exception as e:
    #         logging.error(f"Error processing chunk: {e}")
    #         return None
            
    #     finally:
    #         # Final cleanup
    #         for clip in active_clips:
    #             try:
    #                 clip.close()
    #             except:
    #                 pass
    #         gc.collect()

    def process_chunk(self, tracks, start_time, end_time, chunk_idx):
        """Process chunk with memory optimization"""
        try:
            active_clips = []
            rows, cols = self.get_track_layout()
            
            # Pre-allocate grid
            grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                            color=(0,0,0), 
                            duration=end_time - start_time) 
                    for _ in range(cols)] 
                    for _ in range(rows)]

            # Convert tracks to dict if needed
            track_dict = tracks if isinstance(tracks, dict) else {
                str(idx): track for idx, track in enumerate(tracks)
            }

            for track_id, track in track_dict.items():
                try:
                    if not isinstance(track, dict):
                        logging.warning(f"Skipping invalid track {track_id}")
                        continue

                    track_idx = int(track_id)
                    chunk_notes = self.get_chunk_notes(track, start_time, end_time)
                    
                    # Process in smaller batches
                    BATCH_SIZE = 5
                    for i in range(0, len(chunk_notes), BATCH_SIZE):
                        batch = chunk_notes[i:i + BATCH_SIZE]
                        
                        for note in batch:
                            try:
                                # Handle drum tracks differently
                                if track.get('isDrum'):
                                    clip = self.get_drum_clip(track, note)
                                else:
                                    clip = self.get_note_clip(track, note)

                                if clip:
                                    active_clips.append(clip)
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

                except Exception as e:
                    logging.error(f"Error processing track {track_id}: {str(e)}")
                    continue

            # Create chunk with optimized parameters
            chunk = clips_array(grid)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
            chunk.write_videofile(
                str(chunk_path),
                fps=30,
                codec='h264_nvenc',
                audio_codec='aac',
                preset='fast',
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
        

    # def create_composition(self):
    #     try:
    #         full_chunks, final_duration = self.calculate_chunk_lengths()
    #         if full_chunks == 0 and final_duration == 0:
    #             raise ValueError("No valid chunks calculated")
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
    #         rows, cols = self.get_track_layout()
            
    #         chunk_files = []
    #         active_clips = []
    #         for chunk_idx in range(total_chunks):
    #             start_time = chunk_idx * 10
    #             end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                
    #             # Debug grid creation
    #             logging.info(f"\nProcessing Chunk {chunk_idx}")
    #             logging.info(f"Time Range: {start_time}-{end_time}")
                
    #             grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                             color=(0,0,0), 
    #                             duration=end_time - start_time) 
    #                     for _ in range(cols)] 
    #                     for _ in range(rows)]
                
    #             # Process each track
    #             for track_idx, track in enumerate(self.midi_data['tracks']):
    #                 chunk_notes = [
    #                     note for note in track.get('notes', [])
    #                     if start_time <= float(note['time']) < end_time
    #                 ]
                    
    #                 logging.info(f"\nTrack {track_idx}: {track.get('instrument', {}).get('name', 'unknown')}")
    #                 logging.info(f"Notes in chunk: {len(chunk_notes)}")

    #                 if not chunk_notes:
    #                     continue
                    
    #                 if is_drum_kit(track.get('instrument', {})):
    #                     for note in track.get('notes', []):
    #                         drum_group = self.get_drum_group(note.get('midi', 0))  # Add this line
    #                         position_key = f"drum_{drum_group}"
    #                     drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
    #                     if drum_dir.exists():
    #                         drum_notes = {}
    #                         for note in chunk_notes:
    #                             drum_group = self.get_drum_group_for_note(note['midi'])
    #                             logging.info(f"Processing drum note {note['midi']} -> group {drum_group}")
    #                             if drum_group:
    #                                 if drum_group not in drum_notes:
    #                                     drum_notes[drum_group] = []
    #                                 drum_notes[drum_group].append(note)

    #                         for drum_group, notes in drum_notes.items():
    #                             # position_key = f"drum_{track_idx}_{drum_group}"
    #                             logging.info(f"Processing drum group {drum_group} at position {position_key}")
    #                             if position_key in self.grid_positions:
    #                                 position = self.grid_positions[position_key]
    #                                 row = position // cols
    #                                 col = position % cols
    #                                 drum_file = drum_dir / f"{drum_group}.mp4"
    #                                 if drum_file.exists():
    #                                     for note in notes:
    #                                         try:
    #                                             clip = VideoFileClip(str(drum_file))
    #                                             active_clips.append(clip)
                                                
    #                                             # Extract and normalize velocity
    #                                             velocity = float(note.get('velocity', 100))
    #                                             volume = self.get_note_volume(velocity, is_drum=True)
                                                
    #                                             # Apply volume adjustment
    #                                             clip = clip.volumex(volume)
                                                
    #                                             # Log for debugging
    #                                             logging.info(f"Drum {drum_group} velocity={velocity:.2f}, volume={volume:.2f}")
                                                
    #                                             time = float(note['time']) - start_time
    #                                             duration = min(float(note['duration']), clip.duration)
    #                                             clip = clip.subclip(0, duration).set_start(time)
                                                
    #                                             if isinstance(grid[row][col], ColorClip):
    #                                                 grid[row][col] = clip
    #                                             else:
    #                                                 existing = grid[row][col]
    #                                                 grid[row][col] = CompositeVideoClip([existing, clip])
                                                
    #                                             logging.info(f"Added drum {drum_group} at [{row}][{col}] t={time}")
    #                                         except Exception as e:
    #                                             logging.error(f"Error processing drum clip: {e}")
    #                                             continue
                
    #                 else:
    #                     position_key = f"track_{track_idx}"
    #                     if position_key in self.grid_positions:
    #                         position = self.grid_positions[position_key]
    #                         row = position // cols
    #                         col = position % cols
                            
    #                         clips_for_instrument = []
    #                         for note in chunk_notes:
    #                             midi_note = int(float(note['midi']))
    #                             note_file = self.processed_videos_dir / f"track_{track_idx}_{normalize_instrument_name(track['instrument']['name'])}" / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                                
    #                             logging.info(f"Loading note: {note_file}")
                                
    #                             if note_file.exists():
    #                                 clip = VideoFileClip(str(note_file))
    #                                 active_clips.append(clip)
                                    
    #                                 # Get and normalize MIDI velocity
    #                                 velocity = float(note.get('velocity', 100))
    #                                 volume = self.get_note_volume(velocity, is_drum=False)  # Ensure minimum audible volume
                                    
    #                                 # Apply volume to clip
    #                                 clip = clip.volumex(volume)
                                    
    #                                 # Log volume level for debugging
    #                                 logging.info(f"Note {note.get('midi')}: velocity={velocity:.2f}, volume={volume:.2f}")
                                    
    #                                 time = float(note['time']) - start_time
    #                                 duration = min(float(note['duration']), clip.duration)
    #                                 clip = clip.subclip(0, duration).set_start(time)
    #                                 clips_for_instrument.append(clip)
                            
    #                         if clips_for_instrument:
    #                             composite = CompositeVideoClip(clips_for_instrument)
    #                             grid[row][col] = composite.set_duration(end_time - start_time)
    #                             logging.info(f"Added instrument at [{row}][{col}]")
                
    #             # Create chunk
    #             chunk = clips_array(grid)
    #             chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
    #             chunk.write_videofile(
    #                 str(chunk_path),
    #                 fps=30,
    #                 codec='h264_nvenc',  # NVIDIA GPU encoder
    #                 audio_codec='aac',
    #                 preset='medium',
    #                 ffmpeg_params=[
    #                     "-vsync", "1",
    #                     "-async", "1",
    #                     "-b:v", "5M",
    #                     "-maxrate", "10M",
    #                     "-bufsize", "10M",
    #                     "-rc", "vbr",
    #                     "-tune", "hq"
    #                 ]
    #             )
    #             chunk_files.append(str(chunk_path))
                
    #             # Aggressive cleanup after each chunk
    #             for clip in active_clips:
    #                 try:
    #                     clip.close()
    #                 except:
    #                     pass
    #             active_clips.clear()
                
    #             # Force garbage collection
    #             gc.collect()
            
    #         # Combine chunks
    #         if chunk_files:
    #             logging.info(f"Attempting to concatenate {len(chunk_files)} chunks")
    #             logging.info(f"Chunk files: {chunk_files}")
    #             return self._combine_chunks(chunk_files)
           
                
    #     except Exception as e:
    #         logging.error(f"Composition error: {str(e)}")
    #         return None
    def create_composition(self):
        """Main composition method"""
        try:
            # Calculate chunks
            full_chunks, final_duration = self.calculate_chunk_lengths()
            if full_chunks == 0 and final_duration == 0:
                raise ValueError("No valid chunks calculated")
                
            # Initialize grid dimensions
            total_tracks = len(self.tracks) + len(self.drum_tracks)
            self.cols = min(4, math.ceil(math.sqrt(total_tracks)))
            self.rows = math.ceil(total_tracks / self.cols)
            
            # Process chunks
            chunk_files = self._process_chunks(full_chunks, final_duration)
            
            # Combine final result
            if chunk_files:
                logging.info(f"Concatenating {len(chunk_files)} chunks")
                return self._combine_chunks(chunk_files)
            return None
                
        except Exception as e:
            logging.error(f"Composition error: {str(e)}")
            return None

    def _save_chunk(self, grid, chunk_idx, duration):
        """Save processed chunk to file"""
        try:
            chunk = clips_array(grid)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
            chunk.write_videofile(
                str(chunk_path),
                fps=30,
                codec='h264_nvenc',
                audio_codec='aac',
                preset='fast',
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
            logging.error(f"Error saving chunk: {str(e)}")
            return None
    
    def _process_chunks(self, full_chunks, final_duration):
        """Process each time chunk"""
        chunk_files = []
        active_clips = []
        total_chunks = full_chunks + (1 if final_duration > 0 else 0)
        
        for chunk_idx in range(total_chunks):
            try:
                start_time = chunk_idx * self.CHUNK_DURATION
                end_time = start_time + (final_duration if chunk_idx == full_chunks else self.CHUNK_DURATION)
                
                # Initialize empty grid
                grid = self._initialize_grid(end_time - start_time)
                
                # Process tracks for this time chunk
                grid = self._process_chunk_tracks(grid, start_time, end_time)
                
                # Create and save chunk
                chunk_path = self._save_chunk(grid, chunk_idx, end_time - start_time)
                if chunk_path:
                    chunk_files.append(str(chunk_path))
                
                # Cleanup
                self._cleanup_clips(active_clips)
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_idx}: {e}")
                continue
                
        return chunk_files

    def _initialize_grid(self, duration):
        """Initialize empty grid with proper dimensions"""
        return [[ColorClip(size=(1920//self.cols, 1080//self.rows), 
                color=(0,0,0), 
                duration=duration) 
                for _ in range(self.cols)] 
                for _ in range(self.rows)]

    def _process_chunk_tracks(self, grid, start_time, end_time):
        """Process all tracks for current time chunk"""
        try:
            # Process regular tracks first
            for track_id, track in self.tracks.items():
                try:
                    track_path = self.processed_videos_dir / f"{track.get('name', '')}_notes"
                    if not track_path.exists():
                        continue
                        
                    chunk_notes = [
                        note for note in track.get('notes', [])
                        if start_time <= float(note['time']) < end_time
                    ]
                    
                    if chunk_notes:
                        position = self.grid_positions.get(str(track_id))
                        if position is not None:
                            row = position // self.cols
                            col = position % self.cols
                            clips = []
                            
                            for note in chunk_notes:
                                midi_note = note['midi']
                                note_name = midi_to_note(midi_note)
                                note_file = track_path / f"note_{midi_note}_{note_name}.mp4"
                                
                                if note_file.exists():
                                    clip = VideoFileClip(str(note_file))
                                    time = float(note['time']) - start_time
                                    clip = clip.set_start(time)
                                    clips.append(clip)
                            
                            if clips:
                                grid[row][col] = CompositeVideoClip(clips)
                                
                except Exception as e:
                    logging.error(f"Error processing track {track_id}: {str(e)}")
                    continue

            # Then process drum tracks
            for track_idx, track in enumerate(self.drum_tracks):
                chunk_notes = [note for note in track.get('notes', [])
                            if start_time <= float(note['time']) < end_time]
                
                if chunk_notes:
                    grid = self._process_drum_track(grid, track_idx, track, chunk_notes, start_time)
                    
            return grid
                
        except Exception as e:
            logging.error(f"Error in _process_chunk_tracks: {str(e)}")
            return grid

    def _process_drum_track(self, grid, track_idx, track, chunk_notes, start_time):
        try:
            logging.info(f"\n=== DRUM TRACK PROCESSING ===")
            logging.info(f"Track Index: {track_idx}")
            
            # Create drum directory
            drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
            drum_dir.mkdir(exist_ok=True)
            logging.info(f"Created drum directory: {drum_dir}")

            # List ALL files in uploads directory first
            upload_files = list(self.uploads_dir.glob('*.mp4'))
            logging.info(f"\nUploads directory contents at {self.uploads_dir}:")
            for file in upload_files:
                logging.info(f"Found uploaded file: {file.name}")

            # Get unique drum types needed
            needed_drums = set()
            for note in chunk_notes:
                midi_note = note['midi']
                drum_name = DRUM_NOTES.get(midi_note)
                if drum_name:
                    needed_drums.add((midi_note, drum_name))
                    logging.info(f"Need drum video for: {drum_name} (MIDI: {midi_note})")

            # Process each needed drum
            for midi_note, drum_name in needed_drums:
                normalized_name = f"drum_{drum_name.lower().replace(' ', '_')}"
                logging.info(f"\nLooking for drum: {normalized_name}")
                
                # Look for matching file
                source_file = None
                for file in upload_files:
                    # Just check if normalized name is in the filename
                    if normalized_name in file.name.lower():
                        source_file = file
                        logging.info(f"Found matching drum file: {file.name}")
                        break

                if source_file:
                    dest_file = drum_dir / f"{normalized_name}.mp4"
                    if not dest_file.exists():
                        shutil.copy2(str(source_file), str(dest_file))
                        logging.info(f"Copied drum file: {source_file.name} -> {dest_file.name}")
                else:
                    logging.error(f"Missing drum file for {drum_name}")
                    logging.error(f"Looked for pattern: {normalized_name} in files: {[f.name for f in upload_files]}")

            # Verify copied files
            logging.info("\nVerifying copied drum files:")
            for file in drum_dir.glob('*.mp4'):
                logging.info(f"Found copied file: {file}")

            return grid
                
        except Exception as e:
            logging.error(f"Error in _process_drum_track: {str(e)}")
            return grid
                        
    def _verify_drum_files(self, drum_dir):
        """Verify drum files were copied correctly"""
        logging.info(f"\nVerifying drum files in: {drum_dir}")
        if drum_dir.exists():
            for file in drum_dir.glob('*.mp4'):
                logging.info(f"Found drum file: {file}")
                if file.stat().st_size == 0:
                    logging.error(f"Empty drum file: {file}")
        else:
            logging.error(f"Drum directory not found: {drum_dir}")

    def _process_regular_track(self, grid, track_idx, track, chunk_notes, start_time):
        """Process regular instrument track clips"""
        position_key = f"track_{track_idx}"
        if position_key not in self.grid_positions:
            return grid
            
        position = self.grid_positions[position_key]
        row = position // self.cols
        col = position % self.cols
        
        # Bounds check
        if row >= len(grid) or col >= len(grid[0]):
            logging.error(f"Position {position} out of grid bounds ({self.rows}x{self.cols})")
            return grid
            
        clips = self._create_instrument_clips(track_idx, track, chunk_notes, start_time)
        if clips:
            grid[row][col] = CompositeVideoClip(clips)
            
        return grid
            

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
