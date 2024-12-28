import os
import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import math

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
from drum_utils import is_drum_kit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_composition.log', mode='w')
    ]
)

class VideoComposer:

    def __init__(self, processed_videos_dir, midi_data, output_path):
        """Initialize VideoComposer with proper path handling"""
        try:
            # Debug logging
            logging.info(f"Input type: {type(processed_videos_dir)}")
            logging.info(f"Input value: {processed_videos_dir}")
            
            # Extract path from input
            if isinstance(processed_videos_dir, dict):
                logging.info("Processing dict input")
                if 'processed_videos_dir' in processed_videos_dir:
                    dir_path = str(processed_videos_dir['processed_videos_dir'])
                elif 'base_path' in processed_videos_dir:
                    dir_path = str(processed_videos_dir['base_path'])
                else:
                    logging.error(f"Invalid dict structure: {processed_videos_dir}")
                    raise ValueError("Missing path in dictionary")
            else:
                dir_path = str(processed_videos_dir)
                
            # Convert and validate path
            self.processed_videos_dir = Path(dir_path).resolve()
            if not isinstance(self.processed_videos_dir, Path):
                raise TypeError(f"Failed to convert to Path: {dir_path}")
                
            logging.info(f"Resolved path: {self.processed_videos_dir}")

            # Get max duration across all tracks
            track_durations = [
                self.get_track_duration(track) 
                for track in midi_data['tracks']
            ]
            if not track_durations:
                raise ValueError("No valid tracks found")
            
            total_duration = max(track_durations)
            # Setup remaining attributes
            self.output_path = output_path
            self.midi_data = midi_data
            self.frame_rate = 30
            self.chunk_duration = 10
            self.full_chunks = int(total_duration // self.chunk_duration)
            self.final_duration = total_duration % self.chunk_duration
            self.total_chunks = self.full_chunks + (1 if self.final_duration > 0 else 0)
            self.temp_dir = self.processed_videos_dir / "temp_composition"
            self.temp_dir.mkdir(exist_ok=True, parents=True)
            
            # Validate path exists
            if not self.processed_videos_dir.exists():
                raise ValueError(f"Directory not found: {self.processed_videos_dir}")
                
        except Exception as e:
            logging.error(f"VideoComposer init error: {str(e)}")
            raise

    def get_drum_position_key(self, midi_note, drum_name):
        """Helper to generate consistent drum position keys"""
        return f"drum_{midi_note}_{drum_name.lower()}"

    def get_chunk_notes(self, track, start_time, end_time):
        """Get notes within chunk timeframe"""
        try:
            return [
                note for note in track.get('notes', [])
                if (float(note['time']) >= start_time and 
                    float(note['time']) < end_time)
            ]
        except Exception as e:
            logging.error(f"Error getting chunk notes: {e}")
            return []

    def get_track_duration(self, track):
        """Get duration for any track type"""
        try:
            notes = track.get('notes', [])
            if not notes:
                return 0
            return max(float(note['time']) + float(note['duration']) 
                    for note in notes)
        except Exception as e:
            logging.error(f"Error calculating track duration: {e}")
            return 0

    # def create_track_chunk(self, track, track_idx, start_time, end_time, chunk_notes):
    #     rows, cols = self.get_track_layout()
    #     clips = []
    #     try:
    #         # Debug drum detection
    #         is_drum = is_drum_kit(track.get('instrument', {}))
    #         logging.info(f"""
    #             Track Analysis:
    #             - Track Index: {track_idx}
    #             - Is Drum Kit: {is_drum}
    #             - Instrument: {track.get('instrument', {})}
    #             - Notes in chunk: {len(chunk_notes)}
    #         """)

    #         if is_drum:
    #             for note in chunk_notes:
    #                 midi_note = int(float(note['midi']))
    #                 drum_name = note.get('drum_name', '').lower()
                    
    #                 # Debug drum note
    #                 logging.info(f"""
    #                     Drum Note:
    #                     - MIDI: {midi_note}
    #                     - Name: {drum_name}
    #                     - Time: {note['time']}
    #                     - Duration: {note['duration']}
    #                     - Video File: track_{track_idx}_drums/{drum_name}.mp4
    #                 """)
                
                    
    #                 if position_key not in self.grid_positions:
    #                     logging.error(f"No grid position found for {position_key}")
    #                     continue
                        
    #                 position = self.grid_positions[position_key]
    #                 row = position // cols
    #                 col = position % cols
                    
    #                 drum_file = self.processed_videos_dir / f"track_{track_idx}_drums" / f"{drum_name}.mp4"
    #                 if drum_file.exists():
    #                     try:
    #                         # Load base clip
    #                         base_clip = VideoFileClip(str(drum_file))
                            
    #                         # Get note timing within chunk
    #                         time = float(note['time']) - start_time
    #                         duration = float(note['duration'])

    #                         # Safety check
    #                         if time + duration > (end_time - start_time):
    #                             logging.warning(f"Note extends beyond chunk: {time + duration} > {end_time - start_time}")
    #                             duration = end_time - start_time - time
                            
    #                         # Adjust clip for note duration
    #                         if base_clip.duration < duration:
    #                             clip = base_clip.loop(duration=duration)
    #                         else:
    #                             clip = base_clip.subclip(0, duration)
                                
    #                         # Position clip at correct time in chunk
    #                         clip = clip.set_start(time)
    #                         clips.append(clip)
                            
    #                         logging.info(f"Added {drum_name} clip at time {time}, duration {duration}")
    #                     except Exception as e:
    #                         logging.error(f"Error processing drum {drum_file}: {e}")
    #                         continue
                            
    #         # Handle instrument tracks
    #         else:

    #             position_key = f"track_{track_idx}"
    #             if position_key not in self.grid_positions:
    #                 logging.error(f"No grid position found for {position_key}")
    #                 return None
                    
    #             position = self.grid_positions[position_key]
    #             row = position // cols
    #             col = position % cols

    #             if not chunk_notes:
    #                 return None
                    
    #             instrument_name = normalize_instrument_name(track['instrument']['name'])
    #             for note in chunk_notes:
    #                 try:
    #                     midi_note = int(float(note['midi']))
    #                     time = float(note['time']) - start_time
    #                     duration = float(note['duration'])
                        
    #                     note_file = (self.processed_videos_dir / 
    #                             f"track_{track_idx}_{instrument_name}" /
    #                             f"note_{midi_note}_{midi_to_note(midi_note)}.mp4")
                        
    #                     if note_file.exists():
    #                         clip = VideoFileClip(str(note_file))
    #                         clip = clip.set_start(time)
    #                         clip = clip.set_duration(duration)
    #                         clips.append(clip)
    #                 except Exception as e:
    #                     logging.error(f"Error processing note {note}: {e}")
    #                     continue
                        
    #         if clips:
    #             return CompositeVideoClip(clips, size=(1920, 1080))
    #         return None
            
    #     except Exception as e:
    #         logging.error(f"Error in create_track_chunk: {e}")
    #         return None
    #     finally:
    #         for clip in clips:
    #             try:
    #                 if hasattr(clip, 'close'):
    #                     clip.close()
    #             except:
    #                 pass


    def get_target_dimensions(self, grid_size):
        """Calculate target dimensions for each clip in grid"""
        rows, cols = grid_size
        target_width = 1920 // cols  # Assuming 1920x1080 output
        target_height = 1080 // rows
        return (target_width, target_height)


    def validate_clip(self, clip):
        try:
            if not clip:
                logging.error("Clip is None")
                return False
            if not hasattr(clip, 'get_frame'):
                logging.error("Clip missing get_frame")
                return False
            frame = clip.get_frame(0)
            if frame is None:
                logging.error("Frame is None")
                return False
            return True
        except Exception as e:
            logging.error(f"Clip validation failed: {str(e)}")
            return False


    # def get_track_layout(self):
    #     try:
    #         total_slots = 0
    #         self.grid_positions = {}
            
    #         # Regular tracks first
    #         for track_idx, track in enumerate(self.midi_data['tracks']):
    #             if not is_drum_kit(track.get('instrument', {})):
    #                 self.grid_positions[f"track_{track_idx}"] = total_slots
    #                 total_slots += 1
                    
    #         # Handle drums - treat each drum note as separate instrument
    #         for track in self.midi_data['tracks']:
    #             if is_drum_kit(track.get('instrument', {})):
    #                 # Get unique drum notes from track
    #                 drum_notes = set(note['midi'] for note in track.get('notes', []))
    #                 for drum_note in drum_notes:
    #                     position_key = f"drum_{drum_note}"  # Use MIDI note as identifier
    #                     self.grid_positions[position_key] = total_slots
    #                     total_slots += 1
            
    #         # Calculate grid dimensions
    #         if total_slots == 0:
    #             return (1, 1)
    #         elif total_slots <= 2:
    #             return (1, 2)
    #         elif total_slots <= 4:
    #             return (2, 2)
    #         else:
    #             cols = min(4, math.ceil(math.sqrt(total_slots)))
    #             rows = math.ceil(total_slots / cols)
    #             return (rows, cols)
            
    #     except Exception as e:
    #         logging.error(f"Layout error: {str(e)}")
    #         if total_slots > 0:
    #             cols = min(4, math.ceil(math.sqrt(total_slots)))
    #             rows = math.ceil(total_slots / cols)
    #             return (rows, cols)
    #         return (4, 4)  # Fallback default
    

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


    # def create_track_chunk(self, track, track_idx, start_time, end_time, chunk_notes):
    #     try:
    #         clips = []
    #         if is_drum_kit(track.get('instrument', {})):
    #             for note in chunk_notes:
    #                 midi_note = int(float(note['midi']))
    #                 drum_name = note.get('drum_name', '').lower()  # From MIDI parser
    #                 position_key = f"drum_{midi_note}"  # Use MIDI note for position
                    
    #                 if position_key in self.grid_positions:
    #                     position = self.grid_positions[position_key]
    #                     # Use drum_name to find video file
    #                     drum_file = self.processed_videos_dir / f"track_{track_idx}_drums" / f"{drum_name}.mp4"
                        
    #                     if drum_file.exists():
    #                         time = float(note['time']) - start_time
    #                         duration = float(note['duration'])
    #                         clip = VideoFileClip(str(drum_file))
    #                         clip = clip.set_start(time).set_duration(duration)
    #                         clips.append((position, clip))
    #                         logging.info(f"Added {drum_name} clip at position {position} from note {midi_note}")
    #         else:
    #             position_key = f"track_{track_idx}"
    #             if position_key in self.grid_positions:
    #                 position = self.grid_positions[position_key]
    #                 for note in chunk_notes:
    #                     midi_note = int(float(note['midi']))
    #                     time = float(note['time']) - start_time
    #                     duration = float(note['duration'])
    #                     note_file = (self.processed_videos_dir / 
    #                             f"track_{track_idx}_{normalize_instrument_name(track['instrument']['name'])}" /
    #                             f"note_{midi_note}_{midi_to_note(midi_note)}.mp4")
    #                     if note_file.exists():
    #                         clip = VideoFileClip(str(note_file))
    #                         clip = clip.set_start(time).set_duration(duration)
    #                         clips.append((position, clip))
    #             pass
    #         return clips 
    #     except Exception as e:
    #         logging.error(f"Error in track chunk: {str(e)}")
    #         return []

        
    def create_track_video(self, track, track_idx, duration):
        try:
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            # Check all possible video locations
            video_locations = [
                self.processed_videos_dir / f"track_{track_idx}_{instrument_name}",
                self.processed_videos_dir / f"{instrument_name}_notes",
                self.processed_videos_dir / f"track_{track_idx}_drums"
            ]
            
            clips = []
            background = None
            
            # Handle drum tracks
            if is_drum_kit(instrument):
                drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                if drum_dir.exists():
                    for drum_file in drum_dir.glob('*.mp4'):
                        try:
                            clip = VideoFileClip(str(drum_file))
                            # Loop drum clips to match duration
                            num_loops = int(np.ceil(duration / clip.duration))
                            extended_clip = clip.loop(n=num_loops)
                            final_clip = extended_clip.subclip(0, duration)
                            clips.append(final_clip)
                            logging.info(f"Added drum clip: {drum_file}")
                        except Exception as e:
                            logging.error(f"Error loading drum clip {drum_file}: {e}")
                
                if clips:
                    return CompositeVideoClip(clips).set_duration(duration)
                    
            # Handle instrument tracks
            else:
                for video_loc in video_locations:
                    if video_loc.exists():
                        for note in track.get('notes', []):
                            try:
                                midi_note = int(float(note['midi']))
                                note_file = video_loc / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                                
                                if note_file.exists():
                                    clip = VideoFileClip(str(note_file))
                                    start_time = float(note['time'])
                                    clip = clip.set_start(start_time)
                                    clip = clip.set_duration(float(note['duration']))
                                    clips.append(clip)
                                    
                                    if background is None:
                                        background = ColorClip(
                                            size=(clip.w, clip.h),
                                            color=(0, 0, 0),
                                            duration=duration
                                        )
                            except Exception as e:
                                logging.error(f"Error loading note clip {midi_note}: {e}")
                                
                if clips:
                    if background:
                        return CompositeVideoClip([background] + clips).set_duration(duration)
                    return clips[0].set_duration(duration)
                    
            return None
            
        except Exception as e:
            logging.error(f"Error creating track video: {str(e)}")
            return None


    # def get_track_layout(self):
    #     try:
    #         total_slots = 0
    #         self.grid_positions = {}
            
    #         # Regular tracks first
    #         for track_idx, track in enumerate(self.midi_data['tracks']):
    #             if not is_drum_kit(track.get('instrument', {})):
    #                 self.grid_positions[f"track_{track_idx}"] = total_slots
    #                 logging.info(f"Assigned position {total_slots} to track {track_idx}")
    #                 total_slots += 1
                    
    #         # Then handle drums
    #         for track_idx, track in enumerate(self.midi_data['tracks']):
    #             if is_drum_kit(track.get('instrument', {})):
    #                 drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
    #                 if drum_dir.exists():
    #                     for drum_file in drum_dir.glob('*.mp4'):
    #                         drum_name = drum_file.stem
    #                         position_key = f"drum_{track_idx}_{drum_name}"
    #                         self.grid_positions[position_key] = total_slots
    #                         logging.info(f"Assigned position {total_slots} to drum {position_key}")
    #                         total_slots += 1
            
    #         # Calculate grid dimensions
    #         if total_slots == 0:
    #             return (1, 1)
    #         elif total_slots <= 2:
    #             return (1, 2)
    #         elif total_slots <= 4:
    #             return (2, 2)
    #         else:
    #             cols = min(4, math.ceil(math.sqrt(total_slots)))
    #             rows = math.ceil(total_slots / cols)
    #             logging.info(f"Grid dimensions: {rows}x{cols}")
    #             return (rows, cols)
                
    #     except Exception as e:
    #         logging.error(f"Layout error: {str(e)}")
    #         return (1, 1)  # Fallback dimensions

    def get_track_layout(self):
        try:
            total_slots = 0
            self.grid_positions = {}
            
            # Regular tracks
            for track_idx, track in enumerate(self.midi_data['tracks']):
                if not is_drum_kit(track.get('instrument', {})):
                    self.grid_positions[f"track_{track_idx}"] = total_slots
                    total_slots += 1
                else:
                    # Handle drums based on actual files
                    drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                    if drum_dir.exists():
                        for drum_file in drum_dir.glob('*.mp4'):
                            drum_name = drum_file.stem
                            position_key = f"drum_{track_idx}_{drum_name}"
                            self.grid_positions[position_key] = total_slots
                            total_slots += 1
                            logging.info(f"Assigned position {total_slots} to {position_key}")
            
            # Calculate grid dimensions
            cols = min(4, math.ceil(math.sqrt(total_slots)))
            rows = math.ceil(total_slots / cols)
            return (rows, cols)
                
        except Exception as e:
            logging.error(f"Layout error: {str(e)}")
            return (1, 1)

    # def create_composition(self):
    #     try:
    #         rows, cols = self.get_track_layout()
    #         if not rows or not cols:
    #             raise ValueError("Invalid grid dimensions")
                
    #         chunk_files = []
    #         full_chunks, final_duration = self.calculate_chunk_lengths(self.midi_data)
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
            
    #         for chunk_idx in range(total_chunks):
    #             grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                             color=(0,0,0), 
    #                             duration=10) 
    #                     for _ in range(cols)] 
    #                     for _ in range(rows)]
                
    #             start_time = chunk_idx * 10
    #             end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)

    #             logging.info(f"""
    #                 Chunk Details:
    #                 - Index: {chunk_idx}
    #                 - Start Time: {start_time}
    #                 - End Time: {end_time}
    #             """)
                
    #             # Process each track
    #             for track_idx, track in enumerate(self.midi_data['tracks']):
    #                 if 'track_idx' not in track:
    #                     track['track_idx'] = track_idx  # Ensure track has index
    #                 logging.info(f"Processing track {track['track_idx']} for chunk {chunk_idx}")
                    
    #                 start_time = chunk_idx * 10
    #                 end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                    
    #                 # Get notes for this chunk's time window
    #                 chunk_notes = [
    #                     note for note in track.get('notes', [])
    #                     if start_time <= float(note['time']) < end_time
    #                 ]
                    
    #                 # Process drums
    #                 if is_drum_kit(track.get('instrument', {})):
    #                     for note in chunk_notes:
    #                         logging.info(f"""
    #                             Note Details:
    #                             - Time: {note['time']}
    #                             - Duration: {note['duration']}
    #                             - Relative Time: {float(note['time']) - start_time}
    #                             - Note End: {float(note['time']) + float(note['duration'])}
    #                         """)

    #                         drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
    #                         if drum_dir.exists():
    #                             for drum_file in drum_dir.glob('*.mp4'):
    #                                 drum_name = drum_file.stem  # Gets filename without extension
    #                                 position_key = f"drum_{track_idx}_{drum_name}"
    #                                 if position_key in self.grid_positions:
    #                                     position = self.grid_positions[position_key]
    #                                     row = position // cols
    #                                     col = position % cols
    #                                     clip = VideoFileClip(str(drum_file))
    #                                     clip = clip.loop(duration=end_time - start_time)
    #                                     grid[row][col] = clip
    #                                     logging.info(f"Added drum {drum_name} to position [{row}][{col}]")
    #                                 else:
    #                                     logging.warning(f"No position found for drum: {drum_name}")
                    
    #                 # Process instruments
    #                 else:
    #                     position_key = f"track_{track_idx}"  # Fix: Use proper key format
    #                     if position_key in self.grid_positions:
    #                         position = self.grid_positions[position_key]  # Use position_key instead of raw index
    #                         row = position // cols
    #                         col = position % cols
    #                         if chunk_notes:
    #                             clips = []
    #                             for note in chunk_notes:
    #                                 midi_note = int(float(note['midi']))
    #                                 time = float(note['time']) - start_time
    #                                 duration = float(note['duration'])
    #                                 note_file = (self.processed_videos_dir / 
    #                                         f"track_{track_idx}_{normalize_instrument_name(track['instrument']['name'])}" /
    #                                         f"note_{midi_note}_{midi_to_note(midi_note)}.mp4")
                                    
    #                                 if note_file.exists():
    #                                     clip = VideoFileClip(str(note_file))
    #                                     clip = clip.set_start(time).set_duration(duration)
    #                                     clips.append(clip)
                                        
    #                             if clips:
    #                                 grid[row][col] = CompositeVideoClip(clips).set_duration(end_time - start_time)
    #                                 logging.info(f"Added track {position_key} to position [{row}][{col}]")

    def create_composition(self):
        try:
            rows, cols = self.get_track_layout()
            fps = 30
            chunk_files = []
            full_chunks, final_duration = self.calculate_chunk_lengths(self.midi_data)
            total_chunks = full_chunks + (1 if final_duration > 0 else 0)
            for chunk_idx in range(total_chunks):
                # Create grid with black clips
                grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                                color=(0,0,0), 
                                duration=10) 
                        for _ in range(cols)] 
                        for _ in range(rows)]
                
                # Set fps after creation
                for row in grid:
                    for clip in row:
                        clip.fps = fps
                
                start_time = chunk_idx * 10
                end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
                
                # Process each track
                for track_idx, track in enumerate(self.midi_data['tracks']):
                    if is_drum_kit(track.get('instrument', {})):
                        # Get all drum notes for this chunk
                        chunk_notes = [note for note in track.get('notes', [])
                                    if start_time <= float(note['time']) < end_time]
                        
                        for note in chunk_notes:
                            drum_name = note.get('drum_name', '').lower()
                            position_key = f"drum_{track_idx}_{drum_name}"
                            
                            if position_key in self.grid_positions:
                                position = self.grid_positions[position_key]
                                row = position // cols
                                col = position % cols
                                
                                # Load and process drum clip
                                drum_file = self.processed_videos_dir / f"track_{track_idx}_drums" / f"{drum_name}.mp4"
                                if drum_file.exists():
                                    base_clip = VideoFileClip(str(drum_file))
                                    time = float(note['time']) - start_time
                                    duration = min(float(note['duration']), base_clip.duration)
                                    
                                    clip = base_clip.subclip(0, duration).set_start(time)
                                    
                                    # Combine with existing clip at position
                                    existing_clip = grid[row][col]
                                    if isinstance(existing_clip, ColorClip):
                                        grid[row][col] = clip
                                    else:
                                        grid[row][col] = CompositeVideoClip([existing_clip, clip])
                                    
                                    logging.info(f"Added {drum_name} clip at [{row}][{col}], time={time}, duration={duration}")
                
                # Create chunk
                chunk = clips_array(grid)
                chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
                chunk.write_videofile(str(chunk_path), fps=fps)
                chunk_files.append(str(chunk_path))
                
            # Combine chunks
            final = concatenate_videoclips([VideoFileClip(f) for f in chunk_files])
            final.write_videofile(str(self.output_path), fps=fps)
            return self.output_path
            
        except Exception as e:
            logging.error(f"Composition error: {str(e)}")
            return None

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
