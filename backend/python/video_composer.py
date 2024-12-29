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


    def get_track_layout(self):
        try:
            total_slots = 0
            self.grid_positions = {}
            
            # Log grid layout header
            logging.info("\nGrid Layout Planning:")
            
            # Regular tracks
            for track_idx, track in enumerate(self.midi_data['tracks']):
                if not is_drum_kit(track.get('instrument', {})):
                    self.grid_positions[f"track_{track_idx}"] = total_slots
                    instrument_name = normalize_instrument_name(track['instrument']['name'])
                    logging.info(f"Position {total_slots}: {instrument_name} (track_{track_idx})")
                    total_slots += 1
                
            # Handle drums
            for track_idx, track in enumerate(self.midi_data['tracks']):
                if is_drum_kit(track.get('instrument', {})):
                    drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                    if drum_dir.exists():
                        for drum_file in drum_dir.glob('*.mp4'):
                            drum_name = drum_file.stem
                            position_key = f"drum_{track_idx}_{drum_name}"
                            self.grid_positions[position_key] = total_slots
                            logging.info(f"Position {total_slots}: {drum_name} (track_{track_idx}_drums)")
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

    

    # def create_composition(self):
    #     try:
    #         # Calculate chunks first
    #         full_chunks, final_duration = self.calculate_chunk_lengths(self.midi_data)
    #         if full_chunks is None:
    #             raise ValueError("Failed to calculate chunks")
                
    #         total_chunks = full_chunks + (1 if final_duration > 0 else 0)
    #         logging.info(f"Processing {total_chunks} chunks ({full_chunks} full + {1 if final_duration > 0 else 0} partial)")
            
    #         # Get grid layout
    #         rows, cols = self.get_track_layout()
    #         logging.info(f"Creating grid with dimensions {rows}x{cols}")
            
    #         chunk_files = []
    #         for chunk_idx in range(total_chunks):
    #             # Process chunk
    #             start_time = chunk_idx * 10
    #             end_time = start_time + (final_duration if chunk_idx == full_chunks else 10)
    #             logging.info(f"Processing chunk {chunk_idx}: {start_time}-{end_time}s")
                
    #             # Initialize grid with black clips
    #             grid = [[ColorClip(size=(1920//cols, 1080//rows), 
    #                         color=(0,0,0), 
    #                         duration=end_time - start_time) 
    #                     for _ in range(cols)] 
    #                     for _ in range(rows)]
                
    #             # Process each track
    #             for track_idx, track in enumerate(self.midi_data['tracks']):
    #                 if is_drum_kit(track.get('instrument', {})):
    #                     # Get notes for this chunk
    #                     chunk_notes = [
    #                         note for note in track.get('notes', [])
    #                         if start_time <= float(note['time']) < end_time
    #                     ]
                        
    #                     logging.info(f"Processing {len(chunk_notes)} drum notes for chunk {chunk_idx}")
                        
    #                     for note in chunk_notes:
    #                         drum_name = note.get('drum_name', '').lower()
    #                         position_key = f"drum_{track_idx}_{drum_name}"
                            
    #                         if position_key in self.grid_positions:
    #                             position = self.grid_positions[position_key]
    #                             row = position // cols
    #                             col = position % cols
                                
    #                             drum_file = self.processed_videos_dir / f"track_{track_idx}_drums" / f"{drum_name}.mp4"
    #                             if drum_file.exists():
    #                                 clip = VideoFileClip(str(drum_file))
    #                                 time = float(note['time']) - start_time
    #                                 duration = min(float(note['duration']), clip.duration)
                                    
    #                                 clip = clip.subclip(0, duration).set_start(time)
    #                                 grid[row][col] = clip
    #                                 logging.info(f"Added {drum_name} at [{row}][{col}] t={time:.2f}")
                    
    #             # Create chunk with proper FPS
    #             chunk = clips_array(grid)
    #             chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
    #             chunk.write_videofile(str(chunk_path), fps=30)
                
    #             # Clean up
    #             for row in grid:
    #                 for clip in row:
    #                     if hasattr(clip, 'close'):
    #                         clip.close()
    #         return self.output_path

    #     except Exception as e:
    #          logging.error(f"Layout error: {str(e)}")
    #          return 

    def _combine_chunks(self, chunk_files):
        """Separate method for final video assembly with cleanup"""
        try:
            clips = []
            for f in chunk_files:
                clip = VideoFileClip(f)
                clips.append(clip)
            
            final = concatenate_videoclips(clips)
            final.write_videofile(str(self.output_path), fps=30)
            
            # Cleanup
            for clip in clips:
                clip.close()
            final.close()
            
            # Remove temp files
            for f in chunk_files:
                os.unlink(f)
                
            return self.output_path
            
        except Exception as e:
            logging.error(f"Concatenation error: {e}")
            return None

    def create_composition(self):
        try:
            full_chunks, final_duration = self.calculate_chunk_lengths(self.midi_data)
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
                    
                    if is_drum_kit(track.get('instrument', {})):
                        for note in chunk_notes:
                            drum_name = note.get('drum_name', '').lower()
                            position_key = f"drum_{track_idx}_{drum_name}"
                            
                            if position_key in self.grid_positions:
                                position = self.grid_positions[position_key]
                                row = position // cols
                                col = position % cols
                                
                                drum_file = self.processed_videos_dir / f"track_{track_idx}_drums" / f"{drum_name}.mp4"
                                logging.info(f"Loading drum: {drum_file}")
                                
                                if drum_file.exists():
                                    clip = VideoFileClip(str(drum_file))
                                    active_clips.append(clip)
                                    time = float(note['time']) - start_time
                                    duration = min(float(note['duration']), clip.duration)
                                    clip = clip.subclip(0, duration).set_start(time)
                                    
                                    # Combine with existing or set new
                                    if isinstance(grid[row][col], ColorClip):
                                        grid[row][col] = clip
                                    else:
                                        existing = grid[row][col]
                                        grid[row][col] = CompositeVideoClip([existing, clip])
                                    
                                    logging.info(f"Added drum {drum_name} at [{row}][{col}] t={time}")
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
                chunk.write_videofile(str(chunk_path), fps=30)
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
            #     try:
            #         final = concatenate_videoclips([VideoFileClip(f) for f in chunk_files])
            #         logging.info("Clips loaded for concatenation")
                    
            #         logging.info(f"Writing final video to: {self.output_path}")
            #         final.write_videofile(str(self.output_path), fps=30)
            #         logging.info("Final video written successfully")
                    
            #         # Clean up
            #         final.close()
            #         return self.output_path
                    
            #     except Exception as e:
            #         logging.error(f"Concatenation error: {str(e)}")
            #         return None
            # else:
            #     logging.error("No chunk files created")
            #     return None
                
        except Exception as e:
            logging.error(f"Composition error: {str(e)}")
            return None
        # finally:
        #     # Clean up chunks
        #     for chunk_file in chunk_files:
        #         try:
        #             if os.path.exists(chunk_file):
        #                 os.unlink(chunk_file)
        #         except Exception as e:
        #             logging.error(f"Error cleaning up chunk {chunk_file}: {str(e)}")

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
