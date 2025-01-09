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
            self.drum_groups = {
                'kick': [35, 36],  # Acoustic Bass Drum, Bass Drum 1
                'snare': [38, 40, 37],  # Acoustic Snare, Electric Snare, Side Stick
                'hihat': [42, 44, 46],  # Closed HH, Pedal HH, Open HH
                'cymbal': [49, 51, 52, 55, 57, 53, 59],  # Crash 1, Ride 1, etc
                'tom': [41, 43, 45, 47, 48, 50],  # Low Floor Tom, etc
                'percussion': [39, 54, 56, 58, 60, 61, 62, 63, 64]  # Clap, etc
            }
            self.OVERLAP_DURATION = 0.3  # 300ms overlap
            self.CROSSFADE_DURATION = 0.25  # 250ms crossfade
            # Validate path exists
            if not self.processed_videos_dir.exists():
                raise ValueError(f"Directory not found: {self.processed_videos_dir}")
                
        except Exception as e:
            logging.error(f"VideoComposer init error: {str(e)}")
            raise

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
            notes = track.get('notes', [])
            if not notes:
                return 0
            return max(float(note['time']) + float(note['duration']) 
                    for note in notes)
        except Exception as e:
            logging.error(f"Error calculating track duration: {e}")
            return 0

    
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
        
    def get_drum_group_for_note(self, midi_note):
            """Map MIDI note number to drum group name"""
            for group, notes in self.drum_groups.items():
                if midi_note in notes:
                    return group
            logging.info(f"No drum group found for MIDI note {midi_note}")
            return None

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
            
            # Log grid layout header
            logging.info("\nGrid Layout Planning:")
            logging.info(f"\nFound {len(valid_tracks)} tracks with notes:")
            
            # First pass - Count all tracks including drums
            for track_idx, track in valid_tracks:
                if is_drum_kit(track.get('instrument', {})):
                    # For drum tracks, count each drum group separately
                    drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                    if drum_dir.exists():
                        # Look for actual drum group videos (kick.mp4, snare.mp4, etc)
                        for drum_file in drum_dir.glob('*.mp4'):
                            drum_group = drum_file.stem  # Gets 'kick' from 'kick.mp4'
                            position_key = f"drum_{track_idx}_{drum_group}"
                            self.grid_positions[position_key] = total_slots
                            logging.info(f"Position {total_slots}: Drum {drum_group} (track_{track_idx})")
                            total_slots += 1
                else:
                    # Regular instrument track
                    self.grid_positions[f"track_{track_idx}"] = total_slots
                    instrument_name = normalize_instrument_name(track['instrument']['name'])
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
        
    def process_chunk(self, tracks, start_time, end_time, chunk_idx):
        """Process chunk with overlapping notes"""
        try:
            
            # Extend chunk time range to include overlap
            actual_start = start_time - (self.OVERLAP_DURATION if chunk_idx > 0 else 0)
            actual_end = end_time + self.OVERLAP_DURATION
            
            # Get grid dimensions
            rows, cols = self.get_track_layout()
            grid = [[ColorClip(size=(1920//cols, 1080//rows), 
                            color=(0,0,0), 
                            duration=actual_end - actual_start) 
                    for _ in range(cols)] 
                    for _ in range(rows)]

            # Get notes including overlap periods
            for track_idx, track in enumerate(self.midi_data['tracks']):
                chunk_notes = self.get_chunk_notes(
                    track, 
                    actual_start, 
                    actual_end, 
                    include_overlap=True
                )
                
                # Process notes with adjusted timing
                for note in chunk_notes:
                    # Adjust note timing relative to chunk start
                    note_time = float(note['time']) - actual_start
                    note_duration = float(note.get('duration', 0))
                    # Ensure notes at chunk boundaries aren't cut off
                    if chunk_idx > 0 and note_time < self.OVERLAP_DURATION:
                        note_time = max(0, note_time)

            # Get grid dimensions
            rows = math.ceil(len(tracks) / 2)
            cols = 2
            grid = [[None for _ in range(cols)] for _ in range(rows)]
            
            for track_idx, track in enumerate(tracks):
                row = track_idx // 2
                col = track_idx % 2
                
                # Get notes including overlaps
                notes = self.get_chunk_notes(track, start_time, end_time, include_overlap=True)
                clips_for_instrument = []
                
                for note in notes:
                    # Adjust timing relative to chunk start
                    note_start = float(note['time']) - start_time
                    note_duration = float(note.get('duration', 0))
                    
                    # Load and position clip
                    note_clip = self.get_note_clip(track, note)
                    if note_clip:
                        positioned_clip = note_clip.set_start(note_start)
                        positioned_clip = positioned_clip.set_duration(note_duration)
                        clips_for_instrument.append(positioned_clip)
                
                if clips_for_instrument:
                    composite = CompositeVideoClip(clips_for_instrument)
                    grid[row][col] = composite.set_duration(end_time - start_time)
                    
            # Create chunk with crossfade
            chunk = clips_array(grid)
            chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
            
            # Add small crossfade buffer
            if chunk_idx > 0:
                crossfade_duration = 0.1  # 100ms crossfade
                chunk = chunk.crossfadein(crossfade_duration)
                
            chunk.write_videofile(
                str(chunk_path),
                fps=30,
                codec='h264_nvenc',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=[
                    "-vsync", "1",
                    "-async", "1", 
                    "-b:v", "5M"
                ]
            )

             # Add crossfade
            if chunk_idx > 0:
                chunk = chunk.crossfadein(self.CROSSFADE_DURATION)
            
            return str(chunk_path)
            
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
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

                    if not chunk_notes:
                        continue
                    
                    if is_drum_kit(track.get('instrument', {})):
                        drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                        if drum_dir.exists():
                            drum_notes = {}
                            for note in chunk_notes:
                                drum_group = self.get_drum_group_for_note(note['midi'])
                                logging.info(f"Processing drum note {note['midi']} -> group {drum_group}")
                                if drum_group:
                                    if drum_group not in drum_notes:
                                        drum_notes[drum_group] = []
                                    drum_notes[drum_group].append(note)

                            for drum_group, notes in drum_notes.items():
                                position_key = f"drum_{track_idx}_{drum_group}"
                                logging.info(f"Processing drum group {drum_group} at position {position_key}")
                                if position_key in self.grid_positions:
                                    position = self.grid_positions[position_key]
                                    row = position // cols
                                    col = position % cols
                                    drum_file = drum_dir / f"{drum_group}.mp4"
                                    if drum_file.exists():
                                        for note in notes:
                                            try:
                                                clip = VideoFileClip(str(drum_file))
                                                active_clips.append(clip)
                                                
                                                # Extract and normalize velocity
                                                velocity = float(note.get('velocity', 100)) / 127.0
                                                # Scale drums down slightly to balance with instruments
                                                volume = max(0.1, velocity * 0.7)  # 70% of original volume
                                                
                                                # Apply volume adjustment
                                                clip = clip.volumex(volume)
                                                
                                                # Log for debugging
                                                logging.info(f"Drum {drum_group} velocity={velocity:.2f}, volume={volume:.2f}")
                                                
                                                time = float(note['time']) - start_time
                                                duration = min(float(note['duration']), clip.duration)
                                                clip = clip.subclip(0, duration).set_start(time)
                                                
                                                if isinstance(grid[row][col], ColorClip):
                                                    grid[row][col] = clip
                                                else:
                                                    existing = grid[row][col]
                                                    grid[row][col] = CompositeVideoClip([existing, clip])
                                                
                                                logging.info(f"Added drum {drum_group} at [{row}][{col}] t={time}")
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
                                    velocity = float(note.get('velocity', 100)) / 127.0
                                    volume = max(0.1, velocity)  # Ensure minimum audible volume
                                    
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
