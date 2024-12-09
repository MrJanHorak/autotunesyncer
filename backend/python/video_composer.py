# video_composer.py
import numpy as np
from pathlib import Path
import logging
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip, ColorClip, concatenate_videoclips
from utils import normalize_instrument_name, midi_to_note
from drum_utils import is_drum_kit
import gc
import os

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
        self.processed_videos_dir = Path(processed_videos_dir)
        self.midi_data = midi_data
        self.output_path = output_path
        self.frame_rate = 30
        self.chunk_duration = 10  # Process 10 seconds at a time
        self.temp_dir = Path(self.processed_videos_dir) / "temp_composition"
        self.temp_dir.mkdir(exist_ok=True)

    def validate_clip(self, clip):
        """Validate that clip was loaded properly"""
        try:
            if clip is None:
                return False
                
            # Try to access first frame
            test_frame = clip.get_frame(0)
            if test_frame is None or len(test_frame.shape) < 2:
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Invalid clip: {str(e)}")
            return False

    def get_track_layout(self):
        valid_tracks = []
        
        logging.info(f"Checking tracks in: {self.processed_videos_dir}")
        
        for track_idx, track in enumerate(self.midi_data['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            # Check all possible video locations
            video_locations = [
                f"track_{track_idx}_{instrument_name}",
                f"{instrument_name}_notes",
                f"track_{track_idx}_drums",
            ]
            
            has_videos = False
            for loc in video_locations:
                dir_path = self.processed_videos_dir / loc
                if dir_path.exists():
                    video_files = list(dir_path.glob('*.mp4'))
                    if video_files:
                        has_videos = True
                        logging.info(f"Found videos for track {track_idx} in {loc}: {len(video_files)} files")
                        break
            
            if has_videos:
                valid_tracks.append(track)
                logging.info(f"Found valid track {track_idx}: {instrument_name}")
        
        track_count = len(valid_tracks)
        logging.info(f"Total valid tracks: {track_count}")
        
        # Calculate layout
        if track_count == 0:
            raise ValueError("No valid tracks found")
        elif track_count == 1:
            return (1, 1)
        elif track_count == 2:
            return (1, 2)
        elif track_count <= 4:
            return (2, 2)
        else:
            cols = min(3, track_count)
            rows = (track_count + cols - 1) // cols
            return (rows, cols)
    
    def create_track_chunk(self, track, track_idx, start_time, end_time, chunk_notes):
        """Process a specific time chunk of a track"""
        try:
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            # Handle drum tracks
            if is_drum_kit(instrument):
                drum_dir = self.processed_videos_dir / f"track_{track_idx}_drums"
                if drum_dir.exists():
                    drum_clips = []
                    for drum_file in drum_dir.glob('*.mp4'):
                        try:
                            with VideoFileClip(str(drum_file)) as base_clip:
                                if not self.validate_clip(base_clip):
                                    logging.error(f"Invalid drum clip: {drum_file}")
                                    continue
                                    
                                # Calculate how many times to loop in this chunk
                                chunk_duration = end_time - start_time
                                num_loops = int(np.ceil(chunk_duration / base_clip.duration))
                                
                                # Create looped clip for this chunk
                                looped_clip = base_clip.loop(n=num_loops)
                                chunk_clip = looped_clip.subclip(0, chunk_duration)
                                drum_clips.append(chunk_clip)
                                
                        except Exception as e:
                            logging.error(f"Error processing drum clip {drum_file}: {e}")
                            continue
                    
                    if drum_clips:
                        return CompositeVideoClip(drum_clips).set_duration(end_time - start_time)
                    return None
                    
            # Handle instrument tracks
            else:
                clips = []
                background = None
                notes_dir = self.processed_videos_dir / f"{instrument_name}_notes"
                
                if notes_dir.exists():
                    for note in chunk_notes:
                        try:
                            midi_note = int(float(note['midi']))
                            note_file = notes_dir / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                            
                            if note_file.exists():
                                with VideoFileClip(str(note_file)) as clip:
                                    if not self.validate_clip(clip):
                                        logging.error(f"Invalid note clip: {note_file}")
                                        continue
                                        
                                    # Adjust note timing relative to chunk
                                    relative_start = float(note['time']) - start_time
                                    clip = clip.set_start(relative_start)
                                    clip = clip.set_duration(float(note['duration']))
                                    clips.append(clip)
                                    
                                    if background is None:
                                        background = ColorClip(
                                            clip.size,
                                            color=(0, 0, 0),
                                            duration=end_time - start_time
                                        )
                                        
                        except Exception as e:
                            logging.error(f"Error processing note {midi_note}: {e}")
                            continue
                
                if clips:
                    if background:
                        return CompositeVideoClip([background] + clips)
                    return clips[0]
                
                return None
                
        except Exception as e:
            logging.error(f"Error creating track chunk: {str(e)}")
            return None
        
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
        
    def create_composition(self):
            try:
                rows, cols = self.get_track_layout()
                max_time = max(
                    float(note['time']) + float(note['duration'])
                    for track in self.midi_data['tracks']
                    for note in track.get('notes', [])
                )
                
                # Calculate number of chunks needed
                num_chunks = int(np.ceil(max_time / self.chunk_duration))
                chunk_files = []
                
                logging.info(f"Processing composition in {num_chunks} chunks")
                
                # Process each time chunk
                for chunk_idx in range(num_chunks):
                    start_time = chunk_idx * self.chunk_duration
                    end_time = min((chunk_idx + 1) * self.chunk_duration, max_time)
                    
                    chunk_clips = []
                    
                    # Process each track for this time chunk
                    for track_idx, track in enumerate(self.midi_data['tracks']):
                        try:
                            # Get only notes that fall within this chunk
                            chunk_notes = [
                                note for note in track.get('notes', [])
                                if float(note['time']) < end_time and 
                                float(note['time']) + float(note['duration']) > start_time
                            ]
                            
                            if not chunk_notes and not is_drum_kit(track.get('instrument', {})):
                                continue
                                
                            clip = self.create_track_chunk(
                                track, track_idx, 
                                start_time, end_time, 
                                chunk_notes
                            )
                            
                            if clip is not None:
                                if chunk_clips:
                                    clip = clip.resize(chunk_clips[0].size)
                                chunk_clips.append(clip)
                                
                        except Exception as e:
                            logging.error(f"Error processing track {track_idx} chunk {chunk_idx}: {e}")
                            continue
                    
                    if chunk_clips:
                        # Create grid for this chunk
                        grid = []
                        for i in range(rows):
                            row = []
                            for j in range(cols):
                                idx = i * cols + j
                                if idx < len(chunk_clips):
                                    row.append(chunk_clips[idx])
                                else:
                                    row.append(ColorClip(
                                        chunk_clips[0].size,
                                        color=(0, 0, 0),
                                        duration=end_time - start_time
                                    ))
                            grid.append(row)
                        
                        # Write chunk to temporary file
                        chunk_path = self.temp_dir / f"chunk_{chunk_idx}.mp4"
                        chunk_composition = clips_array(grid)
                        chunk_composition.write_videofile(
                            str(chunk_path),
                            fps=self.frame_rate,
                            codec='libx264',
                            audio=True,
                            audio_codec='aac',
                            preset='ultrafast'  # Faster encoding for temp files
                        )
                        
                        # Clean up clips
                        for clip in chunk_clips:
                            clip.close()
                        chunk_composition.close()
                        
                        chunk_files.append(str(chunk_path))
                        
                    # Force garbage collection
                    gc.collect()
                
                # Concatenate all chunks
                if chunk_files:
                    concatenate_videoclips([VideoFileClip(f) for f in chunk_files]).write_videofile(
                        self.output_path,
                        fps=self.frame_rate,
                        codec='libx264',
                        audio=True,
                        audio_codec='aac',
                        preset='medium'
                    )
                
                # Clean up temp files
                for chunk_file in chunk_files:
                    try:
                        os.remove(chunk_file)
                    except:
                        pass
                
                return self.output_path
                
            except Exception as e:
                logging.error(f"Error in composition: {e}")
                raise
        
def compose_from_processor_output(processor_result, output_path):
    """Bridge function to be called from audio_processor.py"""
    try:
        base_dir = processor_result['processed_videos_dir']
        logging.info(f"Using base directory: {base_dir}")
        
        # Log available tracks and paths
        if 'tracks' in processor_result['processed_files']:
            for instrument, data in processor_result['processed_files']['tracks'].items():
                logging.info(f"Found instrument track: {instrument}")
                logging.info(f"Base path: {data['base_path']}")
                logging.info(f"Available notes: {list(data['notes'].keys())}")
                
        if 'drum_tracks' in processor_result['processed_files']:
            for drum, data in processor_result['processed_files']['drum_tracks'].items():
                logging.info(f"Found drum track: {drum}")
                logging.info(f"Path: {data['path']}")
        
        composer = VideoComposer(
            processed_videos_dir=base_dir,
            midi_data=processor_result['tracks'],
            output_path=output_path
        )
        
        track_count = len(processor_result['processed_files']['tracks']) + len(processor_result['processed_files']['drum_tracks'])
        logging.info(f"Starting video composition with {track_count} tracks")
        
        composer.create_composition()
        
        return {
            'output_path': output_path,
            'track_count': track_count
        }
    except Exception as e:
        logging.error(f"Error in video composition: {str(e)}")
        raise
