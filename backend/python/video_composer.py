# video_composer.py
from pathlib import Path
import logging
from moviepy.editor import VideoFileClip, clips_array, CompositeVideoClip, ColorClip
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
        self.processed_videos_dir = Path(processed_videos_dir)
        self.midi_data = midi_data
        self.output_path = output_path
        self.frame_rate = 30

    def get_track_layout(self):
        """Determine grid layout based on number of valid tracks"""
        # Only count tracks that have videos
        valid_tracks = []

            # Log directory contents
        logging.info(f"Scanning directory: {self.processed_videos_dir}")
        for path in self.processed_videos_dir.rglob('*.mp4'):
            logging.info(f"Found video file: {path}")

        for track_idx, track in enumerate(self.midi_data['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            track_dir = self.processed_videos_dir / f"track_{track_idx}_{instrument_name}"
            
            if track_dir.exists() and any(track_dir.glob('*.mp4')):
                valid_tracks.append(track)
        
        track_count = len(valid_tracks)
        logging.info(f"Found {track_count} valid tracks with videos")
        
        if track_count == 0:
            raise ValueError("No valid tracks with videos found")
        elif track_count == 1:
            return (1, 1)
        elif track_count <= 2:
            return (1, 2)
        elif track_count <= 4:
            return (2, 2)
        else:
            cols = min(3, track_count)
            rows = (track_count + cols - 1) // cols
            return (rows, cols)
        
    def create_track_video(self, track, track_idx, duration):
        try:
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            track_dir = self.processed_videos_dir / f"track_{track_idx}_{instrument_name}"
            
            logging.info(f"Creating video for track {track_idx}: {instrument_name}")
            
            if not track_dir.exists():
                logging.warning(f"No video directory found for track {track_idx}")
                return None
                
            clips = []
            
            # Handle drum tracks differently
            if is_drum_kit(instrument):
                for group_file in track_dir.glob('*.mp4'):
                    try:
                        clip = VideoFileClip(str(group_file), audio=True)
                        clip = clip.set_duration(duration)
                        clips.append(clip)
                    except Exception as e:
                        logging.error(f"Error loading drum clip {group_file}: {e}")
                        continue
                if clips:
                    return clips[0]  # Return first valid drum clip
                return None
                
            # Handle instrument tracks
            background = None
            for note in track.get('notes', []):
                try:
                    midi_note = int(float(note['midi']))
                    note_file = track_dir / f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                    
                    if note_file.exists():
                        clip = VideoFileClip(str(note_file), audio=True)
                        clip = clip.set_start(note['time'])
                        clip = clip.set_duration(note['duration'])
                        clips.append(clip)
                        
                        if background is None:
                            background = ColorClip(
                                size=(clip.w, clip.h),
                                color=(0, 0, 0),
                                duration=duration
                            )
                except Exception as e:
                    logging.error(f"Error loading note clip {midi_note}: {e}")
                    continue
                    
            if not clips:
                return None
                
            try:
                # Handle memory more efficiently
                final_clip = CompositeVideoClip(
                    [background] + clips,
                    size=background.size
                ).set_duration(duration)
                return final_clip
            except Exception as e:
                logging.error(f"Error compositing clips: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Error creating track video: {str(e)}")
            return None
        
    def create_composition(self):
        try:
            rows, cols = self.get_track_layout()
            valid_clips = []
            
            max_time = max(
                note['time'] + note['duration']
                for track in self.midi_data['tracks']
                for note in track.get('notes', [])
            )
            
            for track_idx, track in enumerate(self.midi_data['tracks']):
                try:
                    clip = self.create_track_video(track, track_idx, max_time)
                    if clip is not None:
                        valid_clips.append(clip)
                except Exception as e:
                    logging.error(f"Error processing track {track_idx}: {e}")
                    continue
                    
            if not valid_clips:
                raise ValueError("No valid clips created")
                
            # Process in smaller batches if needed
            grid = []
            for i in range(0, len(valid_clips), cols):
                row = valid_clips[i:i + cols]
                if row:
                    grid.append(row)
                    
            final_clip = clips_array(grid)
            final_clip.write_videofile(
                self.output_path,
                fps=self.frame_rate,
                codec='libx264',
                audio=True,
                audio_codec='aac',
                threads=4,
                preset='medium'  # Balance between speed and quality
            )
            
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