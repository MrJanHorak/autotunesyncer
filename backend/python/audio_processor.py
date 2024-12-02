# audio_processor.py
import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import os
import tempfile
import shutil
import logging
from pathlib import Path
import uuid
import json
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Simple format without "Python error:"
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('video_processing.log', mode='w')  # 'w' mode to clear previous logs
    ]
)

def is_drum_kit(instrument):
    """Check if instrument is a drum kit based on name or channel 10 (9 in zero-based)"""
    drum_keywords = ['standard kit', 'drum kit', 'drums', 'percussion']
    name = instrument.get('name', '').lower()
    channel = instrument.get('channel', 0)
    
    return (
        any(keyword in name for keyword in drum_keywords) or
        'drum' in name or
        channel == 9  # MIDI channel 10 (0-based) is reserved for drums
    )

def midi_to_note(midi_num):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes[midi_num % 12]
    octave = (midi_num // 12) - 1
    return f"{note_name}{octave}"
class AudioVideoProcessor:
    def __init__(self):
        self.sample_rate = 44100
        self.frame_length = 2048
        self.temp_dir = None
        self.session_id = str(uuid.uuid4())
        self.videos_dir = None
        # Add base directory for all processed videos
        self.base_videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_videos')
    
    def setup_temp_directory(self):
        """Create temporary directories for processing files"""
        self.temp_dir = tempfile.mkdtemp(prefix='autotune_temp_')
        # Create a separate directory for processed videos using absolute path
        os.makedirs(self.base_videos_dir, exist_ok=True)
        self.videos_dir = os.path.join(self.base_videos_dir, self.session_id)
        os.makedirs(self.videos_dir, exist_ok=True)
        logging.info(f"Created videos directory: {self.videos_dir}")
        return self.temp_dir
    
    def cleanup_temp_directory(self):
        """Remove temporary directory but keep the videos directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def analyze_pitch(self, audio_path):
        # Load audio and extract pitch
        y, sr = librosa.load(audio_path)
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr,
            n_fft=self.frame_length,
            hop_length=self.frame_length//4
        )
        
        # Get predominant pitch
        pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)*0.1])
        return librosa.hz_to_midi(pitch_mean)
    
    def create_tuned_video(self, video_path, target_note, output_path):
        try:
            logging.info(f"Processing video: {video_path} for note: {target_note}")
            target_note = int(target_note)
            
            # Create paths in temp directory
            temp_audio = os.path.join(self.temp_dir, f"temp_{os.path.basename(video_path)}.wav")
            tuned_audio = os.path.join(self.temp_dir, f"tuned_{os.path.basename(video_path)}.wav")
            
            # Extract audio
            ffmpeg_extract = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',  # Convert to mono
                temp_audio
            ]
            
            logging.info(f"Extracting audio: {' '.join(ffmpeg_extract)}")
            result = subprocess.run(ffmpeg_extract, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg extract failed: {result.stderr}")

            # Analyze and shift pitch
            current_pitch = self.analyze_pitch(temp_audio)
            pitch_shift = target_note - current_pitch
            
            logging.info(f"Current pitch: {current_pitch:.1f}, Target: {target_note}, Shift: {pitch_shift:.1f} semitones")
            
            if abs(pitch_shift) < 0.1:
                logging.info("Pitch shift too small, copying original video")
                shutil.copy2(video_path, output_path)
                return output_path
                
            # Use rubberband for pitch shifting with quality settings
            # Update the rubberband command in create_tuned_video method:
            # rubberband_cmd = [
            #     'rubberband',
            #     '--pitch', f"{pitch_shift:.3f}",  # Keep pitch shift value
            #     '--time', '1.0',                  # Keep original duration (ratio 1.0)
            #     temp_audio,
            #     tuned_audio
            # ]
            
            rubberband_cmd = [
                'rubberband',
                '-p', f"{pitch_shift:.3f}",  # Use -p instead of --pitch
                '-t', '1.0',                  # Keep original duration
                '-F',                         # Enable formant preservation
                '-c', '4',                    # Set crispness level for better quality
                '-2',                         # Use R2 engine (faster)
                temp_audio,
                tuned_audio
            ]

            logging.info(f"Pitch shifting with rubberband: {' '.join(rubberband_cmd)}")
            result = subprocess.run(rubberband_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Rubberband failed: {result.stderr}")

            # Combine video and audio with better quality settings
            ffmpeg_combine = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-i', tuned_audio,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '320k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            
            logging.info(f"Combining video and audio: {' '.join(ffmpeg_combine)}")
            result = subprocess.run(ffmpeg_combine, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg combine failed: {result.stderr}")

            # Cleanup temp files
            os.remove(temp_audio)
            os.remove(tuned_audio)

            return output_path

        except Exception as e:
            logging.error(f"Error in create_tuned_video: {str(e)}")
            raise

def process_track_videos(tracks, videos):
    processed_videos = {}
    processor = AudioVideoProcessor()
    processor.setup_temp_directory()
    
    try:
        if not isinstance(tracks, dict) or 'tracks' not in tracks:
            raise ValueError("Invalid tracks format in config")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for track_idx, track in enumerate(tracks['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = instrument.get('name', 'default')
            
            if instrument_name not in videos:
                continue
                
            video_path = os.path.join(base_dir, videos[instrument_name])
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
            
            logging.info(f"Processing track {track_idx}: {instrument_name}")
            
            if is_drum_kit(instrument):
                logging.info(f"Skipping drum kit: {instrument_name}")
                continue
            
            # Get unique notes only
            unique_notes = {note['midi'] for note in track.get('notes', [])}
            processed_videos[instrument_name] = []
            
            for midi_note in unique_notes:
                try:
                    output_path = os.path.join(
                        processor.videos_dir,
                        f"track_{track_idx}_{instrument_name}",
                        f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                    )
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    processed_path = processor.create_tuned_video(
                        video_path,
                        midi_note,
                        output_path
                    )
                    
                    processed_videos[instrument_name].append({
                        'track': track_idx,
                        'note': midi_note,
                        'note_name': midi_to_note(midi_note),
                        'path': processed_path
                    })
                    logging.info(f"Processed note {midi_note} for track {track_idx}")
                except Exception as e:
                    logging.error(f"Failed to process note {midi_note}: {str(e)}")
                    continue

        return processed_videos
                    
    except Exception as e:
        logging.error(f"Error processing track videos: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    import argparse
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path', help='Path to config JSON file')
        args = parser.parse_args()
        
        logging.info("Python script started")
        logging.info(f"Loaded config from: {args.config_path}")
        
        with open(args.config_path) as f:
            config = json.load(f)
        
        processor = AudioVideoProcessor()
        processor.setup_temp_directory()
        
        tracks = config.get('tracks', {})
        videos = config.get('videos', {})
        
        result = process_track_videos(tracks, videos)
        
        if not result:
            raise Exception("No videos were processed")
            
        print(json.dumps({
            'success': True,
            'data': result
        }), flush=True)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }), flush=True)