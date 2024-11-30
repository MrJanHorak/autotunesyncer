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
        """
        Extract audio, tune it, and recombine with video
        Returns path to tuned video
        """
        try:
            logging.info(f"Processing video: {video_path} for note: {target_note}")
            
            # Create paths in temp directory for intermediate files
            temp_audio = os.path.join(self.temp_dir, f"temp_{os.path.basename(video_path)}.wav")
            tuned_audio = os.path.join(self.temp_dir, f"tuned_{os.path.basename(video_path)}.wav")
            
            # Extract audio from video
            ffmpeg_extract = f'ffmpeg -i "{video_path}" -vn "{temp_audio}"'
            logging.info(f"Extracting audio: {ffmpeg_extract}")
            result = os.system(ffmpeg_extract)
            if result != 0:
                raise Exception(f"FFmpeg audio extraction failed with code {result}")
            
            if not os.path.exists(temp_audio):
                raise Exception(f"Audio extraction failed - {temp_audio} not created")

            # Analyze and tune audio
            current_pitch = self.analyze_pitch(temp_audio)
            pitch_shift = target_note - current_pitch
            logging.info(f"Shifting pitch from {current_pitch} to {target_note} (shift: {pitch_shift})")
            
            # Load and shift audio
            y, sr = librosa.load(temp_audio)
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            
            # Save tuned audio
            sf.write(tuned_audio, y_shifted, sr)
            
            if not os.path.exists(tuned_audio):
                raise Exception(f"Tuned audio file not created: {tuned_audio}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Combine tuned audio with original video
            ffmpeg_combine = f'ffmpeg -i "{video_path}" -i "{tuned_audio}" -c:v copy -c:a aac "{output_path}"'
            logging.info(f"Combining video and audio: {ffmpeg_combine}")
            result = os.system(ffmpeg_combine)
            if result != 0:
                raise Exception(f"FFmpeg combine failed with code {result}")
            
            if not os.path.exists(output_path):
                raise Exception(f"Final video not created: {output_path}")
            
            # Cleanup only temporary audio files
            os.remove(temp_audio)
            os.remove(tuned_audio)
            
            logging.info(f"Successfully created tuned video: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            raise

def process_track_videos(midi_tracks, instrument_videos):
    processor = AudioVideoProcessor()
    processor.setup_temp_directory()
    
    try:
        logging.info(f"Processing tracks with session ID: {processor.session_id}")
        logging.info(f"Input tracks: {midi_tracks}")
        logging.info(f"Input videos: {instrument_videos}")
        
        output_videos = {
            'drums': [],
            'instruments': {},
            'session_id': processor.session_id  # Return session_id instead of temp_dir
        }
        
        for track in midi_tracks:
            if track.get('isDrum'):
                output_videos['drums'].extend(instrument_videos[track['id']])
            else:
                instrument_id = track['id']
                if instrument_id not in output_videos['instruments']:
                    output_videos['instruments'][instrument_id] = []
                    
                notes = set(note['note'] for note in track['notes'])
                source_video = instrument_videos[instrument_id][0]
                
                for note in notes:
                    output_path = os.path.join(processor.videos_dir, f"tuned_{instrument_id}_note_{note}.mp4")
                    tuned_video = processor.create_tuned_video(
                        source_video,
                        note,
                        output_path
                    )
                    output_videos['instruments'][instrument_id].append({
                        'note': note,
                        'session_id': processor.session_id,
                        'filename': os.path.basename(tuned_video)
                    })
        
        return output_videos
        
    except Exception as e:
        logging.error(f"Error in process_track_videos: {str(e)}")
        processor.cleanup_temp_directory()
        raise

if __name__ == '__main__':
    import sys
    import json
    
    # Configure logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Read input from Node.js
        input_data = json.loads(sys.argv[1])
        
        # Process
        result = process_track_videos(
            input_data['tracks'],
            input_data['videos']
        )
        
        # Return results to Node.js
        print(json.dumps({
            'success': True,
            'data': result
        }))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))