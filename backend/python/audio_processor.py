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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_processing.log')
    ]
)

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
            
            ffmpeg_extract = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 "{temp_audio}"'
            os.system(ffmpeg_extract)
        
            current_pitch = self.analyze_pitch(temp_audio)
            pitch_shift = target_note - current_pitch
        
            # Log detailed pitch information
            logging.info(f"""
            Pitch analysis:
            - Original pitch: {current_pitch} MIDI ({midi_to_note(current_pitch)})
            - Target pitch: {target_note} MIDI ({midi_to_note(target_note)})
            - Required shift: {pitch_shift} semitones
            """)
        
            if abs(pitch_shift) < 0.1:
                logging.info("Pitch shift too small, copying original video")
                shutil.copy2(video_path, output_path)
                return output_path
            
            tuned_audio = os.path.join(self.temp_dir, f"tuned_{os.path.basename(video_path)}.wav")
            
            # Extract audio from video with higher quality settings
            ffmpeg_extract = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 "{temp_audio}"'
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
            
            # Load and shift audio with higher quality settings
            y, sr = librosa.load(temp_audio, sr=44100)
            # Use rubberband for better quality pitch shifting
            y_shifted = librosa.effects.pitch_shift(
                y, 
                sr=sr,
                n_steps=pitch_shift,
                bins_per_octave=12,
                res_type='kaiser_best'
            )
            
            # Save tuned audio with higher quality
            sf.write(tuned_audio, y_shifted, sr, subtype='PCM_24')
            
            if not os.path.exists(tuned_audio):
                raise Exception(f"Tuned audio file not created: {tuned_audio}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Combine tuned audio with original video using higher quality settings
            ffmpeg_combine = f'ffmpeg -i "{video_path}" -i "{tuned_audio}" -c:v copy -c:a aac -b:a 320k "{output_path}"'
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

def process_track_videos(midi_tracks, instrument_videos):
    try:
        logging.info("=== Starting video processing ===")
        logging.info(f"Received MIDI tracks: {json.dumps(midi_tracks, indent=2)}")

        logging.info(f"Received videos: {json.dumps(instrument_videos, indent=2)}")
        
        # Debug print to stdout as well
        print("Python processor started", flush=True)

        # Create shorter paths by using relative paths where possible
        base_dir = os.path.dirname(os.path.abspath(__file__))
        processed_videos = {}
        
        for instrument_name, video_path in instrument_videos.items():
            try:
                # Convert to relative path if possible
                proper_path = os.path.normpath(video_path)
                rel_path = os.path.relpath(proper_path, base_dir)
                if len(rel_path) < len(proper_path):
                    proper_path = rel_path
            except ValueError:
                # Keep absolute path if relative path not possible
                proper_path = os.path.normpath(video_path)
                
            if not os.path.exists(proper_path):
                raise Exception(f"Video file not found: {proper_path}")
            processed_videos[instrument_name] = proper_path
            logging.info(f"Processing video for {instrument_name}: {proper_path}")

        # Write configuration to temporary file
        config = {
            "midi_tracks": midi_tracks,
            "instrument_videos": processed_videos
        }
        
        temp_config = os.path.join(tempfile.gettempdir(), f"video_config_{uuid.uuid4()}.json")
        with open(temp_config, 'w') as f:
            json.dump(config, f)

         # Add detailed note analysis per track
        for track in midi_tracks.get('tracks', []):
            instrument = track.get('instrument', {})
            instrument_name = instrument.get('name', '').lower().replace(' ', '_')

             # Skip pitch processing for drum tracks
            if is_drum_kit(instrument):
                logging.info(f"Detected drum kit: {instrument_name}")
                # Just copy the drum video to processed folder without pitch shifting
                if instrument_name in processed_videos:
                    source_video = processed_videos[instrument_name]
                    drum_output = os.path.join(
                        processor.videos_dir,
                        f"drum_{instrument_name}.mp4"
                    )
                    shutil.copy2(source_video, drum_output)
                    output_videos['instruments'][instrument_name] = [{
                        'type': 'drum',
                        'session_id': processor.session_id,
                        'filename': os.path.basename(drum_output)
                    }]
                continue

            notes = set(note['midi'] for note in track.get('notes', []))
            note_names = [midi_to_note(note) for note in sorted(notes)]
            logging.info(f"Instrument '{instrument_name}' uses notes: {', '.join(note_names)}")
            
        logging.info(f"Received videos: {json.dumps(instrument_videos, indent=2)}")
        
        # Convert Windows paths to proper format and validate
        processed_videos = {}
        for instrument_name, video_path in instrument_videos.items():
            # Convert backslashes to forward slashes
            proper_path = os.path.normpath(video_path)
            if not os.path.exists(proper_path):
                raise Exception(f"Video file not found: {proper_path}")
            processed_videos[instrument_name] = proper_path
            logging.info(f"Verified video exists: {proper_path}")

        processor = AudioVideoProcessor()
        processor.setup_temp_directory()
        
        logging.info(f"Processing tracks with session ID: {processor.session_id}")
        
        output_videos = {
            'instruments': {},
            'session_id': processor.session_id
        }
        
        # Process each track
        for track in midi_tracks.get('tracks', []):
            instrument = track.get('instrument', {})
            instrument_name = instrument.get('name', '').lower().replace(' ', '_')
            
            if not instrument_name or instrument_name not in processed_videos:
                logging.warning(f"Skipping track - no matching video for instrument: {instrument_name}")
                continue
                
            if instrument_name not in output_videos['instruments']:
                output_videos['instruments'][instrument_name] = []
                
            # Get unique MIDI notes from the track
            notes = set(note['midi'] for note in track.get('notes', []))
            source_video = processed_videos[instrument_name]
            
            for note in notes:
                output_path = os.path.join(
                    processor.videos_dir, 
                    f"tuned_{instrument_name}_note_{note}.mp4"
                )
                try:
                    tuned_video = processor.create_tuned_video(
                        source_video,
                        note,
                        output_path
                    )
                    output_videos['instruments'][instrument_name].append({
                        'note': note,
                        'session_id': processor.session_id,
                        'filename': os.path.basename(tuned_video)
                    })
                    logging.info(f"Successfully processed note {note} for {instrument_name}")
                except Exception as e:
                    logging.error(f"Failed to process note {note} for {instrument_name}: {str(e)}")
                    continue
        
        return output_videos
        
    except Exception as e:
        logging.error(f"Error in process_track_videos: {str(e)}")
        raise
    finally:
        if 'processor' in locals():
            processor.cleanup_temp_directory()

# if __name__ == '__main__':
#     import sys
#     import json
#     import argparse
    
#     # logging.basicConfig(
#     #     level=logging.INFO,
#     #     format='%(asctime)s - %(levelname)s - %(message)s',
#     #     handlers=[
#     #         logging.StreamHandler(),
#     #         logging.FileHandler('video_processing.log')
#     #     ]
#     # )

#     logging.info("Python script started")

#     try:
#         logging.info(f"Received args: {sys.argv}")
#         input_data = json.loads(sys.argv[1])
#         logging.info(f"Parsed input data: {json.dumps(input_data, indent=2)}")
        
#         if not input_data.get('tracks') or not input_data.get('videos'):
#             raise Exception("Missing required tracks or videos data")
        
#         # Process
#         result = process_track_videos(
#             input_data['tracks'],
#             input_data['videos']
#         )
        
#         # Return results to Node.js
#         print(json.dumps({
#             'success': True,
#             'data': result
#         }))
        
#     except Exception as e:
#         logging.error(f"Error in main: {str(e)}")
#         print(json.dumps({
#             'success': False,
#             'error': str(e)
#         }), flush=True)

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    logging.info("Python script started")
    
    try:
        # Read config from file
        with open(args.config, 'r') as f:
            input_data = json.load(f)
            
        logging.info(f"Loaded config from: {args.config}")
        logging.info(f"Config data: {json.dumps(input_data, indent=2)}")
        
        if not input_data.get('tracks') or not input_data.get('videos'):
            raise Exception("Missing required tracks or videos data")
            
        result = process_track_videos(
            input_data['tracks'],
            input_data['videos']
        )
        
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