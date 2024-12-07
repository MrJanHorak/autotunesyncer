# audio_processor.py
import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import os
import tempfile
import shutil
import mimetypes
import logging
from pathlib import Path
import uuid
import json
import subprocess
import argparse
from utils import normalize_instrument_name, midi_to_note
from drum_utils import is_drum_kit, get_drum_groups

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Simple format without "Python error:"
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('video_processing.log', mode='w')  # 'w' mode to clear previous logs
    ]
)

def get_drum_groups(track):
    """Match frontend's DRUM_GROUPS logic"""
    drum_groups = {
        'kick': [35, 36],
        'snare': [38, 40],
        'hihat': [42, 44, 46],
        'cymbal': [49, 51, 52, 55, 57],
        'tom': [41, 43, 45, 47, 48, 50]
    }
    # Extract unique drum groups from track notes
    groups = set()
    for note in track.get('notes', []):
        for group, midi_numbers in drum_groups.items():
            if note['midi'] in midi_numbers:
                groups.add(group)
    return groups

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

# In audio_processor.py, add this function:
def convert_video_format(input_path, output_path):
    """Convert video to MP4 format with proper timestamps"""
    try:
        logging.info(f"Starting video conversion: {input_path} -> {output_path}")
        
        # First check input file
        if not os.path.exists(input_path):
            raise Exception(f"Input file does not exist: {input_path}")
            
        # Log file size
        input_size = os.path.getsize(input_path)
        logging.info(f"Input file size: {input_size} bytes")
        
        # Convert video with detailed logging
        convert_cmd = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-progress', 'pipe:1',  # Show progress
            output_path
        ]
        
        logging.info(f"Running FFmpeg command: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg stderr: {result.stderr}")
            logging.error(f"FFmpeg stdout: {result.stdout}")
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
            
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            logging.info(f"Conversion successful. Output size: {output_size} bytes")
        else:
            raise Exception("Output file was not created")
            
        return output_path
        
    except Exception as e:
        logging.error(f"Video conversion failed: {str(e)}")
        raise
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

            # Add file type logging
            file_ext = os.path.splitext(video_path)[1]
            logging.info(f"Input video format: {file_ext}")
            
            # Check if input is WebM or has no extension
            temp_converted = None
            if file_ext.lower() in ['.webm', ''] or 'webm' in mimetypes.guess_type(video_path)[0]:
                temp_converted = os.path.join(self.temp_dir, f"converted_{os.path.basename(video_path)}.mp4")
                logging.info(f"Converting WebM to MP4: {temp_converted}")
                try:
                    video_path = convert_video_format(video_path, temp_converted)
                    logging.info("Conversion successful")
                except Exception as e:
                    logging.error(f"Conversion failed: {str(e)}")
                    raise

            # Add video validation logging
            logging.info(f"Validating video file: {video_path}")
               # Add video validation
            ffmpeg_check = [
                'ffmpeg',
                '-v', 'error',
                '-i', video_path,
                '-f', 'null',
                '-'
            ]
            result = subprocess.run(ffmpeg_check, capture_output=True, text=True)
            if result.stderr:
                logging.error(f"Video validation failed: {result.stderr}")
                raise Exception(f"Input video corrupted: {result.stderr}")
            logging.info("Video validation successful")
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
            # ffmpeg_combine = [
            #     'ffmpeg',
            #     '-y',
            #     '-i', video_path,
            #     '-i', tuned_audio,
            #     '-c:v', 'copy',
            #     '-c:a', 'aac',
            #     '-b:a', '320k',
            #     '-map', '0:v:0',
            #     '-map', '1:a:0',
            #     output_path
            # ]
            ffmpeg_combine = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-i', tuned_audio,
                '-c:v', 'copy',     # Copy video stream
                '-c:a', 'aac',      # Use AAC codec
                '-strict', 'experimental',  # Allow experimental codecs
                '-b:a', '320k',     # High audio quality
                '-map', '0:v:0',    # Map video stream
                '-map', '1:a:0',    # Map audio stream
                '-movflags', '+faststart',  # Optimize for web playback
                output_path
            ]
            
            logging.info(f"Combining video and audio: {' '.join(ffmpeg_combine)}")
            result = subprocess.run(ffmpeg_combine, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg combine failed: {result.stderr}")
            
              # Verify output
            check_output = subprocess.run(ffmpeg_check[:-1] + [output_path, '-f', 'null', '-'], 
                                        capture_output=True, text=True)
            if check_output.stderr:
                raise Exception(f"Output video corrupted: {check_output.stderr}")

            # Cleanup temp files
            os.remove(temp_audio)
            os.remove(tuned_audio)

            # Clean up temp converted file
            if temp_converted and os.path.exists(temp_converted):
                os.remove(temp_converted)

            return output_path

        except Exception as e:
            logging.error(f"Error in create_tuned_video: {str(e)}")
            raise

def process_track_videos(tracks, videos, processor):
    """Process video tracks with provided processor instance"""
    processed_videos = {
        'tracks': {},
        'drum_tracks': {},
        'metadata': {
            'base_dir': processor.videos_dir,
            'session_id': processor.session_id,
            'valid_track_count': 0
        }
    }
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logging.info(f"Processing videos from: {list(videos.keys())}")
        

        # Add validation for empty tracks
        if not tracks.get('tracks'):
            logging.error("No tracks found in input")
            raise ValueError("No tracks provided")
            
        # Add validation for missing videos
        if not videos:
            logging.error("No videos provided")
            raise ValueError("No videos provided")
            
        # Log input data
        logging.info(f"Processing {len(tracks['tracks'])} tracks")
        logging.info(f"Available videos: {list(videos.keys())}")
        # First pass: Process all unique instrument notes
        instrument_notes = {}  # Cache for processed notes
        
        # Process each track first
        for track_idx, track in enumerate(tracks['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            if not track.get('notes'):
                continue
            
            if is_drum_kit(instrument):
                # Handle drum tracks
                drum_processed = False
                for group in get_drum_groups(track):
                    instrument_key = f"drum_{group}"
                    if instrument_key in videos:
                        video_path = os.path.join(base_dir, videos[instrument_key])
                        if not os.path.exists(video_path):
                            logging.error(f"Drum video not found: {video_path}")
                            continue

                        output_dir = os.path.join(
                            processor.videos_dir,
                            f"track_{track_idx}_drums"
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Just copy drum video without pitch shifting
                        output_path = os.path.join(output_dir, f"{group}.mp4")
                        shutil.copy2(video_path, output_path)
                        
                        processed_videos['drum_tracks'][f"track_{track_idx}_{group}"] = {
                            'track_idx': track_idx,
                            'group': group,
                            'path': output_path,
                            'relative_path': os.path.relpath(output_path, processor.videos_dir)
                        }
                        logging.info(f"Processed drum track {track_idx}: {group}")
                        drum_processed = True
                
                if drum_processed:
                    processed_videos['metadata']['valid_track_count'] += 1
                    
            else:
                # Handle instrument tracks
                if instrument_name not in videos:
                    logging.info(f"Skipping track {track_idx}: {instrument_name} (no video)")
                    continue
                    
                video_path = os.path.join(base_dir, videos[instrument_name])
                if not os.path.exists(video_path):
                    logging.error(f"Video not found: {video_path}")
                    continue
                
                 # Initialize instrument cache if needed
            if instrument_name not in instrument_notes:
                instrument_notes[instrument_name] = {
                    'video_path': video_path,
                    'notes': {}
                }
            
            # Process unique notes for this instrument
            unique_notes = {int(float(note['midi'])) for note in track.get('notes', [])}
            for midi_note in unique_notes:
                if midi_note not in instrument_notes[instrument_name]['notes']:
                    try:
                        output_path = os.path.join(
                            processor.videos_dir,
                            f"{instrument_name}_notes",  # Group by instrument
                            f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                        )
                        
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        processed_path = processor.create_tuned_video(
                            video_path,
                            midi_note,
                            output_path
                        )
                        
                        instrument_notes[instrument_name]['notes'][midi_note] = {
                            'path': processed_path,
                            'relative_path': os.path.relpath(processed_path, processor.videos_dir),
                            'note_name': midi_to_note(midi_note)
                        }
                    except Exception as e:
                        logging.error(f"Failed to process note {midi_note}: {str(e)}")
                        continue
        
        # Second pass: Create track directories with symlinks
        for track_idx, track in enumerate(tracks['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            if is_drum_kit(instrument):
                # Handle drums (existing code)...
                continue
                
            if instrument_name not in instrument_notes:
                continue
                
            # Create track directory
            track_dir = os.path.join(
                processor.videos_dir,
                f"track_{track_idx}_{instrument_name}"
            )
            os.makedirs(track_dir, exist_ok=True)
            
            # Link needed notes
            unique_notes = {int(float(note['midi'])) for note in track.get('notes', [])}
            track_output = {
                'track_idx': track_idx,
                'notes': {},
                'base_path': track_dir
            }
            
            for midi_note in unique_notes:
                if midi_note in instrument_notes[instrument_name]['notes']:
                    source = instrument_notes[instrument_name]['notes'][midi_note]['path']
                    dest = os.path.join(track_dir, os.path.basename(source))
                    
                    if os.path.exists(dest):
                        os.remove(dest)  # Remove existing link/file
                    shutil.copy2(source, dest)  # Create symlink
                    
                    track_output['notes'][midi_note] = {
                        'path': dest,
                        'relative_path': os.path.relpath(dest, processor.videos_dir),
                        'note_name': midi_to_note(midi_note)
                    }
            
            processed_videos['tracks'][f"{instrument_name}_{track_idx}"] = track_output
            processed_videos['metadata']['valid_track_count'] += 1
            logging.info(f"Processed track {track_idx}: {instrument_name}")

        if processed_videos['metadata']['valid_track_count'] == 0:
            logging.error("No valid tracks were processed")
            raise ValueError("No valid tracks were processed")

        return processed_videos
                    
    except Exception as e:
        logging.error(f"Error processing track videos: {str(e)}")
        raise

# def process_track_videos(tracks, videos, processor):
#     """Process video tracks with provided processor instance"""
#     processed_videos = {
#         'tracks': {},
#         'drum_tracks': {},
#         'paths': {
#             'base_dir': processor.videos_dir,
#             'session_id': processor.session_id
#         }
#     }
    
#     # processor = AudioVideoProcessor()
#     # processor.setup_temp_directory()
    
#     try:
#         if not isinstance(tracks, dict) or 'tracks' not in tracks:
#             raise ValueError("Invalid tracks format in config")

#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         logging.info(f"Available videos: {list(videos.keys())}")
        
#         for track_idx, track in enumerate(tracks['tracks']):
#             instrument = track.get('instrument', {})
            
#             if is_drum_kit(instrument):
#                 # Handle drum tracks
#                 for group in get_drum_groups(track):
#                     instrument_key = f"drum_{group}"
#                     if instrument_key in videos:
#                         video_path = os.path.join(base_dir, videos[instrument_key])
#                         if not os.path.exists(video_path):
#                             logging.error(f"Drum video not found: {video_path}")
#                             continue

#                          # Create drum output directory
#                         output_dir = os.path.join(
#                             processor.videos_dir,
#                             f"track_{track_idx}_drums",
#                             group
#                         )
#                         os.makedirs(output_dir, exist_ok=True)
                        
#                         # Copy drum video to processed folder
#                         output_path = os.path.join(output_dir, f"{instrument_key}.mp4")
#                         shutil.copy2(video_path, output_path)
                            
#                         processed_videos['drum_tracks'][instrument_key] = {
#                             'track_idx': track_idx,
#                             'group': group,
#                             'path': output_path,  # Use full path
#                             'relative_path': os.path.relpath(output_path, processor.videos_dir)
#                         }
#                         logging.info(f"Processed drum track {track_idx}: {group}")
#             else:
#                 # Handle instrument tracks
#                 instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
#                 processed_videos['tracks'][instrument_name] = {
#                     'track_idx': track_idx,
#                     'notes': {},
#                     'base_path': os.path.join(processor.videos_dir, f"track_{track_idx}_{instrument_name}")
#                 }
#                 if instrument_name not in videos:
#                     logging.warning(f"No video found for instrument: {instrument_name}")
#                     continue
                    
#                 video_path = os.path.join(base_dir, videos[instrument_name])
#                 if not os.path.exists(video_path):
#                     logging.error(f"Video not found: {video_path}")
#                     continue
                
#                 logging.info(f"Processing track {track_idx}: {instrument_name}")
                
#                 # Rest of your existing instrument processing code
#                 unique_notes = {int(float(note['midi'])) for note in track.get('notes', [])}
#                 processed_videos[instrument_name] = []
                
#                 for midi_note in unique_notes:
#                     try:
#                         output_path = os.path.join(
#                             processor.videos_dir,
#                             f"track_{track_idx}_{instrument_name}",
#                             f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
#                         )
                        
#                         os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
#                         processed_path = processor.create_tuned_video(
#                             video_path,
#                             midi_note,
#                             output_path
#                         )
                        
#                         processed_videos[instrument_name].append({
#                             'track': track_idx,
#                             'note': midi_note,
#                             'note_name': midi_to_note(midi_note),
#                             'path': processed_path
#                         })
#                         processed_videos['tracks'][instrument_name]['notes'][midi_note] = {
#                             'path': processed_path,  # Use full path
#                             'relative_path': os.path.relpath(processed_path, processor.videos_dir)
#                         }
#                         logging.info(f"Processed note {midi_note} for track {track_idx}")
#                     except Exception as e:
#                         logging.error(f"Failed to process note {midi_note}: {str(e)}")
#                         continue

#         return processed_videos
                    
#     except Exception as e:
#         logging.error(f"Error processing track videos: {str(e)}")
#         raise

if __name__ == "__main__":


    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path', help='Path to config JSON file')
        args = parser.parse_args()
        
        with open(args.config_path) as f:
            config = json.load(f)
            
        processor = AudioVideoProcessor()
        processor.setup_temp_directory()
        
        result = process_track_videos(
            config.get('tracks', {}), 
            config.get('videos', {}),
            processor
        )
        
        output_path = os.path.join(
            os.path.dirname(processor.videos_dir),
            f"final_composition_{processor.session_id}.mp4"
        )
        
        composition_result = compose_from_processor_output(
            {
                'processed_videos_dir': processor.videos_dir,
                'tracks': config['tracks'],
                'processed_files': result
            },
            output_path
        )
        
        print(json.dumps({
            'success': True,
            'data': {
                'processed': result,
                'composition': composition_result
            }
        }), flush=True)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }), flush=True)