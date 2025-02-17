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
import concurrent.futures 

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

# In audio_processor.py, add this function:
def convert_video_format(input_path, output_path):
    """Convert video to MP4 format with proper timestamps"""
    try:
        logging.info(f"Starting video conversion: {input_path} -> {output_path}")

          # Probe input video
        probe_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-show_entries', 'stream=width,height,codec_name',
            '-of', 'json',
            '-i', input_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        logging.info(f"Input video probe: {probe_result.stdout}")
        
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
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
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

         # Validate output video
        validate_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', output_path,
            '-f', 'null',
            '-'
        ]
        validate_result = subprocess.run(validate_cmd, capture_output=True, text=True)

        if validate_result.stderr:
            logging.error(f"Output validation failed: {validate_result.stderr}")
            raise Exception("Output video validation failed")
            
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
        self.base_videos_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "processed_videos"
        )
    
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
    
    def validate_output_video(self, video_path):
        """Validate video file integrity"""
        try:
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            duration = float(subprocess.check_output(probe_cmd).decode().strip())
            return duration > 0
        except:
            return False
    
    def create_tuned_video(self, video_path, target_note, output_path):
        try:
            logging.info(f"Processing video: {video_path} for note: {target_note}")
            target_note = int(target_note)

            # Add file type logging
            file_ext = os.path.splitext(video_path)[1]
            logging.info(f"Input video format: {file_ext}")
            
            # Create paths in temp directory
            temp_audio = os.path.join(self.temp_dir, f"temp_{os.path.basename(video_path)}.wav")
            tuned_audio = os.path.join(self.temp_dir, f"tuned_{os.path.basename(video_path)}.wav")
            temp_video = os.path.join(self.temp_dir, f"temp_video_{os.path.basename(video_path)}.mp4")
            
            # First transcode video to H.264
            ffmpeg_transcode = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-c:v', 'h264_nvenc',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-an',  # No audio
                temp_video
            ]
            
            logging.info(f"Transcoding video to H.264: {' '.join(ffmpeg_transcode)}")
            result = subprocess.run(ffmpeg_transcode, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg transcode failed: {result.stderr}")

            # Extract audio
            ffmpeg_extract = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
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
                ffmpeg_transcode = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-c:v', 'h264_nvenc',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    '-b:a', '320k',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    output_path
                ]
                subprocess.run(ffmpeg_transcode, check=True)
                
                # Validate output
                if not self.validate_output_video(output_path):
                    raise Exception("Output validation failed")
                    
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

            ffmpeg_combine = [
                'ffmpeg',
                '-y',
                '-i', temp_video,
                '-i', tuned_audio,
                '-c:v', 'h264_nvenc',  # Re-encode instead of copy
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-b:a', '320k',
                '-pix_fmt', 'yuv420p',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-movflags', '+faststart',
                output_path
            ]
            
            logging.info(f"Combining video and audio: {' '.join(ffmpeg_combine)}")
            result = subprocess.run(ffmpeg_combine, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg combine failed: {result.stderr}")
            
            # Enhanced validation
            validate_cmd = [
                'ffmpeg',
                '-v', 'error',
                '-i', output_path,
                '-c', 'copy',
                '-f', 'null',
                '-'
            ]
            
            logging.info(f"Validating output video: {' '.join(validate_cmd)}")
            result = subprocess.run(validate_cmd, capture_output=True, text=True)
            if result.returncode != 0 or result.stderr:
                raise Exception(f"Output video validation failed: {result.stderr}")

            # Add extra frame validation
            probe_cmd = [
                'ffmpeg',
                '-v', 'error',
                '-i', output_path,
                '-f', 'null',
                '-'
            ]
        
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode != 0:
                raise Exception(f"Video validation failed: {probe_result.stderr}")
                
            return output_path

        except Exception as e:
            # Cleanup on error
            logging.error(f"Error processing video: {str(e)}")
            for temp_file in [temp_audio, tuned_audio, temp_video]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            logging.error(f"Error in create_tuned_video: {str(e)}")
            raise

def process_drum_track(video_path, output_path):
    """Process and validate drum track video"""
    try:
        # Convert to consistent format first
        temp_output = output_path + '.temp.mp4'
        convert_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            temp_output
        ]
        
        subprocess.run(convert_cmd, check=True)
        
        # Validate converted video
        validate_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', temp_output,
            '-f', 'null',
            '-'
        ]
        
        subprocess.run(validate_cmd, check=True)
        
        # If validation passes, move to final location
        shutil.move(temp_output, output_path)
        return True
        
    except Exception as e:
        logging.error(f"Failed to process drum track: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False
    
# def process_track_videos(tracks, videos, processor, nvenc=True):
#     """Process video tracks with provided processor instance"""
#     processed_videos = {
#         'tracks': {},
#         'drum_tracks': {},
#         'metadata': {
#             'base_dir': processor.videos_dir,
#             'session_id': processor.session_id,
#             'valid_track_count': 0
#         }
#     }

#     logging.info("Starting track processing...")
#     if isinstance(tracks, list):
#         # Convert list to dictionary if needed
#         tracks = {idx: track for idx, track in enumerate(tracks)}
        
#     for track_idx, track in tracks.items():
#         logging.info(f"Processing track {track_idx}")
#         if not isinstance(track, dict):
#                 track = {"notes": track} if hasattr(track, "notes") else {}
#         if is_drum_kit(track):
#             logging.info(f"Found drum track {track_idx}: {track.get('instrument', {}).get('name')}")
    
#     try:
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         logging.info(f"Processing videos from: {list(videos.keys())}")
        
#         # Add validation for empty tracks
#         if not tracks.get('tracks'):
#             logging.error("No tracks found in input")
#             raise ValueError("No tracks provided")
            
#         # Add validation for missing videos
#         if not videos:
#             logging.error("No videos provided")
#             raise ValueError("No videos provided")
            
#         # Log input data
#         logging.info(f"Processing {len(tracks['tracks'])} tracks")
#         logging.info(f"Available videos: {list(videos.keys())}")
#         # First pass: Process all unique instrument notes
#         instrument_notes = {}  # Cache for processed notes
        
#         # Process each track first
#         for track_idx, track in enumerate(tracks['tracks']):
#             instrument = track.get('instrument', {})
#             instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
#             if not track.get('notes'):
#                 continue
            
#             if is_drum_kit(instrument):

#                 for group in get_drum_groups(track):
#                     instrument_key = f"drum_{group}"
#                     if instrument_key in videos:
#                         video_path = videos[instrument_key]  # Remove os.path.join(base_dir)
#                         if not os.path.exists(video_path):
#                             logging.error(f"Drum video not found: {video_path}")
#                             continue

#                         output_dir = os.path.join(
#                             processor.videos_dir,
#                             f"track_{track_idx}_drums"
#                         )
#                         os.makedirs(output_dir, exist_ok=True)
#                         output_path = os.path.join(output_dir, f"{group}.mp4")
                        
#                         if process_drum_track(video_path, output_path):
#                             processed_videos['drum_tracks'][f"track_{track_idx}_{group}"] = {
#                                 'track_idx': track_idx,
#                                 'group': group,
#                                 'path': output_path,
#                                 'relative_path': os.path.relpath(output_path, processor.videos_dir)
#                             }
#                             logging.info(f"Processed drum track {track_idx}: {group}")
#                             drum_processed = True
#             else:
#                 # Handle instrument tracks
#                 if instrument_name not in videos:
#                     logging.info(f"Skipping track {track_idx}: {instrument_name} (no video)")
#                     continue
                    
#                 video_path = os.path.join(base_dir, videos[instrument_name])
#                 if not os.path.exists(video_path):
#                     logging.error(f"Video not found: {video_path}")
#                     continue
                
#                  # Initialize instrument cache if needed
#                 if instrument_name not in instrument_notes:
#                     instrument_notes[instrument_name] = {
#                         'video_path': video_path,
#                         'notes': {}
#                     }
                
#                 # Process unique notes for this instrument
#                 unique_notes = {int(float(note['midi'])) for note in track.get('notes', [])}
#                 for midi_note in unique_notes:
#                     if midi_note not in instrument_notes[instrument_name]['notes']:
#                         try:
#                             output_path = os.path.join(
#                                 processor.videos_dir,
#                                 f"{instrument_name}_notes",  # Group by instrument
#                                 f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
#                             )
                            
#                             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#                             processed_path = processor.create_tuned_video(
#                                 video_path,
#                                 midi_note,
#                                 output_path
#                             )
                            
#                             instrument_notes[instrument_name]['notes'][midi_note] = {
#                                 'path': processed_path,
#                                 'relative_path': os.path.relpath(processed_path, processor.videos_dir),
#                                 'note_name': midi_to_note(midi_note)
#                             }
#                         except Exception as e:
#                             logging.error(f"Failed to process note {midi_note}: {str(e)}")
#                             continue
        
#                 # Create track entry immediately after processing its notes
#                 track_dir = os.path.join(
#                     processor.videos_dir,
#                     f"track_{track_idx}_{instrument_name}"
#                 )
#                 os.makedirs(track_dir, exist_ok=True)
                
#                 track_output = {
#                     'track_idx': track_idx,
#                     'notes': {},
#                     'base_path': track_dir
#                 }
                
#                 # Link needed notes for this track
#                 for midi_note in unique_notes:
#                     if midi_note in instrument_notes[instrument_name]['notes']:
#                         source = instrument_notes[instrument_name]['notes'][midi_note]['path']
#                         dest = os.path.join(track_dir, os.path.basename(source))
                        
#                         if os.path.exists(dest):
#                             os.remove(dest)
#                         shutil.copy2(source, dest)
                        
#                         track_output['notes'][midi_note] = {
#                             'path': dest,
#                             'relative_path': os.path.relpath(dest, processor.videos_dir),
#                             'note_name': midi_to_note(midi_note)
#                         }
                
#                 processed_videos['tracks'][f"{instrument_name}_{track_idx}"] = track_output
#                 processed_videos['metadata']['valid_track_count'] += 1
#                 logging.info(f"Processed track {track_idx}: {instrument_name}")

#         if processed_videos['metadata']['valid_track_count'] == 0:
#             logging.error("No valid tracks were processed")
#             raise ValueError("No valid tracks were processed")

#         return processed_videos
                    
#     except Exception as e:
#         logging.error(f"Error processing track videos: {str(e)}")
#         raise

def process_track_videos(tracks, videos, processor, nvenc=True):
    """Process video tracks with provided processor instance, using multithreading."""
    processed_videos = {
        'tracks': {},
        'drum_tracks': {},
        'metadata': {
            'base_dir': processor.videos_dir,
            'session_id': processor.session_id,
            'valid_track_count': 0
        }
    }

    logging.info("Starting track processing...")

    if isinstance(tracks, list):
        tracks = {idx: track for idx, track in enumerate(tracks)}

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logging.info(f"Processing videos from: {list(videos.keys())}")

        if not tracks.get('tracks'):
            logging.error("No tracks found in input")
            raise ValueError("No tracks provided")

        if not videos:
            logging.error("No videos provided")
            raise ValueError("No videos provided")

        logging.info(f"Processing {len(tracks['tracks'])} tracks")
        logging.info(f"Available videos: {list(videos.keys())}")

        instrument_notes = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for track_idx, track in enumerate(tracks['tracks']):
                instrument = track.get('instrument', {})
                instrument_name = normalize_instrument_name(instrument.get('name', 'default'))

                if not track.get('notes'):
                    continue

                if is_drum_kit(instrument):
                    for group in get_drum_groups(track):
                        instrument_key = f"drum_{group}"
                        if instrument_key in videos:
                            video_path = videos[instrument_key]
                            if not os.path.exists(video_path):
                                logging.error(f"Drum video not found: {video_path}")
                                continue

                            output_dir = os.path.join(
                                processor.videos_dir,
                                f"track_{track_idx}_drums"
                            )
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, f"{group}.mp4")

                            if process_drum_track(video_path, output_path):
                                processed_videos['drum_tracks'][f"track_{track_idx}_{group}"] = {
                                    'track_idx': track_idx,
                                    'group': group,
                                    'path': output_path,
                                    'relative_path': os.path.relpath(output_path, processor.videos_dir)
                                }
                                logging.info(f"Processed drum track {track_idx}: {group}")

                else:
                    if instrument_name not in videos:
                        logging.info(f"Skipping track {track_idx}: {instrument_name} (no video)")
                        continue

                    video_path = os.path.join(base_dir, videos[instrument_name])
                    if not os.path.exists(video_path):
                        logging.error(f"Video not found: {video_path}")
                        continue

                    unique_notes = {int(float(note['midi'])) for note in track.get('notes', [])}
                    for midi_note in unique_notes:
                        if midi_note not in instrument_notes.get(instrument_name, {}).get('notes', {}):
                            instrument_notes.setdefault(instrument_name, {'notes': {}})
                            output_path = os.path.join(
                                processor.videos_dir,
                                f"{instrument_name}_notes",
                                f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                            )
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            future = executor.submit(
                                processor.create_tuned_video, video_path, midi_note, output_path, nvenc
                            )
                            futures.append((instrument_name, midi_note, future))

            for instrument_name, midi_note, future in futures:
                try:
                    processed_path = future.result()
                    instrument_notes[instrument_name]['notes'][midi_note] = {
                        'path': processed_path,
                        'relative_path': os.path.relpath(processed_path, processor.videos_dir),
                        'note_name': midi_to_note(midi_note)
                    }
                except Exception as e:
                    logging.error(f"Failed to process note {midi_note} for {instrument_name}: {str(e)}")

        for track_idx, track in enumerate(tracks['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))

            if is_drum_kit(instrument):
                continue

            if instrument_name not in videos or not track.get('notes'):
                continue

            track_dir = os.path.join(
                processor.videos_dir,
                f"track_{track_idx}_{instrument_name}"
            )
            os.makedirs(track_dir, exist_ok=True)

            track_output = {
                'track_idx': track_idx,
                'notes': {},
                'base_path': track_dir
            }

            unique_notes = {int(float(note['midi'])) for note in track.get('notes', [])}
            for midi_note in unique_notes:
                if instrument_name in instrument_notes and midi_note in instrument_notes[instrument_name]['notes']:
                    source = instrument_notes[instrument_name]['notes'][midi_note]['path']
                    dest = os.path.join(track_dir, os.path.basename(source))

                    if os.path.exists(dest):
                        os.remove(dest)
                    shutil.copy2(source, dest)

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

if __name__ == "__main__":


    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_path', help='Path to config JSON file')
        args = parser.parse_args()
        
        with open(args.config_path) as f:
            config = json.load(f)
            
        processor = AudioVideoProcessor()
        processor.setup_temp_directory()

        nvenc_available = False
        
        try:
            subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True, check=True)
            nvenc_available = True
            logging.info("NVENC hardware acceleration available")
        except subprocess.CalledProcessError:
            logging.info("NVENC hardware acceleration not available, falling back to CPU")
            pass
        
        result = process_track_videos(
            config.get('tracks', {}), 
            config.get('videos', {}),
            processor,
            nvenc_available
        )
        
        output_path = os.path.join(
            os.path.dirname(processor.videos_dir),
            f"final_composition_{processor.session_id}.mp4"
        )
        
        # composition_result = compose_from_processor_output(
        #     {
        #         'processed_videos_dir': processor.videos_dir,
        #         'tracks': config['tracks'],
        #         'processed_files': result
        #     },
        #     output_path
        # )
        
        # print(json.dumps({
        #     'success': True,
        #     'data': {
        #         'processed': result,
        #         'composition': composition_result
        #     }
        # }), flush=True)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }), flush=True)