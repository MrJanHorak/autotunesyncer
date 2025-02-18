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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('video_processing.log', mode='w')  
    ]
)

class EncoderQueue:
    def __init__(self, max_concurrent=2):
        self.queue = Queue()
        self.semaphore = threading.Semaphore(max_concurrent)
        
    def encode(self, ffmpeg_command):
        with self.semaphore:
            logging.info(f"EncoderQueue: Running command: {' '.join(ffmpeg_command)}")
            try:
                result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"EncoderQueue: Command failed: {result.stderr}")
                return result
            except Exception as e:
                logging.error(f"EncoderQueue: Error executing command: {str(e)}")
                raise

encoder_queue = EncoderQueue(max_concurrent=2) 

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

def convert_video_format(input_path, output_path):
    """Convert video to MP4 format with proper timestamps"""
    try:
        logging.info(f"Starting video conversion: {input_path} -> {output_path}")

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
        # result = subprocess.run(convert_cmd, capture_output=True, text=True)
        result = encoder_queue.encode(convert_cmd)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg stderr: {result.stderr}")
            logging.error(f"FFmpeg stdout: {result.stdout}")
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")

        validate_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', output_path,
            '-f', 'null',
            '-'
        ]
        # validate_result = subprocess.run(validate_cmd, capture_output=True, text=True)
        validate_result = encoder_queue.encode(validate_cmd)

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
    
    def create_tuned_video(self, video_path, target_note, output_path, nvenc=True):
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

            # Check if NVENC is available
            try:
                nvenc_check = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-encoders'],
                    capture_output=True,
                    text=True
                )
                nvenc_available = 'h264_nvenc' in nvenc_check.stdout
            except:
                nvenc_available = False
            
            # Choose encoder based on availability
            video_codec = 'h264_nvenc' if nvenc_available and nvenc else 'libx264'
            encoder_params = []
            
            if video_codec == 'h264_nvenc':
                encoder_params.extend([
                    '-rc', 'vbr',              # Variable bitrate mode
                    '-rc-lookahead', '32',     # Look-ahead frames
                    '-gpu', '0',               # Use first GPU
                    '-tune', 'hq',             # High quality tune
                    '-profile:v', 'high'       # High profile
                ])
            else:
                encoder_params.extend([
                    '-preset', 'medium',        # Balance speed/quality
                    '-profile:v', 'high',      # High profile
                    '-tune', 'film'            # Film tune for general content
                ])

            # First transcode video
            ffmpeg_transcode = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', video_codec,
                *encoder_params,
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-an',  # No audio
                temp_video
            ]
            
            logging.info(f"Transcoding video using {video_codec}: {' '.join(ffmpeg_transcode)}")
            # result = subprocess.run(ffmpeg_transcode, capture_output=True, text=True)
            result = encoder_queue.encode(ffmpeg_transcode)
            
            if result.returncode != 0:
                if video_codec == 'h264_nvenc':
                    logging.warning("NVENC failed, falling back to CPU encoding")
                    return self.create_tuned_video(video_path, target_note, output_path, nvenc=False)
                else:
                    raise Exception(f"FFmpeg transcode failed: {result.stderr}")

            # Extract audio
            ffmpeg_extract = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                temp_audio
            ]
            
            logging.info(f"Extracting audio: {' '.join(ffmpeg_extract)}")
            # result = subprocess.run(ffmpeg_extract, capture_output=True, text=True)
            result = encoder_queue.encode(ffmpeg_extract)
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
                    '-c:v', video_codec,
                    *encoder_params,
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-an',  # No audio
                    output_path
                ]
                subprocess.run(ffmpeg_transcode, check=True)
                
                # Validate output
                if not self.validate_output_video(output_path):
                    raise Exception("Output validation failed")
                    
                return output_path
            
            rubberband_cmd = [
                'rubberband',
                '-p', f"{pitch_shift:.3f}",  
                '-t', '1.0',                  
                '-F',                         
                '-c', '4',                    
                '-2',                        
                temp_audio,
                tuned_audio
            ]

            logging.info(f"Pitch shifting with rubberband: {' '.join(rubberband_cmd)}")
            result = subprocess.run(rubberband_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Rubberband failed: {result.stderr}")

            ffmpeg_combine = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', tuned_audio,
                '-c:v', video_codec,  
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
            # result = subprocess.run(ffmpeg_combine, capture_output=True, text=True)
            result = encoder_queue.encode(ffmpeg_combine)
            if result.returncode != 0:
                raise Exception(f"FFmpeg combine failed: {result.stderr}")

            # Validate output
            if not self.validate_output_video(output_path):
                raise Exception("Output validation failed")
                
            return output_path

        except Exception as e:
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
        
        # subprocess.run(convert_cmd, check=True)
        result = encoder_queue.encode(convert_cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, convert_cmd)
        
        validate_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', temp_output,
            '-f', 'null',
            '-'
        ]
        
        # subprocess.run(validate_cmd, check=True)
        result = encoder_queue.encode(validate_cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, validate_cmd)
        
        shutil.move(temp_output, output_path)
        return True
        
    except Exception as e:
        logging.error(f"Failed to process drum track: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False
    
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

    logging.info("Starting track processing...")

    try:
        instrument_notes = {}
        track_note_map = {}  # Track which notes are needed by each track
        
        for track_idx, track in enumerate(tracks['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            if not track.get('notes') or is_drum_kit(track):
                continue
                
            # Map which notes this track needs
            track_id = f"track_{track_idx}_{instrument_name}"
            track_note_map[track_id] = {
                'instrument': instrument_name,
                'notes': {int(float(note['midi'])) for note in track['notes']}
            }
            
            # Collect all unique notes needed for this instrument
            if instrument_name not in instrument_notes:
                instrument_notes[instrument_name] = set()
            instrument_notes[instrument_name].update(track_note_map[track_id]['notes'])

        # Process all needed notes for each instrument
        for instrument_name, notes in instrument_notes.items():
            if instrument_name in videos:
                video_path = videos[instrument_name]
                notes_dir = os.path.join(processor.videos_dir, f"{instrument_name}_notes")
                os.makedirs(notes_dir, exist_ok=True)
                
                # Process each unique note for this instrument
                for midi_note in notes:
                    output_path = os.path.join(
                        notes_dir,
                        f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                    )
                    if not os.path.exists(output_path):
                        processor.create_tuned_video(video_path, midi_note, output_path)

        # Create track-specific directories and link to needed notes
        for track_id, track_info in track_note_map.items():
            instrument_name = track_info['instrument']
            track_dir = os.path.join(processor.videos_dir, track_id)
            os.makedirs(track_dir, exist_ok=True)
            
            # Create track entry
            track_idx = int(track_id.split('_')[1])  # Extract track index
            processed_videos['tracks'][track_id] = {
                'track_idx': track_idx,
                'notes_dir': track_dir,
                'notes': []
            }
            
            # Link only the notes needed by this track
            for midi_note in track_info['notes']:
                source = os.path.join(
                    processor.videos_dir,
                    f"{instrument_name}_notes",
                    f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                )
                if os.path.exists(source):
                    # Add note info to track data
                    processed_videos['tracks'][track_id]['notes'].append({
                        'midi': midi_note,
                        'note_name': midi_to_note(midi_note),
                        'source': source,
                        'relative_path': os.path.relpath(source, processor.videos_dir)
                    })
            
            processed_videos['metadata']['valid_track_count'] += 1
            logging.info(f"Processed track directory: {track_id}")

        # Handle drum tracks separately
        for track_idx, track in enumerate(tracks['tracks']):
            if is_drum_kit(track):
                for group in get_drum_groups(track):
                    drum_key = f"drum_{group}"
                    if drum_key in videos:
                        output_dir = os.path.join(processor.videos_dir, f"track_{track_idx}_drums")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"{group}.mp4")
                        
                        shutil.copy2(videos[drum_key], output_path)
                        processed_videos['drum_tracks'][f"track_{track_idx}_{group}"] = {
                            'track_idx': track_idx,
                            'group': group,
                            'path': output_path
                        }
                        logging.info(f"Processed drum track {track_idx}: {group}")

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
               
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }), flush=True)