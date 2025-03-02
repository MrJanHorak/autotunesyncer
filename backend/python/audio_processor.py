import librosa
import numpy as np
import os
import tempfile
import shutil
import logging
import uuid
import json
import subprocess
import argparse
from utils import normalize_instrument_name, midi_to_note
from drum_utils import is_drum_kit, get_drum_groups
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import OrderedDict
import threading
from processing_utils import ProgressTracker, encoder_queue, GPUManager
from video_utils import run_ffmpeg_command, encode_video, validate_video
from threading import RLock
import mmap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('video_processing.log', mode='w')  
    ]
)

gpu_manager = GPUManager()
class CacheManager:
    def __init__(self, max_size=1024*1024*1024):  # 1GB default
        self.lock = RLock()
        self.max_size = max_size
        self.current_size = 0
        self.cache = OrderedDict()
        self.last_access = {}
        self.mmap_cache = {}
        
    def add(self, key, data, size):
        with self.lock:
            if size > self.max_size * 0.25:  # Use mmap for large files
                self.mmap_cache[key] = self._create_mmap(data)
                return
                
            while self.current_size + size > self.max_size:
                self._evict_oldest()
                
            self.cache[key] = data
            self.current_size += size
        
    def get(self, key):
        if key in self.cache:
            self.last_access[key] = time.time()
            return self.cache[key]
        return None
        
    # def _evict_oldest(self):
    #     if not self.cache:
    #         return
    #     oldest = min(self.last_access.items(), key=lambda x: x[1])[0]
    #     del self.cache[oldest]
    #     del self.last_access[oldest]

    def _evict_oldest(self):
        if not self.cache:
            return
        _, oldest_data = self.cache.popitem(last=False)
        self.current_size -= len(oldest_data)



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
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        input_size = os.path.getsize(input_path)
        logging.info(f"Input file size: {input_size} bytes")
        
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
            '-progress', 'pipe:1',
            output_path
        ]
        
        logging.info(f"Running FFmpeg command: {' '.join(convert_cmd)}")
        run_ffmpeg_command(convert_cmd)
        validate_video(output_path)
        
        if os.path.exists(output_path):
            logging.info(f"Video conversion successful: {output_path}")
        else:
            raise FileNotFoundError(f"Output file not found: {output_path}")
        
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
        self.video_cache = {}
        self.audio_cache = {}
        self.processed_notes_cache = {}
        self.batch_size = 4
        self.mmap_cache = {}
    
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
    
    def _cache_base_audio(self, video_path):
        """Extract and cache base audio"""
        if video_path not in self.audio_cache or isinstance(self.audio_cache[video_path], dict):
            logging.info(f"Audio cache miss: {video_path}")
            temp_audio = os.path.join(self.temp_dir, f"cached_audio_{os.path.basename(video_path)}.wav")
            
            ffmpeg_extract = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                temp_audio
            ]
            result = encoder_queue.encode(ffmpeg_extract)
            if result.returncode == 0:
                # Store just the path initially
                self.audio_cache[video_path] = temp_audio
                
        else:
            logging.info(f"Audio cache hit: {video_path}")
        return (self.audio_cache[video_path]['audio_path'] 
                if isinstance(self.audio_cache[video_path], dict)
                else self.audio_cache[video_path])
        
    def _cache_base_video(self, video_path):
        """Pre-process and cache video without audio"""
        if video_path not in self.video_cache:
            temp_video = os.path.join(self.temp_dir, f"cached_video_{os.path.basename(video_path)}.mp4")
            
            ffmpeg_transcode = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'h264_nvenc',
                '-an',  # No audio
                '-pix_fmt', 'yuv420p',
                temp_video
            ]
            result = encoder_queue.encode(ffmpeg_transcode)
            if result.returncode == 0:
                self.video_cache[video_path] = temp_video
                
        return self.video_cache.get(video_path)
    
    # Modify the process_notes_batch function:
    def process_notes_batch(self, notes_batch, video_path):
        """Process multiple notes in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            for note in notes_batch:
                output_path = os.path.join(
                    self.videos_dir,
                    f"note_{note}_{midi_to_note(note)}.mp4"
                )
                if not os.path.exists(output_path):
                    futures.append(
                        executor.submit(
                            self.create_tuned_video,
                            video_path,
                            note,
                            output_path
                        )
                    )
            results.extend([f.result() for f in as_completed(futures)])
        return results

    # def process_notes_batch(self, notes_batch, video_path):
    #     """Process multiple notes in parallel"""
    #     results = []
    #     with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
    #         futures = []
    #         for note in notes_batch:
    #             output_path = os.path.join(
    #                 self.videos_dir,
    #                 f"note_{note}_{midi_to_note(note)}.mp4"
    #             )
    #             if not os.path.exists(output_path):
    #                 futures.append(
    #                     executor.submit(
    #                         self.create_tuned_video,
    #                         video_path,
    #                         note,
    #                         output_path
    #                     )
    #                 )
    #         results.extend([f.result() for f in as_completed(futures)])
    #     return results

    # def _cache_base_audio(self, video_path):
    #     """Extract and cache base audio"""
    #     if video_path not in self.audio_cache or isinstance(self.audio_cache[video_path], dict):
    #         temp_audio = os.path.join(self.temp_dir, f"cached_audio_{os.path.basename(video_path)}.wav")
            
    #         ffmpeg_extract = [
    #             'ffmpeg', '-y',
    #             '-i', video_path,
    #             '-vn',
    #             '-acodec', 'pcm_s16le',
    #             '-ar', '44100',
    #             '-ac', '1',
    #             temp_audio
    #         ]
    #         result = encoder_queue.encode(ffmpeg_extract)
    #         if result.returncode == 0:
    #             # Store just the path initially
    #             self.audio_cache[video_path] = temp_audio
                
    #     return (self.audio_cache[video_path]['audio_path'] 
    #             if isinstance(self.audio_cache[video_path], dict)
    #             else self.audio_cache[video_path])

    def _cache_base_video(self, video_path):
        """Pre-process and cache video without audio"""
        if video_path not in self.video_cache:
            logging.info(f"Video cache miss: {video_path}")
            temp_video = os.path.join(self.temp_dir, f"cached_video_{os.path.basename(video_path)}.mp4")
            
            ffmpeg_transcode = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'h264_nvenc',
                '-an',  # No audio
                '-pix_fmt', 'yuv420p',
                temp_video
            ]
            result = encoder_queue.encode(ffmpeg_transcode)
            if result.returncode == 0:
                self.video_cache[video_path] = temp_video
                
        else:
            logging.info(f"Video cache hit: {video_path}")
        return self.video_cache.get(video_path)
    
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
        
    # In audio_processor.py
    def create_tuned_video(self, video_path, target_note, output_path, nvenc=True):
        try:
            # Validate input video exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Input video not found: {video_path}")

            # Check if output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Get cached video (no audio needed anymore)
            cached_video = self._cache_base_video(video_path)
            if not cached_video or not os.path.exists(cached_video):
                raise Exception("Failed to cache video")

            # Add pitch analysis
            current_pitch = self.analyze_pitch(video_path)  # Analyze directly from video
            target_note = int(target_note)  # Ensure target note is integer
            pitch_shift = target_note - current_pitch  # Calculate pitch shift

            logging.info(f"Pitch analysis - Current: {current_pitch:.1f}, Target: {target_note}, Shift: {pitch_shift:.1f}")

            # Skip processing if pitch shift is minimal
            if abs(pitch_shift) < 0.1:
                logging.info("Pitch shift too small, copying original video")
                shutil.copy2(video_path, output_path)
                return output_path

            # Create final video with rubberband filter
            ffmpeg_combine = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_device', '0',  # Specify GPU device
                '-i', cached_video,
                '-af', f"rubberband=pitch={pitch_shift:.3f}",  # Apply rubberband filter
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-gpu', '0',  # Specify GPU device
                '-b:v', '5M',
                '-maxrate', '8M',
                '-bufsize', '10M',
                '-tune', 'hq',
                '-rc', 'vbr',
                output_path
            ]
            result = encoder_queue.encode(ffmpeg_combine)
            if result.returncode != 0:
                raise Exception(f"FFmpeg combine failed: {result.stderr}")

            return output_path

        except Exception as e:
            logging.error(f"Error creating tuned video for note {target_note}: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
                raise
    

    # def create_tuned_video(self, video_path, target_note, output_path, nvenc=True):
    #     try:
    #         # Validate input video exists
    #         if not os.path.exists(video_path):
    #             raise FileNotFoundError(f"Input video not found: {video_path}")
                
    #         # Check if output directory exists
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #         # Get cached video and audio with validation
    #         cached_video = self._cache_base_video(video_path)
    #         if not cached_video or not os.path.exists(cached_video):
    #             raise Exception("Failed to cache video")
                
    #         cached_audio = self._cache_base_audio(video_path)
    #         if not cached_audio or not os.path.exists(cached_audio):
    #             raise Exception("Failed to cache audio")

    #         # Add pitch analysis
    #         current_pitch = self.analyze_pitch(cached_audio)
    #         target_note = int(target_note)  # Ensure target note is integer
    #         pitch_shift = target_note - current_pitch  # Calculate pitch shift

    #         logging.info(f"Pitch analysis - Current: {current_pitch:.1f}, Target: {target_note}, Shift: {pitch_shift:.1f}")

    #         # Skip processing if pitch shift is minimal
    #         if abs(pitch_shift) < 0.1:
    #             logging.info("Pitch shift too small, copying original video")
    #             shutil.copy2(video_path, output_path)
    #             return output_path

    #         # Process pitch shift
    #         tuned_audio = os.path.join(self.temp_dir, f"tuned_{target_note}_{os.path.basename(video_path)}.wav")
    #         rubberband_cmd = [
    #             'rubberband',
    #             '-p', f"{pitch_shift:.3f}",
    #             '-t', '1.0',
    #             '-F',
    #             '-c', '4',
    #             cached_audio,
    #             tuned_audio
    #         ]
    #         result = subprocess.run(rubberband_cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             raise Exception(f"Rubberband failed: {result.stderr}")

    #         # Create final video with proper GPU parameters
    #         ffmpeg_combine = [
    #             'ffmpeg', '-y',
    #             '-hwaccel', 'cuda',
    #             '-i', cached_video,
    #             '-i', tuned_audio,
    #             '-c:v', 'h264_nvenc',
    #             '-preset', 'p4',
    #             '-b:v', '5M',
    #             '-maxrate', '8M',
    #             '-bufsize', '10M',
    #             '-tune', 'hq',
    #             '-rc', 'vbr',
    #             output_path
    #         ]
    #         result = encoder_queue.encode(ffmpeg_combine)
    #         if result.returncode != 0:
    #             raise Exception(f"FFmpeg combine failed: {result.stderr}")
                
    #         return output_path

    #     except Exception as e:
    #         logging.error(f"Error creating tuned video for note {target_note}: {str(e)}")
    #         if os.path.exists(output_path):
    #             try:
    #                 os.remove(output_path)
    #             except:
    #                 pass
    #         raise

def process_drum_track(video_path, output_path):
    """Process and validate drum track video"""
    try:
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
        
        run_ffmpeg_command(convert_cmd)
        validate_video(temp_output)
        
        os.rename(temp_output, output_path)
        logging.info(f"Drum track processed successfully: {output_path}")
        
    except Exception as e:
        logging.error(f"Error processing drum track: {str(e)}")
        raise

# def process_drum_track(video_path, output_path):
#     """Process and validate drum track video"""
#     try:
#         # Convert to consistent format first
#         temp_output = output_path + '.temp.mp4'
#         convert_cmd = [
#             'ffmpeg', '-y',
#             '-i', video_path,
#             '-c:v', 'h264_nvenc',
#             '-preset', 'fast',
#             '-crf', '23',
#             '-c:a', 'aac',
#             '-strict', 'experimental',
#             '-b:a', '192k',
#             '-pix_fmt', 'yuv420p',
#             temp_output
#         ]
        
#         # subprocess.run(convert_cmd, check=True)
#         result = encoder_queue.encode(convert_cmd)
#         if result.returncode != 0:
#             raise subprocess.CalledProcessError(result.returncode, convert_cmd)
        
#         validate_cmd = [
#             'ffmpeg',
#             '-v', 'error',
#             '-i', temp_output,
#             '-f', 'null',
#             '-'
#         ]
        
#         # subprocess.run(validate_cmd, check=True)
#         result = encoder_queue.encode(validate_cmd)
#         if result.returncode != 0:
#             raise subprocess.CalledProcessError(result.returncode, validate_cmd)
        
#         shutil.move(temp_output, output_path)
#         return True
        
#     except Exception as e:
#         logging.error(f"Failed to process drum track: {str(e)}")
#         if os.path.exists(temp_output):
#             os.remove(temp_output)
#         return False


    
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
        # Process instruments in parallel
        instrument_notes = {}
        track_note_map = {}
        
        # First collect all notes needed
        for track_idx, track in enumerate(tracks['tracks']):
            instrument = track.get('instrument', {})
            instrument_name = normalize_instrument_name(instrument.get('name', 'default'))
            
            if not track.get('notes') or is_drum_kit(track):
                continue
                
            track_id = f"track_{track_idx}_{instrument_name}"
            track_note_map[track_id] = {
                'instrument': instrument_name,
                'notes': {int(float(note['midi'])) for note in track['notes']}
            }
            
            if instrument_name not in instrument_notes:
                instrument_notes[instrument_name] = set()
            instrument_notes[instrument_name].update(track_note_map[track_id]['notes'])

        # Process instruments with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for instrument_name, notes in instrument_notes.items():
                if instrument_name in videos:
                    video_path = videos[instrument_name]
                    notes_dir = os.path.join(processor.videos_dir, f"{instrument_name}_notes")
                    os.makedirs(notes_dir, exist_ok=True)
                    
                    # Pre-cache video and audio
                    processor._cache_base_video(video_path)
                    processor._cache_base_audio(video_path)
                    
                    for midi_note in notes:
                        output_path = os.path.join(
                            notes_dir,
                            f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                        )
                        if not os.path.exists(output_path):
                            futures.append(
                                executor.submit(
                                    processor.create_tuned_video,
                                    video_path,
                                    midi_note,
                                    output_path
                                )
                            )
            total_notes = sum(len(notes) for notes in instrument_notes.values())
            progress = ProgressTracker(total_notes)
                            
            # Wait for all note processing to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    progress.update(result is not None)
                except Exception as e:
                    progress.update(False)
                    logging.error(f"Note processing failed: {str(e)}")
                finally:
                    progress.close()

        # Create track entries for instruments
        for track_id, track_info in track_note_map.items():
            instrument_name = track_info['instrument']
            track_dir = os.path.join(processor.videos_dir, track_id)
            os.makedirs(track_dir, exist_ok=True)
            
            track_idx = int(track_id.split('_')[1])
            processed_videos['tracks'][track_id] = {
                'track_idx': track_idx,
                'notes_dir': track_dir,
                'notes': []
            }
            
            for midi_note in track_info['notes']:
                source = os.path.join(
                    processor.videos_dir,
                    f"{instrument_name}_notes",
                    f"note_{midi_note}_{midi_to_note(midi_note)}.mp4"
                )
                if os.path.exists(source):
                    processed_videos['tracks'][track_id]['notes'].append({
                        'midi': midi_note,
                        'note_name': midi_to_note(midi_note),
                        'source': source,
                        'relative_path': os.path.relpath(source, processor.videos_dir)
                    })
            
            processed_videos['metadata']['valid_track_count'] += 1
            logging.info(f"Processed track directory: {track_id}")

        # Process drum tracks
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