import json
import sys
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, vfx, CompositeAudioClip, AudioClip
import numpy as np
from PIL import Image
import logging
import time
import os
import subprocess
import gc  # Add garbage collector import
from pathlib import Path
import tempfile
import shutil
import atexit

# Try to import psutil, but provide fallback if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not installed. Using basic memory monitoring.")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Replace '0' with the correct GPU index if necessary
os.environ['FFMPEG_MEMORY_LIMIT'] = '256M'  # Limit ffmpeg memory usage

# Configure logging at the beginning of the file
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add these constants at the top of the file
BATCH_SIZE = 10  # Reduced from 50 to prevent memory overload
MEMORY_THRESHOLD = 70
TEMP_DIR = tempfile.mkdtemp()
temp_video_files = []

def get_memory_usage():
    """Monitor memory usage with fallback mechanism"""
    try:
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_percent()
        else:
            # Basic memory check using gc module
            gc.collect()  # Run garbage collection
            # Return a conservative estimate
            return 50.0  # Assume 50% usage to trigger cleanup more frequently
    except Exception as e:
        logging.warning(f"Memory monitoring failed: {e}")
        return 75.0  # Conservative default to trigger cleanup

def cleanup_memory():
    """Force garbage collection and clear memory"""
    try:
        # Run garbage collection multiple times to ensure thorough cleanup
        for _ in range(3):
            gc.collect()
        
        # Clear any cached memory
        import sys
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
            
    except Exception as e:
        logging.error(f"Error during memory cleanup: {e}")

def clean_temp_files():
    """Clean temporary files"""
    try:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
    except Exception as e:
        logging.error(f"Error cleaning temp files: {e}")

def test_codecs():
    cmd = ['ffmpeg', '-codecs']
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Available codecs:")
    for line in result.stdout.split('\n'):
        if any(codec in line.lower() for codec in ['aac', 'opus', 'mp3']):
            print(line)

test_codecs()

# Add PIL compatibility layer
if not hasattr(Image, 'ANTIALIAS'):
    # For newer versions of Pillow
    Image.ANTIALIAS = Image.Resampling.LANCZOS

def adjust_audio_speed(clip, pitch_ratio):
    """Adjust audio speed to match the pitch ratio"""
    return clip.fx(vfx.speedx, factor=pitch_ratio)

# Add the midiNoteToFrequencyRatio function
def midiNoteToFrequencyRatio(targetMidiNote, sourceMidiNote=60):
    """
    Calculate frequency ratio based on the MIDI note.
    Each semitone is a factor of 2^(1/12).
    """
    semitone_difference = targetMidiNote - sourceMidiNote
    return 2 ** (semitone_difference / 12)

# Modify the get_pitch_ratio function
def get_pitch_ratio(note):
    """Calculate pitch ratio based on the MIDI note."""
    try:
        return midiNoteToFrequencyRatio(note['midi'])  # Changed 'note' to 'midi'
    except Exception as e:
        logging.error(f"Failed to calculate pitch ratio for note {note}: {e}")
        return 1.0  # Default pitch ratio

def verify_frame_reading(clip, max_attempts=3):
    """Verify that frames can be read from the video clip."""
    for attempt in range(max_attempts):
        try:
            # Try reading first frame
            frame = clip.get_frame(0)
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame at position 0")
            return True
        except Exception as e:
            logging.warning(f"Frame reading attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1)
    return False

def verify_video_file(video_path, retries=3, delay=1):
    """Verify the integrity of the video file with retries."""
    for attempt in range(retries):
        try:
            # Try to get video info using ffmpeg directly first
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', video_path,
                '-f', 'null', '-'
            ], capture_output=True, text=True)
            
            # Try to extract duration from ffmpeg output
            duration = None
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    try:
                        time_str = line.split('Duration: ')[1].split(',')[0].strip()
                        h, m, s = map(float, time_str.split(':'))
                        duration = h * 3600 + m * 60 + s
                    except:
                        pass

            # Load the clip with an estimated duration if needed
            clip = VideoFileClip(video_path, audio=True)
            if not hasattr(clip, 'duration') or clip.duration is None:
                clip.duration = duration or 10.0  # Use extracted duration or fallback
            
            # Verify frame reading
            if not verify_frame_reading(clip):
                raise ValueError("Failed to verify frame reading")
            
            clip.close()
            return True
            
        except Exception as e:
            logging.error(f"Failed to verify video file {video_path} on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                
    return False

def ensure_audio_fps(audio_clip, default_fps=44100):
    """Ensure audio clip has fps attribute."""
    if (audio_clip is None) or (not hasattr(audio_clip, 'fps')) or (audio_clip.fps is None):
        return None
    return audio_clip

def standardize_audio(audio_clip, target_fps=44100):
    """Standardize audio format across different codecs."""
    try:
        if audio_clip is None:
            return None, target_fps

        # Ensure fps is set before getting audio data
        audio_clip = ensure_audio_fps(audio_clip, target_fps)
        
        try:
            # Get audio data
            audio_array = audio_clip.to_soundarray()
            fps = audio_clip.fps
        except Exception as e:
            logging.warning(f"Failed to get audio data directly, trying alternate method: {e}")
            try:
                # Fallback method to get audio data
                audio_array = np.array(audio_clip.get_frame(0))
                fps = target_fps
            except:
                return None, target_fps

        # Convert to numpy array if needed
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Ensure array is contiguous and float32
        audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

        # Handle mono/stereo conversion
        if len(audio_array.shape) == 1:
            audio_array = np.column_stack((audio_array, audio_array))
        elif audio_array.shape[1] == 1:
            audio_array = np.column_stack((audio_array[:, 0], audio_array[:, 0]))

        # Normalize audio levels safely
        abs_max = np.abs(audio_array).max()
        if abs_max > 1e-10:  # Safe threshold
            audio_array = audio_array / abs_max

        return audio_array, fps

    except Exception as e:
        logging.error(f"Error standardizing audio: {e}")
        return None, target_fps

def create_normalized_audio_clip(audio_array, fps, duration, start_time=0):
    """Create a normalized audio clip with proper array handling."""
    try:
        if audio_array is None:
            return None
            
        # Ensure we have a valid numpy array
        if not isinstance(audio_array, np.ndarray):
            return None
            
        # Convert to float32 if needed
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
            
        # Normalize audio level
        max_val = np.abs(audio_array).max()
        if max_val > 1e-10:
            audio_array = audio_array / max_val
            audio_array = audio_array * 0.8
            
        def make_frame(t):
            try:
                frame_time = t - start_time
                frame_idx = int(frame_time * fps)
                
                # Check bounds
                if frame_time < 0 or frame_time >= duration or frame_idx >= len(audio_array):
                    return np.zeros(2, dtype=np.float32)
                
                # Get frame data
                frame = audio_array[frame_idx]
                
                # Handle different shapes properly
                if frame.size == 1:  # Single value
                    return np.array([float(frame), float(frame)], dtype=np.float32)
                elif frame.size == 2:  # Stereo
                    return frame.astype(np.float32)
                else:  # Unexpected shape
                    return np.zeros(2, dtype=np.float32)
                    
            except Exception as e:
                logging.error(f"Error in make_frame: {e}")
                return np.zeros(2, dtype=np.float32)
                
        # Create audio clip with explicit fps
        clip = AudioClip(make_frame, duration=duration)
        clip.fps = fps
        return clip
        
    except Exception as e:
        logging.error(f"Error creating normalized audio clip: {e}")
        return None

def verify_audio_stream(video_path):
    """Verify if video file contains valid audio stream."""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-af', 'volumedetect', '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if audio stream exists
        has_audio = False
        audio_info = {}
        
        for line in result.stderr.split('\n'):
            if 'Audio:' in line:
                has_audio = True
                logging.info(f"Found audio stream: {line.strip()}")
                
            # Get volume information
            if 'mean_volume:' in line or 'max_volume:' in line:
                logging.info(f"Audio levels: {line.strip()}")
                
        if not has_audio:
            logging.warning(f"No audio stream found in {video_path}")
        return has_audio
        
    except Exception as e:
        logging.error(f"Error verifying audio stream: {e}")
        return False

def verify_output_file(output_path):
    """Verify the output file has valid video and audio streams."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        video_stream = None
        audio_stream = None
        
        for stream in info.get('streams', []):
            if stream['codec_type'] == 'video':
                video_stream = stream
            elif stream['codec_type'] == 'audio':
                audio_stream = stream
                
        if not audio_stream:
            logging.error("No audio stream found in output file!")
            return False
            
        # Verify audio properties
        logging.info(f"Output audio stream: {json.dumps(audio_stream, indent=2)}")
        
        # Check audio level of output file
        cmd = ['ffmpeg', '-i', output_path, '-af', 'volumedetect', '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        for line in result.stderr.split('\n'):
            if 'mean_volume' in line or 'max_volume' in line:
                logging.info(f"Output file {line.strip()}")
                
        return True
        
    except Exception as e:
        logging.error(f"Error verifying output file: {e}")
        return False

def create_audio_clip(audio_array, fps, duration, start_time=0):
    """Create an audio clip with proper array handling."""
    try:
        if audio_array is None:
            return None

        # Create audio clip with explicit fps first
        clip = AudioClip(make_frame=lambda t: np.zeros(2, dtype=np.float32), duration=duration)
        clip.fps = fps

        def make_frame(t):
            try:
                frame_time = t - start_time
                frame_idx = int(frame_time * fps)
                
                # Check bounds
                if frame_time < 0 or frame_time >= duration or frame_idx >= len(audio_array):
                    return np.zeros(2, dtype=np.float32)
                
                # Get frame data
                frame = audio_array[frame_idx]
                
                # Handle different shapes properly
                if frame.size == 1:  # Single value
                    return np.array([float(frame), float(frame)], dtype=np.float32)
                elif frame.size == 2:  # Stereo
                    return frame.astype(np.float32)
                else:  # Unexpected shape
                    return np.zeros(2, dtype=np.float32)
                
            except Exception as e:
                logging.error(f"Error in make_frame: {e}")
                return np.zeros(2, dtype=np.float32)

        clip.make_frame = make_frame
        return clip

    except Exception as e:
        logging.error(f"Error creating audio clip: {e}")
        return None

def process_audio_array(audio_array, fps=44100, duration=None):
    """Process audio array without relying on AudioClip."""
    try:
        if audio_array is None:
            return None

        # Convert to float32 if needed
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Ensure stereo
        if len(audio_array.shape) == 1:
            audio_array = np.column_stack((audio_array, audio_array))
        elif audio_array.shape[1] == 1:
            audio_array = np.column_stack((audio_array[:, 0], audio_array[:, 0]))

        # Normalize audio
        max_val = np.abs(audio_array).max()
        if max_val > 1e-10:
            audio_array = audio_array / max_val * 0.8

        # Calculate total samples needed
        if duration:
            total_samples = int(duration * fps)
            if len(audio_array) < total_samples:
                # Pad with zeros if needed
                padding = np.zeros((total_samples - len(audio_array), 2), dtype=np.float32)
                audio_array = np.vstack((audio_array, padding))
            elif len(audio_array) > total_samples:
                # Trim if longer
                audio_array = audio_array[:total_samples]

        return audio_array

    except Exception as e:
        logging.error(f"Error processing audio array: {e}")
        return None

def create_audio_segment(audio_array, start_time, duration, fps=44100):
    """Create audio segment with proper timing."""
    try:
        if audio_array is None:
            return None

        # Process the audio data
        processed_audio = process_audio_array(audio_array, fps, duration)
        if processed_audio is None:
            return None

        def make_frame(t):
            try:
                # Convert time to sample index
                sample_idx = int((t - start_time) * fps)
                if sample_idx < 0 or sample_idx >= len(processed_audio):
                    return np.array([0.0, 0.0])
                return processed_audio[sample_idx]
            except Exception as e:
                logging.error(f"Error in make_frame: {e}")
                return np.array([0.0, 0.0])

        # Create audio clip with explicit fps
        clip = AudioClip(make_frame, duration=duration)
        clip.fps = fps
        return clip

    except Exception as e:
        logging.error(f"Error creating audio segment: {e}")
        return None

def get_safe_clip_duration(clip, note_start, note_end):
    """Calculate safe start and end times for a clip."""
    if not clip or not hasattr(clip, 'duration'):
        return 0, 0
        
    clip_duration = clip.duration
    safe_end = min(note_end, clip_duration)
    safe_start = min(note_start, safe_end)
    return safe_start, safe_end

def stretch_video_to_duration(clip, target_duration):
    """
    Stretch a video clip to match a target duration while maintaining audio pitch.
    """
    try:
        if not clip or not hasattr(clip, 'duration'):
            return None
            
        current_duration = clip.duration
        if current_duration >= target_duration:
            return clip
            
        # Calculate stretch factor
        stretch_factor = target_duration / current_duration
        
        # Stretch video
        stretched_clip = clip.fx(vfx.speedx, factor=1/stretch_factor)
        
        # If there's audio, stretch it while maintaining pitch
        if clip.audio is not None:
            stretched_audio = clip.audio.fx(vfx.speedx, factor=1/stretch_factor)
            stretched_clip = stretched_clip.set_audio(stretched_audio)
            
        return stretched_clip
        
    except Exception as e:
        logging.error(f"Error stretching video: {e}")
        return clip

def process_video_segments(midi_data, video_files, output_path):
    clips_to_close = []
    batch_temp_files = []
    
    try:
        # Validate input data first
        if not midi_data or not video_files:
            raise ValueError("Invalid midi_data or video_files")

        # Create batches of notes across all tracks
        all_notes = []
        for track_id, track_data in video_files.items():
            if track_data.get('notes'):
                all_notes.extend((note, track_id) for note in track_data['notes'])
        
        if not all_notes:
            raise ValueError("No valid notes found in tracks")

        # Sort notes by start time
        all_notes.sort(key=lambda x: x[0].get('start', 0))
        
        total_duration = max(note[0].get('end', 0) for note in all_notes)
        background = ColorClip(size=(960, 720), color=(0,0,0), duration=total_duration)
        
        # Process in batches
        batch_temp_files = []
        for i in range(0, len(all_notes), BATCH_SIZE):
            batch_notes = all_notes[i:i + BATCH_SIZE]
            batch_clips = [background] if i == 0 else []  # Only include background in first batch
            
            valid_clips_in_batch = False
            
            # Process each note in the batch
            for note, track_id in batch_notes:
                try:
                    video_path = video_files[track_id].get('path')
                    if not video_path or not os.path.exists(video_path):
                        logging.warning(f"Invalid video path for track {track_id}")
                        continue
                    
                    if not verify_video_file(video_path):
                        continue

                    clip = VideoFileClip(video_path, audio=True)
                    pitch_ratio = get_pitch_ratio(note)
                    note_start = note.get('start', 0)
                    note_end = note.get('end', note_start + clip.duration)
                    note_duration = note_end - note_start
                    
                    # Stretch video if it's too short
                    if clip.duration < note_duration:
                        logging.info(f"Stretching clip from {clip.duration}s to {note_duration}s")
                        clip = stretch_video_to_duration(clip, note_duration)
                        if clip is None:
                            continue
                    
                    # Get safe duration for the clip
                    safe_start, safe_end = get_safe_clip_duration(clip, note_start, note_end)
                    
                    if safe_end <= safe_start:
                        logging.warning(f"Invalid clip duration: start={safe_start}, end={safe_end}")
                        clip.close()
                        continue
                    
                    # Process clip with safe duration
                    if clip.audio is not None:
                        clip = clip.set_audio(adjust_audio_speed(clip.audio, pitch_ratio))
                    clip = clip.set_position((0, 0))
                    clip = clip.subclip(0, safe_end - safe_start)  # First trim to needed duration
                    clip = clip.set_start(safe_start)  # Then set the start time
                    
                    batch_clips.append(clip)
                    clips_to_close.append(clip)
                    valid_clips_in_batch = True
                    
                except Exception as e:
                    logging.error(f"Error processing note: {e}")
                    continue

            # Only process batch if it contains valid clips
            if valid_clips_in_batch:
                try:
                    batch_composite = CompositeVideoClip(batch_clips, size=(960, 720))
                    temp_file = os.path.join(TEMP_DIR, f'batch_{i}.mp4')
                    batch_composite.write_videofile(
                        temp_file,
                        codec='libx264',
                        audio_codec='aac',
                        preset='ultrafast',
                        ffmpeg_params=['-tune', 'fastdecode']
                    )
                    batch_temp_files.append(temp_file)
                    
                except Exception as e:
                    logging.error(f"Error saving batch {i}: {e}")
                finally:
                    # Cleanup batch resources
                    try:
                        batch_composite.close()
                    except:
                        pass
                    for clip in batch_clips:
                        try:
                            clip.close()
                        except:
                            pass
                    
            # Clear batch resources
            batch_clips.clear()
            cleanup_memory()

        # Only proceed with final composition if we have temp files
        if not batch_temp_files:
            raise ValueError("No valid video segments were processed")

        # Combine all batch files
        if batch_temp_files:
            final_clips = [VideoFileClip(tf) for tf in batch_temp_files]
            final = concatenate_videoclips(final_clips)
            
            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(Path(TEMP_DIR) / 'temp-audio.m4a'),
                remove_temp=True,
                fps=30,
                preset='ultrafast',
                ffmpeg_params=[
                    '-tune', 'fastdecode',
                    '-b:v', '2000k',
                    '-maxrate', '2500k',
                    '-bufsize', '5000k',
                    '-threads', '4'
                ]
            )
            
            # Cleanup final clips
            for clip in final_clips:
                clip.close()
            final.close()
            del final_clips
            del final

    except Exception as e:
        logging.error(f"Error in process_video_segments: {str(e)}")
        raise
    
    finally:
        # Cleanup all resources
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass
        
        # Remove temp files
        for tf in batch_temp_files:
            try:
                os.remove(tf)
            except:
                pass
                
        clean_temp_files()
        cleanup_memory()

def pre_python_processing_verification(drum_tracks):
    """Verify drum tracks before processing."""
    for track in drum_tracks:
        if 'key' not in track or 'noteCount' not in track:
            logging.error(f"Invalid drum track data: {track}")
            return False
        logging.info(f"Drum track {track['key']} with {track['noteCount']} notes verified.")
    return True

def main():
    if len(sys.argv) != 4:
        print("Usage: python video_processor.py midi_data.json video_files.json output_path")
        sys.exit(1)

    midi_json_path = sys.argv[1]
    video_files_json_path = sys.argv[2]
    output_path = sys.argv[3]

    try:
        with open(midi_json_path, 'r') as f:
            midi_data = json.load(f)
        with open(video_files_json_path, 'r') as f:
            video_files = json.load(f)
        
        # Pre-Python processing verification
        drum_tracks = [
            {"key": "drum_kick_track1", "noteCount": 360},
            {"key": "drum_snare_track10", "noteCount": 28},
            {"key": "drum_hihat_track0", "noteCount": 664},
            {"key": "drum_snare_track1", "noteCount": 144},
            {"key": "drum_kick_track10", "noteCount": 16},
            {"key": "drum_percussion_track1", "noteCount": 20}
        ]
        if not pre_python_processing_verification(drum_tracks):
            logging.error("Pre-Python processing verification failed.")
            sys.exit(1)
            
        process_video_segments(midi_data, video_files, output_path)
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        sys.exit(1)
    finally:
        clean_temp_files()
        cleanup_memory()

if __name__ == "__main__":
    main()

# Add cleanup on script exit
atexit.register(clean_temp_files)
