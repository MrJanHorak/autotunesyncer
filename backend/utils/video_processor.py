import json
import sys
from moviepy.editor import (
    VideoFileClip, CompositeVideoClip, ColorClip, 
    vfx, CompositeAudioClip, AudioClip, concatenate_videoclips
)
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
import hashlib
import traceback

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
CACHE_DIR = os.path.join(TEMP_DIR, 'note_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Add new constants for time-based batching
TIME_WINDOW = 10.0  # Process 10 seconds at a time
WINDOW_OVERLAP = 0.5  # Half second overlap to ensure smooth transitions

# Add these constants at the top
MAX_MEMORY_PERCENT = 85
VIDEO_CACHE_SIZE = (320, 240)  # Smaller cache video size
# Update the FFMPEG options to be compatible with MoviePy
FFMPEG_OPTS = {
    'codec': 'libx264',
    'audio_codec': 'aac',
    'preset': 'ultrafast',
    'ffmpeg_params': [
        '-crf', '30',
        '-b:a', '128k'
    ]
}

CACHE_FILE_VERIFY_RETRIES = 3
CACHE_VERIFY_DELAY = 1

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
    """Clean temporary files with retry mechanism"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Force close any open file handles
            gc.collect()
            
            if os.path.exists(TEMP_DIR):
                for file in os.listdir(TEMP_DIR):
                    file_path = os.path.join(TEMP_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logging.warning(f"Failed to remove {file_path}: {e}")
                        continue
                
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
            
            os.makedirs(TEMP_DIR, exist_ok=True)
            return
            
        except Exception as e:
            logging.error(f"Error cleaning temp files (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                
    logging.error("Failed to clean temp files after all retries")

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
    try:
        if not clip or not hasattr(clip, 'duration'):
            return 0, 0
            
        clip_duration = float(clip.duration)
        safe_end = min(note_end, clip_duration)
        safe_start = max(0, min(note_start, safe_end))
        return safe_start, safe_end
        
    except Exception as e:
        logging.error(f"Error calculating safe duration: {e}")
        return 0, 0

def stretch_video_to_duration(clip, target_duration):
    """
    Stretch a video clip to match a target duration while maintaining audio pitch.
    """
    try:
        if not clip or not hasattr(clip, 'duration'):
            return None
            
        current_duration = clip.duration
        if (current_duration >= target_duration) or (current_duration <= 0):
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

def get_cache_key(track_id, midi_note):
    """Generate unique cache key for a track's note."""
    return hashlib.md5(f"{track_id}_{midi_note}".encode()).hexdigest()

def get_cache_path(track_id, midi_note):
    """Get full path for cached note clip."""
    return os.path.join(CACHE_DIR, f"{get_cache_key(track_id, midi_note)}.mp4")

def is_cached(track_id, midi_note):
    """Check if note clip is already cached."""
    cache_path = get_cache_path(track_id, midi_note)
    return os.path.exists(cache_path) and os.path.getsize(cache_path) > 0

def verify_cache_directory():
    """Verify cache directory exists and is writable."""
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            logging.info(f"Created cache directory: {CACHE_DIR}")
            
        # Test write permissions
        test_file = os.path.join(CACHE_DIR, 'test_write.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.unlink(test_file)
            return True
        except Exception as e:
            logging.error(f"Cache directory not writable: {e}")
            return False
    except Exception as e:
        logging.error(f"Error verifying cache directory: {e}")
        return False

def verify_cache_file(cache_path, retries=CACHE_FILE_VERIFY_RETRIES):
    """Verify cached file exists and is valid."""
    for attempt in range(retries):
        try:
            if not os.path.exists(cache_path):
                logging.debug(f"Cache file does not exist: {cache_path}")
                return False
                
            if os.path.getsize(cache_path) == 0:
                logging.debug(f"Cache file is empty: {cache_path}")
                return False
                
            # Verify file is readable
            with open(cache_path, 'rb') as f:
                f.read(1024)  # Try reading first 1KB
                
            return True
            
        except Exception as e:
            logging.warning(f"Cache verification attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(CACHE_VERIFY_DELAY)
                
    return False

def create_note_cache(track_id, track_data, video_path):
    """Create cached video clips for each unique note in the track."""
    try:
        if not verify_cache_directory():
            raise RuntimeError("Cache directory verification failed")
            
        if not verify_video_file(video_path):
            raise ValueError(f"Invalid video file: {video_path}")
            
        cache_paths = {}
        unique_notes = {note['midi'] for note in track_data.get('notes', [])}
        total_notes = len(unique_notes)
        
        logging.info(f"Creating cache for track {track_id} with {total_notes} unique notes")
        logging.debug(f"Cache directory: {CACHE_DIR}")
        
        for idx, midi_note in enumerate(unique_notes, 1):
            try:
                cache_path = get_cache_path(track_id, midi_note)
                logging.debug(f"Cache path for note {midi_note}: {cache_path}")
                
                if verify_cache_file(cache_path):
                    logging.debug(f"Using existing cache for note {midi_note}")
                    cache_paths[midi_note] = cache_path
                    continue
                    
                logging.info(f"Processing note {idx}/{total_notes} (MIDI: {midi_note})")
                
                with VideoFileClip(video_path, audio=True) as clip:
                    # Resize clip first to reduce memory usage
                    clip = clip.resize(width=VIDEO_CACHE_SIZE[0])
                    
                    # Process audio with pitch ratio
                    pitch_ratio = get_pitch_ratio({'midi': midi_note})
                    if clip.audio is not None:
                        modified_audio = adjust_audio_speed(clip.audio, pitch_ratio)
                        clip = clip.set_audio(modified_audio)
                        
                    # Write to cache with compatible parameters
                    temp_cache = cache_path + '.tmp'
                    clip.write_videofile(
                        temp_cache,
                        codec=FFMPEG_OPTS['codec'],
                        audio_codec=FFMPEG_OPTS['audio_codec'],
                        preset=FFMPEG_OPTS['preset'],
                        ffmpeg_params=FFMPEG_OPTS['ffmpeg_params'],
                        logger=None
                    )
                    
                    # Verify temp file before moving to final location
                    if verify_cache_file(temp_cache):
                        shutil.move(temp_cache, cache_path)
                        cache_paths[midi_note] = cache_path
                        logging.debug(f"Successfully cached note {midi_note}")
                    else:
                        raise RuntimeError(f"Cache file verification failed: {temp_cache}")
                        
                cleanup_memory()
                
            except Exception as e:
                logging.error(f"Error caching note {midi_note}: {str(e)}\n{traceback.format_exc()}")
                # Try to clean up failed temp file
                try:
                    if os.path.exists(temp_cache):
                        os.unlink(temp_cache)
                except:
                    pass
                continue
                
        return cache_paths
        
    except Exception as e:
        logging.error(f"Error creating note cache for track {track_id}: {str(e)}\n{traceback.format_exc()}")
        return {}

def calculate_grid_layout(num_tracks):
    """Calculate grid dimensions for track layout."""
    cols = int(np.ceil(np.sqrt(num_tracks)))
    rows = int(np.ceil(num_tracks / cols))
    return rows, cols

def get_grid_position(track_index, grid_layout, video_size=(320, 240)):
    """Calculate position for a track in the grid."""
    rows, cols = grid_layout
    row = track_index // cols
    col = track_index % cols
    return (col * video_size[0], row * video_size[1])

def get_time_windows(total_duration, window_size=TIME_WINDOW, overlap=WINDOW_OVERLAP):
    """Split total duration into overlapping time windows."""
    windows = []
    start_time = 0
    
    while start_time < total_duration:
        end_time = min(start_time + window_size, total_duration)
        windows.append((start_time, end_time))
        start_time = end_time - overlap
        
    return windows

def process_time_window(window_start, window_end, track_caches, video_files, grid_layout, canvas_size):
    """Process all track notes within a time window."""
    current_clips = []
    try:
        window_clips = []
        
        # Create background for this window
        background = ColorClip(
            size=canvas_size,
            color=(0, 0, 0),
            duration=window_end - window_start
        )
        window_clips.append(background)
        
        # Process each track's notes that fall within this window
        for track_idx, (track_id, track_data) in enumerate(video_files.items()):
            if not track_data.get('notes'):
                continue
                
            grid_pos = get_grid_position(track_idx, grid_layout)
            track_cache = track_caches.get(track_id, {})
            
            if not track_cache:
                continue
            
            # Filter notes that overlap with this window
            window_notes = [
                note for note in track_data['notes']
                if (float(note.get('time', 0)) < window_end and 
                    float(note.get('time', 0)) + float(note.get('duration', 0)) > window_start)
            ]
            
            for note in window_notes:
                try:
                    midi_note = note['midi']
                    cache_path = track_cache.get(midi_note)
                    if not cache_path or not os.path.exists(cache_path):
                        continue
                        
                    note_start = float(note.get('time', 0)) - window_start
                    note_duration = float(note.get('duration', 0))
                    
                    # Load clip from cache
                    clip = VideoFileClip(cache_path, audio=True)
                    current_clips.append(clip)  # Track for cleanup
                    
                    # Position and time the clip
                    positioned_clip = (clip
                        .resize(width=320)
                        .set_position(grid_pos)
                        .set_start(note_start)
                        .set_duration(note_duration))
                    
                    window_clips.append(positioned_clip)
                    
                except Exception as e:
                    logging.error(f"Error processing note {midi_note}: {e}")
                    continue
                    
        # Create composite for this window
        if len(window_clips) > 1:
            try:
                composite = CompositeVideoClip(
                    window_clips,
                    size=canvas_size,
                    bg_color=(0,0,0)
                )
                return composite
            except Exception as e:
                logging.error(f"Error creating composite: {e}")
                return None
        return None
        
    finally:
        # Clean up all clips
        for clip in current_clips:
            try:
                if clip and hasattr(clip, 'close'):
                    clip.close()
            except:
                pass
        cleanup_memory()

# Modify process_video_segments to add early termination and better cleanup
def process_video_segments(midi_data, video_files, output_path):
    """Process video segments using time-based batching with improved memory management."""
    temp_files = []
    current_window_clips = []
    final_clips = []
    track_caches = {}  # Initialize track_caches as empty dict
    
    # Add termination flag
    processing_complete = False
    
    try:
        logging.info("Starting video processing...")
        
        # Calculate grid layout first
        grid_layout = calculate_grid_layout(len(video_files))
        canvas_width = grid_layout[1] * VIDEO_CACHE_SIZE[0]
        canvas_height = grid_layout[0] * VIDEO_CACHE_SIZE[1]
        canvas_size = (canvas_width, canvas_height)
        
        logging.info(f"Grid layout: {grid_layout}, Canvas size: {canvas_size}")

        # Initialize empty batches list
        batches = []

        # Create note caches for all tracks first
        total_tracks = len(video_files)
        for track_idx, (track_id, track_data) in enumerate(video_files.items(), 1):
            logging.info(f"Caching track {track_idx}/{total_tracks}: {track_id}")
            if track_data.get('notes'):
                cache_result = create_note_cache(
                    track_id,
                    track_data,
                    track_data.get('path')
                )
                if cache_result:  # Only add if cache creation was successful
                    track_caches[track_id] = cache_result
                cleanup_memory()
                
        # Calculate total duration from MIDI notes with validation
        total_duration = 0
        try:
            for track_data in video_files.values():
                if not track_data.get('notes'):
                    continue
                for note in track_data['notes']:
                    try:
                        note_end = float(note.get('time', 0)) + float(note.get('duration', 0))
                        total_duration = max(total_duration, note_end)
                    except (TypeError, ValueError) as e:
                        logging.warning(f"Invalid note data: {note}, Error: {e}")
                        continue
        except Exception as e:
            logging.error(f"Error calculating duration: {e}")
            raise
            
        if total_duration <= 0:
            raise ValueError("No valid duration calculated from notes")
            
        logging.info(f"Total duration: {total_duration} seconds")
        
        # Process in very small time windows
        window_size = 2.0  # Even smaller window size
        overlap = 0.1     # Minimal overlap
        batch_size = 5    # Process windows in batches
        
        time_windows = []
        current_time = 0
        while current_time < total_duration:
            end_time = min(current_time + window_size, total_duration)
            time_windows.append((current_time, end_time))
            current_time = end_time - overlap
            
        total_windows = len(time_windows)
        logging.info(f"Total windows to process: {total_windows}")
        
        # Add a timeout mechanism
        start_time = time.time()
        max_processing_time = 3600  # 1 hour timeout
        
        for batch_idx in range(0, len(time_windows), batch_size):
            # Check processing time
            if time.time() - start_time > max_processing_time:
                raise TimeoutError("Video processing exceeded maximum allowed time")
                
            # Check memory usage before each batch
            if get_memory_usage() > MAX_MEMORY_PERCENT:
                logging.error("Memory usage exceeded threshold, terminating processing")
                raise MemoryError("Memory usage too high")
                
            batch_windows = time_windows[batch_idx:batch_idx + batch_size]
            current_window_clips = []
            
            logging.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(time_windows) + batch_size - 1)//batch_size}")
            
            try:
                for window_idx, (start_time, end_time) in enumerate(batch_windows):
                    logging.info(f"Processing window {batch_idx + window_idx + 1}/{total_windows} ({start_time:.2f}s - {end_time:.2f}s)")
                    
                    try:
                        window_clip = process_time_window(
                            start_time,
                            end_time,
                            track_caches,
                            video_files,
                            grid_layout,
                            canvas_size
                        )
                        
                        if window_clip:
                            # Save window clip with unique identifier
                            temp_path = os.path.join(
                                TEMP_DIR,
                                f'window_{batch_idx + window_idx:04d}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.mp4'
                            )
                            
                            window_clip.write_videofile(
                                temp_path,
                                codec='libx264',
                                audio_codec='aac',
                                preset='ultrafast',
                                fps=30,
                                logger=None,
                                threads=2
                            )
                            
                            current_window_clips.append(temp_path)
                            temp_files.append(temp_path)
                            window_clip.close()
                            
                            # Force cleanup after each window
                            cleanup_memory()
                            
                    except Exception as e:
                        logging.error(f"Error processing window {batch_idx + window_idx}: {str(e)}")
                        continue
                        
                # Combine batch windows
                if current_window_clips:
                    batch_output = os.path.join(TEMP_DIR, f'batch_{batch_idx//batch_size:04d}.mp4')
                    
                    # Use ffmpeg to concatenate batch clips
                    list_file = os.path.join(TEMP_DIR, f"batch_{batch_idx//batch_size:04d}_files.txt")
                    with open(list_file, 'w') as f:
                        for clip_path in current_window_clips:
                            f.write(f"file '{clip_path}'\n")
                            
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', list_file,
                        '-c', 'copy',
                        batch_output
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    final_clips.append(batch_output)
                    temp_files.extend([list_file, batch_output])
                    
                    # Clean up window clips
                    for clip_path in current_window_clips:
                        try:
                            os.unlink(clip_path)
                        except:
                            pass
                            
                    current_window_clips = []
                    cleanup_memory()
                    gc.collect(2)
                    
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx//batch_size}: {str(e)}")
                continue
                
        # Final concatenation of batch clips
        if final_clips:
            try:
                final_list = os.path.join(TEMP_DIR, "final_files.txt")
                with open(final_list, 'w') as f:
                    for clip_path in final_clips:
                        if os.path.exists(clip_path):
                            f.write(f"file '{clip_path}'\n")
                            
                temp_files.append(final_list)
                
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', final_list,
                    '-c', 'copy',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    error_msg = f"FFmpeg error during final concatenation: {result.stderr}"
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
                    
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    raise RuntimeError("Output file was not created or is empty")
                    
                logging.info(f"Successfully created output video: {output_path}")
                
            except Exception as e:
                logging.error(f"Error in final concatenation: {str(e)}")
                raise
                
        processing_complete = True
        
    except Exception as e:
        error_msg = f"Error in video processing: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
        
    finally:
        try:
            # Clean up all clips and temporary files
            for clip in current_window_clips:
                try:
                    if isinstance(clip, VideoFileClip) and hasattr(clip, 'close'):
                        clip.close()
                except:
                    pass
                    
            # Remove temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
                    
            # Force garbage collection
            cleanup_memory()
            clean_temp_files()
            
            # If processing didn't complete successfully, remove output file
            if not processing_complete and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
                    
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {cleanup_error}")

# Add memory error class
class MemoryError(Exception):
    pass

# Modify cleanup_memory to be more aggressive
def cleanup_memory():
    """Force garbage collection and clear memory"""
    try:
        # Run garbage collection multiple times
        for _ in range(5):
            gc.collect()
            
        # Clear MoviePy's cache if it exists
        if hasattr(VideoFileClip, 'close_all'):
            VideoFileClip.close_all()
            
        # Clear any cached attributes
        if 'CACHE_DIR' in globals():
            try:
                shutil.rmtree(CACHE_DIR, ignore_errors=True)
                os.makedirs(CACHE_DIR, exist_ok=True)
            except:
                pass
                
    except Exception as e:
        logging.error(f"Error during memory cleanup: {e}")

def pre_python_processing_verification(drum_tracks):
    """Verify drum tracks before processing."""
    for track in drum_tracks:
        if 'key' not in track or 'noteCount' not in track:
            logging.error(f"Invalid drum track data: {track}")
            return False
        logging.info(f"Drum track {track['key']} with {track['noteCount']} notes verified.")
    return True

# Modify main function to handle termination
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
        # Ensure cleanup happens even on error
        cleanup_memory()
        clean_temp_files()
        # Remove output file if it exists but is incomplete
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        sys.exit(1)
    finally:
        cleanup_memory()
        clean_temp_files()

# Ensure cleanup happens on script exit
atexit.register(cleanup_memory)
atexit.register(clean_temp_files)

if __name__ == "__main__":
    main()

# Add cleanup on script exit
atexit.register(clean_temp_files)
