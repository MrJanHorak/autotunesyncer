import json
import sys
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, vfx, CompositeAudioClip, AudioClip
import numpy as np
from PIL import Image
import logging
import time
import os
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Replace '0' with the correct GPU index if necessary

# Configure logging at the beginning of the file
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    if audio_clip is None:
        return None
    if not hasattr(audio_clip, 'fps') or audio_clip.fps is None:
        audio_clip.fps = default_fps
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

def process_video_segments(midi_data, video_files, output_path):
    clips_to_close = []
    try:
        duration = float(midi_data.get('duration', 10))
        if duration <= 0:
            logging.error("Invalid MIDI duration")
            duration = 10.0

        tracks = midi_data.get('tracks', [])
        
        background = ColorClip(size=(960, 720), color=(0, 0, 0), duration=duration)
        clips_to_close.append(background)
        
        # Filter video files to only include those with notes
        active_video_files = {
            track_id: data for track_id, data in video_files.items()
            if data.get('notes') and len(data['notes']) > 0
        }
        
        total_tracks = len(active_video_files)
        logging.info(f"Active tracks with notes: {total_tracks}")

        # Handle special layout cases
        source_clips = {}
        track_positions = {}
        clip_dimensions = {}  # Store dimensions for each track

        if total_tracks == 1:
            # Single track - use full screen layout
            track_id, track_data = next(iter(active_video_files.items()))
            try:
                source_clip = VideoFileClip(track_data['path'], audio=True)
                if not verify_frame_reading(source_clip):
                    raise ValueError(f"Failed to verify frame reading for {track_id}")
                
                # Center crop the video to fill the screen while maintaining aspect ratio
                source_clip = source_clip.resize((960, 720))
                source_clips[track_id] = source_clip
                track_positions[track_id] = (0, 0)
                clip_dimensions[track_id] = {'width': 960, 'height': 720}
                clips_to_close.append(source_clip)
                
            except Exception as e:
                logging.error(f"Failed to load single video track: {e}")
                raise
                
        elif total_tracks == 2:
            # Two tracks - split screen horizontally
            for i, (track_id, track_data) in enumerate(active_video_files.items()):
                try:
                    source_clip = VideoFileClip(track_data['path'], audio=True)
                    if not verify_frame_reading(source_clip):
                        continue
                    
                    # Position horizontally side by side
                    x_pos = i * 480  # 960/2 = 480 for each half
                    source_clip = source_clip.resize((480, 720))
                    
                    source_clips[track_id] = source_clip
                    track_positions[track_id] = (x_pos, 0)
                    clip_dimensions[track_id] = {'width': 480, 'height': 720}
                    clips_to_close.append(source_clip)
                    
                except Exception as e:
                    logging.error(f"Failed to load video for track {track_id}: {e}")
                    continue
                    
        else:
            # Default grid layout for 3+ tracks
            grid_size = max(2, int(np.ceil(np.sqrt(total_tracks))))
            clip_width = 960 // grid_size
            clip_height = 720 // grid_size
            position_idx = 0
            
            for track_id, track_data in active_video_files.items():
                try:
                    source_clip = VideoFileClip(track_data['path'], audio=True)
                    if not verify_frame_reading(source_clip):
                        continue
                        
                    x_pos = (position_idx % grid_size) * clip_width
                    y_pos = (position_idx // grid_size) * clip_height
                    source_clip = source_clip.resize((clip_width, clip_height))
                    
                    source_clips[track_id] = source_clip
                    track_positions[track_id] = (x_pos, y_pos)
                    clip_dimensions[track_id] = {'width': clip_width, 'height': clip_height}
                    clips_to_close.append(source_clip)
                    position_idx += 1
                    
                except Exception as e:
                    logging.error(f"Failed to load video for grid layout: {e}")
                    continue

        # Verify audio in source files
        for track_id, track_data in active_video_files.items():
            if not verify_audio_stream(track_data['path']):
                logging.warning(f"No valid audio found in track {track_id}")
                continue
                
            try:
                source_clip = VideoFileClip(track_data['path'], audio=True)
                if not verify_frame_reading(source_clip):
                    raise ValueError(f"Failed to verify frame reading for {track_id}")
                
                # Verify audio data
                if source_clip.audio is None:
                    logging.error(f"Failed to load audio for track {track_id}")
                    continue
                    
                # Test audio reading
                try:
                    audio_array = source_clip.audio.to_soundarray()
                    logging.info(f"Audio array shape for {track_id}: {audio_array.shape}")
                    logging.info(f"Audio FPS for {track_id}: {source_clip.audio.fps}")
                except Exception as e:
                    logging.error(f"Failed to read audio data for {track_id}: {e}")
                    continue

                # Add audio standardization to all source clips
                for track_id, track_data in active_video_files.items():
                    try:
                        source_clip = VideoFileClip(track_data['path'], audio=True)
                        if not verify_frame_reading(source_clip):
                            raise ValueError(f"Failed to verify frame reading for {track_id}")
                        
                        # Standardize audio using AudioClip methods instead
                        if source_clip.audio is not None:
                            try:
                                # Get standardized audio data
                                audio_array = source_clip.audio.to_soundarray()
                                if not isinstance(audio_array, np.ndarray):
                                    logging.error(f"Audio array for track {track_id} is not a valid numpy array")
                                    continue
                                fps = source_clip.audio.fps or 44100  # Use default if not set
                                
                                # Create normalized clip
                                normalized_clip = create_normalized_audio_clip(
                                    audio_array,
                                    fps,
                                    source_clip.duration,
                                    0
                                )
                                
                                if normalized_clip is not None:
                                    source_clip.audio = normalized_clip
                                
                            except Exception as e:
                                logging.warning(f"Could not normalize audio for track {track_id}: {e}")
                        
                        source_clips[track_id] = source_clip
                        clips_to_close.append(source_clip)
                        
                    except Exception as e:
                        logging.error(f"Failed to load video track {track_id}: {e}")
                        continue

                # Process all tracks and notes
                all_clips = []
                temp_clips = []  # Store temporary clip paths
                
                for track_id, track_data in active_video_files.items():
                    if track_id not in source_clips or track_id not in track_positions:
                        continue
                        
                    source_clip = source_clips[track_id]
                    is_drum = track_data.get('isDrum', False)
                    x_pos, y_pos = track_positions[track_id]
                    dimensions = clip_dimensions[track_id]
                    
                    # Process notes for this track
                    for i, note in enumerate(track_data.get('notes', [])):
                        try:
                            start_time = float(note.get('time', 0))
                            note_duration = float(note.get('duration', 0))
                            
                            if note_duration <= 0 or start_time >= duration:
                                continue
                            
                            note_duration = min(note_duration, duration - start_time)
                            
                            # Create temp file path for this note
                            temp_note_path = os.path.join(os.path.dirname(output_path), f'temp_note_{track_id}_{i}.mp4')
                            temp_clips.append(temp_note_path)
                            
                            # Extract the segment using FFmpeg for precise audio handling
                            pitch_ratio = get_pitch_ratio(note) if not is_drum else 1.0
                            volume = float(note.get('velocity', 1.0))
                            if is_drum:
                                volume *= 0.7

                            # Use FFmpeg command for audio processing
                            ffmpeg_cmd = [
                                'ffmpeg', '-y',
                                '-i', track_data['path'],
                                '-ss', '0',
                                '-t', str(note_duration),
                                '-filter_complex',
                                f'[0:v]scale={dimensions["width"]}:{dimensions["height"]}[v];' +
                                f'[0:a]asetrate={44100*pitch_ratio},aformat=sample_fmts=fltp,volume={volume}[a]',
                                '-map', '[v]',
                                '-map', '[a]',
                                '-c:v', 'libx264',
                                '-c:a', 'aac',
                                '-ar', '44100',
                                temp_note_path
                            ]
                            
                            subprocess.run(ffmpeg_cmd, check=True)
                            
                            # Load processed clip
                            processed_clip = VideoFileClip(temp_note_path)
                            positioned_clip = (processed_clip
                                .set_position((x_pos, y_pos))
                                .set_start(start_time))
                            
                            all_clips.append(positioned_clip)
                            clips_to_close.append(processed_clip)
                            
                        except Exception as e:
                            logging.error(f"Error processing note in {track_id}: {e}")
                            continue

                if not all_clips:
                    raise ValueError("No valid clips were created")
                
                # Create final composition
                final_clip = CompositeVideoClip([background] + all_clips, size=(960, 720))
                final_clip.duration = duration
                clips_to_close.append(final_clip)
                
                # Write final video using FFmpeg directly
                try:
                    final_clip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile=None,  # Disable MoviePy's audio handling
                        remove_temp=True,
                        fps=30,
                        ffmpeg_params=[
                            '-c:a', 'aac',
                            '-ar', '44100',
                            '-ac', '2',
                            '-b:a', '192k'
                        ]
                    )
                    return True
                    
                except Exception as e:
                    logging.error(f"Error writing final video: {e}")
                    raise
                    
                finally:
                    # Clean up temp files
                    for temp_file in temp_clips:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except Exception as e:
                            logging.warning(f"Failed to remove temp file {temp_file}: {e}")

            except Exception as e:
                logging.error(f"Error in video processing: {str(e)}")
                raise
            finally:
                # Clean up clips
                for clip in clips_to_close:
                    try:
                        if clip is not None:
                            clip.close()
                    except Exception as e:
                        logging.warning(f"Failed to close clip: {e}")
                try:
                    if os.path.exists('temp-audio-full.m4a'):
                        os.remove('temp-audio-full.m4a')
                except:
                    pass

    except Exception as e:
        logging.error(f"Error in process_video_segments: {str(e)}")
        raise

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
            
        process_video_segments(midi_data, video_files, output_path)
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
