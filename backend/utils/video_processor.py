import json
import sys
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, vfx, CompositeAudioClip, AudioClip
import numpy as np
from PIL import Image
import logging
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Replace '0' with the correct GPU index if necessary

# Configure logging at the beginning of the file
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def standardize_audio(audio_clip, target_fps=44100):
    """Standardize audio format across different codecs."""
    try:
        if (audio_clip is None):
            return None, target_fps
            
        # Get audio data
        audio_array = audio_clip.to_soundarray()
        original_fps = getattr(audio_clip, 'fps', target_fps)
        
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
            
        return audio_array, original_fps
        
    except Exception as e:
        logging.error(f"Error standardizing audio: {e}")
        return None, target_fps

def create_normalized_audio_clip(audio_array, fps, duration, start_time=0):
    """Create a normalized audio clip with improved handling."""
    try:
        if audio_array is None or not isinstance(audio_array, np.ndarray):
            return None
            
        # Ensure array is contiguous and float32
        audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
        
        # Ensure stereo format
        if len(audio_array.shape) == 1:
            audio_array = np.column_stack((audio_array, audio_array))
        elif audio_array.shape[1] == 1:
            audio_array = np.column_stack((audio_array[:, 0], audio_array[:, 0]))
            
        # Calculate lengths
        target_length = int(duration * fps)
        original_length = len(audio_array)
        
        if original_length == 0:
            return None

        # Create make_frame function that handles audio properly            
        def make_frame(t):
            try:
                frame_time = float(t) - float(start_time)
                if frame_time < 0 or frame_time >= duration:
                    return np.zeros(2)
                    
                idx = int(frame_time * fps) % original_length
                return audio_array[idx]
            except:
                return np.zeros(2)

        return AudioClip(make_frame, duration=duration, fps=fps)

    except Exception as e:
        logging.error(f"Error creating normalized audio clip: {e}")
        return None

def process_video_segments(midi_data, video_files, output_path):
    clips_to_close = []
    try:
        # Get midi duration and ensure it's valid
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
        all_audio_clips = []
        
        for track_id, track_data in active_video_files.items():
            if track_id not in source_clips or track_id not in track_positions:
                continue
                
            source_clip = source_clips[track_id]
            is_drum = track_data.get('isDrum', False)
            
            # Extract and store audio from source clip
            if hasattr(source_clip, 'audio') and source_clip.audio is not None:
                try:
                    audio_array = source_clip.audio.to_soundarray()
                    fps = source_clip.audio.fps or 44100
                    
                    # Create normalized source audio
                    source_audio = create_normalized_audio_clip(
                        audio_array,
                        fps,
                        source_clip.duration
                    )
                    if source_audio is not None:
                        source_audio = source_audio.set_fps(44100)
                        all_audio_clips.append(source_audio)
                        clips_to_close.append(source_audio)
                except Exception as e:
                    logging.warning(f"Could not process source audio for {track_id}: {e}")

            # Process notes
            for note in track_data.get('notes', []):
                try:
                    start_time = float(note.get('time', 0))
                    note_duration = float(note.get('duration', 0))
                    
                    if note_duration <= 0 or start_time >= duration:
                        continue
                        
                    note_duration = min(note_duration, duration - start_time)
                    
                    # Create video snippet
                    snippet = source_clip.subclip(0, min(note_duration, source_clip.duration))
                    if not is_drum:
                        pitch_ratio = get_pitch_ratio(note)
                        snippet = snippet.fx(vfx.speedx, factor=pitch_ratio)
                    
                    snippet = (snippet
                        .resize((clip_dimensions[track_id]['width'], clip_dimensions[track_id]['height']))
                        .set_position((track_positions[track_id][0], track_positions[track_id][1]))
                        .set_start(start_time))
                    
                    all_clips.append(snippet)
                    clips_to_close.append(snippet)
                    
                    # Create audio snippet with proper normalization
                    if hasattr(snippet, 'audio') and snippet.audio is not None:
                        try:
                            audio_array = snippet.audio.to_soundarray()
                            fps = snippet.audio.fps or 44100
                            
                            audio_clip = create_normalized_audio_clip(
                                audio_array,
                                fps,
                                snippet.duration,
                                start_time
                            )
                            
                            if audio_clip is not None:
                                audio_clip = audio_clip.set_fps(44100)
                                volume = float(note.get('velocity', 1.0))
                                if is_drum:
                                    volume *= 0.7
                                audio_clip = audio_clip.volumex(volume)
                                all_audio_clips.append(audio_clip)
                                clips_to_close.append(audio_clip)
                        except Exception as e:
                            logging.error(f"Failed to process audio for note in {track_id}: {e}")
                            
                except Exception as e:
                    logging.error(f"Error processing note in {track_id}: {e}")
                    continue

        # Filter out None entries from all_audio_clips
        all_audio_clips = [clip for clip in all_audio_clips if clip is not None]
        
        # Create final video with audio
        final_clip = CompositeVideoClip([background] + all_clips, size=(960, 720))
        final_clip.duration = duration
        clips_to_close.append(final_clip)

        # Create final audio mix with additional error checking
        if all_audio_clips:
            try:
                logging.debug(f"Combining {len(all_audio_clips)} audio clips")
                valid_audio_clips = []
                
                # Validate each audio clip before combining
                for clip in all_audio_clips:
                    if clip is not None:
                        try:
                            # Ensure clip has necessary attributes
                            if not hasattr(clip, 'fps') or clip.fps is None:
                                clip.fps = 44100
                            if not hasattr(clip, 'duration') or clip.duration is None:
                                continue
                            # Set all clips to same sample rate
                            clip = clip.set_fps(44100)
                            valid_audio_clips.append(clip)
                        except Exception as e:
                            logging.warning(f"Skipping invalid audio clip: {e}")
                            continue

                if valid_audio_clips:
                    logging.debug(f"Creating composite audio from {len(valid_audio_clips)} valid clips")
                    try:
                        final_audio = CompositeAudioClip(valid_audio_clips)
                        if final_audio is not None:
                            final_audio.fps = 44100
                            # Ensure final audio duration matches video
                            final_audio = final_audio.set_duration(duration)
                            final_clip = final_clip.set_audio(final_audio)
                            clips_to_close.append(final_audio)
                            logging.debug("Successfully set final audio")
                    except Exception as e:
                        logging.error(f"Error creating composite audio: {e}")
                else:
                    logging.warning("No valid audio clips to combine")
            except Exception as e:
                logging.error(f"Error in final audio mixing: {e}")
        else:
            logging.warning("No audio clips to combine for final audio")

        # Write final video with explicit temp file handling
        temp_audio = None
        try:
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='libmp3lame',  # Changed to 'libmp3lame' for consistency
                audio=True,
                audio_fps=44100,
                # audio_nbytes=4,  # Removed to prevent potential audio issues
                audio_bufsize=2000,
                temp_audiofile=None,  # Let MoviePy handle temp files
                remove_temp=True,
                fps=30,
                preset='ultrafast',
                threads=4,
                ffmpeg_params=[
                    '-ac', '2',
                    '-ar', '44100',
                    '-b:a', '192k',
                    '-c:a', 'libmp3lame',  # Updated from 'aac' to match audio_codec
                    '-strict', '-2'
                ],
                logger='bar',
                verbose=True
            )
        except Exception as e:
            logging.error(f"Error writing final video: {e}")
            raise

        return True

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
