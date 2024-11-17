import json
import sys
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, vfx
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

def process_video_segments(midi_data, video_files, output_path):
    clips_to_close = []
    try:
        duration = float(midi_data.get('duration', 10))
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

        # Process all tracks and notes
        all_clips = []
        
        # Process each video file's notes directly
        for track_id, track_data in active_video_files.items():
            if track_id not in source_clips or track_id not in track_positions:
                continue
                
            source_clip = source_clips[track_id]
            is_drum = track_data.get('isDrum', False)
            x_pos, y_pos = track_positions[track_id]
            dimensions = clip_dimensions[track_id]
            
            # Process notes for this track
            for note in track_data.get('notes', []):
                try:
                    start_time = float(note.get('time', 0))
                    note_duration = float(note.get('duration', 0))
                    
                    if note_duration <= 0:
                        continue
                    
                    snippet = source_clip.subclip(0, min(note_duration, source_clip.duration))
                    
                    if not is_drum and snippet.audio is not None:
                        pitch_ratio = midiNoteToFrequencyRatio(note.get('midi', 60))
                        snippet = snippet.speedx(pitch_ratio)
                    
                    snippet = snippet.volumex(note.get('velocity', 1.0))
                    
                    snippet = (snippet
                        .resize((dimensions['width'], dimensions['height']))
                        .set_position((x_pos, y_pos))
                        .set_start(start_time))
                    
                    all_clips.append(snippet)
                    clips_to_close.append(snippet)
                    
                    if is_drum:
                        logging.debug(f"Added drum snippet for {track_id} at time {start_time} with duration {note_duration}")
                    
                except Exception as e:
                    logging.error(f"Error processing note in {track_id}: {e}")
                    continue

        if not all_clips:
            raise ValueError("No valid clips were created")
        
        # Create final composition
        logging.debug(f"Creating composite of {len(all_clips)} clips...")
        final_clip = CompositeVideoClip([background] + all_clips, size=(960, 720))
        final_clip.duration = duration
        clips_to_close.append(final_clip)
        
        # Try NVENC first, fallback to CPU encoding if not available
        try:
            logging.info("Starting video encoding with NVENC")
            final_clip.write_videofile(
                output_path,
                codec='h264_nvenc',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                preset='llhp',# Valid NVENC preset: default, slow, medium, fast, hp, hq, bd, ll, llhq, llhp
                audio=True,
                verbose=True,
                logger='bar',  # Set to None to enable verbose output
                ffmpeg_params=['-gpu', '0']
            )
            logging.info("Completed video encoding with NVENC")
        except Exception as e:
            print(f"NVENC encoding failed: {e}")
            logging.warning(f"NVENC encoding failed, falling back to CPU encoding: {e}")
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                threads=4,
                preset='ultrafast',  # ultrafast preset is valid for libx264
                audio=True,
                verbose=True,
                logger='bar'  # Set to None to enable verbose output
            )
        
        return True
        
    except Exception as e:
        logging.error(f"Error in video processing: {str(e)}")
        raise
    finally:
        # Clean up clips
        for clip in clips_to_close:
            try:
                clip.close()
            except Exception as e:
                logging.warning(f"Failed to close clip: {e}")

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