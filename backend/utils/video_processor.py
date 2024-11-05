from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, concatenate_videoclips, AudioFileClip
from PIL import Image
import numpy as np
import sys
import json
import os
from pathlib import Path

# Add PIL compatibility layer
if not hasattr(Image, 'ANTIALIAS'):
    # For newer versions of Pillow
    Image.ANTIALIAS = Image.Resampling.LANCZOS

def process_video_segments(midi_data, video_files, output_path):
    clips_to_close = []  # Track clips for proper cleanup
    try:
        # Parse MIDI tracks and notes
        tracks = midi_data['tracks']
        duration = midi_data['duration']
        
        # Initialize empty canvas
        background = ColorClip(size=(960, 720), color=(0, 0, 0), duration=duration)
        clips_to_close.append(background)
        
        # Process each track's notes
        all_clips = []
        grid_size = int(np.ceil(np.sqrt(len(tracks))))
        clip_width = 960 // grid_size
        clip_height = 720 // grid_size
        
        for track_idx, track in enumerate(tracks):
            instrument_name = track['instrument']['name'].replace(' ', '_').lower()
            if instrument_name not in video_files:
                print(f"No video file for {instrument_name}")
                continue
                
            video_path = video_files[instrument_name]
            source_clip = VideoFileClip(video_path)
            clips_to_close.append(source_clip)
            
            # Calculate position in grid
            x_pos = (track_idx % grid_size) * clip_width
            y_pos = (track_idx // grid_size) * clip_height
            
            # Process each note in the track
            for note in track['notes']:
                start_time = note.get('time', 0)
                duration = note.get('duration', 0)
                pitch = note.get('midi', 60)  # MIDI note number
                velocity = note.get('velocity', 1.0)
                
                # Create snippet for this note
                try:
                    # Extract snippet from source video
                    snippet = source_clip.subclip(0, duration)
                    
                    # Resize and position snippet
                    snippet = snippet.resize((clip_width, clip_height))
                    snippet = snippet.set_position((x_pos, y_pos))
                    snippet = snippet.set_start(start_time)
                    
                    # Apply pitch adjustment
                    if hasattr(snippet, 'audio') and snippet.audio is not None:
                        # Convert MIDI note to frequency ratio
                        base_note = 60  # Middle C
                        semitone_ratio = 2 ** (1/12)
                        pitch_ratio = semitone_ratio ** (pitch - base_note)
                        
                        # Apply pitch shift by adjusting speed
                        # This affects both video and audio, but it's the simplest way to pitch-shift
                        snippet = snippet.speedx(pitch_ratio)
                        
                        # Adjust volume
                        snippet = snippet.volumex(velocity)
                    
                    all_clips.append(snippet)
                    clips_to_close.append(snippet)
                except Exception as e:
                    print(f"Error processing note in {instrument_name}: {e}")
                    continue
        
        # Combine all clips
        if all_clips:
            final_clip = CompositeVideoClip([background] + all_clips)
            clips_to_close.append(final_clip)
            final_clip.write_videofile(output_path, codec='libx264', fps=24)
            print(f"Video saved to {output_path}")
        else:
            raise ValueError("No valid clips to compose")
            
    except Exception as e:
        print(f"Error in video processing: {e}")
        raise
    finally:
        # Clean up all clips
        for clip in clips_to_close:
            try:
                clip.close()
            except:
                pass

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python video_processor.py <midi_data.json> <video_files.json> <output_path>")
        sys.exit(1)
    
    midi_data_path = sys.argv[1]
    video_files_path = sys.argv[2]
    output_path = sys.argv[3]
    
    with open(midi_data_path, 'r') as f:
        midi_data = json.load(f)
    
    with open(video_files_path, 'r') as f:
        video_files = json.load(f)
    
    process_video_segments(midi_data, video_files, output_path)