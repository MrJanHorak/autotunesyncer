from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
from PIL import Image
import numpy as np
import sys
import json
import os
from pathlib import Path

# Version-compatible resampling
try:
    # For Pillow version >= 9.0.0
    from PIL.Image import Resampling
    RESIZE_FILTER = Resampling.LANCZOS
except ImportError:
    # For Pillow version < 9.0.0
    RESIZE_FILTER = Image.ANTIALIAS

# Monkey patch moviepy's resize to use the correct filter
from moviepy.video.fx.resize import resize
def new_resizer(pic, newsize):
    pilim = Image.fromarray(pic)
    resized_pil = pilim.resize(newsize[::-1], RESIZE_FILTER)
    return np.array(resized_pil)

# Replace moviepy's resizer function
import moviepy.video.fx.resize
moviepy.video.fx.resize.resizer = new_resizer

def process_video_segments(midi_data, video_files, output_path):
    try:
        # Parse MIDI data
        midi_tracks = midi_data['tracks']
        
        # Calculate grid layout
        total_tracks = len(midi_tracks)
        grid_size = int(np.ceil(np.sqrt(total_tracks)))
        clip_width = 960 // grid_size
        clip_height = 720 // grid_size
        
        print(f"Processing {total_tracks} tracks in {grid_size}x{grid_size} grid")
        
        # Create background with explicit RGB color
        background_color = (0, 0, 0)  # Black background
        background = ColorClip(size=(960, 720), color=background_color, duration=midi_data['duration'])
        
        # Load video clips and resize them
        video_clips = []
        for track in midi_tracks:
            instrument_name = track['instrument']['name'].replace(' ', '_').lower()
            if instrument_name in video_files:
                video_path = video_files[instrument_name]
                print(f"Loaded video for {instrument_name}")
                video_clip = VideoFileClip(video_path).resize((clip_width, clip_height))
                video_clips.append(video_clip)
            else:
                print(f"No video file for {instrument_name}")
        
        if not video_clips:
            raise ValueError("No valid clips to compose")
        
        # Create composite video
        composite_clips = []
        for i, clip in enumerate(video_clips):
            x = (i % grid_size) * clip_width
            y = (i // grid_size) * clip_height
            composite_clips.append(clip.set_position((x, y)))
        
        final_clip = CompositeVideoClip([background] + composite_clips)
        final_clip.write_videofile(output_path, codec='libx264', fps=24)
        
        print(f"Video saved to {output_path}")
        
    except Exception as e:
        print(f"Error in video processing: {e}")
        raise

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