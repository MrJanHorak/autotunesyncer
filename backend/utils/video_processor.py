from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
import numpy as np
import sys
import json
import os
from pathlib import Path

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
        background = ColorClip(size=(960, 720), color=(0,0,0), duration=float(midi_data['duration']))
        
        clips = [background]
        base_clips = {}  # Cache for base clips to avoid reloading
        
        # Process each track
        for track_idx, track in enumerate(midi_tracks):
            if not track.get('notes'):
                print(f"Skipping track {track_idx} - no notes")
                continue
                
            instrument_name = track['instrument']['name'].lower().replace(' ', '_')
            if instrument_name not in video_files:
                print(f"No video found for instrument: {instrument_name}")
                continue
                
            video_path = video_files[instrument_name]
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue
            
            # Load base clip if not already loaded
            if video_path not in base_clips:
                try:
                    base_clips[video_path] = VideoFileClip(video_path)
                    print(f"Loaded video for {instrument_name}")
                except Exception as e:
                    print(f"Error loading video {video_path}: {str(e)}")
                    continue
            
            base_clip = base_clips[video_path]
            
            # Calculate position in grid
            x = (track_idx % grid_size) * clip_width
            y = (track_idx // grid_size) * clip_height
            
            # Process notes
            for note_idx, note in enumerate(track['notes']):
                try:
                    start_time = float(note['time'])
                    duration = float(note['duration'])
                    
                    # Create segment with explicit duration
                    segment = (base_clip
                             .subclip(0, min(duration, base_clip.duration))
                             .resize(width=clip_width, height=clip_height)
                             .set_start(start_time)
                             .set_position((x, y)))
                    
                    print(f"Added note segment {note_idx} for track {track_idx}")
                    clips.append(segment)
                except Exception as e:
                    print(f"Error processing note in track {track_idx}: {str(e)}")
                    continue
        
        if len(clips) <= 1:
            raise ValueError("No valid clips to compose")
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Compose final video with explicit duration
        print("Compositing final video...")
        final_clip = CompositeVideoClip(clips, size=(960, 720))
        final_duration = float(midi_data['duration'])
        final_clip = final_clip.set_duration(final_duration)
        
        # Write output with progress reporting
        print(f"Writing output to {output_path}")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=30,
            verbose=False,
            logger=None
        )
        
        print("Video processing completed successfully")
        
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        raise
        
    finally:
        # Cleanup
        try:
            for clip in clips:
                clip.close()
            for clip in base_clips.values():
                clip.close()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python video_processor.py <midi_json> <video_files_json> <output_path>")
        sys.exit(1)
        
    try:
        with open(sys.argv[1], 'r') as f:
            midi_data = json.load(f)
        
        with open(sys.argv[2], 'r') as f:
            video_files = json.load(f)
            
        process_video_segments(midi_data, video_files, sys.argv[3])
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
