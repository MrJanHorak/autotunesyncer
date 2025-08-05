#!/usr/bin/env python3
"""
MIDI Synchronized Video Compositor
Creates videos where instruments are triggered by MIDI notes
"""

import subprocess
import os
import json
import tempfile
import logging
from pathlib import Path

class MidiSynchronizedCompositor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)
        
    def create_midi_triggered_video(self, midi_data, video_paths, output_path, total_duration):
        """
        Create a video where instruments are triggered by MIDI notes
        
        Args:
            midi_data: MIDI data structure with tracks and notes
            video_paths: Dictionary mapping instrument names to video file paths
            output_path: Output video file path
            total_duration: Total duration of the composition
        """
        self.logger.info(f"🎵 Creating MIDI-triggered video (duration: {total_duration:.2f}s)")
        
        # Create individual triggered video segments for each track
        triggered_videos = []
        
        for track_idx, track in enumerate(midi_data.get('tracks', [])):
            instrument_name = track.get('instrument', f'track_{track_idx}')
            notes = track.get('notes', [])
            
            if instrument_name in video_paths:
                video_path = video_paths[instrument_name]
                triggered_video = self.create_triggered_track(
                    video_path, notes, total_duration, instrument_name
                )
                if triggered_video:
                    triggered_videos.append(triggered_video)
        
        if not triggered_videos:
            self.logger.error("No triggered videos created")
            return False
        
        # Composite all triggered videos into final grid
        return self.composite_triggered_videos(triggered_videos, output_path)
    
    def create_triggered_track(self, video_path, notes, total_duration, track_name):
        """
        Create a video track where the instrument video is triggered by MIDI notes
        
        Args:
            video_path: Path to the source video file
            notes: List of note events with timing and duration
            total_duration: Total duration of the track
            track_name: Name of the track for logging
        """
        output_path = os.path.join(self.temp_dir, f"{track_name}_triggered.mp4")
        
        if not notes:
            self.logger.warning(f"No notes for track {track_name}")
            return None
        
        try:
            # Create filter complex for note-triggered playback
            filter_parts = []
            
            # Create black base video
            filter_parts.append(f"color=black:size=640x360:duration={total_duration}[base]")
            
            # Add triggered segments for each note
            current_overlay = "base"
            for i, note in enumerate(notes):
                start_time = note.get('time', 0)
                duration = min(note.get('duration', 0.5), 2.0)  # Cap at 2 seconds
                
                if duration <= 0:
                    continue
                
                # Create segment for this note
                filter_parts.append(
                    f"[0:v]trim=start=0:duration={duration},"
                    f"setpts=PTS-STARTPTS[note_{i}]"
                )
                
                # Overlay this note segment at the correct time
                next_overlay = f"out_{i}"
                filter_parts.append(
                    f"[{current_overlay}][note_{i}]overlay=enable='between(t,{start_time},{start_time + duration})'[{next_overlay}]"
                )
                current_overlay = next_overlay
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-filter_complex', ';'.join(filter_parts),
                '-map', f'[{current_overlay}]',
                '-t', str(total_duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-an',  # No audio for individual tracks
                output_path
            ]
            
            self.logger.info(f"Creating triggered track: {track_name}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"✅ Created triggered track: {track_name}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to create triggered track {track_name}: {e}")
            self.logger.error(f"FFmpeg stderr: {e.stderr}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error creating triggered track {track_name}: {e}")
            return None
    
    def composite_triggered_videos(self, triggered_videos, output_path):
        """
        Composite all triggered videos into final grid layout
        
        Args:
            triggered_videos: List of triggered video file paths
            output_path: Final output video path
        """
        if not triggered_videos:
            return False
        
        try:
            # Calculate grid layout
            grid_size = self.calculate_grid_size(len(triggered_videos))
            
            # Build ffmpeg command for grid composition
            cmd = ['ffmpeg', '-y']
            
            # Add all input videos
            for video in triggered_videos:
                cmd.extend(['-i', video])
            
            # Create filter complex for grid layout
            filter_complex = self.build_grid_filter(len(triggered_videos), grid_size)
            
            cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[grid]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-an',  # No audio in final video for now
                output_path
            ])
            
            self.logger.info("Creating final grid composition...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"✅ Created final composed video: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Failed to create final composition: {e}")
            self.logger.error(f"FFmpeg stderr: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error creating final composition: {e}")
            return False
    
    def calculate_grid_size(self, num_videos):
        """Calculate optimal grid size for given number of videos"""
        if num_videos <= 1:
            return (1, 1)
        elif num_videos <= 4:
            return (2, 2)
        elif num_videos <= 9:
            return (3, 3)
        elif num_videos <= 16:
            return (4, 4)
        else:
            return (5, 5)
    
    def build_grid_filter(self, num_videos, grid_size):
        """Build ffmpeg filter for grid layout"""
        rows, cols = grid_size
        cell_width = 640 // cols
        cell_height = 360 // rows
        
        # Scale all inputs to fit grid cells
        scale_filters = []
        for i in range(num_videos):
            scale_filters.append(f"[{i}:v]scale={cell_width}:{cell_height}[v{i}]")
        
        # Create grid layout positions
        layout_parts = []
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if idx < num_videos:
                    x = col * cell_width
                    y = row * cell_height
                    layout_parts.append(f"{x}_{y}")
        
        # Build xstack filter
        input_streams = ''.join(f'[v{i}]' for i in range(num_videos))
        xstack_filter = f"{input_streams}xstack=inputs={num_videos}:layout={'|'.join(layout_parts)}[grid]"
        
        return ';'.join(scale_filters + [xstack_filter])
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
