#!/usr/bin/env python3
"""
Simple Video Compositor
Creates video grid layouts without complex processing
"""

import subprocess
import os
import tempfile
import logging
from pathlib import Path

class SimpleVideoCompositor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)
    
    def create_simple_grid(self, video_files, output_path, grid_size=(3, 3), duration=4.0):
        """
        Create a simple grid layout from video files
        
        Args:
            video_files: List of video file paths
            output_path: Output video file path
            grid_size: Tuple of (rows, cols) for grid layout
            duration: Duration of output video
        """
        if not video_files:
            self.logger.error("No video files provided")
            return False
        
        # Filter out files that don't have video streams
        valid_video_files = []
        for video_file in video_files:
            if not os.path.exists(video_file):
                self.logger.warning(f"Video file not found: {video_file}")
                continue
            
            # Check if file has video stream
            try:
                import subprocess
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'v:0',
                    '-of', 'csv=p=0', video_file
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    valid_video_files.append(video_file)
                else:
                    self.logger.warning(f"File has no video stream: {video_file}")
            except Exception as e:
                self.logger.warning(f"Error checking video file {video_file}: {e}")
        
        if not valid_video_files:
            self.logger.error("No valid video files found")
            return False
        
        # Limit to grid size
        max_videos = grid_size[0] * grid_size[1]
        valid_video_files = valid_video_files[:max_videos]
        
        self.logger.info(f"Creating grid with {len(valid_video_files)} videos, size: {grid_size}")
        
        cmd = ['ffmpeg', '-y']
        
        # Add input files
        for video_file in valid_video_files:
            cmd.extend(['-i', video_file])
        
        # Create simple grid filter
        cell_width = 640 // grid_size[1]
        cell_height = 360 // grid_size[0]
        
        # Scale inputs
        scale_filters = []
        for i in range(len(valid_video_files)):
            scale_filters.append(f'[{i}:v]scale={cell_width}:{cell_height}[v{i}]')
        
        # Create grid layout
        grid_inputs = ''.join(f'[v{i}]' for i in range(len(valid_video_files)))
        
        # Build layout string
        layout_parts = []
        for i, video_file in enumerate(valid_video_files):
            row = i // grid_size[1]
            col = i % grid_size[1]
            x = col * cell_width
            y = row * cell_height
            layout_parts.append(f'{x}_{y}')
        
        layout = '|'.join(layout_parts)
        
        # Combine filters
        if len(valid_video_files) == 1:
            # Single video, just scale it
            filter_complex = scale_filters[0].replace(f'[v0]', '[video_out]')
        else:
            # Multiple videos, use xstack
            filter_complex = ';'.join(scale_filters)
            filter_complex += f';{grid_inputs}xstack=inputs={len(valid_video_files)}:layout={layout}[video_out]'
        
        # Add audio mixing only if there are audio streams
        audio_inputs = []
        for i, video_file in enumerate(valid_video_files):
            # Check if video has audio stream
            try:
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a:0',
                    '-of', 'csv=p=0', video_file
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    audio_inputs.append(f'[{i}:a]')
            except:
                pass
        
        if audio_inputs:
            filter_complex += f';{"".join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=longest[audio_out]'
            cmd.extend(['-filter_complex', filter_complex])
            cmd.extend(['-map', '[video_out]'])
            cmd.extend(['-map', '[audio_out]'])
        else:
            # No audio, just video
            cmd.extend(['-filter_complex', filter_complex])
            cmd.extend(['-map', '[video_out]'])
            cmd.extend(['-an'])  # No audio
        
        cmd.extend(['-c:v', 'libx264'])
        if audio_inputs:
            cmd.extend(['-c:a', 'aac'])
        cmd.extend(['-t', str(duration)])
        cmd.append(output_path)
        
        try:
            self.logger.info(f"Running FFmpeg command: {' '.join(cmd[:15])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"✅ Grid video created successfully: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Simple compositor failed: {e}")
            self.logger.error(f"Command: {' '.join(cmd)}")
            self.logger.error(f"FFmpeg stderr: {e.stderr}")
            return False
    
    def test_with_sample_videos(self, video_dir):
        """
        Test compositor with sample videos
        
        Args:
            video_dir: Directory containing video files
        """
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(Path(video_dir).glob(ext))
        
        if not video_files:
            self.logger.warning("No video files found for testing")
            return False
        
        video_files = [str(f) for f in video_files[:9]]  # Max 9 for 3x3 grid
        output_path = os.path.join(self.temp_dir, 'test_output.mp4')
        
        self.logger.info(f"Testing with {len(video_files)} video files")
        success = self.create_simple_grid(video_files, output_path)
        if success:
            self.logger.info(f"✅ Test video created: {output_path}")
            return output_path
        else:
            self.logger.error("❌ Test video creation failed")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
