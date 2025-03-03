import torch
import cv2
import numpy as np
from pathlib import Path
import os
import subprocess
from typing import List, Dict, Tuple
import logging

class GPUPipelineProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pre-allocate reusable tensors
        self.frame_cache = {}
        
    def load_video_frames_to_gpu(self, video_path, start_time=0, duration=None):
        """Load video frames directly to GPU memory without MoviePy"""
        # Use OpenCV to read frames directly
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame ranges
        start_frame = int(start_time * fps)
        if duration:
            end_frame = start_frame + int(duration * fps)
        else:
            end_frame = frame_count
            
        # Seek to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames directly to GPU
        frames = []
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Send directly to GPU
            frame_tensor = torch.from_numpy(frame).to(self.device)
            frames.append(frame_tensor)
            
        cap.release()
        
        # Return stacked tensor of all frames
        if frames:
            return torch.stack(frames)
        return None
    
    def process_chunk_pure_gpu(self, grid_config, output_path, fps=30.0, duration=4.0):
        """Process chunk completely on GPU without CPU transfers"""
        # Determine output dimensions from grid
        rows, cols = len(grid_config), len(grid_config[0])
        h, w = 1080, 1920
        cell_h, cell_w = h // rows, w // cols
        frame_count = int(duration * fps)
        
        # Pre-allocate output tensor
        output_frames = torch.zeros((frame_count, h, w, 3), 
                                    dtype=torch.uint8, 
                                    device=self.device)
        
        # Extract audio tracks
        temp_dir = os.path.join(os.path.dirname(output_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        audio_tracks = []
        
        # Process each cell in the grid
        for row in range(rows):
            for col in range(cols):
                cell_config = grid_config[row][col]
                if not cell_config or cell_config.get('empty', True):
                    continue
                    
                # Get video path and timing
                video_path = cell_config.get('path')
                start_time = cell_config.get('start_time', 0)
                cell_duration = cell_config.get('duration', duration)
                
                if not video_path or not os.path.exists(video_path):
                    continue
                    
                # Process audio
                self._extract_audio(video_path, temp_dir, audio_tracks, 
                                   f"r{row}c{col}_{Path(output_path).stem}")
                
                # Load frames to GPU if not in cache
                cache_key = f"{video_path}:{start_time}:{cell_duration}"
                if cache_key not in self.frame_cache:
                    frames = self.load_video_frames_to_gpu(
                        video_path, start_time, cell_duration)
                    if frames is not None:
                        self.frame_cache[cache_key] = frames
                
                if cache_key in self.frame_cache:
                    cell_frames = self.frame_cache[cache_key]
                    
                    # Calculate frame positions
                    cell_frame_count = min(cell_frames.shape[0], frame_count)
                    
                    # Resize frames to cell size using GPU operations
                    if cell_frames.shape[1:3] != (cell_h, cell_w):
                        cell_frames = torch.nn.functional.interpolate(
                            cell_frames.permute(0,3,1,2),  # NHWC -> NCHW
                            size=(cell_h, cell_w),
                            mode='bilinear'
                        ).permute(0,2,3,1).to(torch.uint8)  # NCHW -> NHWC
                    
                    # Copy frames to output tensor - direct memory operations on GPU
                    for i in range(cell_frame_count):
                        output_frames[i, 
                                     row*cell_h:(row+1)*cell_h, 
                                     col*cell_w:(col+1)*cell_w] = cell_frames[i]
        
        # Mix audio and export video
        mixed_audio = self._mix_audio_tracks(audio_tracks, temp_dir, 
                                           f"mixed_{Path(output_path).stem}.aac")
        
        # Write frames to disk - batched CPU transfer at the end
        self._write_frames_to_disk(output_frames, output_path, fps, mixed_audio)
        
        # Clean up
        for track in audio_tracks:
            if os.path.exists(track):
                try:
                    os.remove(track)
                except Exception:
                    pass
                    
        return output_path
        
    def _extract_audio(self, video_path, temp_dir, audio_tracks, identifier):
        """Extract audio from video to a temporary file"""
        audio_path = os.path.join(temp_dir, f"audio_{identifier}.aac")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn', '-acodec', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode == 0 and os.path.exists(audio_path):
                audio_tracks.append(audio_path)
                return True
        except Exception as e:
            logging.error(f"Audio extraction error: {e}")
        
        return False
        
    def _mix_audio_tracks(self, audio_tracks, temp_dir, output_filename):
        """Mix multiple audio tracks together with FFmpeg"""
        if not audio_tracks:
            return None
            
        if len(audio_tracks) == 1:
            return audio_tracks[0]
            
        output_path = os.path.join(temp_dir, output_filename)
        
        cmd = ['ffmpeg', '-y']
        for track in audio_tracks:
            cmd.extend(['-i', track])
            
        cmd.extend([
            '-filter_complex', f'amix=inputs={len(audio_tracks)}:duration=longest',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
        except Exception as e:
            logging.error(f"Audio mixing error: {e}")
            
        return audio_tracks[0] if audio_tracks else None
        
    def _write_frames_to_disk(self, frames, output_path, fps, audio_path=None):
        """Write frames to disk using NVENC directly"""
        # Create temporary raw video file
        temp_yuv = str(Path(output_path).with_suffix('.yuv'))
        
        try:
            # Transfer to CPU in large batches to minimize transfers
            batch_size = 60  # 2 seconds worth at 30fps
            with open(temp_yuv, 'wb') as f:
                for i in range(0, frames.shape[0], batch_size):
                    end = min(i + batch_size, frames.shape[0])
                    batch = frames[i:end].cpu().numpy()
                    
                    # Convert RGB to YUV420 and write
                    for frame in batch:
                        yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
                        f.write(yuv.tobytes())
            
            # Use FFmpeg to encode with NVENC
            h, w = frames.shape[1:3]
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'yuv420p',
                '-s', f'{w}x{h}',
                '-r', str(fps),
                '-i', temp_yuv
            ]
            
            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                cmd.extend(['-i', audio_path])
                cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
            
            # Add video encoding parameters
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-tune', 'hq',
                '-rc', 'vbr_hq',
                '-cq', '20',
                '-bufsize', '10M',
                '-vsync', 'cfr',
                output_path
            ])
            
            subprocess.run(cmd, check=True)
            
        finally:
            # Clean up
            if os.path.exists(temp_yuv):
                os.remove(temp_yuv)