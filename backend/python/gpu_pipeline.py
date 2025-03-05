import shutil
import time
import traceback
import torch
import cv2
import numpy as np
from pathlib import Path
import os
import subprocess
import logging
from path_registry import PathRegistry

class GPUPipelineProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pre-allocate reusable tensors
        self.frame_cache = {}
        self.registry = PathRegistry.get_instance()
        self.CROSSFADE_DURATION = 0.5
        self.CHUNK_DURATION = 4.0
        
    def load_video_frames_to_gpu(self, video_path, start_time=0, duration=None):
        """Load ALL frames from video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return None
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
                
            # Determine number of frames to read
            if duration is not None:
                frames_to_read = int(duration * fps)
            else:
                frames_to_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
            frames = []
            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Move to GPU
                frame_tensor = torch.from_numpy(frame).to(self.device)
                frames.append(frame_tensor)
                
            cap.release()
            
            if not frames:
                logging.error(f"No frames read from {video_path}")
                return None
                
            frames_tensor = torch.stack(frames)
            logging.info(f"Loaded {len(frames)} frames from {os.path.basename(video_path)}")
            return frames_tensor
            
        except Exception as e:
            logging.error(f"Error loading video: {str(e)}")
            return None
    
    def process_chunk_pure_gpu(self, grid_config, output_path, fps=30.0, duration=4.0, audio_path=None):
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
                    
                # Handle multiple clips in a cell (composites)
                clips_to_process = []
                if 'clips' in cell_config:
                    clips_to_process = cell_config['clips']
                else:
                    clips_to_process = [cell_config]
                    
                for clip in clips_to_process:
                    # Get video path and timing
                    video_path = clip.get('path')
                    start_time = clip.get('start_time', 0)
                    cell_duration = clip.get('duration', duration)
                    offset = clip.get('offset', 0)  # Time offset within chunk
                    
                    if not video_path or not os.path.exists(video_path):
                        continue
                        
                    # Process audio - extract from video file
                    unique_id = f"{row}_{col}_{Path(video_path).stem}_{Path(output_path).stem}"
                    self._extract_audio(video_path, temp_dir, audio_tracks, unique_id, float(offset))
                    
                    # Load frames to GPU
                    frames = self.load_video_frames_to_gpu(
                        video_path, start_time, cell_duration)
                    
                    if frames is not None:
                        frame_count = frames.shape[0]
                        logging.info(f"SUCCESS: Loaded {frame_count} frames from {os.path.basename(video_path)} at position [{row},{col}]")
                    else:
                        logging.error(f"FAILED: Could not load frames from {os.path.basename(video_path)} at position [{row},{col}]")
                        # Try to diagnose why loading failed
                        try:
                            cap = cv2.VideoCapture(video_path)
                            if not cap.isOpened():
                                logging.error(f"  - Video cannot be opened: {video_path}")
                            else:
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                logging.error(f"  - Video opened but no frames read: {video_path}")
                                logging.error(f"  - Stats: FPS={fps}, frames={total_frames}, dimensions={width}x{height}")
                                ret, frame = cap.read()
                                if not ret:
                                    logging.error("  - First frame read failed")
                                else:
                                    logging.error(f"  - First frame read succeeded with shape {frame.shape}")
                                cap.release()
                        except Exception as e:
                            logging.error(f"  - Diagnosis failed: {str(e)}")
                    
                    if frames is not None:
                        # Calculate frame positions
                        offset_frames = int(offset * fps)
                        cell_frame_count = min(frames.shape[0], frame_count - offset_frames)
                        
                        # Resize frames to cell size
                        if frames.shape[1:3] != (cell_h, cell_w):
                            frames = torch.nn.functional.interpolate(
                                frames.permute(0,3,1,2),  # NHWC -> NCHW
                                size=(cell_h, cell_w),
                                mode='bilinear'
                            ).permute(0,2,3,1).to(torch.uint8)  # NCHW -> NHWC
                        
                        # Copy frames to output tensor with proper offset
                        for i in range(min(frames.shape[0], int(duration * fps))):
    # Determine the position in the output timeline
                            output_frame_idx = offset_frames + i
                            
                            # Skip if we're past the end of the chunk
                            if output_frame_idx >= output_frames.shape[0]:
                                break
                                
                            # Place frame in the grid
                            output_frames[output_frame_idx, row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = frames[i]

        # Set chunk duration for audio processing
        self.CHUNK_DURATION = duration
        
        # Mix audio and export video
        mixed_audio = self._mix_audio_tracks(audio_tracks, temp_dir, 
                                           f"mixed_{Path(output_path).stem}.aac")
        
        # Write frames to disk
        # self._write_frames_to_disk(output_frames, output_path, fps, mixed_audio)

        self._write_frames_to_disk(output_frames, output_path, fps)
    
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            self._add_audio_to_video(output_path, audio_path)
        
        
        # Clean up temporary files
        for track in audio_tracks:
            track_path = track if isinstance(track, str) else track.get('path')
            if track_path and os.path.exists(track_path):
                try:
                    os.remove(track_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temp file {track_path}: {e}")
                            
                return output_path
            
    def _add_audio_to_video(self, video_path, audio_path):
        """Add audio to a video file"""
        temp_output = video_path + ".temp.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            temp_output
        ]
        subprocess.run(cmd, check=True)
        shutil.move(temp_output, video_path)

    def batch_process_audio(self, audio_files, output_path):
        """Process multiple audio operations in a single FFmpeg call"""
        if not audio_files:
            return None
            
        cmd = ['ffmpeg', '-y']
        filter_parts = []
        
        # Add all inputs first
        for i, audio_file in enumerate(audio_files):
            # Fix: Check for either 'path' or 'video_path' key
            path = audio_file.get('path')
            if path is None:
                path = audio_file.get('video_path')  # Try alternate key name
                
            if not path:
                logging.warning(f"Skipping audio file with no path: {audio_file}")
                continue
                
            cmd.extend(['-i', path])
            
            # Calculate offset in milliseconds
            delay_ms = int(audio_file.get('offset', 0) * 1000)
            
            # Create input label and apply delay if needed
            if delay_ms > 0:
                filter_parts.append(f'[{i}]adelay={delay_ms}|{delay_ms}[a{i}]')
            else:
                filter_parts.append(f'[{i}]acopy[a{i}]')
        
        # Create mix part with proper crossfading
        inputs = ''.join(f'[a{i}]' for i in range(len(audio_files)))
        filter_parts.append(
            f'{inputs}amix=inputs={len(audio_files)}:duration=longest:normalize=0,'
            f'afade=t=in:st=0:d=0.5,'
            f'afade=t=out:st=3.5:d=0.5'
            '[aout]'
        )
        
        # Complete the command
        cmd.extend([
            '-filter_complex', ';'.join(filter_parts),
            '-map', '[aout]',
            '-c:a', 'aac', '-b:a', '192k',
            output_path
        ])
        
        # Execute in a single call
        subprocess.run(cmd, check=True)
        return output_path
    

    def _mix_audio_tracks(self, audio_tracks, temp_dir, output_filename):
        """Mix audio tracks with proper timing"""
        if not audio_tracks:
            return None
                
        if len(audio_tracks) == 1:
            # If single track, just copy it
            output_path = os.path.join(temp_dir, output_filename)
            shutil.copy2(audio_tracks[0]['path'], output_path)
            return output_path
                
        # For multiple tracks, use complex filter
        output_path = os.path.join(temp_dir, output_filename)
        
        # Prepare ffmpeg command
        cmd = ['ffmpeg', '-y']
        filter_parts = []
        
        # Add inputs and create filter components
        for i, track_info in enumerate(audio_tracks):
            cmd.extend(['-i', track_info['path']])
            delay_ms = int(track_info['offset'] * 1000)
            if delay_ms > 0:
                filter_parts.append(f'[{i}]adelay={delay_ms}|{delay_ms}[a{i}]')
            else:
                filter_parts.append(f'[{i}]acopy[a{i}]')
        
        # Create mix part of filter
        filter_str = f'amix=inputs={len(audio_tracks)}:duration=longest:normalize=0,afade=t=in:st=0:d={self.CROSSFADE_DURATION},afade=t=out:st={self.CHUNK_DURATION-self.CROSSFADE_DURATION}:d={self.CROSSFADE_DURATION}'
        mix_inputs = ''.join(f'[a{i}]' for i in range(len(audio_tracks)))
        filter_parts.append(f'{mix_inputs}amix=inputs={len(audio_tracks)}:duration=longest[out]')
        
        # Complete command
        cmd.extend([
            '-filter_complex', ';'.join(filter_parts),
            '-map', '[out]',
            '-c:a', 'aac', '-b:a', '192k',
            output_path
        ])
        
        try:
            result = subprocess.run(cmd, check=False, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
        except Exception as e:
            logging.error(f"Audio mixing error: {e}")
        
        return None

    def _extract_audio(self, video_path, temp_dir, audio_tracks, identifier, offset=0):
        """Extract audio with proper timing offset"""
        try:
            # Extract base audio
            audio_path = os.path.join(temp_dir, f"audio_{identifier}.wav")
            extract_cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                audio_path
            ]
            subprocess.run(extract_cmd, check=False, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
                        
            if not os.path.exists(audio_path):
                return False
                
            # Add timing information
            audio_tracks.append({
                'path': audio_path,
                'offset': offset
            })
            return True
                
        except Exception as e:
            logging.error(f"Audio extraction error: {e}")
            return False
    

    def get_video_path(self, video_type, *args):
        """Generic video path finder"""
        if video_type == 'drum':
            return self.registry.get_drum_path(args[0])
        elif video_type == 'instrument':
            return self.registry.get_instrument_path(args[0], args[1])
        return None
        
    # def _write_frames_to_disk(self, frames, output_path, fps, audio_path=None):
    #     """Write frames to disk with reliable video+audio sync"""
    #     temp_video = str(Path(output_path).with_suffix('.temp.mp4'))
    def _write_frames_to_disk(self, frames, output_path, fps, audio_path=None):
        """Write frames to disk with optimized GPU→CPU transfers and reduced memory pressure"""
        temp_video = str(Path(output_path).with_suffix('.temp.mp4'))
        start_time = time.time()
        
        try:
            # Use CUDA-direct memory access if possible
            use_cuda_direct = torch.cuda.is_available()
            
            if use_cuda_direct:
                # Pin memory for faster transfers and make contiguous
                frames = frames.contiguous()
                
                # Better NVENC parameters for higher throughput
                video_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{frames.shape[2]}x{frames.shape[1]}',
                    '-pix_fmt', 'rgb24',
                    '-r', str(fps),
                    '-i', 'pipe:',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',  # Performance preset
                    '-gpu', '0',
                    '-surfaces', '64',  # Increased from 32 for better parallelism
                    '-tune', 'hq',
                    '-rc', 'vbr',
                    '-b:v', '8M',
                    '-bufsize', '16M',  # Larger buffer for smoother encoding
                    '-pix_fmt', 'yuv420p',
                    '-g', '30',  # GOP size matching fps for better streaming
                    '-threads', '8',  # Use more threads
                    temp_video
                ]
                
                # Increased batch size for better throughput
                batch_size = 60  # Up from 30
                
                # Create a dedicated CUDA stream for the transfer
                with torch.cuda.stream(torch.cuda.Stream(priority=0)):  # High priority stream
                    proc = subprocess.Popen(video_cmd, stdin=subprocess.PIPE)
                    
                    # Pre-allocate a buffer for batches to avoid reallocations
                    # Process frames in larger batches with progress logging
                    total_frames = frames.shape[0]
                    bytes_written = 0
                    
                    for i in range(0, total_frames, batch_size):
                        end_idx = min(i + batch_size, total_frames)
                        batch = frames[i:end_idx]
                        
                        # Use pinned memory for faster CPU→GPU transfer
                        with torch.cuda.device(self.device):
                            cpu_batch = batch.cpu()
                            
                        # Convert to numpy with optimal memory layout
                        frame_bytes = cpu_batch.numpy().astype(np.uint8, order='C').tobytes()
                        bytes_written += len(frame_bytes)
                        
                        # Write to FFmpeg
                        proc.stdin.write(frame_bytes)
                        
                        # Log progress for every ~25% completion
                        if i % max(1, total_frames // 4) == 0 or i + batch_size >= total_frames:
                            logging.info(f"Writing frames: {min(i + batch_size, total_frames)}/{total_frames} ({min(100, int((i + batch_size) / total_frames * 100))}%)")
                    
                    # Properly close stdin and wait for process to complete
                    proc.stdin.close()
                    returncode = proc.wait()
                    
                    if returncode != 0:
                        logging.error(f"FFmpeg encoding failed with code {returncode}")
                        raise RuntimeError(f"FFmpeg encoding failed with code {returncode}")
                    
                    logging.info(f"Video encoding completed in {time.time() - start_time:.2f}s, wrote {bytes_written/(1024*1024):.1f}MB")
            else:
                # Fallback non-CUDA path
                # [Existing non-CUDA implementation]
                try:
                    # First pass: write video with raw encoding
                    video_cmd = [
                        'ffmpeg', '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-s', f'{frames.shape[2]}x{frames.shape[1]}',
                        '-pix_fmt', 'rgb24',
                        '-r', str(fps),
                        '-i', '-',  # Read from stdin
                        '-an',      # No audio in first pass
                        '-c:v', 'h264_nvenc' if torch.cuda.is_available() else 'libx264',
                        '-preset', 'fast',
                        '-threads', '8',
                        '-pix_fmt', 'yuv420p',
                        temp_video
                    ]
                    
                    logging.info(f"Writing {frames.shape[0]} frames to {temp_video}")
                    
                    proc = subprocess.Popen(
                        video_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Process frames with better memory handling
                    for i in range(frames.shape[0]):
                        # Log progress every ~25%
                        if i % max(1, frames.shape[0] // 4) == 0:
                            logging.info(f"Processing frame {i}/{frames.shape[0]}")
                            
                        try:
                            # Get frame data as contiguous bytes
                            frame_data = frames[i].cpu().numpy().astype(np.uint8, order='C')
                            proc.stdin.write(frame_data.tobytes())
                        except Exception as e:
                            logging.error(f"Error writing frame {i}: {e}")
                            break
                            
                    # Properly close stdin and wait
                    proc.stdin.close()
                    stdout, stderr = proc.communicate()
                    
                    if proc.returncode != 0:
                        logging.error(f"FFmpeg video encoding error: {stderr.decode() if stderr else 'Unknown error'}")
                        raise RuntimeError("FFmpeg encoding failed")
                except Exception as e:
                    logging.error(f"Error in _write_frames_to_disk: {e}", exc_info=True)
                    if os.path.exists(temp_video) and not os.path.exists(output_path):
                        shutil.copy2(temp_video, output_path)
                    return output_path
                    
            # Second pass: add audio if available - using optimized parameters
            if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logging.info(f"Adding audio {audio_path} to {output_path}")
                audio_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', audio_path,
                    '-c:v', 'copy',      # Copy video to avoid re-encoding
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-af', 'aresample=async=1000',  # Better AV sync
                    '-shortest',
                    output_path
                ]
                
                result = subprocess.run(audio_cmd, capture_output=True)
                if result.returncode != 0:
                    logging.error(f"Audio muxing failed: {result.stderr.decode()}")
                    # Fall back to video-only if audio fails
                    shutil.copy2(temp_video, output_path)
                else:
                    logging.info(f"Successfully created video with audio: {output_path}")
            else:
                # Just copy if no audio
                shutil.copy2(temp_video, output_path)
                
            return output_path
                
        except Exception as e:
            logging.error(f"Error in _write_frames_to_disk: {e}", exc_info=True)
            if os.path.exists(temp_video) and not os.path.exists(output_path):
                shutil.copy2(temp_video, output_path)
            return output_path
        finally:
            # Clean up temp file
            if os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                except Exception as e:
                    logging.warning(f"Failed to remove temp file {temp_video}: {e}")
                    
            # Log total processing time
            logging.info(f"Total frame writing process took {time.time() - start_time:.2f}s")
