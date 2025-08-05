# CRITICAL PERFORMANCE FIXES FOR AUTOTUNESYNCER GPU PIPELINE
# This file contains the complete working implementation with all performance fixes applied

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
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None
try:
    from .path_registry import PathRegistry
except ImportError:
    from path_registry import PathRegistry

class GPUPipelineProcessor:
    def __init__(self, composer = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Pre-allocate reusable tensors and caching
        self.frame_cache = {}
        self.tensor_cache = {}
        self.max_cache_size = 50  # Limit cache size
        
        # Enhanced GPU stream management
        if torch.cuda.is_available():
            self.num_streams = min(4, torch.cuda.get_device_properties(0).multi_processor_count // 4)
            self.cuda_streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
            self.current_stream = 0
            logging.info(f"Initialized {self.num_streams} CUDA streams")
        else:
            self.cuda_streams = []
            self.num_streams = 0
            
        self.registry = PathRegistry.get_instance()
        self.CROSSFADE_DURATION = 0.5
        self.CHUNK_DURATION = 4.0
        self.composer = composer
        
        # Performance optimization parameters
        self.enable_tensor_reuse = True
        self.async_processing = torch.cuda.is_available()
        self.batch_processing = True
        
        # CRITICAL FIX: Initialize enhanced memory management
        self.memory_pressure_threshold = 0.85  # 85% memory usage triggers cleanup
        self.last_memory_check = 0
        self.memory_check_interval = 5.0  # Check every 5 seconds
        
        # Initialize optimization features
        self.initialize_memory_pools()
        self.optimize_gpu_utilization()
        
        # Performance tracking
        self.performance_stats = {
            'frames_processed': 0,
            'memory_cleanups': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logging.info(f"GPUPipelineProcessor initialized on {self.device} with enhanced memory management")
        
    def load_video_frames_to_gpu(self, video_path, start_time=0, duration=None):
        """Load ALL frames from video file with enhanced caching and performance optimization"""
        # Generate cache key
        cache_key = f"{video_path}:{start_time}:{duration}"
        
        # Check cache first
        if cache_key in self.frame_cache:
            logging.info(f"Frame cache hit for {os.path.basename(video_path)}")
            self.performance_stats['cache_hits'] += 1
            return self.frame_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        
        # Clear cache if needed to prevent memory issues
        self.clear_cache_if_needed()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return None
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
                
            # Skip to start time if specified
            if start_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                
            # Determine number of frames to read
            if duration is not None:
                frames_to_read = int(duration * fps)
            else:
                frames_to_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
            frames = []
            
            # Use async processing if available
            stream = self.get_next_cuda_stream() if self.async_processing else None
            
            # Pre-allocate tensor if we know the frame count
            if frames_to_read > 0:
                # Read first frame to get dimensions
                ret, first_frame = cap.read()
                if ret:
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    h, w, c = first_frame.shape
                    
                    # Pre-allocate tensor on GPU for better performance
                    if stream:
                        with torch.cuda.stream(stream):
                            frames_tensor = torch.zeros((frames_to_read, h, w, c), 
                                                       dtype=torch.uint8, 
                                                       device=self.device)
                    else:
                        frames_tensor = torch.zeros((frames_to_read, h, w, c), 
                                                   dtype=torch.uint8, 
                                                   device=self.device)
                    
                    # Add first frame
                    first_frame_tensor = torch.from_numpy(first_frame)
                    first_frame_tensor = self.optimize_tensor_for_gpu(first_frame_tensor)
                    frames_tensor[0] = first_frame_tensor
                    
                    # Read remaining frames efficiently
                    frame_idx = 1
                    while frame_idx < frames_to_read:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Convert to RGB and optimize for GPU
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame)
                        frame_tensor = self.optimize_tensor_for_gpu(frame_tensor)
                        frames_tensor[frame_idx] = frame_tensor
                        frame_idx += 1
                    
                    # Trim tensor if we read fewer frames than expected
                    if frame_idx < frames_to_read:
                        frames_tensor = frames_tensor[:frame_idx]
                    
                    cap.release()
                    
                    # Convert to float for processing
                    frames_tensor = frames_tensor.float()
                    
                    # Cache the result
                    if self.enable_tensor_reuse:
                        self.frame_cache[cache_key] = frames_tensor
                    
                    self.performance_stats['frames_processed'] += frame_idx
                    logging.info(f"Loaded {frame_idx} frames from {os.path.basename(video_path)}")
                    return frames_tensor
                else:
                    cap.release()
                    logging.error(f"Could not read first frame from {video_path}")
                    return None
            else:
                cap.release()
                logging.error(f"No frames to read from {video_path}")
                return None
                
        except Exception as e:
            logging.error(f"Error loading video: {str(e)}")
            return None

    def process_chunk_pure_gpu(self, grid_config, output_path, fps=30.0, duration=4.0, audio_path=None):
        """Process chunk completely on GPU without CPU transfers - WITH CRITICAL PERFORMANCE FIXES"""
        
        # CRITICAL FIX: Memory pressure monitoring before processing
        if not self.check_and_manage_gpu_memory():
            logging.error("Insufficient GPU memory for processing, falling back to CPU")
            self.device = torch.device('cpu')
        
        # Determine output dimensions from grid
        rows, cols = len(grid_config), len(grid_config[0])
        h, w = 1080, 1920
        cell_h, cell_w = h // rows, w // cols
        frame_count = int(duration * fps)
        
        # CRITICAL FIX: Check GPU memory before allocation to prevent CUDA OOM
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()  # Clear any cached memory
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                required_memory = frame_count * h * w * 3  # bytes for uint8 tensor
                
                # Safety margin - only use 70% of available memory
                safe_memory = free_memory * 0.7
                
                logging.info(f"Memory check: Required={required_memory/(1024*1024):.1f}MB, Available={free_memory/(1024*1024):.1f}MB, Safe limit={safe_memory/(1024*1024):.1f}MB")
                
                if required_memory > safe_memory:
                    # Reduce frame count to fit available memory
                    max_frames = int(safe_memory / (h * w * 3))
                    if max_frames < 30:  # Minimum 1 second at 30fps
                        logging.error(f"Insufficient GPU memory: only {max_frames} frames possible")
                        raise RuntimeError(f"GPU memory insufficient: need {required_memory/(1024*1024):.1f}MB, have {free_memory/(1024*1024):.1f}MB")
                    
                    frame_count = min(frame_count, max_frames)
                    duration = frame_count / fps  # Adjust duration to match frame count
                    logging.warning(f"Reduced frame count to {frame_count} ({duration:.1f}s) due to memory constraints")
                    
            except Exception as e:
                logging.warning(f"GPU memory check failed: {e}")
        
        # Pre-allocate output tensor with validated size
        try:
            output_frames = torch.zeros((frame_count, h, w, 3), 
                                        dtype=torch.uint8, 
                                        device=self.device)
            logging.info(f"Successfully allocated output tensor: {frame_count}x{h}x{w}x3 = {output_frames.numel() * output_frames.element_size() / (1024*1024):.1f}MB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(f"CUDA OOM during tensor allocation: {e}")
                # Fallback to CPU processing
                self.device = torch.device('cpu')
                output_frames = torch.zeros((frame_count, h, w, 3), dtype=torch.uint8, device=self.device)
                logging.warning("Fell back to CPU processing due to GPU memory constraints")
            else:
                raise
        
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

                    # Use a SINGLE duration for both audio and video
                    clip_duration = clip.get('duration', duration) # Use base duration
                    offset = clip.get('offset', 0)  # Time offset within chunk

                    if not video_path or not os.path.exists(video_path):
                        continue

                    # Process audio - extract from video file with correct duration
                    unique_id = f"{row}_{col}_{Path(video_path).stem}_{Path(output_path).stem}"
                    self._extract_audio(video_path, temp_dir, audio_tracks, unique_id, float(offset),
                                    duration=clip_duration)  # Pass audio duration here

                    # Load frames with extended video duration
                    frames = self.load_video_frames_to_gpu(
                        video_path, start_time, clip_duration)  # Use video duration here

                    
                    if frames is not None:
                        frame_count_loaded = frames.shape[0]
                        logging.info(f"SUCCESS: Loaded {frame_count_loaded} frames from {os.path.basename(video_path)} at position [{row},{col}]")
                    else:
                        logging.error(f"FAILED: Could not load frames from {os.path.basename(video_path)} at position [{row},{col}]")
                        # Try to diagnose why loading failed
                        try:
                            cap = cv2.VideoCapture(video_path)
                            if not cap.isOpened():
                                logging.error(f"  - Video cannot be opened: {video_path}")
                            else:
                                fps_check = cap.get(cv2.CAP_PROP_FPS)
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                logging.error(f"  - Video opened but no frames read: {video_path}")
                                logging.error(f"  - Stats: FPS={fps_check}, frames={total_frames}, dimensions={width}x{height}")
                                ret, frame = cap.read()
                                if not ret:
                                    logging.error("  - First frame read failed")
                                else:
                                    logging.error(f"  - First frame read succeeded with shape {frame.shape}")
                                cap.release()
                        except Exception as e:
                            logging.error(f"  - Diagnosis failed: {str(e)}")
                    
                    if frames is not None:
                        # CRITICAL FIX: Improved frame positioning with proper bounds validation
                        offset_frames = int(offset * fps)
                        available_output_frames = frame_count - offset_frames
                        
                        # Validate offset is within bounds
                        if offset_frames < 0:
                            logging.warning(f"Negative offset {offset}s -> {offset_frames} frames, clamping to 0")
                            offset_frames = 0
                        elif offset_frames >= frame_count:
                            logging.warning(f"Offset {offset}s -> {offset_frames} frames exceeds output length {frame_count}, skipping clip")
                            continue
                        
                        # Calculate how many frames we can actually copy
                        source_frame_count = frames.shape[0]
                        copyable_frames = min(source_frame_count, available_output_frames)
                        
                        if copyable_frames <= 0:
                            logging.warning(f"No frames to copy: source={source_frame_count}, available_output={available_output_frames}, offset={offset_frames}")
                            continue
                        
                        logging.info(f"Frame copy plan: source={source_frame_count}, offset={offset_frames}, copyable={copyable_frames}, output_size={frame_count}")

                        # Convert to float32 before interpolation
                        frames = frames.float()
                        
                        # Resize frames to cell size
                        if frames.shape[1:3] != (cell_h, cell_w):
                            frames = torch.nn.functional.interpolate(
                                frames.permute(0,3,1,2),  # NHWC -> NCHW
                                size=(cell_h, cell_w),
                                mode='bilinear'
                            ).permute(0,2,3,1).to(torch.uint8)  # NCHW -> NHWC
                        
                        # Ensure frames are uint8 for proper copying
                        frames = frames.byte()
                        
                        # FIXED: Copy frames with guaranteed bounds checking
                        for i in range(copyable_frames):
                            output_frame_idx = offset_frames + i
                            
                            # Double-check bounds (should never trigger due to copyable_frames calculation)
                            if output_frame_idx >= frame_count:
                                logging.error(f"INTERNAL ERROR: frame index {output_frame_idx} >= {frame_count}")
                                break
                            
                            if i >= source_frame_count:
                                logging.error(f"INTERNAL ERROR: source index {i} >= {source_frame_count}")
                                break
                            
                            # Validate grid positioning 
                            row_start, row_end = row * cell_h, (row + 1) * cell_h
                            col_start, col_end = col * cell_w, (col + 1) * cell_w
                            
                            if row_end > h or col_end > w:
                                logging.error(f"Grid cell out of bounds: row=[{row_start}:{row_end}], col=[{col_start}:{col_end}], output=[{h}x{w}]")
                                continue
                            
                            # Safe frame copy with error handling
                            try:
                                output_frames[output_frame_idx, row_start:row_end, col_start:col_end] = frames[i]
                            except Exception as e:
                                logging.error(f"Frame copy failed: output_idx={output_frame_idx}, source_idx={i}, grid=[{row},{col}], error={e}")
                                continue
                        
                        logging.info(f"Successfully copied {copyable_frames} frames to grid position [{row},{col}] with offset {offset_frames}")

        # Set chunk duration for audio processing
        self.CHUNK_DURATION = duration
        
        # Mix audio and export video
        mixed_audio = self._mix_audio_tracks(audio_tracks, temp_dir, 
                                           f"mixed_{Path(output_path).stem}.aac")
        
        # Write frames to disk
        self._write_frames_to_disk(output_frames, output_path, fps)
    
        # Add audio - prioritize mixed_audio from MIDI processing over external audio_path
        audio_to_add = None
        if mixed_audio and os.path.exists(mixed_audio):
            audio_to_add = mixed_audio
            logging.info(f"Using mixed audio from MIDI processing: {mixed_audio}")
        elif audio_path and os.path.exists(audio_path):
            audio_to_add = audio_path
            logging.info(f"Using external audio path: {audio_path}")
        
        if audio_to_add:
            self._add_audio_to_video(output_path, audio_to_add)
            logging.info(f"Successfully added audio to video: {audio_to_add}")
        else:
            logging.warning("No audio tracks found - video will be silent")
        
        
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

    def process_audio_operation(self, op, temp_dir):
        """Process a single audio operation using VideoComposer's extraction method"""
        # Generate a unique output path for this audio sample
        output_path = os.path.join(
            temp_dir, 
            f"audio_{op['position'][0]}_{op['position'][1]}_{os.path.basename(op['video_path']).split('.')[0]}.wav"
        )
        
        # Extract audio using the composer's method
        success = self.composer._extract_audio_with_offset(
            op['video_path'],
            output_path,
            op['audio_duration'],
            sample_offset=op.get('sample_offset', 0),
            volume=op.get('volume', 1.0)
        )
        
        if success:
            return output_path
        return None

    def batch_process_audio(self, audio_files, output_path):
        """Process multiple audio operations by first extracting audio from video files"""
        if not audio_files:
            logging.warning("No audio files provided to batch_process_audio")
            return None

        logging.info(f"Processing {len(audio_files)} audio operations")
        
        # Create a temporary directory for extracted audio
        temp_dir = Path(output_path).parent / "temp_audio"
        temp_dir.mkdir(exist_ok=True)
        
        extracted_audio_files = []
        
        # First, extract audio from all video files
        for i, audio_file in enumerate(audio_files):
            # Get video path - try different keys
            video_path = audio_file.get('video_path') or audio_file.get('path')
            
            if not video_path or not os.path.exists(video_path):
                logging.warning(f"Skipping audio file with invalid path: {audio_file}")
                continue
            
            # Extract audio to temporary file
            audio_output = temp_dir / f"extracted_{i}.wav"
            extract_cmd = [
                'ffmpeg', '-y',                '-i', video_path,
                '-vn',  # No video
                '-c:a', 'pcm_s16le',  # PCM audio for best quality
                '-ar', '44100',  # 44.1 kHz
                '-ac', '2',  # Stereo
                str(audio_output)
            ]
            try:
                result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True,
                                      encoding='utf-8', errors='replace')
                if os.path.exists(audio_output) and os.path.getsize(audio_output) > 0:
                    extracted_audio_files.append({
                        'path': str(audio_output),
                        'offset': audio_file.get('offset', 0),
                        'duration': audio_file.get('audio_duration', audio_file.get('duration', 4.0)),
                        'volume': audio_file.get('volume', 1.0)
                    })
                    logging.info(f"Extracted audio from {video_path} -> {audio_output}")
                else:
                    logging.warning(f"Failed to extract audio from {video_path} - output file empty or missing")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to extract audio from {video_path}: {e.stderr}")
                continue
        
        if not extracted_audio_files:
            logging.warning("No audio files were successfully extracted")
            return None
        
        # Now mix all extracted audio files
        cmd = ['ffmpeg', '-y']
        filter_parts = []
        
        # Add all extracted audio inputs
        for i, audio_info in enumerate(extracted_audio_files):
            cmd.extend(['-i', audio_info['path']])
            
            # Calculate offset in milliseconds
            delay_ms = int(audio_info['offset'] * 1000)
            volume = audio_info.get('volume', 1.0)
            
            # Apply delay and volume adjustments
            filter_chain = f'[{i}]'
            if delay_ms > 0:
                filter_chain += f'adelay={delay_ms}|{delay_ms},'
            if volume != 1.0:
                filter_chain += f'volume={volume},'
            filter_chain += f'aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]'
            filter_parts.append(filter_chain)

        # Create mix part with proper crossfading
        if len(extracted_audio_files) > 1:
            inputs = ''.join(f'[a{i}]' for i in range(len(extracted_audio_files)))
            filter_parts.append(
                f'{inputs}amix=inputs={len(extracted_audio_files)}:duration=longest:normalize=0,'
                f'afade=t=in:st=0:d=0.1,'
                f'afade=t=out:st=3.9:d=0.1'
                '[aout]'
            )
            map_output = '[aout]'
        else:
            # Single audio file case
            map_output = '[a0]'

        # Complete the command
        cmd.extend([
            '-filter_complex', ';'.join(filter_parts),            '-map', map_output,
            '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
            output_path
        ])
        
        try:
            logging.info(f"Mixing audio with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                  encoding='utf-8', errors='replace')
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(f"Successfully created mixed audio: {output_path} ({os.path.getsize(output_path)} bytes)")
                
                # Clean up temporary files
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logging.warning(f"Failed to clean up temp directory: {e}")
                
                return output_path
            else:
                logging.error("Mixed audio file was not created or is empty")
                return None
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Audio mixing failed: {e.stderr}")
            return None
    

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
        filter_str = f'amix=inputs={len(audio_tracks)}:duration=longest:normalize=0,afade=t=in:st=0:d=0.2,afade=t=out:st={self.CHUNK_DURATION-0.2}:d=0.2'
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
        

    def _extract_audio(self, video_path, temp_dir, audio_tracks, identifier, offset=0, duration=None, volume=1.0, autotuned_audio_path=None):
        """Extract audio with proper timing offset and duration"""
        try:
            # Use autotuned audio if available
            if autotuned_audio_path and os.path.exists(autotuned_audio_path):
                audio_path = autotuned_audio_path
            else:
                # Extract base audio with volume adjustment
                audio_path = os.path.join(temp_dir, f"audio_{identifier}.wav")
                extract_cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-vn', '-af', f'volume={volume}',  # Add volume filter
                    '-acodec', 'pcm_s16le'
                ]
                
                # Add duration parameter if specified
                if duration:
                    extract_cmd.extend(['-t', str(duration)])
                    
                extract_cmd.append(audio_path)
            
            try:
                subprocess.run(extract_cmd, check=True, 
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
            except subprocess.CalledProcessError as e:
                logging.error(f"Audio extraction failed: {e.stderr.decode()}")
                return False
                
        except Exception as e:
            logging.error(f"Audio extraction error: {e}")
            return None

    def get_video_path(self, video_type, *args):
        """Generic video path finder"""
        if video_type == 'drum':
            return self.registry.get_drum_path(args[0])
        elif video_type == 'instrument':
            return self.registry.get_instrument_path(args[0], args[1])
        return None
        
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
                
                # Increased batch size for better throughput but use memory-aware sizing
                batch_size = self.get_optimal_batch_size(60)  # Default 60, but adjust based on available memory
                
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

    # CRITICAL PERFORMANCE FIX METHODS

    def get_next_cuda_stream(self):
        """Get next available CUDA stream for parallel processing"""
        if not self.cuda_streams:
            return None
        stream = self.cuda_streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream
    
    def clear_cache_if_needed(self):
        """Clear caches if they exceed maximum size to prevent memory issues"""
        if len(self.frame_cache) > self.max_cache_size:
            # Clear oldest entries (simple LRU-like behavior)
            items_to_remove = len(self.frame_cache) - self.max_cache_size + 10
            keys_to_remove = list(self.frame_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.frame_cache[key]
            logging.info(f"Cleared {items_to_remove} entries from frame cache")
            self.performance_stats['memory_cleanups'] += 1
        
        if len(self.tensor_cache) > self.max_cache_size:
            items_to_remove = len(self.tensor_cache) - self.max_cache_size + 10
            keys_to_remove = list(self.tensor_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.tensor_cache[key]
            logging.info(f"Cleared {items_to_remove} entries from tensor cache")
    
    def optimize_tensor_for_gpu(self, tensor):
        """Optimize tensor for GPU processing with memory efficiency"""
        if not torch.cuda.is_available():
            return tensor
            
        # Use memory-efficient operations
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pin memory for faster transfers if on CPU
        if tensor.device.type == 'cpu':
            tensor = tensor.pin_memory()
            
        return tensor.to(self.device, non_blocking=True)
    
    def initialize_memory_pools(self):
        """Initialize tensor pools for efficient memory management"""
        if torch.cuda.is_available():
            try:
                # Set memory fraction to prevent OOM errors
                torch.cuda.set_per_process_memory_fraction(0.8)
                
                # Enable memory pooling
                torch.cuda.empty_cache()
                
                # Pre-allocate common tensor sizes as reusable pool
                common_sizes = [
                    (30, 1080, 1920, 3),  # 1 second of 1080p video at 30fps
                    (120, 1080, 1920, 3), # 4 seconds of 1080p video
                    (60, 540, 960, 3),    # 2 seconds of 540p video
                ]
                
                self.tensor_pool = {}
                for size in common_sizes:
                    tensor = torch.zeros(size, dtype=torch.uint8, device=self.device)
                    self.tensor_pool[size] = tensor
                
                logging.info(f"Tensor pools initialized with {len(self.tensor_pool)} common sizes")
                
            except Exception as e:
                logging.warning(f"Could not initialize tensor pools: {str(e)}")
                self.tensor_pool = {}
        else:
            self.tensor_pool = {}
    
    def get_pooled_tensor(self, shape):
        """Get a pre-allocated tensor from tensor pool if available"""
        if shape in self.tensor_pool:
            tensor = self.tensor_pool[shape]
            # Zero out the tensor for reuse
            tensor.zero_()
            return tensor
        else:
            # Create new tensor if not in pool
            return torch.zeros(shape, dtype=torch.uint8, device=self.device)
    
    def optimize_gpu_utilization(self):
        """Optimize GPU settings for maximum utilization"""
        if torch.cuda.is_available():
            try:
                # Enable TensorFloat-32 (TF32) for better performance on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Enable cuDNN benchmarking for consistent input sizes
                torch.backends.cudnn.benchmark = True
                
                # Set optimal number of threads
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(min(8, os.cpu_count()))
                
                # Optimize memory allocation
                torch.cuda.empty_cache()
                
                logging.info("GPU utilization optimized")
                
            except Exception as e:
                logging.warning(f"Could not optimize GPU utilization: {str(e)}")
                
        # Optimize OpenCV threading
        cv2.setNumThreads(min(4, os.cpu_count()))
    
    def cleanup_resources(self):
        """Clean up GPU resources and tensor pools"""
        try:
            # Clear caches
            if hasattr(self, 'frame_cache'):
                self.frame_cache.clear()
            if hasattr(self, 'tensor_cache'):
                self.tensor_cache.clear()
            
            # Clear tensor pool
            if hasattr(self, 'tensor_pool'):
                for tensor in self.tensor_pool.values():
                    del tensor
                self.tensor_pool.clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logging.info("GPU resources cleaned up successfully")
            
        except Exception as e:
            logging.warning(f"Error during resource cleanup: {str(e)}")
    
    def check_and_manage_gpu_memory(self):
        """Monitor GPU memory and perform cleanup if needed - CRITICAL FIX"""
        if not torch.cuda.is_available():
            return True
        
        try:
            # Get memory info
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - allocated_memory
            
            memory_usage_percent = (allocated_memory / total_memory) * 100
            
            logging.info(f"GPU Memory Status: {allocated_memory/(1024*1024):.1f}MB allocated, "
                        f"{cached_memory/(1024*1024):.1f}MB cached, "
                        f"{free_memory/(1024*1024):.1f}MB free ({memory_usage_percent:.1f}% used)")
            
            # If memory usage is high, perform cleanup
            if memory_usage_percent > self.memory_pressure_threshold * 100:
                logging.warning(f"High GPU memory usage ({memory_usage_percent:.1f}%), performing cleanup")
                self.force_memory_cleanup()
                
                # Check again after cleanup
                new_allocated = torch.cuda.memory_allocated()
                freed_memory = allocated_memory - new_allocated
                logging.info(f"Cleanup freed {freed_memory/(1024*1024):.1f}MB")
                
                return new_allocated / total_memory < 0.9  # Allow up to 90% usage
            
            return True
            
        except Exception as e:
            logging.warning(f"GPU memory check failed: {e}")
            return True
    
    def force_memory_cleanup(self):
        """Aggressive memory cleanup when under pressure - CRITICAL FIX"""
        try:
            # Clear all caches
            self.frame_cache.clear()
            self.tensor_cache.clear()
            
            # Clear tensor pool if it exists
            if hasattr(self, 'tensor_pool'):
                for tensor in self.tensor_pool.values():
                    del tensor
                self.tensor_pool.clear()
                
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            self.performance_stats['memory_cleanups'] += 1
            logging.info("Performed aggressive memory cleanup")
            
        except Exception as e:
            logging.error(f"Error during memory cleanup: {e}")

    def get_optimal_batch_size(self, base_size=30):
        """Calculate optimal batch size based on available GPU memory - CRITICAL FIX"""
        if not torch.cuda.is_available():
            return base_size
        
        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            
            # Estimate memory per frame (1920x1080x3 RGB)
            memory_per_frame = 1920 * 1080 * 3  # bytes
            
            # Use 50% of available memory for batching
            safe_memory = free_memory * 0.5
            max_batch_size = int(safe_memory / memory_per_frame)
            
            # Clamp to reasonable range
            optimal_batch_size = max(5, min(max_batch_size, base_size * 2))
            
            logging.info(f"Optimal batch size: {optimal_batch_size} frames (based on {free_memory/(1024*1024):.1f}MB free)")
            return optimal_batch_size
            
        except Exception as e:
            logging.warning(f"Could not calculate optimal batch size: {e}")
            return base_size

    def validate_and_fix_frame_counts(self, video_configs, expected_duration, fps):
        """Validate video frame counts and fix mismatches before processing"""
        fixed_configs = []
        total_adjustments = 0
        
        logging.info(f"Validating frame counts for {len(video_configs)} videos (expected: {expected_duration:.1f}s @ {fps}fps = {int(expected_duration * fps)} frames)")
        
        for config in video_configs:
            video_path = config.get('path')
            if not video_path or not os.path.exists(video_path):
                continue
            
            try:
                # Quick probe to get actual video info
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logging.warning(f"Cannot open video for validation: {video_path}")
                    fixed_configs.append(config)
                    continue
                
                actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
                actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                actual_duration = actual_frame_count / actual_fps if actual_fps > 0 else expected_duration
                cap.release()
                
                expected_frames = int(expected_duration * fps)
                
                # Check for significant mismatch
                frame_diff = abs(actual_frame_count - expected_frames)
                if frame_diff > fps * 0.1:  # More than 0.1 second difference
                    logging.warning(f"Frame count mismatch in {os.path.basename(video_path)}: "
                                  f"expected {expected_frames}, got {actual_frame_count} "
                                  f"(diff: {frame_diff} frames, {frame_diff/fps:.2f}s)")
                    
                    # Create corrected config
                    corrected_config = config.copy()
                    corrected_config['actual_duration'] = actual_duration
                    corrected_config['actual_frame_count'] = actual_frame_count
                    corrected_config['frame_mismatch'] = True
                    
                    total_adjustments += 1
                    fixed_configs.append(corrected_config)
                else:
                    # No significant mismatch
                    config['actual_duration'] = actual_duration
                    config['actual_frame_count'] = actual_frame_count
                    config['frame_mismatch'] = False
                    fixed_configs.append(config)
                    
            except Exception as e:
                logging.error(f"Error validating video {video_path}: {e}")
                fixed_configs.append(config)  # Keep original if validation fails
        
        if total_adjustments > 0:
            logging.warning(f"Fixed {total_adjustments} frame count mismatches")
        else:
            logging.info("All video frame counts are consistent")
        
        return fixed_configs

    def optimize_processing_pipeline(self, grid_config, duration, fps):
        """Optimize the processing pipeline based on content analysis"""
        optimization_report = {
            'memory_optimizations': [],
            'processing_optimizations': [],
            'frame_optimizations': []
        }
        
        try:
            # 1. Analyze content complexity
            total_videos = 0
            total_size = 0
            
            for row in grid_config:
                for cell in row:
                    if cell and not cell.get('empty', True):
                        clips = cell.get('clips', [cell])
                        for clip in clips:
                            if clip.get('path'):
                                total_videos += 1
                                try:
                                    # Quick size check
                                    file_size = os.path.getsize(clip['path'])
                                    total_size += file_size
                                except:
                                    pass
            
            # 2. Memory optimization
            estimated_memory_mb = (total_videos * duration * fps * 1920 * 1080 * 3) / (1024 * 1024)
            if estimated_memory_mb > 1000:  # > 1GB
                optimization_report['memory_optimizations'].append('high_memory_usage_detected')
                self.force_memory_cleanup()
            
            # 3. Processing optimization based on content
            if total_videos > 6:
                optimization_report['processing_optimizations'].append('many_videos_detected')
                # Use smaller batch sizes for many videos
                self.batch_processing = True
            
            # 4. Frame optimization
            expected_total_frames = int(duration * fps * total_videos)
            if expected_total_frames > 1000:
                optimization_report['frame_optimizations'].append('many_frames_detected')
                # Enable frame streaming instead of loading all at once
                
            logging.info(f"Pipeline optimization: {optimization_report}")
            return optimization_report
            
        except Exception as e:
            logging.warning(f"Pipeline optimization failed: {e}")
            return optimization_report

    def get_performance_stats(self):
        """Get current performance statistics"""
        return self.performance_stats.copy()

    def log_performance_summary(self):
        """Log a summary of performance statistics"""
        stats = self.performance_stats
        logging.info(f"Performance Summary:")
        logging.info(f"  Frames processed: {stats['frames_processed']:,}")
        logging.info(f"  Memory cleanups: {stats['memory_cleanups']}")
        logging.info(f"  Cache hits: {stats['cache_hits']}")
        logging.info(f"  Cache misses: {stats['cache_misses']}")
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100
            logging.info(f"  Cache hit rate: {hit_rate:.1f}%")

# End of GPUPipelineProcessor class with critical performance fixes
