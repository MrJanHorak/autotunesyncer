# import torch
# import numpy as np
# import cv2
# from pathlib import Path
# import logging
# import os
# import tempfile

# from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter, ffmpeg_escape_filename, cross_platform_popen_params, ffmpeg_read_image, FFMPEG_VideoReader

# class CudaVideoProcessor:
#     # def __init__(self, device='cuda:0'):
#     #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #     self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
#     #     logging.info(f"CUDA Video Processor initialized on {self.device}")
        
#     #     # Report CUDA capabilities
#     #     if torch.cuda.is_available():
#     #         for i in range(torch.cuda.device_count()):
#     #             prop = torch.cuda.get_device_properties(i)
#     #             logging.info(f"GPU {i}: {prop.name}, Mem: {prop.total_memory/(1024**3):.2f}GB")
#     def __init__(self, device='cuda:0'):
#         try:
#             if torch.cuda.is_available():
#                 self.device = torch.device(device)
#                 torch.cuda.set_device(self.device)  # Explicitly set device
#                 # Test CUDA
#                 _ = torch.zeros(1).cuda()
#                 logging.info(f"CUDA initialized successfully on {self.device}")
#             else:
#                 self.device = torch.device('cpu')
#                 logging.warning("CUDA not available, using CPU")
#         except Exception as e:
#             self.device = torch.device('cpu')
#             logging.error(f"Error initializing CUDA: {e}, falling back to CPU")
#         self.stream = torch.cuda.Stream() if torch.cuda.is_available() and str(self.device).startswith('cuda') else None
#         logging.info(f"CUDA Video Processor initialized on {self.device}")
                
#     def to_tensor(self, frame):
#         """Convert numpy array to torch tensor on GPU"""
#         with torch.cuda.stream(self.stream) if self.stream else torch.no_grad():
#             return torch.from_numpy(frame).to(self.device)
    
#     def to_numpy(self, tensor):
#         """Convert torch tensor to numpy array"""
#         return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    
#     # def composite_frames(self, base_frame, overlay_frame, mask=None, position=(0, 0)):
#     #     """Composite overlay onto base frame using GPU acceleration"""
#     #     with torch.cuda.stream(self.stream) if self.stream else torch.no_grad():
#     #         # Convert to tensors if needed
#     #         if not torch.is_tensor(base_frame):
#     #             base_frame = self.to_tensor(base_frame)
#     #         if not torch.is_tensor(overlay_frame):
#     #             overlay_frame = self.to_tensor(overlay_frame)
#     #         if mask is not None and not torch.is_tensor(mask):
#     #             mask = self.to_tensor(mask)
            
#     #         # Get dimensions
#     #         y, x = position
#     #         h, w = overlay_frame.shape[:2]
            
#     #         # Create region of interest
#     #         roi = base_frame[y:y+h, x:x+w]
            
#     #         if mask is not None:
#     #             # Apply mask with alpha blending on GPU
#     #             alpha = mask / 255.0 if mask.max() > 1.0 else mask
#     #             alpha = alpha.unsqueeze(-1) if len(alpha.shape) == 2 else alpha
#     #             result = roi * (1.0 - alpha) + overlay_frame * alpha
#     #         else:
#     #             # Direct overlay
#     #             result = overlay_frame
            
#     #         # Update base frame
#     #         base_frame[y:y+h, x:x+w] = result
            
#     #         return base_frame

#     # Fix the composite_frames method:
#     def composite_frames(self, base_frame, overlay_frame, mask=None, position=(0, 0), target_size=None):
#         """Composite overlay onto base frame using GPU acceleration"""
#         with torch.cuda.stream(self.stream) if self.stream else torch.no_grad():
#             # Convert to tensors if needed
#             if not torch.is_tensor(base_frame):
#                 base_frame = self.to_tensor(base_frame)
#             if not torch.is_tensor(overlay_frame):
#                 overlay_frame = self.to_tensor(overlay_frame)
#             if mask is not None and not torch.is_tensor(mask):
#                 mask = self.to_tensor(mask)
            
#             # Get dimensions
#             y, x = position
            
#             # Resize overlay if target size is specified
#             if target_size:
#                 h, w = target_size
#                 if overlay_frame.shape[:2] != (h, w):
#                     overlay_frame = torch.nn.functional.interpolate(
#                         overlay_frame.permute(2,0,1).unsqueeze(0),
#                         size=(h, w),
#                         mode='bilinear',
#                         align_corners=False
#                     ).squeeze(0).permute(1,2,0)
#             else:
#                 h, w = overlay_frame.shape[:2]
            
#             # Create region of interest
#             roi = base_frame[y:y+h, x:x+w]
            
#             if mask is not None:
#                 # Optimize with pre-multiplied alpha for better performance
#                 alpha = mask.float() / 255.0 if mask.max() > 1.0 else mask.float()
#                 if len(alpha.shape) == 2:
#                     alpha = alpha.unsqueeze(-1)
                
#                 # Pre-multiply alpha (faster than separate operations)
#                 overlay_premult = overlay_frame * alpha
                
#                 # Single operation blend (much faster than multiple operations)
#                 result = overlay_premult + roi * (1.0 - alpha)
#             else:
#                 # Direct overlay
#                 result = overlay_frame
            
#             # Update base frame (in-place)
#             base_frame[y:y+h, x:x+w] = result.to(torch.uint8)
            
#             return base_frame
        
#     # Modified process_chunk method - remove audio extraction

#     def process_chunk(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
#         """Process chunk with GPU acceleration"""
#         h, w = frame_size
#         frame_count = int(duration * fps)
        
#         # Pre-allocate all frames on GPU memory
#         all_frames = torch.zeros((frame_count, h, w, 3), dtype=torch.uint8, device=self.device)
        
#         # Process all frames in a single batch
#         for frame_idx in range(frame_count):
#             time_pos = frame_idx / fps
            
#             # Create frame directly on GPU
#             base_frame = all_frames[frame_idx]
#             cell_h, cell_w = h // rows, w // cols
            
#             # Process all cells for this frame
#             for row in range(rows):
#                 for col in range(cols):
#                     cell = grid[row][col]
#                     if hasattr(cell, 'get_frame'):
#                         try:
#                             if time_pos >= cell.start and (cell.end is None or time_pos < cell.end):
#                                 # Process frame directly on GPU
#                                 self._process_cell_on_gpu(
#                                     base_frame, cell, time_pos - cell.start,
#                                     row * cell_h, col * cell_w, 
#                                     cell_h, cell_w
#                                 )
#                         except Exception as e:
#                             logging.error(f"Error processing cell [{row},{col}] at {time_pos}s: {e}")
        
#         # Set up FFmpeg writer WITHOUT audio - this is the key fix
#         ffmpeg_params = [
#             "-vsync", "cfr", 
#             "-surfaces", "16", 
#             "-tune", "hq", 
#             "-rc", "vbr_hq",
#             "-cq", "20", 
#             "-bufsize", "10M"
#         ]

#         writer = FFMPEG_VideoWriter(
#             filename=output_path,
#             size=(w, h),
#             fps=fps,
#             codec='h264_nvenc',
#             preset='p4',
#             ffmpeg_params=ffmpeg_params
#         )
        
#         # Write all frames with minimal transfers
#         batch_size = 10  # Process frames in batches
#         try:
#             for i in range(0, frame_count, batch_size):
#                 batch_end = min(i + batch_size, frame_count)
#                 # Move only the batch we're writing back to CPU
#                 for j in range(i, batch_end):
#                     writer.write_frame(self.to_numpy(all_frames[j]))
                
#                 if i % 30 == 0:
#                     logging.info(f"Written {i}/{frame_count} frames ({i/frame_count*100:.1f}%)")
#         finally:
#             writer.close()
#             # Free GPU memory
#             torch.cuda.empty_cache()
                
#         return output_path
#     # Add to cuda_compositing.py process_chunk_optimized
#     # def process_chunk(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
#     #     """Process chunk with minimized CPU-GPU transfers"""
#     #     h, w = frame_size
#     #     frame_count = int(duration * fps)
        
#     #     # Pre-allocate all frames on GPU memory
#     #     all_frames = torch.zeros((frame_count, h, w, 3), dtype=torch.uint8, device=self.device)

#     #     # Try to extract audio from the first clip that has it
#     #     first_chunk_audio = None
#     #     for row in range(rows):
#     #         for col in range(cols):
#     #             cell = grid[row][col]
#     #             if hasattr(cell, 'audio') and cell.audio is not None:
#     #                 try:
#     #                     temp_dir = tempfile.gettempdir()
#     #                     audio_path = os.path.join(temp_dir, f"audio_tmp_{os.path.basename(output_path)}.aac")
                        
#     #                     # Try to extract audio with more robust error handling
#     #                     logging.info(f"Extracting audio to {audio_path}")
#     #                     try:
#     #                         # Try modern API first
#     #                         cell.audio.write_audiofile(
#     #                             audio_path, 
#     #                             codec='aac',
#     #                             ffmpeg_params=["-strict", "-2"]  # Remove verbose and logger
#     #                         )
#     #                     except TypeError:
#     #                         # Fall back to legacy API
#     #                         try:
#     #                             cell.audio.write_audiofile(
#     #                                 audio_path, 
#     #                                 codec='aac'
#     #                             )
#     #                         except Exception as e2:
#     #                             logging.error(f"Audio extraction failed with both APIs: {e2}")
#     #                             continue
                        
#     #                     # Verify the file exists and has content
#     #                     if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
#     #                         first_chunk_audio = audio_path
#     #                         logging.info(f"Successfully extracted audio from clip at position [{row},{col}] to {audio_path}")
#     #                         break
#     #                     else:
#     #                         logging.warning(f"Audio file was created but is empty: {audio_path}")
#     #                 except Exception as e:
#     #                     logging.error(f"Error extracting audio: {str(e)}")
#     #                     continue
#     #         if first_chunk_audio:
#     #             break
        
#     #     # Process all frames in a single batch
#     #     for frame_idx in range(frame_count):
#     #         time_pos = frame_idx / fps
            
#     #         # Create frame directly on GPU
#     #         base_frame = all_frames[frame_idx]
#     #         cell_h, cell_w = h // rows, w // cols
            
#     #         # Process all cells for this frame
#     #         for row in range(rows):
#     #             for col in range(cols):
#     #                 cell = grid[row][col]
#     #                 if hasattr(cell, 'get_frame'):
#     #                     try:
#     #                         if time_pos >= cell.start and (cell.end is None or time_pos < cell.end):
#     #                             # Process frame directly on GPU
#     #                             self._process_cell_on_gpu(
#     #                                 base_frame, cell, time_pos - cell.start,
#     #                                 row * cell_h, col * cell_w, 
#     #                                 cell_h, cell_w
#     #                             )
#     #                     except Exception as e:
#     #                         logging.error(f"Error processing cell [{row},{col}] at {time_pos}s: {e}")
        
#     #     # Set up FFmpeg writer
#     #     ffmpeg_params = [
#     #         "-vsync", "cfr", 
#     #         "-surfaces", "16", 
#     #         "-tune", "hq", 
#     #         "-rc", "vbr_hq",
#     #         "-cq", "20", 
#     #         "-bufsize", "10M"
#     #     ]

#     #     writer_args = {
#     #         'filename': output_path,
#     #         'size': (w, h),
#     #         'fps': fps,
#     #         'codec': 'h264_nvenc',
#     #         'preset': 'p4',
#     #         'ffmpeg_params': ffmpeg_params
#     #     }

#     #     # Only include audiofile if it exists
#     #     if first_chunk_audio and os.path.exists(first_chunk_audio):
#     #         writer_args['audiofile'] = first_chunk_audio
#     #         # Add audio codec parameters
#     #         ffmpeg_params.extend(["-c:a", "aac", "-b:a", "192k"])

#     #     writer = FFMPEG_VideoWriter(**writer_args)

#     #     # Clean up temporary audio file if created
#     #     if first_chunk_audio and Path(first_chunk_audio).exists():
#     #         try:
#     #             Path(first_chunk_audio).unlink()
#     #         except Exception as e:
#     #             logging.warning(f"Could not delete temporary audio file: {e}")
        
#     #     # Write all frames with minimal transfers
#     #     batch_size = 10  # Process frames in batches
#     #     try:
#     #         for i in range(0, frame_count, batch_size):
#     #             batch_end = min(i + batch_size, frame_count)
#     #             # Move only the batch we're writing back to CPU
#     #             for j in range(i, batch_end):
#     #                 writer.write_frame(self.to_numpy(all_frames[j]))
                
#     #             if i % 30 == 0:
#     #                 logging.info(f"Written {i}/{frame_count} frames ({i/frame_count*100:.1f}%)")
#     #     finally:
#     #         writer.close()
#     #         # Free GPU memory
#     #         torch.cuda.empty_cache()
                
#     #     return output_path

#     def _process_cell_on_gpu(self, base_frame, cell, clip_time, y, x, h, w):
#         """Process a grid cell directly on GPU"""
#         # Get the frame data
#         frame_data = cell.get_frame(clip_time)
#         frame_tensor = self.to_tensor(frame_data)
        
#         # Get mask if available
#         mask_tensor = None
#         if hasattr(cell, 'mask') and cell.mask is not None:
#             mask_data = cell.mask.get_frame(clip_time)
#             mask_tensor = self.to_tensor(mask_data)
        
#         # Composite using efficient alpha blending
#         self.composite_frames(base_frame, frame_tensor, mask_tensor, (y, x), (h, w))


#     # Add this option to process_chunk in cuda_compositing.py
#     # # Complete the process_grid_no_alpha method
#     def process_grid_no_alpha(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
#         """Process grid without alpha compositing (last clip wins)"""
#         h, w = frame_size
#         frame_count = int(duration * fps)
#         output_frames = torch.zeros((frame_count, h, w, 3), dtype=torch.uint8, device=self.device)
#         cell_h, cell_w = h // rows, w // cols

#         first_chunk_audio = None
#         for row in range(rows):
#             for col in range(cols):
#                 cell = grid[row][col]
#                 if hasattr(cell, 'audio') and cell.audio is not None:
#                     try:
#                         temp_dir = tempfile.gettempdir()
#                         audio_path = os.path.join(temp_dir, f"audio_tmp_{os.path.basename(output_path)}.aac")
                        
#                         # Try to extract audio with more robust error handling
#                         logging.info(f"Extracting audio to {audio_path}")
#                         cell.audio.write_audiofile(
#                             audio_path, 
#                             codec='aac',
#                             ffmpeg_params=["-strict", "-2"],  # Allow experimental codecs
#                             verbose=False, 
#                             logger=None
#                         )
                        
#                         # Verify the file exists and has content
#                         if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
#                             first_chunk_audio = audio_path
#                             logging.info(f"Successfully extracted audio from clip at position [{row},{col}] to {audio_path}")
#                             break
#                         else:
#                             logging.warning(f"Audio file was created but is empty: {audio_path}")
#                     except Exception as e:
#                         logging.error(f"Error extracting audio: {str(e)}")
#                         continue

#             if first_chunk_audio:
#                 break
        
#         # Sort clips by start time in each cell (latest start time first)
#         for row in range(rows):
#             for col in range(cols):
#                 if isinstance(grid[row][col], CompositeVideoClip):
#                     # Get individual clips in the cell
#                     clips = grid[row][col].clips
                    
#                     # For each frame, use the latest active clip
#                     for frame_idx in range(frame_count):
#                         time_pos = frame_idx / fps
#                         cell_region = output_frames[frame_idx, row*cell_h:(row+1)*cell_h, 
#                                                     col*cell_w:(col+1)*cell_w]
                        
#                         # Find the last active clip for this time
#                         for clip in reversed(clips):  # Process in reverse order
#                             if time_pos >= clip.start and time_pos < clip.end:
#                                 # Use this clip without alpha blending
#                                 clip_time = time_pos - clip.start
#                                 frame_data = clip.get_frame(clip_time)
#                                 frame_tensor = self.to_tensor(frame_data)
                                
#                                 # Resize if needed
#                                 if frame_tensor.shape[:2] != (cell_h, cell_w):
#                                     frame_tensor = torch.nn.functional.interpolate(
#                                         frame_tensor.permute(2,0,1).unsqueeze(0),
#                                         size=(cell_h, cell_w),
#                                         mode='bilinear'
#                                     ).squeeze(0).permute(1,2,0)
                                
#                                 # Direct placement - no alpha blending
#                                 cell_region.copy_(frame_tensor)
#                                 break  # Stop after first active clip
        
#         # Write frames to output
#         # writer = FFMPEG_VideoWriter(
#         #     output_path, size=(w, h), fps=fps,
#         #     codec='h264_nvenc', preset='p4',
#         #     audiofile=first_chunk_audio,  # Add audio from first chunk
#         #     ffmpeg_params=[
#         #         "-vsync", "cfr", "-surfaces", "16", 
#         #         "-tune", "hq", "-rc", "vbr_hq",
#         #         "-c:a", "aac", "-b:a", "192k",  # Audio codec
#         #         "-cq", "20", "-bufsize", "10M"
#         #     ]
#         # )
#         ffmpeg_params = [
#             "-vsync", "cfr", 
#             "-surfaces", "16", 
#             "-tune", "hq", 
#             "-rc", "vbr_hq",
#             "-cq", "20", 
#             "-bufsize", "10M"
#         ]

#         writer_args = {
#             'filename': output_path,
#             'size': (w, h),
#             'fps': fps,
#             'codec': 'h264_nvenc',
#             'preset': 'p4',
#             'ffmpeg_params': ffmpeg_params
#         }

#         # Only include audiofile if it exists
#         if first_chunk_audio and os.path.exists(first_chunk_audio):
#             writer_args['audiofile'] = first_chunk_audio
#             # Add audio codec parameters
#             ffmpeg_params.extend(["-c:a", "aac", "-b:a", "192k"])

#         writer = FFMPEG_VideoWriter(**writer_args)
        
#         # Write all frames with minimal transfers
#         batch_size = 10  # Process frames in batches
#         try:
#             for i in range(0, frame_count, batch_size):
#                 batch_end = min(i + batch_size, frame_count)
#                 # Move only the batch we're writing back to CPU
#                 for j in range(i, batch_end):
#                     writer.write_frame(self.to_numpy(output_frames[j]))
                
#                 if i % 30 == 0:
#                     logging.info(f"Written {i}/{frame_count} frames ({i/frame_count*100:.1f}%)")
#         finally:
#             writer.close()
#             # Free GPU memory
#             torch.cuda.empty_cache()
        
#         return output_path
        
#     # def process_chunk(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
#     #     """Process an entire video chunk on GPU"""
#     #     h, w = frame_size
#     #     frame_count = int(duration * fps)
        
#     #     # Set up FFmpeg writer
#     #     writer = FFMPEG_VideoWriter(
#     #         output_path, 
#     #         size=(w, h),
#     #         fps=fps,
#     #         codec='h264_nvenc',
#     #         preset='p4',
#     #         ffmpeg_params=[
#     #             "-vsync", "cfr",
#     #             "-surfaces", "16",
#     #             "-tune", "hq",
#     #             "-rc", "vbr_hq",
#     #             "-cq", "20",
#     #             "-bufsize", "10M"
#     #         ]
#     #     )
        
#     #     try:
#     #         # Process each frame
#     #         for frame_idx in range(frame_count):
#     #             time_pos = frame_idx / fps
                
#     #             # Create blank base frame on GPU
#     #             base_frame = torch.zeros((h, w, 3), dtype=torch.uint8, device=self.device)
                
#     #             # Get cell dimensions
#     #             cell_h, cell_w = h // rows, w // cols
                
#     #             # Composite cells
#     #             for row in range(rows):
#     #                 for col in range(cols):
#     #                     cell = grid[row][col]
#     #                     if hasattr(cell, 'get_frame'):
#     #                         try:
#     #                             # Get frame data for this point in time
#     #                             if time_pos >= cell.start and (cell.end is None or time_pos < cell.end):
#     #                                 # Adjust time relative to clip start
#     #                                 clip_time = time_pos - cell.start
                                    
#     #                                 # Extract frame
#     #                                 frame_data = cell.get_frame(clip_time)
#     #                                 mask_data = None
                                    
#     #                                 # Get mask if available
#     #                                 if hasattr(cell, 'mask') and cell.mask is not None:
#     #                                     mask_data = cell.mask.get_frame(clip_time)
                                    
#     #                                 # Position in grid
#     #                                 pos_y = row * cell_h
#     #                                 pos_x = col * cell_w
                                    
#     #                                 # Composite on GPU
#     #                                 frame_tensor = self.to_tensor(frame_data)
#     #                                 mask_tensor = self.to_tensor(mask_data) if mask_data is not None else None
                                    
#     #                                 base_frame = self.composite_frames(
#     #                                     base_frame, 
#     #                                     frame_tensor, 
#     #                                     mask_tensor, 
#     #                                     (pos_y, pos_x)
#     #                                 )
#     #                         except Exception as e:
#     #                             logging.error(f"Error compositing frame at {time_pos}: {e}")
                
#     #             # Write frame
#     #             numpy_frame = self.to_numpy(base_frame)
#     #             writer.write_frame(numpy_frame)
                
#     #             # Print progress every 30 frames
#     #             if frame_idx % 30 == 0:
#     #                 logging.info(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
                    
#     #     finally:
#     #         writer.close()
            

#     #     return output_path

import os
import numpy as np
import torch
import logging
from pathlib import Path
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from gpu_pipeline import GPUPipelineProcessor

class CudaVideoProcessor:
    def __init__(self, device=None):
        # More robust CUDA initialization
        try:
            if torch.cuda.is_available():
                # Let user specify device or default to first GPU
                self.device = torch.device(device or "cuda:0")
                # Force initialization to detect problems early
                torch.cuda.set_device(self.device)
                # Create sample tensor to verify CUDA works
                test_tensor = torch.zeros(1, device=self.device)
                # Try a simple operation to verify CUDA is working
                test_result = test_tensor + 1
                if test_result.item() != 1:
                    raise RuntimeError("CUDA operation failed")
                
                # Set performance optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                logging.info(f"CUDA initialized successfully on {torch.cuda.get_device_name(self.device)}")
                self.stream = torch.cuda.Stream(device=self.device)
            else:
                self.device = torch.device('cpu')
                self.stream = None
                logging.warning("CUDA not available, using CPU")
        except Exception as e:
            logging.error(f"CUDA initialization error: {e}")
            self.device = torch.device('cpu')
            self.stream = None
            logging.warning("CUDA initialization failed, falling back to CPU")

    def to_tensor(self, array):
        """Convert numpy array to tensor with proper error handling"""
        try:
            if isinstance(array, np.ndarray):
                tensor = torch.from_numpy(array).to(self.device)
                return tensor
            return array
        except Exception as e:
            logging.error(f"Error converting to tensor: {e}")
            return torch.zeros((1, 1, 3), device=self.device)

    def to_numpy(self, tensor):
        """Convert tensor to numpy array with proper error handling"""
        try:
            return tensor.cpu().numpy()
        except Exception as e:
            logging.error(f"Error converting to numpy: {e}")
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def composite_frames(self, base_frame, overlay_frame, mask=None, position=(0, 0), target_size=None):
        """Optimized alpha compositing with pre-multiplication"""
        with torch.cuda.stream(self.stream):
            y, x = position
            h, w = overlay_frame.shape[:2]
            
            # Create region of interest
            try:
                roi = base_frame[y:y+h, x:x+w]
                
                if mask is not None:
                    # Convert mask to float and normalize if needed
                    alpha = mask.float() / 255.0 if mask.max() > 1.0 else mask.float()
                    if len(alpha.shape) == 2:
                        alpha = alpha.unsqueeze(-1)
                    
                    # Single operation blend using pre-multiplied alpha (more efficient)
                    overlay_premult = overlay_frame.float() * alpha
                    inv_alpha = 1.0 - alpha
                    result = overlay_premult + roi.float() * inv_alpha
                    
                    # Update in-place (more efficient)
                    roi.copy_(result.to(torch.uint8))
                else:
                    # Direct copy when no alpha blending needed
                    roi.copy_(overlay_frame)
                
                return base_frame
            except Exception as e:
                logging.error(f"Error in composite_frames: {e}")
                return base_frame
            
    def process_chunk(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
        # Convert grid to grid_config format
        grid_config = []
        for row in range(rows):
            grid_row = []
            for col in range(cols):
                cell = grid[row][col]
                if hasattr(cell, 'get_frame'):
                    config = {
                        'path': cell.filename if hasattr(cell, 'filename') else None,
                        'start_time': cell.start if hasattr(cell, 'start') else 0,
                        'duration': cell.duration if hasattr(cell, 'duration') else duration,
                        'empty': False
                    }
                else:
                    config = {'empty': True}
                grid_row.append(config)
            grid_config.append(grid_row)
            
        # Use pure GPU pipeline
        pipeline = GPUPipelineProcessor()
        return pipeline.process_chunk_pure_gpu(grid_config, output_path, fps, duration)

    # def process_chunk(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
    #     """Process chunk with proper audio handling and increased batch size"""
    #     h, w = frame_size
    #     frame_count = int(duration * fps)
    #     cell_h, cell_w = h // rows, w // cols
        
    #     # Extract audio from ALL clips that have audio
    #     temp_dir = os.path.join(os.path.dirname(output_path), "temp")
    #     os.makedirs(temp_dir, exist_ok=True)
        
    #     # Extract audio from all clips
    #     audio_tracks = []
    #     for row in range(rows):
    #         for col in range(cols):
    #             cell = grid[row][col]
    #             if hasattr(cell, 'audio') and cell.audio is not None:
    #                 try:
    #                     # Use a unique filename to avoid conflicts
    #                     unique_id = os.path.splitext(os.path.basename(output_path))[0]
    #                     audio_path = os.path.join(temp_dir, f"audio_{unique_id}_{row}_{col}.aac")
                        
    #                     # Extract audio using cell's audio function
    #                     cell.audio.write_audiofile(audio_path, codec='aac')
                        
    #                     if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
    #                         audio_tracks.append(audio_path)
    #                         logging.info(f"Extracted audio from clip at [{row},{col}]")
    #                 except Exception as e:
    #                     logging.warning(f"Audio extraction error: {e}")
    #                     continue
        
    #     # Mix audio if we have multiple tracks
    #     final_audio = None
    #     if len(audio_tracks) > 1:
    #         try:
    #             import subprocess
    #             mixed_audio = os.path.join(temp_dir, f"mixed_{os.path.basename(output_path)}.aac")
                
    #             # Create FFmpeg command for mixing
    #             cmd = ['ffmpeg', '-y']
    #             for track in audio_tracks:
    #                 cmd.extend(['-i', track])
                
    #             # Add filter for mixing all inputs
    #             cmd.extend([
    #                 '-filter_complex', f'amix=inputs={len(audio_tracks)}:duration=longest',
    #                 '-c:a', 'aac',
    #                 '-b:a', '192k',
    #                 mixed_audio
    #             ])
                
    #             # Run FFmpeg to mix audio
    #             result = subprocess.run(cmd, check=True, capture_output=True)
                
    #             if os.path.exists(mixed_audio) and os.path.getsize(mixed_audio) > 0:
    #                 final_audio = mixed_audio
    #                 logging.info(f"Mixed {len(audio_tracks)} audio tracks successfully")
    #             else:
    #                 logging.warning("Failed to mix audio tracks, using first track")
    #                 final_audio = audio_tracks[0]
    #         except Exception as e:
    #             logging.error(f"Error mixing audio: {e}")
    #             # Fall back to first audio track if mixing fails
    #             final_audio = audio_tracks[0] if audio_tracks else None
    #     elif audio_tracks:
    #         final_audio = audio_tracks[0]
        
    #     # Pre-allocate all frames on GPU
    #     all_frames = torch.zeros((frame_count, h, w, 3), dtype=torch.uint8, device=self.device)
        
    #     try:
    #         # Process all frames
    #         for frame_idx in range(frame_count):
    #             time_pos = frame_idx / fps
                
    #             # Process grid cells for this frame
    #             for row in range(rows):
    #                 for col in range(cols):
    #                     cell = grid[row][col]
    #                     if hasattr(cell, 'get_frame'):
    #                         try:
    #                             # Check if clip is active at this time
    #                             if hasattr(cell, 'start') and hasattr(cell, 'end') and time_pos >= cell.start and time_pos < cell.end:
    #                                 # Process clip frame
    #                                 self._process_cell_on_gpu(
    #                                     all_frames[frame_idx], 
    #                                     cell, 
    #                                     time_pos - cell.start,
    #                                     row * cell_h, 
    #                                     col * cell_w,
    #                                     cell_h, 
    #                                     cell_w
    #                                 )
    #                         except Exception as e:
    #                             logging.error(f"Error processing cell [{row},{col}]: {e}")

    #         # Set up FFmpeg writer with audio
    #         writer_args = {
    #             'filename': output_path,
    #             'size': (w, h),
    #             'fps': fps,
    #             'codec': 'h264_nvenc' if self.device.type == 'cuda' else 'libx264',
    #             'preset': 'p4' if self.device.type == 'cuda' else 'medium',
    #             'ffmpeg_params': [
    #                 '-pix_fmt', 'yuv420p',
    #                 '-vsync', 'cfr'
    #             ]
    #         }
            
    #         # Only include audio if we have a valid file
    #         if final_audio and os.path.exists(final_audio):
    #             writer_args['audiofile'] = final_audio
    #             logging.info(f"Adding audio to video: {final_audio}")
                
    #         writer = FFMPEG_VideoWriter(**writer_args)
            
    #         # Write frames in larger batches (increased from 10 to 30)
    #         batch_size = 30
    #         for i in range(0, frame_count, batch_size):
    #             batch_end = min(i + batch_size, frame_count)
    #             for j in range(i, batch_end):
    #                 writer.write_frame(self.to_numpy(all_frames[j]))
                
    #             if i % 90 == 0:  # Log less frequently
    #                 logging.info(f"Written {i}/{frame_count} frames ({i/frame_count*100:.1f}%)")
                    
    #         writer.close()
            
    #         # Release GPU memory
    #         del all_frames
    #         torch.cuda.empty_cache()
            
    #         return output_path
            
    #     except Exception as e:
    #         logging.error(f"Error in process_chunk: {e}", exc_info=True)
    #         # Clean up resources on error
    #         if 'all_frames' in locals():
    #             del all_frames
    #         if 'writer' in locals():
    #             try:
    #                 writer.close()
    #             except Exception:
    #                 pass
    #         torch.cuda.empty_cache()
    #         raise
            
    def _process_cell_on_gpu(self, base_frame, cell, clip_time, y, x, h, w):
        """Process a single grid cell directly on GPU"""
        try:
            # Get the frame data
            frame_data = cell.get_frame(clip_time)
            if frame_data is None:
                return
                
            frame_tensor = self.to_tensor(frame_data)
            
            # Resize if necessary
            if frame_tensor.shape[0] != h or frame_tensor.shape[1] != w:
                frame_tensor = torch.nn.functional.interpolate(
                    frame_tensor.permute(2,0,1).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear'
                ).squeeze(0).permute(1,2,0).to(torch.uint8)
            
            # Get mask if available
            mask_tensor = None
            if hasattr(cell, 'mask') and cell.mask is not None:
                mask_data = cell.mask.get_frame(clip_time)
                if mask_data is not None:
                    mask_tensor = self.to_tensor(mask_data)
            
            # Composite using efficient alpha blending
            self.composite_frames(base_frame, frame_tensor, mask_tensor, (y, x))
            
        except Exception as e:
            logging.error(f"Error in _process_cell_on_gpu: {e}")

    def direct_copy_compositor(frame, video_frame, position, size):
        """Direct frame copy with no alpha blending - much faster than compositing"""
        y, x = position
        h, w = size
        
        # No alpha blending, just direct copy
        frame[y:y+h, x:x+w] = video_frame
        return frame
    

    def process_grid_no_alpha(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
        """Process grid without alpha compositing (pure GPU implementation)"""
        # Convert MoviePy grid to our grid_config format
        grid_config = []
        for row in range(rows):
            grid_row = []
            for col in range(cols):
                cell = grid[row][col]
                if hasattr(cell, 'get_frame'):
                    config = {
                        'path': None,  # Initialize to None
                        'start_time': cell.start if hasattr(cell, 'start') else 0,
                        'duration': cell.duration if hasattr(cell, 'duration') else duration,
                        'empty': False
                    }
                    
                    # Try multiple ways to get the file path
                    if hasattr(cell, 'filename') and cell.filename:
                        config['path'] = cell.filename
                        logging.info(f"Using filename from cell: {cell.filename}")
                    elif hasattr(cell, 'source') and hasattr(cell.source, 'filename'):
                        config['path'] = cell.source.filename
                        logging.info(f"Using source filename: {cell.source.filename}")
                    elif hasattr(cell, 'video_fp') and cell.video_fp:
                        config['path'] = cell.video_fp
                        logging.info(f"Using video_fp: {cell.video_fp}")
                    # For VideoFileClip from MoviePy
                    elif hasattr(cell, 'reader') and hasattr(cell.reader, 'filename'):
                        config['path'] = cell.reader.filename
                        logging.info(f"Using reader filename: {cell.reader.filename}")
                    
                    # As a last resort, check for custom attributes specific to your app
                    elif hasattr(cell, '_video_path'):
                        config['path'] = cell._video_path
                        logging.info(f"Using _video_path: {cell._video_path}")
                    
                    if config['path'] is None:
                        logging.warning(f"No path found for cell at [{row},{col}]. Adding debug info:")
                        logging.warning(f"Cell type: {type(cell)}")
                        logging.warning(f"Cell attributes: {dir(cell)}")
                else:
                    config = {'empty': True}
                    
                grid_row.append(config)
            grid_config.append(grid_row)
            
        # Use pure GPU pipeline
        pipeline = GPUPipelineProcessor()
        return pipeline.process_chunk_pure_gpu(grid_config, output_path, fps, duration)
            
    # def process_grid_no_alpha(self, grid, rows, cols, duration, fps, output_path, frame_size=(1080, 1920)):
    #     """Process grid without alpha compositing (last clip wins)"""
    #     h, w = frame_size
    #     cell_h, cell_w = h // rows, w // cols
    #     frame_count = int(duration * fps)
        
    #     # Extract audio from ALL clips that have audio
    #     temp_dir = os.path.join(os.path.dirname(output_path), "temp")
    #     os.makedirs(temp_dir, exist_ok=True)
        
    #     # Extract audio from all clips
    #     audio_tracks = []
    #     for row in range(rows):
    #         for col in range(cols):
    #             cell = grid[row][col]
    #             if hasattr(cell, 'audio') and cell.audio is not None:
    #                 try:
    #                     # Use a unique filename to avoid conflicts
    #                     unique_id = os.path.splitext(os.path.basename(output_path))[0]
    #                     audio_path = os.path.join(temp_dir, f"audio_{unique_id}_{row}_{col}.aac")
                        
    #                     # Extract audio using cell's audio function
    #                     cell.audio.write_audiofile(audio_path, codec='aac')
                        
    #                     if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
    #                         audio_tracks.append(audio_path)
    #                         logging.info(f"Extracted audio from clip at [{row},{col}]")
    #                 except Exception as e:
    #                     logging.warning(f"Audio extraction error: {e}")
    #                     continue
        
    #     # Mix audio if we have multiple tracks
    #     final_audio = None
    #     if len(audio_tracks) > 1:
    #         try:
    #             import subprocess
    #             mixed_audio = os.path.join(temp_dir, f"mixed_{os.path.basename(output_path)}.aac")
                
    #             # Create FFmpeg command for mixing
    #             cmd = ['ffmpeg', '-y']
    #             for track in audio_tracks:
    #                 cmd.extend(['-i', track])
                
    #             # Add filter for mixing all inputs
    #             cmd.extend([
    #                 '-filter_complex', f'amix=inputs={len(audio_tracks)}:duration=longest',
    #                 '-c:a', 'aac',
    #                 '-b:a', '192k',
    #                 mixed_audio
    #             ])
                
    #             # Run FFmpeg to mix audio
    #             result = subprocess.run(cmd, check=True, capture_output=True)
                
    #             if os.path.exists(mixed_audio) and os.path.getsize(mixed_audio) > 0:
    #                 final_audio = mixed_audio
    #                 logging.info(f"Mixed {len(audio_tracks)} audio tracks successfully")
    #             else:
    #                 logging.warning("Failed to mix audio tracks, using first track")
    #                 final_audio = audio_tracks[0]
    #         except Exception as e:
    #             logging.error(f"Error mixing audio: {e}")
    #             # Fall back to first audio track if mixing fails
    #             final_audio = audio_tracks[0] if audio_tracks else None
    #     elif audio_tracks:
    #         final_audio = audio_tracks[0]
                
    #     # Pre-allocate output tensor for all frames
    #     output_frames = torch.zeros((frame_count, h, w, 3), dtype=torch.uint8, device=self.device)
        
    #     try:
    #         # Process all frames
    #         for frame_idx in range(frame_count):
    #             time_pos = frame_idx / fps
                
    #             # Initialize this frame with black background
    #             frame = output_frames[frame_idx]
                
    #             # Process each cell in the grid
    #             for row in range(rows):
    #                 for col in range(cols):
    #                     cell = grid[row][col]
    #                     cell_region = frame[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
                        
    #                     # Handle composite clips with "last clip wins" approach
    #                     if isinstance(cell, CompositeVideoClip):
    #                         # Get clips in this cell - process in reverse order (last one wins)
    #                         clips = reversed(cell.clips)
    #                         for clip in clips:
    #                             if hasattr(clip, 'start') and hasattr(clip, 'end'):
    #                                 if time_pos >= clip.start and time_pos < clip.end:
    #                                     try:
    #                                         # Get clip frame
    #                                         clip_time = time_pos - clip.start
    #                                         frame_data = clip.get_frame(clip_time)
                                            
    #                                         if frame_data is not None:
    #                                             frame_tensor = self.to_tensor(frame_data)
                                                
    #                                             # Resize if needed
    #                                             if frame_tensor.shape[0] != cell_h or frame_tensor.shape[1] != cell_w:
    #                                                 frame_tensor = torch.nn.functional.interpolate(
    #                                                     frame_tensor.permute(2,0,1).unsqueeze(0),
    #                                                     size=(cell_h, cell_w),
    #                                                     mode='bilinear'
    #                                                 ).squeeze(0).permute(1,2,0).to(torch.uint8)
                                                
    #                                             # Direct copy - no alpha blending
    #                                             cell_region.copy_(frame_tensor)
                                                
    #                                             # We found and processed a clip for this time, stop looking
    #                                             break
    #                                     except Exception as e:
    #                                         logging.error(f"Error processing clip in cell [{row},{col}]: {e}")
    #                                         continue
    #                     # Handle single clip
    #                     elif hasattr(cell, 'get_frame'):
    #                         try:
    #                             if hasattr(cell, 'start') and hasattr(cell, 'end'):
    #                                 if time_pos >= cell.start and time_pos < cell.end:
    #                                     # Get frame directly
    #                                     clip_time = time_pos - cell.start
    #                                     frame_data = cell.get_frame(clip_time)
                                        
    #                                     if frame_data is not None:
    #                                         frame_tensor = self.to_tensor(frame_data)
                                            
    #                                         # Resize if needed
    #                                         if frame_tensor.shape[0] != cell_h or frame_tensor.shape[1] != cell_w:
    #                                             frame_tensor = torch.nn.functional.interpolate(
    #                                                 frame_tensor.permute(2,0,1).unsqueeze(0),
    #                                                 size=(cell_h, cell_w),
    #                                                 mode='bilinear'
    #                                             ).squeeze(0).permute(1,2,0).to(torch.uint8)
                                            
    #                                         # Direct copy - no alpha blending
    #                                         cell_region.copy_(frame_tensor)
    #                         except Exception as e:
    #                             logging.error(f"Error processing cell [{row},{col}]: {e}")
    #                             continue
                        
    #         # Write frames to video file
    #         writer_args = {
    #             'filename': output_path,
    #             'size': (w, h),
    #             'fps': fps,
    #             'codec': 'h264_nvenc' if self.device.type == 'cuda' else 'libx264',
    #             'preset': 'p4' if self.device.type == 'cuda' else 'medium',
    #             'ffmpeg_params': [
    #                 '-pix_fmt', 'yuv420p',
    #                 '-vsync', 'cfr'
    #             ]
    #         }
            
    #         # Add audio if available - using final_audio instead of first_chunk_audio
    #         if final_audio and os.path.exists(final_audio):
    #             writer_args['audiofile'] = final_audio
    #             logging.info(f"Adding mixed audio to video: {final_audio}")
                
    #         writer = FFMPEG_VideoWriter(**writer_args)
            
    #         # Write frames in larger batches
    #         batch_size = 30
    #         for i in range(0, frame_count, batch_size):
    #             batch_end = min(i + batch_size, frame_count)
    #             for j in range(i, batch_end):
    #                 writer.write_frame(self.to_numpy(output_frames[j]))
                
    #             if i % 90 == 0:
    #                 logging.info(f"Written {i}/{frame_count} frames ({i/frame_count*100:.1f}%)")
                    
    #         writer.close()
            
    #         # Clean up audio files
    #         for track in audio_tracks:
    #             if os.path.exists(track):
    #                 try:
    #                     os.remove(track)
    #                 except Exception:
    #                     pass
                        
    #         if final_audio and final_audio not in audio_tracks and os.path.exists(final_audio):
    #             try:
    #                 os.remove(final_audio)
    #             except Exception:
    #                 pass
                    
    #         # Release GPU memory
    #         del output_frames
    #         torch.cuda.empty_cache()
            
    #         return output_path
            
    #     except Exception as e:
    #         logging.error(f"Error in process_grid_no_alpha: {e}", exc_info=True)
    #         # Clean up resources on error
    #         if 'output_frames' in locals():
    #             del output_frames
    #         if 'writer' in locals():
    #             try:
    #                 writer.close()
    #             except Exception:
    #                 pass
    #         torch.cuda.empty_cache()
    #         raise