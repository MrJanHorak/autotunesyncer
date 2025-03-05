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
            