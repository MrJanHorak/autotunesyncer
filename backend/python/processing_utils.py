from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Optional, Dict, Union
import psutil
import logging
import threading
import subprocess
import os
import torch
from typing import Optional
from queue import Queue
from contextlib import contextmanager
from threading import RLock


class GPUManager:
    """Manages GPU detection and monitoring for video processing"""
    
    def __init__(self):
        self.has_gpu = False
        self.handle = None
        self.pynvml_available = False
        self.streams = []
        self.current_stream = 0
        self.stream_lock = threading.RLock()
        
        # Use enhanced GPU detection
        try:
            from gpu_setup import gpu_available, torch_cuda_available
            self.has_gpu = gpu_available and torch_cuda_available
            if self.has_gpu and torch.cuda.is_available():
                self.streams = [torch.cuda.Stream() for _ in range(4)]
                self._init_streams()
                logging.info(f"✓ GPU Manager initialized with {len(self.streams)} CUDA streams")
        except ImportError:
            # Fallback to direct torch detection
            try:
                self.has_gpu = torch.cuda.is_available() and torch.version.cuda is not None
                if self.has_gpu:
                    self.streams = [torch.cuda.Stream() for _ in range(4)]
                    self._init_streams()
            except:
                self.has_gpu = False
                logging.warning("CUDA initialization failed, falling back to CPU")
        
        self._init_gpu()
    
    def _init_gpu(self) -> bool:
      """Initialize GPU and NVML"""
      try:
          # Try NVML first
          import pynvml
          pynvml.nvmlInit()
          device_count = pynvml.nvmlDeviceGetCount()
          if device_count > 0:
              self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
              self.has_gpu = True
              self.pynvml_available = True
              gpu_name = pynvml.nvmlDeviceGetName(self.handle).decode()
              logging.info(f"GPU initialized: {gpu_name}")
              return True
      except:
          # Fallback to direct nvidia-smi check
          try:
              result = subprocess.run(['nvidia-smi'], capture_output=True, check=True)
              self.has_gpu = True
              logging.info("GPU detected via nvidia-smi")
              return True
          except:
              logging.warning("No GPU detected, falling back to CPU processing")
              self.has_gpu = False
              return False
          
    def get_stream(self):
        # Implement CUDA stream management
        if self.has_gpu:
            return self._get_next_stream()
        return None

    def _init_streams(self):
        """Initialize CUDA streams"""
        if self.has_gpu:
            self.streams = [torch.cuda.Stream() for _ in range(4)]
            logging.info(f"Initialized {len(self.streams)} CUDA streams")
    def get_next_stream(self):
        """Get next available CUDA stream with proper locking"""
        with self.stream_lock:  # Use fine-grained lock just for stream selection
            stream = self.streams[self.current_stream]
            self.current_stream = (self.current_stream + 1) % len(self.streams)
            return stream

    def _get_next_stream(self):
        """Get next available CUDA stream in round-robin fashion"""
        if not self.has_gpu or not self.streams:
            return None
        
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream

    def get_gpu_memory(self) -> int:
        """Get GPU memory in MB"""
        if not self.pynvml_available:
            return 4096  # Default fallback
            
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return info.total // (1024*1024)  # Convert to MB
        except Exception as e:
            logging.error(f"Error getting GPU memory: {e}")
            return 4096

    def get_gpu_info(self) -> Optional[Dict[str, Union[float, int]]]:
        """Get current GPU utilization and memory usage"""
        if not self.pynvml_available:
            return None
            
        try:
            import pynvml
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                'gpu_util': utilization.gpu,
                'gpu_memory': memory.used / (1024*1024)  # Convert to MB
            }
        except:
            return None
    
    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations with proper cleanup"""
        stream = None
        
        try:
            if self.has_gpu:
                stream = self._get_next_stream()
                if stream:
                    with torch.cuda.stream(stream):
                        yield stream
                else:
                    yield None
            else:
                yield None
                
        except Exception as e:
            logging.warning(f"GPU context error: {e}, falling back to CPU")
            # Let the calling code handle fallback, don't try to yield again
            raise
            
        finally:
            if stream and torch.cuda.is_available():
                try:
                    torch.cuda.current_stream().synchronize()
                except Exception as e:
                    logging.error(f"CUDA synchronization error: {e}")

class ProgressTracker:
    def __init__(self, total_notes):
        self.progress_bar = tqdm(total=total_notes, 
                               desc="Processing notes",
                               unit="note")
        self.completed = 0
        self.failed = 0
        
    def update(self, success=True):
        self.completed += 1
        if not success:
            self.failed += 1
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            "success_rate": f"{(self.completed-self.failed)/self.completed*100:.1f}%"
        })

    def close(self):
        self.progress_bar.close()

# Replace your current EncoderQueue with:
class EncoderQueue:
    def __init__(self, num_workers=4, queue_size=32, max_concurrent=None):  # Increase queue size
        self.queue = Queue(maxsize=queue_size)
        self.pool = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = []
        self.lock = RLock()
        cpu_count = os.cpu_count() or 2
        self.max_concurrent = max_concurrent if max_concurrent is not None else cpu_count
        self.semaphore = threading.Semaphore(self.max_concurrent)
        
    def encode(self, ffmpeg_command):
        with self.semaphore:
            logging.info(f"EncoderQueue: Running command: {' '.join(ffmpeg_command)}")
            try:
                result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"EncoderQueue: Command failed: {result.stderr}")
                return result
            except Exception as e:
                logging.error(f"EncoderQueue: Error executing command: {str(e)}")
                raise

class EnhancedEncoderQueue(EncoderQueue):
    def __init__(self):
        self.gpu_manager = GPUManager()
        cpu_count = psutil.cpu_count(logical=False)
        gpu_count = 1 if self.gpu_manager.has_gpu else 0
        max_workers = min(32, (cpu_count + gpu_count) * 2)
        self.queue = Queue()
        self.semaphore = threading.Semaphore(max_workers)
        super().__init__(max_concurrent=max_workers)
        self.gpu_locks = [threading.Lock() for _ in range(max(1, gpu_count))]
        self.gpu_memory = self._get_gpu_memory()
        self.concurrent_streams = min(8, self.gpu_memory // 1024)
        self.result_cache = {}
        self.max_cache_size = 100
        
    def _get_gpu_count(self):
        try:
            nvidia_smi = subprocess.run(['nvidia-smi', '-L'], 
                                      capture_output=True, 
                                      text=True)
            return len(nvidia_smi.stdout.splitlines())
        except:
            return 0
        
    def _get_gpu_memory(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.total // (1024*1024)  # Convert to MB
        except:
            return 4096  # Default 4GB
        
    
    def _get_cache_key(self, ffmpeg_command):
        """Generate a cache key from ffmpeg command"""
        # Only use input/output files and core parameters for cache key
        key_parts = []
        skip_next = False
        for i, part in enumerate(ffmpeg_command):
            if skip_next:
                skip_next = False
                continue
            if part.startswith('-'):
                skip_next = True
                continue
            if part.endswith(('.mp4', '.wav')):
                key_parts.append(part)
        return '_'.join(key_parts)
            
    def encode(self, ffmpeg_command):
        try:
            cache_key = self._get_cache_key(ffmpeg_command)
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
                
            with self.semaphore:
                logging.info(f"EncoderQueue: Running command: {' '.join(ffmpeg_command)}")
                try:
                    with self.gpu_manager.gpu_context():
                        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                        if result.returncode == 0:
                            self.result_cache[cache_key] = result
                        return result
                except Exception as e:
                    logging.error(f"EncoderQueue: Error executing command: {str(e)}")
                    raise
        except Exception as e:
            logging.error(f"Encoding error: {str(e)}")
            raise
            
    def _get_available_gpu(self):
        try:
            nvidia_smi = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            utils = [int(x.strip(' %')) for x in nvidia_smi.stdout.splitlines()]
            return utils.index(min(utils))
        except:
            return None

# Then replace your current encoder_queue instantiation with:
encoder_queue = EnhancedEncoderQueue()