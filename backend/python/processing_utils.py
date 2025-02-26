from tqdm import tqdm
from typing import Optional, Dict, Union
import psutil
import logging
import threading
import subprocess
import os
from typing import Optional
from queue import Queue
from contextlib import contextmanager


class GPUManager:
    """Manages GPU detection and monitoring for video processing"""
    
    def __init__(self):
        self.has_gpu = False
        self.handle = None
        self.pynvml_available = False
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
    def gpu_context(self, ffmpeg_command: list):
        """Context manager for GPU-aware FFmpeg encoding"""
        if self.has_gpu:
            ffmpeg_command.extend([
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda"
            ])
        try:
            yield
        finally:
            pass  # Cleanup if needed


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
    def __init__(self, max_concurrent=2):
        cpu_count = os.cpu_count() or 2
        self.max_concurrent = max_concurrent
        self.queue = Queue()
        self.semaphore = threading.Semaphore(max_concurrent)
        
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
        self.concurrent_streams = min(8, self.gpu_memory // 1024)  # 2GB per stream
        
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
            
    def encode(self, ffmpeg_command):
        with self.semaphore:
            logging.info(f"EncoderQueue: Running command: {' '.join(ffmpeg_command)}")
            try:
                # Use GPU manager context
                with self.gpu_manager.gpu_context(ffmpeg_command):
                    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                    if result.returncode != 0:
                        logging.error(f"EncoderQueue: Command failed: {result.stderr}")
                    return result
            except Exception as e:
                logging.error(f"EncoderQueue: Error executing command: {str(e)}")
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