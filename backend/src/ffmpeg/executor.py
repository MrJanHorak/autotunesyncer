import logging
import subprocess
import shutil
import time
import psutil
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import threading

# Thread-safe execution tracking
executor_lock = threading.RLock()
execution_stats = {
    'total_commands': 0,
    'successful_commands': 0,
    'failed_commands': 0,
    'total_time': 0.0
}

@contextmanager
def performance_monitor(operation_name: str):
    """Monitor performance of FFmpeg operations"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        with executor_lock:
            execution_stats['total_time'] += duration
        logging.info(f"FFmpeg operation [{operation_name}] completed in {duration:.3f}s")

class EnhancedFFmpegExecutor:
    """Enhanced FFmpeg executor with parallel processing and performance monitoring"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, psutil.cpu_count(logical=False))
        self.ffmpeg_path = self._find_ffmpeg()
        
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable path"""
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            raise Exception("FFmpeg executable not found in system PATH.")
        return ffmpeg_path
    
    def execute_command(self, ffmpeg_cmd: List[str], operation_name: str = "unknown") -> subprocess.CompletedProcess:
        """Execute a single FFmpeg command with performance monitoring"""
        with performance_monitor(operation_name):
            with executor_lock:
                execution_stats['total_commands'] += 1
            
            try:
                # Ensure we use the correct FFmpeg path
                cmd = [self.ffmpeg_path] + ffmpeg_cmd[1:]
                
                logging.debug(f"Executing FFmpeg command: {' '.join(cmd)}")
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=300  # 5 minute timeout
                )
                
                logging.debug(f"FFmpeg stdout: {process.stdout}")
                if process.stderr:
                    logging.debug(f"FFmpeg stderr: {process.stderr}")
                
                if process.returncode != 0:
                    with executor_lock:
                        execution_stats['failed_commands'] += 1
                    raise Exception(f"FFmpeg failed: {process.stderr.strip()}")
                
                with executor_lock:
                    execution_stats['successful_commands'] += 1
                    
                return process
                
            except subprocess.TimeoutExpired:
                with executor_lock:
                    execution_stats['failed_commands'] += 1
                raise Exception(f"FFmpeg command timed out after 300 seconds")
            except Exception as e:
                with executor_lock:
                    execution_stats['failed_commands'] += 1
                logging.error(f"Error executing FFmpeg command: {e}")
                raise
    
    def execute_commands_parallel(self, commands: List[Dict]) -> List[subprocess.CompletedProcess]:
        """Execute multiple FFmpeg commands in parallel
        
        Args:
            commands: List of dicts with 'cmd' and 'operation_name' keys
        """
        if not commands:
            return []
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cmd = {}
            
            for cmd_info in commands:
                cmd = cmd_info['cmd']
                operation_name = cmd_info.get('operation_name', 'batch_operation')
                
                future = executor.submit(
                    self.execute_command, 
                    cmd, 
                    operation_name
                )
                future_to_cmd[future] = cmd_info
            
            # Collect results
            for future in as_completed(future_to_cmd):
                cmd_info = future_to_cmd[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Successfully executed: {cmd_info.get('operation_name', 'unknown')}")
                except Exception as e:
                    logging.error(f"Failed to execute {cmd_info.get('operation_name', 'unknown')}: {e}")
                    results.append(None)
        
        return results
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        with executor_lock:
            return execution_stats.copy()

# Global enhanced executor instance
enhanced_executor = EnhancedFFmpegExecutor()

def execute_ffmpeg_command(ffmpeg_cmd: List[str]) -> None:
    """Execute FFmpeg command and log outputs (legacy compatibility)"""
    enhanced_executor.execute_command(ffmpeg_cmd, "legacy_command")

def execute_ffmpeg_commands_parallel(commands: List[List[str]], operation_names: List[str] = None) -> List[subprocess.CompletedProcess]:
    """Execute multiple FFmpeg commands in parallel"""
    if operation_names is None:
        operation_names = [f"parallel_cmd_{i}" for i in range(len(commands))]
    
    command_dicts = [
        {'cmd': cmd, 'operation_name': name}
        for cmd, name in zip(commands, operation_names)
    ]
    
    return enhanced_executor.execute_commands_parallel(command_dicts)

def get_ffmpeg_stats() -> Dict:
    """Get FFmpeg execution statistics"""
    return enhanced_executor.get_execution_stats()