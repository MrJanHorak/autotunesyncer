import subprocess
import logging
import asyncio
import time
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from processing_utils import encoder_queue, GPUManager

gpu_manager = GPUManager()

# Performance monitoring
@contextmanager
def performance_monitor(operation_name):
    """Monitor performance of operations"""
    start_time = time.perf_counter()
    start_memory = psutil.virtual_memory().percent
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.virtual_memory().percent
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logging.info(f"Operation [{operation_name}]: {duration:.3f}s, Memory change: {memory_delta:+.1f}%")
        
        # Trigger GC if memory usage is high
        if end_memory > 80:
            gc.collect()
            if gpu_manager.has_gpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

def get_optimized_ffmpeg_params(use_gpu=True, preset="fast", quality="high"):
    """Get optimized FFmpeg parameters based on system capabilities"""
    
    # Force CPU mode since CUDA is failing - disable GPU until issues are resolved
    use_gpu = False
    
    if use_gpu and gpu_manager.has_gpu:
        # NVENC hardware acceleration
        params = {
            'hwaccel': 'cuda',
            'video_codec': 'h264_nvenc',
            'preset': 'p4',
            'crf': 23 if quality == "high" else 28,
            'gpu_options': [
                '-hwaccel_output_format', 'cuda',
                '-tune', 'hq',
                '-rc', 'vbr',
                '-cq', str(23 if quality == "high" else 28),
                '-b:v', '8M',
                '-maxrate', '12M',
                '-bufsize', '16M',
                '-surfaces', '64',
                '-gpu', '0'
            ]
        }
    else:
        # CPU fallback with optimized settings
        cpu_count = psutil.cpu_count(logical=False)
        params = {
            'video_codec': 'libx264',
            'preset': preset,
            'crf': 23 if quality == "high" else 28,
            'cpu_options': [
                '-threads', str(min(cpu_count, 8)),
                '-tune', 'fastdecode'
            ]
        }
    
    # Common optimizations
    params['common_options'] = [
        '-movflags', '+faststart',  # Progressive download
        '-pix_fmt', 'yuv420p',      # Compatibility
        '-g', '30',                 # GOP size for better seeking
        '-bf', '3',                 # B-frames for efficiency
        '-refs', '3'                # Reference frames
    ]
    
    return params

def get_optimized_ffmpeg_params_list(use_gpu=True, preset="fast", quality="high"):
    """Get optimized FFmpeg parameters as a list for legacy compatibility"""
    params = get_optimized_ffmpeg_params(use_gpu, preset, quality)
    base_params = []
    
    if use_gpu and gpu_manager.has_gpu:
        base_params.extend([
            '-hwaccel', params['hwaccel'],
            '-c:v', params['video_codec'],
            '-preset', params['preset'],
            '-crf', str(params['crf'])
        ])
        if 'gpu_options' in params:
            base_params.extend(params['gpu_options'])
    else:
        base_params.extend([
            '-c:v', params['video_codec'],
            '-preset', params['preset'],
            '-crf', str(params['crf'])
        ])
        if 'cpu_options' in params:
            base_params.extend(params['cpu_options'])
    
    if 'common_options' in params:
        base_params.extend(params['common_options'])
    
    return base_params

async def run_ffmpeg_command_async(cmd, timeout=300):
    """Run FFmpeg command asynchronously with timeout"""
    with performance_monitor(f"FFmpeg: {' '.join(cmd[:3])}"):
        try:
            # Use GPU context if available
            if gpu_manager.has_gpu:
                with gpu_manager.gpu_context():
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(), 
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        raise TimeoutError(f"FFmpeg command timed out after {timeout}s")
            else:
                # CPU fallback
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logging.error(f"FFmpeg command failed: {error_msg}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
            
            return stdout.decode() if stdout else ""
            
        except Exception as e:
            logging.error(f"FFmpeg async command failed: {e}")
            raise

def run_ffmpeg_command(cmd):
    """Enhanced FFmpeg command execution with performance optimizations"""
    with performance_monitor(f"FFmpeg: {' '.join(cmd[:3])}"):
        try:
            # Add performance optimizations to command
            optimized_cmd = list(cmd)
            
            # Add thread optimization if not present
            if '-threads' not in cmd:
                cpu_count = psutil.cpu_count(logical=False)
                optimized_cmd.extend(['-threads', str(min(cpu_count, 8))])
              # Use GPU context if available, with fallback to CPU
            try:
                if gpu_manager.has_gpu:
                    with gpu_manager.gpu_context():
                        result = subprocess.run(
                            optimized_cmd, 
                            check=True, 
                            capture_output=True, 
                            text=True,
                            timeout=300  # 5 minute timeout
                        )
                else:                    result = subprocess.run(
                        optimized_cmd, 
                        check=True, 
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
            except Exception as gpu_error:
                # Fallback to CPU if GPU fails
                logging.warning(f"GPU processing failed: {gpu_error}, falling back to CPU")
                  # Create CPU-safe command by removing all GPU-specific parameters
                cpu_cmd = []
                skip_next = False
                
                for i, arg in enumerate(optimized_cmd):
                    if skip_next:
                        skip_next = False
                        continue
                        
                    # Skip GPU-specific parameters and their values
                    if arg in ['-hwaccel', '-hwaccel_device', '-hwaccel_output_format', '-gpu', '-surfaces', 
                              '-tune', '-rc', '-cq', '-b:v', '-maxrate', '-bufsize']:
                        skip_next = True
                        continue
                    elif arg in ['cuda', 'h264_nvenc', 'hevc_nvenc']:
                        # Replace GPU encoders with CPU equivalents
                        if arg == 'h264_nvenc':
                            cpu_cmd.append('libx264')
                        elif arg == 'hevc_nvenc':
                            cpu_cmd.append('libx265')
                        elif arg == 'cuda':
                            continue  # Skip cuda value for hwaccel
                    elif arg == 'p4':
                        # Replace GPU preset with CPU preset
                        cpu_cmd.append('fast')
                    elif arg in ['hq', 'vbr']:
                        # Skip GPU-specific tune/rate control values
                        continue
                    else:
                        cpu_cmd.append(arg)
                
                result = subprocess.run(
                    cpu_cmd, 
                    check=True, 
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            return result
        except subprocess.TimeoutExpired:
            logging.error(f"FFmpeg command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed: {e.stderr}")
            raise

def encode_video_batch(commands, max_workers=None):
    """Encode multiple videos in parallel"""
    if max_workers is None:
        max_workers = min(4, psutil.cpu_count(logical=False))
    
    with performance_monitor(f"Batch encoding {len(commands)} videos"):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cmd = {
                executor.submit(encode_video, cmd): cmd 
                for cmd in commands
            }
            
            results = []
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Successfully encoded: {cmd[-1]}")
                except Exception as e:
                    logging.error(f"Failed to encode {cmd[-1]}: {e}")
                    results.append(None)
            
            return results

def encode_video(cmd):
    """Enhanced video encoding with queue management"""
    with performance_monitor(f"Encoding: {cmd[-1] if cmd else 'unknown'}"):
        # Add performance optimizations to encoding command
        optimized_cmd = list(cmd)
        
        # Ensure we have optimal encoding settings
        if '-preset' not in cmd and gpu_manager.has_gpu:
            optimized_cmd.extend(['-preset', 'p4'])
        elif '-preset' not in cmd:
            optimized_cmd.extend(['-preset', 'fast'])
        
        result = encoder_queue.encode(optimized_cmd)
        if result.returncode != 0:
            logging.error(f"Encoding failed: {result.stderr}")
            raise Exception(f"Encoding failed: {result.stderr}")
        return result

async def encode_video_async(cmd):
    """Asynchronous video encoding"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(executor, encode_video, cmd)

def validate_video_fast(output_path):
    """Fast video validation using ffprobe"""
    with performance_monitor(f"Validating: {output_path}"):
        try:
            # Use ffprobe for faster validation
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                output_path
            ]
            
            result = subprocess.run(
                probe_cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode != 0:
                logging.error(f"Validation failed: {result.stderr}")
                return False
            
            # Check if we got packet count
            packet_count = int(result.stdout.strip()) if result.stdout.strip() else 0
            is_valid = packet_count > 0
            
            logging.info(f"Video validation: {output_path} - {packet_count} packets, valid: {is_valid}")
            return is_valid
            
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

def validate_video(output_path):
    """Validate video file integrity with fallback"""
    # Try fast validation first
    if validate_video_fast(output_path):
        return True
    
    # Fallback to original method
    with performance_monitor(f"Deep validation: {output_path}"):
        validate_cmd = [
            'ffmpeg',
            '-v', 'error',
            '-i', output_path,
            '-f', 'null',
            '-'
        ]
        result = encoder_queue.encode(validate_cmd)
        if result.returncode != 0:
            logging.error(f"Deep validation failed: {result.stderr}")
            raise Exception(f"Deep validation failed: {result.stderr}")
        return True

# Cache for commonly used operations
_ffmpeg_cache = {}

def get_video_info_cached(video_path):
    """Get video information with caching"""
    if video_path in _ffmpeg_cache:
        return _ffmpeg_cache[video_path]
    
    try:
        probe_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            _ffmpeg_cache[video_path] = info
            return info
    except Exception as e:
        logging.error(f"Failed to get video info for {video_path}: {e}")
    
    return None

def clear_video_cache():
    """Clear video info cache"""
    global _ffmpeg_cache
    _ffmpeg_cache.clear()
    gc.collect()