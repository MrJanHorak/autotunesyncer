# GPU Configuration for AutoTuneSyncer
# Optimized settings for your RTX 3050 Laptop GPU

# GPU Hardware Detection
GPU_ENABLED = True
GPU_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"
CUDA_VERSION = "11.8"

# Simplified GPU configuration - removing problematic options
GPU_CONFIG = {
    'enabled': True,
    'device': 'cuda',
    'memory_fraction': 0.8,
    'allow_growth': True,
    
    # Simplified GPU encoding settings
    'encoding': {
        'codec': 'h264_nvenc',
        'preset': 'fast',
        'crf': 23,
        'pix_fmt': 'yuv420p',
        'profile': 'main',
        'level': '4.0'
    },
    
    # Disabled problematic options
    'disabled_options': {
        'hwaccel_output_format': None,  # Causes compatibility issues
        'surfaces': None,  # Hardware specific, causes issues
        'gpu': None,  # Explicit GPU selection can fail
        'tune': None,  # Not supported on all GPUs
        'rc': None,  # May cause issues
        'cq': None,  # Conflicts with crf
        'bufsize': None,  # Can cause buffer issues
        'maxrate': None  # Can cause rate control issues
    }
}

# ffmpeg GPU Settings - Conservative settings for RTX 3050
FFMPEG_GPU_CONFIG = {
    'hwaccel': 'cuda',
    'encoder': 'h264_nvenc',
    'preset': 'fast',
    'pixel_format': 'yuv420p',
    'bitrate': '5M',
    'max_bitrate': '10M',
    'buffer_size': '10M'
}

# Video Processing Settings
VIDEO_CONFIG = {
    'default_resolution': (1920, 1080),
    'default_fps': 30,
    'grid_resolutions': {
        '1x1': (1920, 1080),
        '2x2': (960, 540),
        '3x3': (640, 360),
        '4x4': (480, 270)
    },
    'max_concurrent_encodes': 2,  # RTX 3050 can handle 2 simultaneous encodes
    'memory_limit_mb': 4096,  # 4GB GPU memory limit
}

# PyTorch GPU Settings
PYTORCH_CONFIG = {
    'device': 'cuda',
    'memory_fraction': 0.8,  # Use 80% of GPU memory
    'benchmark': True,  # Enable cuDNN benchmarking
    'deterministic': False,  # For performance
    'batch_size': 32,
    'num_workers': 4
}

# MIDI Processing Settings
MIDI_CONFIG = {
    'sample_rate': 44100,
    'buffer_size': 1024,
    'use_gpu_timing': True,
    'batch_process_notes': True,
    'max_notes_per_batch': 1000
}

# Performance Optimization
PERFORMANCE_CONFIG = {
    'prefetch_frames': 10,
    'async_processing': True,
    'memory_pool_size': 1024,  # MB
    'cache_decoded_frames': True,
    'parallel_encoding': True
}

# Error Handling
ERROR_CONFIG = {
    'retry_attempts': 3,
    'fallback_to_cpu': True,
    'log_gpu_usage': True,
    'monitor_memory': True
}

def get_optimal_settings_for_resolution(width, height):
    """Get optimal GPU settings based on resolution"""
    total_pixels = width * height
    
    if total_pixels <= 640 * 480:  # SD
        return {
            'preset': 'ultrafast',
            'bitrate': '1M',
            'max_concurrent': 4
        }
    elif total_pixels <= 1280 * 720:  # HD
        return {
            'preset': 'veryfast',
            'bitrate': '3M',
            'max_concurrent': 3
        }
    elif total_pixels <= 1920 * 1080:  # FHD
        return {
            'preset': 'fast',
            'bitrate': '5M',
            'max_concurrent': 2
        }
    else:  # 4K+
        return {
            'preset': 'medium',
            'bitrate': '10M',
            'max_concurrent': 1
        }

def get_grid_config(grid_size):
    """Get configuration for grid layout"""
    rows, cols = grid_size
    total_cells = rows * cols
    
    cell_width = 1920 // cols
    cell_height = 1080 // rows
    
    return {
        'cell_resolution': (cell_width, cell_height),
        'total_cells': total_cells,
        'recommended_bitrate': f'{max(1, 5 // total_cells)}M',
        'max_concurrent': min(2, total_cells)
    }

def get_memory_usage_estimate(resolution, fps, duration_seconds):
    """Estimate GPU memory usage"""
    width, height = resolution
    pixels_per_frame = width * height
    frames_total = fps * duration_seconds
    
    # Rough estimation in MB
    memory_mb = (pixels_per_frame * frames_total * 3) / (1024 * 1024)  # 3 bytes per pixel
    return memory_mb

def validate_gpu_config():
    """Validate GPU configuration"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_memory < 2:
            print("⚠️  Low GPU memory detected, reducing settings")
            VIDEO_CONFIG['max_concurrent_encodes'] = 1
            PYTORCH_CONFIG['memory_fraction'] = 0.6
        
        print(f"✅ GPU Config validated for {GPU_NAME}")
        print(f"✅ Available GPU memory: {gpu_memory:.1f} GB")
        return True
    except ImportError:
        print("❌ PyTorch not available")
        return False
