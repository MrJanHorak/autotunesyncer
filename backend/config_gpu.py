import os
import sys
import ctypes
import platform
import subprocess

def find_dll_paths():
    """Find all relevant DLL paths"""
    cuda_paths = [
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6',
        'C:/Program Files/NVIDIA Corporation/NvToolsExt',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/libnvvp',
    ]
    
    dll_paths = []
    for path in cuda_paths:
        if os.path.exists(path):
            dll_paths.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dll')])
    return dll_paths

def configure_gpu_environment():
    """Configure environment variables for TensorFlow GPU support"""
    cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
    
    # Essential CUDA environment variables
    os.environ['CUDA_PATH'] = cuda_path
    os.environ['CUDA_HOME'] = cuda_path
    
    # Add CUDA paths to system PATH at the beginning
    cuda_dirs = [
        f"{cuda_path}/bin",
        f"{cuda_path}/libnvvp",
        f"{cuda_path}/lib/x64",
        'C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64',
    ]
    
    # Update PATH
    os.environ['PATH'] = ';'.join(cuda_dirs + [os.environ['PATH']])
    
    # TensorFlow specific configurations
    os.environ['TF_CUDA_PATHS'] = cuda_path
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CUDA_VERSION'] = '12.6'
    os.environ['TF_CUDNN_VERSION'] = '8'
    
    # Load all available CUDA DLLs
    dll_paths = find_dll_paths()
    loaded_dlls = []
    
    for dll_path in dll_paths:
        try:
            ctypes.CDLL(dll_path)
            loaded_dlls.append(os.path.basename(dll_path))
        except Exception:
            pass
    
    print("\nLoaded DLLs:", ', '.join(loaded_dlls))
    
    # Verify CUDA installation
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode()
        print("\nNVCC Version:")
        print(nvcc_output)
    except Exception as e:
        print("\nNVCC check failed:", str(e))

if __name__ == '__main__':
    configure_gpu_environment()
    
    # Import and configure tensorflow
    import tensorflow as tf
    
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"\nEnabled memory growth for {device}")
            except RuntimeError as e:
                print(f"\nMemory growth setting failed: {e}")
    
    print("\nTensorFlow Configuration:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Build info: {tf.sysconfig.get_build_info()}")
    
    # Test GPU computation
    try:
        with tf.device('/GPU:0'):
            print("\nTesting GPU computation...")
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("GPU computation successful!")
    except Exception as e:
        print(f"\nGPU computation failed: {e}")
