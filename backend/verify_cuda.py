import sys
import os
import ctypes
import platform

# Set environment variables before importing tensorflow
os.environ['CUDA_PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['CUDA_HOME'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
os.environ['PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin;' + os.environ['PATH']
os.environ['PATH'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/libnvvp;' + os.environ['PATH']
os.environ['TF_CUDA_PATHS'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'

# Additional TensorFlow specific variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load CUDA DLL explicitly
if platform.system() == 'Windows':
    try:
        ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cudart64_12.dll')
    except Exception as e:
        print(f"Failed to load CUDA DLL: {e}")

# Now import tensorflow
import tensorflow as tf

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def get_cuda_version():
    """Get CUDA version from nvcc or from CUDA runtime"""
    try:
        if platform.system() == 'Windows':
            cuda_path = os.environ.get('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6')
            # Try different DLL names
            dll_names = ['cudart64_12.dll', 'cudart64_120.dll', 'cudart64.dll']
            for dll_name in dll_names:
                try:
                    cuda_dll = ctypes.CDLL(f"{cuda_path}/bin/{dll_name}")
                    version = ctypes.c_int()
                    cuda_dll.cudaRuntimeGetVersion(ctypes.byref(version))
                    return f"{version.value//1000}.{(version.value%1000)//10}"
                except Exception:
                    continue
            raise Exception("No compatible CUDA DLL found")
    except Exception as e:
        print(f"Error getting CUDA version: {e}")
        # Try alternative method
        try:
            import subprocess
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
            version = version_line.split('V')[1].strip()
            return version
        except Exception as e2:
            print(f"Alternative CUDA version check failed: {e2}")
    return "Unknown"

print("\nSystem Information:")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA Runtime version: {get_cuda_version()}")

print("\nCUDA Configuration:")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print("\nDetailed GPU info:")
os.system('nvidia-smi')

print("\nCUDA Environment Variables:")
cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']
for var in cuda_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

# Test GPU computation
try:
    with tf.device('/GPU:0'):
        print("\nAttempting GPU computation...")
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("GPU computation successful!")
except Exception as e:
    print(f"\nGPU computation failed: {str(e)}")
