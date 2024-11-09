import os
import sys
import platform

# Import configured GPU state
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))
from gpu_setup import gpu_available, tf

if not gpu_available or tf is None:
    print("Warning: GPU acceleration not available")
    sys.exit(1)

print("\nTensorFlow Configuration:")
print(f"Python version: {platform.python_version()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {tf.test.is_built_with_cuda()}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")

# Try a test computation
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print("\nGPU test computation successful!")
except Exception as e:
    print(f"\nGPU test failed: {e}")
    sys.exit(1)