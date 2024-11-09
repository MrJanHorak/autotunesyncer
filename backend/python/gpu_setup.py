import os
import sys
import json

def setup_cuda_paths():
    """Setup CUDA paths for TensorFlow"""
    # Set environment variables first
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    cuda_versions = ['11.2', '11.0', '10.1', '10.0']
    cuda_base = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
    
    for version in cuda_versions:
        cuda_path = os.path.join(cuda_base, f'v{version}')
        if os.path.exists(cuda_path):
            cuda_paths = {
                'base': cuda_path,
                'cudnn': 'C:\\Program Files\\NVIDIA\\CUDNN\\v8.1',
            }
            
            paths_to_add = [
                cuda_paths['base'],
                os.path.join(cuda_paths['base'], 'bin'),
                os.path.join(cuda_paths['base'], 'libnvvp'),
                os.path.join(cuda_paths['base'], 'extras', 'CUPTI', 'lib64'),
                os.path.join(cuda_paths['base'], 'include'),
            ]
            
            if os.path.exists(cuda_paths['cudnn']):
                paths_to_add.append(os.path.join(cuda_paths['cudnn'], 'bin'))
            
            os.environ['PATH'] = ';'.join(paths_to_add) + ';' + os.environ['PATH']
            os.environ['CUDA_PATH'] = cuda_paths['base']
            return True
    return False

# Initialize TensorFlow with GPU configuration
def initialize_tensorflow():
    if not setup_cuda_paths():
        return False, None
        
    try:
        import tensorflow as tf
        
        # Set TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure memory growth first
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Then configure logical devices
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                )
                
                # Set this GPU as the only visible device
                tf.config.set_visible_devices(gpus[0], 'GPU')
                
                print(f"GPU configuration successful. Found {len(gpus)} GPU(s)")
                return True, tf
            except Exception as e:
                print(f"GPU configuration failed: {e}")
                return False, tf
        return False, tf
    except Exception as e:
        print(f"TensorFlow initialization failed: {e}")
        return False, None

# Global variables to store GPU state
gpu_available, tf = initialize_tensorflow()
