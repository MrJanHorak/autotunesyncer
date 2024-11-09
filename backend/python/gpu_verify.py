import os
import sys
import json

def verify_gpu():
    # Set environment variables first
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Check CUDA paths
    cuda_versions = ['11.2', '11.0', '10.1', '10.0']
    cuda_base = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
    cuda_found = False
    
    for version in cuda_versions:
        cuda_path = os.path.join(cuda_base, f'v{version}')
        if os.path.exists(cuda_path):
            os.environ['CUDA_PATH'] = cuda_path
            os.environ['PATH'] = f"{cuda_path}\\bin;{os.environ['PATH']}"
            cuda_found = True
            break
    
    result = {
        'cuda_available': cuda_found,
        'cuda_path': os.environ.get('CUDA_PATH', 'Not found'),
        'gpu_available': False,
        'gpu_devices': [],
        'error': None
    }
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        result['gpu_available'] = len(gpus) > 0
        result['gpu_devices'] = [gpu.name for gpu in gpus]
        
        # Test GPU computation
        if result['gpu_available']:
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
    except Exception as e:
        result['error'] = str(e)
    
    print(json.dumps(result))
    return result['gpu_available']

if __name__ == '__main__':
    verify_gpu()
