#!/usr/bin/env python3
"""
Enhanced GPU Setup with PyTorch CUDA Priority
Detects and configures GPU acceleration for AutoTuneSyncer
"""

import os
import sys
import logging
import subprocess
import warnings

# Suppress warnings during GPU setup
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize GPU availability flags
gpu_available = False
tf = None
torch_cuda_available = False

def detect_pytorch_cuda():
    """Detect PyTorch CUDA availability (most reliable method)"""
    global torch_cuda_available
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            torch_cuda_available = True
            
            logging.info(f"✓ PyTorch CUDA detected: {device_name}")
            logging.info(f"  CUDA version: {cuda_version}")
            logging.info(f"  Available devices: {device_count}")
            
            # Test GPU memory allocation
            try:
                test_tensor = torch.zeros(100, 100).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                logging.info("  GPU memory allocation test: PASSED")
                return True
            except Exception as e:
                logging.warning(f"  GPU memory test failed: {e}")
                return False
                
    except ImportError:
        logging.debug("PyTorch not available")
    except Exception as e:
        logging.warning(f"PyTorch CUDA check failed: {e}")
    
    torch_cuda_available = False
    return False

def detect_nvidia_smi():
    """Detect NVIDIA GPU using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout
            # Look for GPU information
            lines = output.split('\n')
            gpu_info = []
            for line in lines:
                if any(gpu_type in line for gpu_type in ['GeForce', 'RTX', 'GTX', 'Tesla', 'Quadro']):
                    gpu_info.append(line.strip())
            
            if gpu_info:
                logging.info("✓ NVIDIA GPU detected via nvidia-smi:")
                for info in gpu_info:
                    logging.info(f"  {info}")
                return True
                
    except FileNotFoundError:
        logging.debug("nvidia-smi not found in PATH")
    except subprocess.TimeoutExpired:
        logging.warning("nvidia-smi timeout")
    except Exception as e:
        logging.warning(f"nvidia-smi check failed: {e}")
    
    return False

def detect_tensorflow_gpu():
    """Detect TensorFlow GPU support"""
    try:
        import tensorflow as tf_module
        gpus = tf_module.config.experimental.list_physical_devices('GPU')
        if gpus:
            logging.info(f"✓ TensorFlow GPU detected: {len(gpus)} devices")
            for i, gpu in enumerate(gpus):
                logging.info(f"  Device {i}: {gpu}")
            return True, tf_module
    except ImportError:
        logging.debug("TensorFlow not available")
    except Exception as e:
        logging.warning(f"TensorFlow GPU check failed: {e}")
    
    return False, None

def is_gpu_available():
    """Check if GPU is available using comprehensive detection"""
    global gpu_available, tf
    
    # Priority 1: PyTorch CUDA (most reliable for our use case)
    pytorch_available = detect_pytorch_cuda()
    
    # Priority 2: NVIDIA drivers
    nvidia_available = detect_nvidia_smi()
    
    # Priority 3: TensorFlow GPU
    tf_available, tf_module = detect_tensorflow_gpu()
    if tf_available:
        tf = tf_module
    
    # Set overall GPU availability
    gpu_available = pytorch_available or nvidia_available or tf_available
    
    if gpu_available:
        logging.info("✓ GPU acceleration ENABLED")
        if pytorch_available:
            logging.info("  Primary backend: PyTorch CUDA")
        elif tf_available:
            logging.info("  Primary backend: TensorFlow")
        else:
            logging.info("  GPU detected but no acceleration backend available")
    else:
        logging.info("⚠ GPU acceleration DISABLED - using CPU only")
    
    return gpu_available

def get_gpu_info():
    """Get detailed GPU information"""
    info = {
        'available': gpu_available,
        'pytorch_cuda': torch_cuda_available,
        'tensorflow': tf is not None,
        'devices': []
    }
    
    # Get PyTorch device info
    if torch_cuda_available:
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'backend': 'pytorch'
                }
                info['devices'].append(device_info)
        except Exception as e:
            logging.warning(f"Error getting PyTorch device info: {e}")
    
    return info

def setup_gpu_memory_optimization():
    """Setup GPU memory optimization for efficient processing"""
    if not gpu_available:
        return False
    
    try:
        # PyTorch memory optimization
        if torch_cuda_available:
            import torch
            torch.cuda.empty_cache()
            # Set memory fraction to avoid OOM
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            logging.info("✓ PyTorch GPU memory optimization enabled")
        
        # TensorFlow memory optimization
        if tf is not None:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("✓ TensorFlow GPU memory growth enabled")
        
        return True
        
    except Exception as e:
        logging.warning(f"GPU memory optimization failed: {e}")
        return False

# Initialize GPU detection on import
try:
    gpu_available = is_gpu_available()
    setup_gpu_memory_optimization()
except Exception as e:
    logging.error(f"GPU setup initialization failed: {e}")
    gpu_available = False
    tf = None

# Compatibility functions
def get_tensorflow():
    """Get TensorFlow module if available"""
    return tf

def get_device():
    """Get the optimal device for processing"""
    if torch_cuda_available:
        try:
            import torch
            return torch.device('cuda')
        except:
            pass
    return 'cpu'

if __name__ == "__main__":
    # Test GPU setup when run directly
    print("=== GPU Setup Test ===")
    info = get_gpu_info()
    print(f"GPU Available: {info['available']}")
    print(f"PyTorch CUDA: {info['pytorch_cuda']}")
    print(f"TensorFlow: {info['tensorflow']}")
    print(f"Devices: {len(info['devices'])}")
    for device in info['devices']:
        print(f"  {device['name']} (ID: {device['id']}, Backend: {device['backend']})")
