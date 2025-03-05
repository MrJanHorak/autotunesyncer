import psutil
import GPUtil
import logging
from datetime import datetime

def get_system_metrics():
    """Monitor system resource usage"""
    metrics = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics['gpu_util'] = gpus[0].load * 100
            metrics['gpu_memory'] = gpus[0].memoryUtil * 100
    except:
        pass
        
    return metrics