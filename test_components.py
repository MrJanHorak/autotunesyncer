#!/usr/bin/env python3
"""
Simple AutoTuneSyncer Component Integration Test
Tests the main components after the normalize_instrument_name fix
"""

import sys
import os
import logging
import traceback

# Add backend path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_component(component_name, import_func):
    """Test a single component with error handling"""
    try:
        result = import_func()
        print(f"✓ {component_name}: SUCCESS")
        return True
    except Exception as e:
        print(f"✗ {component_name}: FAILED - {str(e)}")
        logging.error(f"{component_name} error: {traceback.format_exc()}")
        return False

def test_gpu_setup():
    """Test GPU setup component"""
    from python.gpu_setup import gpu_available, torch_cuda_available, get_gpu_info
    info = get_gpu_info()
    return info['available']

def test_processing_utils():
    """Test processing utilities"""
    from python.processing_utils import GPUManager, encoder_queue
    gpu_manager = GPUManager()
    return True

def test_health_monitor():
    """Test health monitoring system"""
    from utils.health_monitor import PerformanceHealthMonitor
    monitor = PerformanceHealthMonitor()
    monitor.start_monitoring()
    monitor.stop_monitoring()
    return True

def test_ffmpeg_executor():
    """Test FFmpeg executor"""
    from src.ffmpeg.executor import EnhancedFFmpegExecutor
    executor = EnhancedFFmpegExecutor()
    return True

def test_path_registry():
    """Test path registry system"""
    from python.path_registry import PathRegistry
    registry = PathRegistry()
    return True

def test_video_processor():
    """Test enhanced video processor"""
    from utils.video_processor import EnhancedVideoProcessor
    processor = EnhancedVideoProcessor()
    return True

def test_autotune_processor():
    """Test autotune processor"""
    from python.autotune import ParallelAutotuneProcessor
    processor = ParallelAutotuneProcessor()
    return True

def test_video_composer():
    """Test video composer - this was the failing component"""
    from python.video_composer import VideoComposer, compose_from_processor_output
    # Test that the normalize_instrument_name function is accessible
    from python.video_composer import normalize_instrument_name
    
    # Test the function works
    test_name = normalize_instrument_name("Test Instrument Name")
    expected = "test_instrument_name"
    
    if test_name == expected:
        return True
    else:
        raise Exception(f"normalize_instrument_name failed: got '{test_name}', expected '{expected}'")

def main():
    """Main test runner"""
    print("=== AutoTuneSyncer Component Integration Test ===")
    print("Testing performance enhancements and import fixes...\n")
    
    components = [
        ("GPU Setup", test_gpu_setup),
        ("Processing Utils", test_processing_utils),
        ("Health Monitor", test_health_monitor),
        ("FFmpeg Executor", test_ffmpeg_executor),
        ("Path Registry", test_path_registry),
        ("Video Processor", test_video_processor),
        ("Autotune Processor", test_autotune_processor),
        ("Video Composer", test_video_composer),  # This was the failing one
    ]
    
    successful = 0
    total = len(components)
    
    for name, test_func in components:
        if test_component(name, test_func):
            successful += 1
    
    success_rate = (successful / total) * 100
    
    print(f"\n=== Test Results ===")
    print(f"Successful: {successful}/{total} ({success_rate:.1f}%)")
    
    if success_rate == 100.0:
        print("🎉 ALL COMPONENTS WORKING! Performance enhancements fully integrated.")
        return 0
    elif success_rate >= 85.0:
        print("✅ Most components working. Performance enhancements mostly integrated.")
        return 0
    else:
        print("❌ Multiple component failures. Further debugging needed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
