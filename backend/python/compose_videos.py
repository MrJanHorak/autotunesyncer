import json
import logging
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import psutil

# Import health monitoring - adjust path as needed
try:
    sys.path.append(str(Path(__file__).parent.parent / 'utils'))
    from health_monitor import HealthMonitor
    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    logging.warning("Health monitoring not available")
    HEALTH_MONITORING_AVAILABLE = False
    HealthMonitor = None

from audio_processor import process_track_videos, AudioVideoProcessor
from video_composer import compose_from_processor_output
from processing_utils import GPUManager

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@contextmanager
def performance_monitoring(operation_name):
    """Context manager for performance monitoring"""
    start_time = time.perf_counter()
    start_memory = psutil.virtual_memory().percent
    
    try:
        logging.info(f"Starting {operation_name}")
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.virtual_memory().percent
        duration = end_time - start_time
        
        logging.info(f"Completed {operation_name} in {duration:.2f}s")
        logging.info(f"Memory usage: {start_memory:.1f}% -> {end_memory:.1f}%")

class EnhancedVideoComposer:
    """Enhanced video composer with performance optimizations"""
    
    def __init__(self, performance_mode=True, max_workers=None):
        self.performance_mode = performance_mode
        self.max_workers = max_workers or min(4, psutil.cpu_count(logical=False))
        self.gpu_manager = GPUManager()
        self.health_monitor = HealthMonitor() if HEALTH_MONITORING_AVAILABLE else None
        
        logging.info(f"EnhancedVideoComposer initialized: performance_mode={performance_mode}, max_workers={self.max_workers}")
    
    def process_composition(self, config_path, output_path=None):
        """Process video composition with enhanced performance monitoring"""
        with performance_monitoring("Video composition"):
            try:
                # Start health monitoring
                if self.health_monitor:
                    self.health_monitor.start_monitoring()
                
                with open(config_path) as f:
                    config = json.load(f)
                
                # Enhanced processor setup
                processor = AudioVideoProcessor()
                processor.setup_temp_directory()
                
                # Process track videos with monitoring
                with performance_monitoring("Track video processing"):
                    result = process_track_videos(
                        config.get('tracks', {}), 
                        config.get('videos', {}),
                        processor
                    )
                
                # Determine output path
                if not output_path:
                    output_path = Path(processor.videos_dir).parent / f"final_composition_{processor.session_id}.mp4"
                
                # Compose final video with monitoring
                with performance_monitoring("Final composition"):
                    composition_result = compose_from_processor_output(
                        {
                            'processed_videos_dir': processor.videos_dir,
                            'tracks': config['tracks'],
                            'processed_files': result
                        },
                        str(output_path)
                    )
                
                if not composition_result:
                    raise Exception("Composition failed - no result returned")
                    
                # Verify final output exists
                if not os.path.exists(output_path):
                    raise Exception(f"Final composition file not found at {output_path}")
                
                # Get performance metrics
                metrics = self.health_monitor.get_session_metrics() if self.health_monitor else {}
                logging.info(f"Composition completed successfully")
                logging.info(f"Performance metrics: {metrics}")
                
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'metrics': metrics,
                    'data': {
                        'processed': result,
                        'composition': composition_result
                    }
                }
                
            except Exception as e:
                logging.error(f"Composition error: {str(e)}")
                return {
                    'success': False,
                    'error': str(e)
                }
            finally:
                if self.health_monitor:
                    self.health_monitor.stop_monitoring()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Enhanced Video Composition with Performance Monitoring')
        parser.add_argument('config_path', help='Path to config JSON file')
        parser.add_argument('--output', '-o', help='Output video path')
        parser.add_argument('--performance-mode', action='store_true', default=True, 
                          help='Enable performance optimizations')
        parser.add_argument('--max-workers', type=int, 
                          help='Maximum number of parallel workers')
        args = parser.parse_args()
        
        # Create enhanced composer
        composer = EnhancedVideoComposer(
            performance_mode=args.performance_mode,
            max_workers=args.max_workers
        )
        
        # Process composition
        result = composer.process_composition(args.config_path, args.output)
        
        # Output JSON result
        print(json.dumps(result), flush=True)
        
        if not result['success']:
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }), flush=True)
        sys.exit(1)
