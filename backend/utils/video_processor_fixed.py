#!/usr/bin/env python3
"""
Enhanced Video Processor with Performance Optimizations
Supports parallel processing, memory management, and progress reporting
"""

import sys
import json
import logging
import os
import subprocess
import argparse
import time
import psutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Any

# Configure logging for performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance during video processing"""
    
    def __init__(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        self.disk_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Memory usage
                memory_info = psutil.virtual_memory()
                self.memory_usage.append({
                    'timestamp': time.time(),
                    'used': memory_info.used,
                    'available': memory_info.available,
                    'percent': memory_info.percent
                })
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_usage.append({
                    'timestamp': time.time(),
                    'percent': cpu_percent
                })
                
                # Disk usage for temp directory
                try:
                    disk_info = psutil.disk_usage(os.getcwd())
                    self.disk_usage.append({
                        'timestamp': time.time(),
                        'used': disk_info.used,
                        'free': disk_info.free,
                        'percent': disk_info.used / disk_info.total * 100
                    })
                except:
                    pass
                
                time.sleep(2)  # Monitor every 2 seconds
                
            except Exception as e:
                logger.warning(f"Performance monitoring error: {e}")
                time.sleep(5)
    
    def get_performance_summary(self):
        """Get performance summary"""
        total_time = time.time() - self.start_time
        
        avg_memory = sum(m['percent'] for m in self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        max_memory = max(m['percent'] for m in self.memory_usage) if self.memory_usage else 0
        avg_cpu = sum(c['percent'] for c in self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = max(c['percent'] for c in self.cpu_usage) if self.cpu_usage else 0
        
        return {
            'total_processing_time': total_time,
            'average_memory_usage': avg_memory,
            'peak_memory_usage': max_memory,
            'average_cpu_usage': avg_cpu,
            'peak_cpu_usage': max_cpu,
            'samples_collected': len(self.memory_usage)
        }

class EnhancedVideoProcessor:
    """Enhanced video processor with performance optimizations"""
    
    def __init__(self, performance_mode=True, memory_limit_gb=4, parallel_tracks=None):
        self.performance_mode = performance_mode
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.parallel_tracks = parallel_tracks or min(4, cpu_count())
        self.monitor = PerformanceMonitor()
        self.temp_files = []
        
        logger.info(f"Initialized processor: performance_mode={performance_mode}, "
                   f"memory_limit={memory_limit_gb}GB, parallel_tracks={self.parallel_tracks}")
    
    def report_progress(self, progress: int, message: str = ""):
        """Report progress to parent process"""
        print(f"PROGRESS:{progress}", flush=True)
        if message:
            logger.info(f"Progress {progress}%: {message}")
    
    def validate_input_files(self, midi_path: str, video_files_path: str) -> bool:
        """Validate input files exist and are readable"""
        try:
            if not Path(midi_path).exists():
                logger.error(f"MIDI file not found: {midi_path}")
                return False
            
            if not Path(video_files_path).exists():
                logger.error(f"Video files JSON not found: {video_files_path}")
                return False
            
            # Validate JSON format
            with open(midi_path) as f:
                json.load(f)
            with open(video_files_path) as f:
                json.load(f)
            
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def get_optimal_ffmpeg_settings(self, hardware_accel=True) -> Dict[str, Any]:
        """Get optimal FFmpeg settings based on system capabilities"""
        settings = {
            'video_codec': 'libx264',
            'preset': 'ultrafast',
            'audio_codec': 'aac',
            'pixel_format': 'yuv420p'
        }
        
        if hardware_accel and self.performance_mode:
            # Check for hardware acceleration support
            try:
                result = subprocess.run(['ffmpeg', '-encoders'], 
                                      capture_output=True, text=True, timeout=10)
                output = result.stdout
                
                if 'h264_nvenc' in output:
                    settings.update({
                        'video_codec': 'h264_nvenc',
                        'preset': 'p2',
                        'extra_options': ['-gpu', '0', '-2pass', '0']
                    })
                    logger.info("Using NVIDIA hardware acceleration")
                elif 'h264_amf' in output:
                    settings.update({
                        'video_codec': 'h264_amf',
                        'preset': 'speed',
                        'extra_options': ['-usage', 'transcoding']
                    })
                    logger.info("Using AMD hardware acceleration")
                elif 'h264_qsv' in output:
                    settings.update({
                        'video_codec': 'h264_qsv',
                        'preset': 'veryfast',
                        'extra_options': ['-look_ahead', '0']
                    })
                    logger.info("Using Intel QuickSync acceleration")
                else:
                    logger.info("No hardware acceleration available, using CPU")
                    
            except Exception as e:
                logger.warning(f"Could not detect hardware acceleration: {e}")
        
        return settings
    
    def process_single_video(self, track_id: str, track_data: Dict, output_dir: Path, 
                           ffmpeg_settings: Dict) -> Optional[str]:
        """Process a single video track with optimizations"""
        try:
            logger.info(f"Processing track: {track_id}")
            
            # Handle video data
            if 'path' in track_data and Path(track_data['path']).exists():
                input_path = track_data['path']
            elif 'videoData' in track_data:
                # Handle video buffer data - save to temp file
                temp_path = output_dir / f"temp_{track_id}_input.mp4"
                video_data = track_data['videoData']
                
                # Handle both raw bytes and base64-encoded strings
                if isinstance(video_data, str):
                    # Assume base64-encoded string
                    import base64
                    try:
                        video_data = base64.b64decode(video_data)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 video data for track {track_id}: {e}")
                        return None
                elif not isinstance(video_data, bytes):
                    logger.error(f"Invalid video data type for track {track_id}: {type(video_data)}")
                    return None
                
                with open(temp_path, 'wb') as f:
                    f.write(video_data)
                input_path = str(temp_path)
                self.temp_files.append(temp_path)
            else:
                logger.error(f"No valid video data for track {track_id}")
                return None
            
            # Process video with optimizations
            output_path = output_dir / f"processed_{track_id}.mp4"
            
            # Build FFmpeg command with optimal settings
            cmd = self._build_ffmpeg_command(
                input_path, str(output_path), track_data, ffmpeg_settings
            )
            
            # Execute with timeout and monitoring
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout per track
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg failed for track {track_id}: {result.stderr}")
                return None
            
            # Validate output
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"Output file invalid for track {track_id}")
                return None
            
            logger.info(f"Successfully processed track {track_id}: {output_path.stat().st_size} bytes")
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing track {track_id}")
            return None
        except Exception as e:
            logger.error(f"Error processing track {track_id}: {e}")
            return None
    
    def _build_ffmpeg_command(self, input_path: str, output_path: str, 
                            track_data: Dict, settings: Dict) -> List[str]:
        """Build optimized FFmpeg command"""
        cmd = ['ffmpeg', '-i', input_path]
        
        # Add hardware acceleration if available
        if 'extra_options' in settings:
            cmd.extend(settings['extra_options'])
        
        # Video processing options
        cmd.extend([
            '-c:v', settings['video_codec'],
            '-preset', settings['preset'],
            '-pix_fmt', settings['pixel_format']
        ])
        
        # Layout and scaling based on track data
        layout = track_data.get('layout', {})
        if layout:
            width = layout.get('width', 960)
            height = layout.get('height', 720)
            cmd.extend(['-vf', f'scale={width}:{height}'])
        else:
            cmd.extend(['-vf', 'scale=960:720'])
        
        # Audio processing
        cmd.extend([
            '-c:a', settings['audio_codec'],
            '-ar', '44100',
            '-ac', '2',
            '-b:a', '192k'
        ])
        
        # Performance optimizations
        cmd.extend([
            '-threads', str(min(4, cpu_count())),
            '-movflags', '+faststart',
            '-y',  # Overwrite output
            output_path
        ])
        
        return cmd
    
    def combine_videos(self, processed_videos: Dict[str, str], midi_data: Dict, 
                      output_path: str) -> bool:
        """Combine processed videos into final composition"""
        try:
            logger.info("Starting video combination")
            self.report_progress(80, "Combining videos")
            
            if not processed_videos:
                logger.error("No processed videos to combine")
                return False
            
            # Single video case
            if len(processed_videos) == 1:
                single_video = list(processed_videos.values())[0]
                subprocess.run(['ffmpeg', '-i', single_video, '-c', 'copy', '-y', output_path], 
                             check=True)
                logger.info("Single video copied to output")
                return True
            
            # Multiple videos - create complex filter
            input_files = list(processed_videos.values())
            filter_complex = self._generate_combination_filter(input_files, midi_data)
            
            # Build combination command
            cmd = ['ffmpeg']
            for video_path in input_files:
                cmd.extend(['-i', video_path])
            
            cmd.extend([
                '-filter_complex', filter_complex,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-map', '[outv]',
                '-map', '[outa]',
                '-y',
                output_path
            ])
            
            logger.info(f"Combining {len(input_files)} videos")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Video combination failed: {result.stderr}")
                return False
            
            self.report_progress(95, "Video combination complete")
            return True
            
        except Exception as e:
            logger.error(f"Error combining videos: {e}")
            return False
    
    def _generate_combination_filter(self, input_files: List[str], midi_data: Dict) -> str:
        """Generate FFmpeg filter complex for combining videos"""
        num_videos = len(input_files)
        
        if num_videos == 2:
            # Side by side
            return '[0:v][1:v]hstack=inputs=2[outv];[0:a][1:a]amix=inputs=2[outa]'
        elif num_videos <= 4:
            # 2x2 grid
            return ('[0:v][1:v]hstack=inputs=2[top];'
                   '[2:v][3:v]hstack=inputs=2[bottom];'
                   '[top][bottom]vstack=inputs=2[outv];'
                   '[0:a][1:a][2:a][3:a]amix=inputs=4[outa]')
        else:
            # Dynamic grid layout for 5+ videos
            cols = min(4, int(num_videos ** 0.5) + 1)  # Max 4 columns for readability
            rows = (num_videos + cols - 1) // cols
            
            filter_parts = []
            
            # Scale all videos to fit grid
            cell_width = 1920 // cols
            cell_height = 1080 // rows
            for i in range(num_videos):
                filter_parts.append(f'[{i}:v]scale={cell_width}:{cell_height}[v{i}]')
            
            # Create grid composition
            if num_videos <= 9:  # 3x3 grid max
                # Build overlay chain for grid layout
                overlay_base = '[v0]'
                for i in range(1, num_videos):
                    row = (i - 1) // cols
                    col = (i - 1) % cols
                    x = col * cell_width
                    y = row * cell_height
                    
                    if i == num_videos - 1:
                        filter_parts.append(f'{overlay_base}[v{i}]overlay={x}:{y}[outv]')
                    else:
                        filter_parts.append(f'{overlay_base}[v{i}]overlay={x}:{y}[tmp{i}]')
                        overlay_base = f'[tmp{i}]'
            else:
                # For many videos, use a simpler stacking approach
                filter_parts.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[outv]')
            
            # Fix audio mixing - remove '+' separators  
            audio_inputs = ''.join([f'[{i}:a]' for i in range(num_videos)])
            filter_parts.append(f'{audio_inputs}amix=inputs={num_videos}[outa]')
            
            return ';'.join(filter_parts)
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")
        self.temp_files.clear()
    
    def process_videos(self, midi_data: Dict, video_files: Dict, output_path: str) -> bool:
        """Main video processing pipeline with performance optimizations"""
        try:
            logger.info("Starting enhanced video processing")
            self.monitor.start_monitoring()
            self.report_progress(10, "Initializing processing")
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get optimal settings
            ffmpeg_settings = self.get_optimal_ffmpeg_settings()
            self.report_progress(20, "Configured processing settings")
            
            # Process videos in parallel
            processed_videos = {}
            if self.performance_mode and len(video_files) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.parallel_tracks) as executor:
                    future_to_track = {
                        executor.submit(
                            self.process_single_video, 
                            track_id, track_data, output_dir, ffmpeg_settings
                        ): track_id 
                        for track_id, track_data in video_files.items()
                    }
                    
                    completed = 0
                    total = len(video_files)
                    
                    for future in as_completed(future_to_track):
                        track_id = future_to_track[future]
                        try:
                            result = future.result()
                            if result:
                                processed_videos[track_id] = result
                            completed += 1
                            progress = 20 + int((completed / total) * 50)
                            self.report_progress(progress, f"Processed track {track_id}")
                        except Exception as e:
                            logger.error(f"Failed to process track {track_id}: {e}")
            else:
                # Sequential processing
                total = len(video_files)
                for i, (track_id, track_data) in enumerate(video_files.items()):
                    result = self.process_single_video(track_id, track_data, output_dir, ffmpeg_settings)
                    if result:
                        processed_videos[track_id] = result
                    progress = 20 + int(((i + 1) / total) * 50)
                    self.report_progress(progress, f"Processed track {track_id}")
            
            if not processed_videos:
                logger.error("No videos were successfully processed")
                return False
            
            self.report_progress(70, f"Processed {len(processed_videos)} videos")
            
            # Combine videos
            success = self.combine_videos(processed_videos, midi_data, output_path)
            if success:
                self.report_progress(100, "Video processing complete")
                
                # Log performance metrics
                self.monitor.stop_monitoring()
                metrics = self.monitor.get_performance_summary()
                logger.info(f"Performance metrics: {metrics}")
                
                return True
            else:
                logger.error("Failed to combine videos")
                return False
                
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
        finally:
            self.cleanup_temp_files()
            self.monitor.stop_monitoring()

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Video Processor')
    parser.add_argument('midi_json', help='Path to MIDI data JSON file')
    parser.add_argument('video_files_json', help='Path to video files JSON file')
    parser.add_argument('output_path', help='Output video file path')
    parser.add_argument('--performance-mode', action='store_true', default=True,
                       help='Enable performance optimizations')
    parser.add_argument('--memory-limit', type=float, default=4.0,
                       help='Memory limit in GB (default: 4.0)')
    parser.add_argument('--parallel-tracks', type=int, default=None,
                       help='Number of parallel tracks to process')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedVideoProcessor(
        performance_mode=args.performance_mode,
        memory_limit_gb=args.memory_limit,
        parallel_tracks=args.parallel_tracks
    )
    
    # Validate inputs
    if not processor.validate_input_files(args.midi_json, args.video_files_json):
        logger.error("Input validation failed")
        sys.exit(1)
    
    # Load data
    try:
        with open(args.midi_json) as f:
            midi_data = json.load(f)
        with open(args.video_files_json) as f:
            video_files = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        sys.exit(1)
    
    # Process videos
    success = processor.process_videos(midi_data, video_files, args.output_path)
    
    if success:
        logger.info(f"Video processing completed successfully: {args.output_path}")
        sys.exit(0)
    else:
        logger.error("Video processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
