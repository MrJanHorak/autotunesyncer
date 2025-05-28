import subprocess
import sys
import logging
import argparse
import time
import os
import psutil
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from video_utils import run_ffmpeg_command, validate_video, performance_monitor, get_optimized_ffmpeg_params
from processing_utils import encoder_queue, GPUManager

# Enhanced logging with performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VideoPreprocessor:
    """Enhanced video preprocessor with performance optimizations"""
    
    def __init__(self, performance_mode=True, max_workers=None):
        self.performance_mode = performance_mode
        self.max_workers = max_workers or min(4, psutil.cpu_count(logical=False))
        self.gpu_manager = GPUManager()
        self.processed_count = 0
        self.total_count = 0
        
        logging.info(f"VideoPreprocessor initialized: performance_mode={performance_mode}, max_workers={self.max_workers}")

    def report_progress(self, progress: int, message: str = ""):
        """Report progress to parent process"""
        print(f"PROGRESS:{progress}", flush=True)
        if message:
            logging.info(f"Progress {progress}%: {message}")

    def preprocess_video_enhanced(self, input_path, output_path, target_size=None, quality="high"):
        """Enhanced video preprocessing with hardware acceleration and fallback"""
        with performance_monitor(f"Preprocessing: {os.path.basename(input_path)}"):
            try:
                # Get optimized FFmpeg parameters
                ffmpeg_params = get_optimized_ffmpeg_params(
                    use_gpu=self.performance_mode, 
                    preset="fast" if self.performance_mode else "medium",
                    quality=quality
                )
                
                cmd = ['ffmpeg', '-y', '-i', input_path]
                
                # Add hardware acceleration if available
                if self.performance_mode and 'hwaccel' in ffmpeg_params:
                    cmd.extend(['-hwaccel', ffmpeg_params['hwaccel']])
                
                # Add video filters for resizing
                if target_size:
                    width, height = target_size.split('x')
                    scale_filter = (
                        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                        f'pad=w={width}:h={height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black'
                    )
                    cmd.extend(['-vf', scale_filter])
                
                # Video encoding settings
                cmd.extend([
                    '-c:v', ffmpeg_params['video_codec'],
                    '-preset', ffmpeg_params['preset'],
                    '-crf', str(ffmpeg_params.get('crf', 23)),
                    '-pix_fmt', 'yuv420p'
                ])
                
                # Add GPU-specific options
                if 'gpu_options' in ffmpeg_params:
                    cmd.extend(ffmpeg_params['gpu_options'])
                
                # Audio settings
                cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2'
                ])
                
                # Performance optimizations
                cmd.extend([
                    '-movflags', '+faststart',
                    '-threads', str(min(4, psutil.cpu_count())),
                    output_path
                ])
                
                logging.info(f"Enhanced preprocessing: {os.path.basename(input_path)}")
                
                # Use encoder queue for better resource management
                if hasattr(encoder_queue, 'encode'):
                    result = encoder_queue.encode(cmd)
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
                else:
                    run_ffmpeg_command(cmd)
                
                # Validate output
                validate_video(output_path)
                
                self.processed_count += 1
                progress = int((self.processed_count / self.total_count) * 100) if self.total_count > 0 else 100
                self.report_progress(progress, f"Processed {self.processed_count}/{self.total_count} videos")
                
                return True
                
            except Exception as e:
                logging.error(f"Enhanced preprocessing failed for {input_path}: {e}")
                # Fallback to basic preprocessing
                return self.preprocess_video_fallback(input_path, output_path, target_size)

    def preprocess_video_fallback(self, input_path, output_path, target_size=None):
        """Fallback preprocessing with basic settings"""
        try:
            logging.info(f"Using fallback preprocessing for: {os.path.basename(input_path)}")
            
            cmd = ['ffmpeg', '-y', '-i', input_path]
            
            if target_size:
                width, height = target_size.split('x')
                scale_filter = (
                    f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                    f'pad=w={width}:h={height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black'
                )
                cmd.extend(['-vf', scale_filter])
            
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                output_path
            ])
            
            run_ffmpeg_command(cmd)
            validate_video(output_path)
            
            return True
            
        except Exception as e:
            logging.error(f"Fallback preprocessing failed for {input_path}: {e}")
            return False

    def preprocess_videos_batch(self, video_list, output_dir, target_size=None):
        """Process multiple videos in parallel"""
        if not video_list:
            return []
        
        self.total_count = len(video_list)
        self.processed_count = 0
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        
        if self.performance_mode and len(video_list) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_video = {}
                
                for video_info in video_list:
                    input_path = video_info['input']
                    output_path = os.path.join(output_dir, video_info['output'])
                    
                    future = executor.submit(
                        self.preprocess_video_enhanced,
                        input_path, output_path, target_size
                    )
                    future_to_video[future] = video_info
                
                for future in as_completed(future_to_video):
                    video_info = future_to_video[future]
                    try:
                        success = future.result()
                        results.append({
                            'input': video_info['input'],
                            'output': os.path.join(output_dir, video_info['output']),
                            'success': success
                        })
                    except Exception as e:
                        logging.error(f"Error processing {video_info['input']}: {e}")
                        results.append({
                            'input': video_info['input'],
                            'output': os.path.join(output_dir, video_info['output']),
                            'success': False,
                            'error': str(e)
                        })
        else:
            # Sequential processing
            for video_info in video_list:
                input_path = video_info['input']
                output_path = os.path.join(output_dir, video_info['output'])
                
                success = self.preprocess_video_enhanced(input_path, output_path, target_size)
                results.append({
                    'input': input_path,
                    'output': output_path,
                    'success': success
                })
        
        return results

def preprocess_video_legacy(input_path, output_path, target_size=None):
    """Legacy function for backwards compatibility"""
    preprocessor = VideoPreprocessor(performance_mode=False)
    return preprocessor.preprocess_video_enhanced(input_path, output_path, target_size)

# Enhanced main function with argument parsing
def main():
    """Main entry point with enhanced argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Video Preprocessor')
    parser.add_argument('input_path', help='Input video file path')
    parser.add_argument('output_path', help='Output video file path')
    parser.add_argument('target_size', nargs='?', help='Target size (e.g., 1920x1080)')
    parser.add_argument('--performance-mode', action='store_true', 
                       help='Enable performance optimizations')
    parser.add_argument('--parallel-tracks', type=int, default=4,
                       help='Number of parallel processing tracks')
    parser.add_argument('--memory-limit', type=int, default=4,
                       help='Memory limit in GB')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='high',
                       help='Processing quality level')
    parser.add_argument('--batch-file', help='JSON file with batch processing list')
    
    args = parser.parse_args()
    
    try:
        if args.batch_file:
            # Batch processing mode
            with open(args.batch_file) as f:
                batch_data = json.load(f)
            
            preprocessor = VideoPreprocessor(
                performance_mode=args.performance_mode,
                max_workers=args.parallel_tracks
            )
            
            results = preprocessor.preprocess_videos_batch(
                batch_data['videos'],
                batch_data['output_dir'],
                args.target_size
            )
            
            # Output results
            print(json.dumps({
                'success': True,
                'results': results,
                'total_processed': len([r for r in results if r['success']]),
                'total_failed': len([r for r in results if not r['success']])
            }))
        else:
            # Single file processing
            preprocessor = VideoPreprocessor(
                performance_mode=args.performance_mode,
                max_workers=1
            )
            
            success = preprocessor.preprocess_video_enhanced(
                args.input_path, 
                args.output_path, 
                args.target_size,
                args.quality
            )
            
            print(json.dumps({
                'success': success,
                'input': args.input_path,
                'output': args.output_path
            }))
            
            sys.exit(0 if success else 1)
            
    except Exception as e:
        logging.error(f"Video preprocessing failed: {e}")
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))
        sys.exit(1)

def preprocess_video(input_path, output_path, target_size=None):
    """Legacy function maintained for backward compatibility"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path
        ]
        
        if target_size:
            width, height = target_size.split('x')
            scale_filter = (
                f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad=w={width}:h={height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black'
            )
            cmd.extend(['-vf', scale_filter])
            
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-rc', 'vbr',
            '-rc-lookahead', '32',
            '-gpu', '0',
            '-tune', 'hq',
            '-profile:v', 'high',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ])
        
        logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
        run_ffmpeg_command(cmd)
        validate_video(output_path)
        
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise

def normalize_track_indices(grid_arrangement):
    """Normalize sparse track indices to sequential ones for optimal grid layout"""
    if not grid_arrangement:
        return {}
    
    # Separate numeric and non-numeric keys
    numeric_indices = []
    non_numeric_keys = []
    
    for key in grid_arrangement.keys():
        try:
            numeric_indices.append(int(key))
        except ValueError:
            non_numeric_keys.append(key)
    
    # Sort numeric indices
    numeric_indices.sort()
    
    # Create mapping from original to normalized indices for numeric keys only
    mapping = {str(old_idx): str(new_idx) for new_idx, old_idx in enumerate(numeric_indices)}
    
    # Map positions using the new indices, preserving non-numeric keys
    normalized = {}
    for old_idx, position in grid_arrangement.items():
        if old_idx in non_numeric_keys:
            # Keep non-numeric keys as-is (like 'drum_crash_cymbal')
            normalized[old_idx] = position
        else:
            # Map numeric keys to new indices
            new_idx = mapping.get(old_idx, old_idx)
            normalized[new_idx] = position
    
    return normalized

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        # Check for performance mode flag
        performance_mode = '--performance-mode' in sys.argv
        
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        target_size = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
        
        if performance_mode:
            # Use enhanced processing
            main()
        else:
            # Use legacy processing for backward compatibility
            preprocess_video(input_path, output_path, target_size)
    else:
        # Use argument parser for full functionality
        main()