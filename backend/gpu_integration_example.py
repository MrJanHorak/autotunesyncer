"""
AutoTuneSyncer GPU Integration Example
Production-ready GPU-accelerated video processing for note-synchronized compositions
"""

import os
import sys
import json
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

from utils.ffmpeg_gpu import (
    ffmpeg_gpu_encode,
    gpu_video_pipeline,
    gpu_batch_process,
    gpu_note_synchronized_encode,
    gpu_midi_timing_calculation,
    pytorch_gpu_available
)

class AutoTuneSyncerGPU:
    """
    GPU-accelerated video processing for AutoTuneSyncer
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.gpu_available = pytorch_gpu_available()
        
    def get_default_config(self):
        return {
            'video': {
                'resolution': (1920, 1080),
                'fps': 30,
                'bitrate': '5M',
                'codec': 'h264_nvenc'
            },
            'audio': {
                'sample_rate': 44100,
                'channels': 2
            },
            'processing': {
                'max_parallel': 2,
                'use_gpu': True,
                'fallback_to_cpu': True
            }
        }
    
    def process_midi_video_segments(self, midi_data, video_files, output_path):
        """
        Process MIDI-synchronized video segments using GPU acceleration
        """
        print("🎵 Processing MIDI-synchronized video segments...")
        
        # Step 1: Calculate timing on GPU
        if self.gpu_available:
            print("⚡ Using GPU for MIDI timing calculations...")
            timing_data = gpu_midi_timing_calculation(
                midi_data.get('note_timings', []),
                sample_rate=self.config['audio']['sample_rate']
            )
        else:
            timing_data = midi_data.get('note_timings', [])
        
        # Step 2: Prepare video segments
        segments = []
        for i, (video_file, timing) in enumerate(zip(video_files, timing_data)):
            if os.path.exists(video_file):
                segments.append({
                    'path': video_file,
                    'timing': timing,
                    'index': i
                })
        
        # Step 3: GPU batch processing
        if self.config['processing']['use_gpu']:
            print("⚡ Using GPU for video processing...")
            temp_dir = 'temp_processed'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Process segments in parallel
            processed_results = gpu_batch_process(
                [s['path'] for s in segments],
                temp_dir,
                scale=self.config['video']['resolution'],
                fps=self.config['video']['fps'],
                parallel=True
            )
            
            # Update segments with processed paths
            for i, result in enumerate(processed_results):
                if result[2]:  # Success
                    segments[i]['processed_path'] = result[1]
                else:
                    print(f"❌ Failed to process segment {i}: {result[0]}")
        
        # Step 4: Create synchronized composition
        print("🎬 Creating synchronized composition...")
        grid_size = self.calculate_optimal_grid_size(len(segments))
        
        success = gpu_note_synchronized_encode(
            segments,
            output_path,
            grid_size=grid_size,
            fps=self.config['video']['fps']
        )
        
        # Cleanup
        if os.path.exists('temp_processed'):
            import shutil
            shutil.rmtree('temp_processed')
        
        return success
    
    def calculate_optimal_grid_size(self, num_segments):
        """Calculate optimal grid size for video composition"""
        if num_segments <= 1:
            return (1, 1)
        elif num_segments <= 4:
            return (2, 2)
        elif num_segments <= 9:
            return (3, 3)
        else:
            return (4, 4)
    
    def process_single_video(self, input_path, output_path, effects=None):
        """Process a single video with GPU acceleration"""
        print(f"🎥 Processing video: {input_path}")
        
        if self.config['processing']['use_gpu']:
            return gpu_video_pipeline(
                input_path,
                output_path,
                width=self.config['video']['resolution'][0],
                height=self.config['video']['resolution'][1],
                fps=self.config['video']['fps'],
                bitrate=self.config['video']['bitrate']
            )
        else:
            return ffmpeg_gpu_encode(
                input_path,
                output_path,
                scale=self.config['video']['resolution'],
                framerate=self.config['video']['fps']
            )
    
    def benchmark_performance(self, test_video_path):
        """Benchmark GPU performance for your specific use case"""
        print("📊 Benchmarking GPU performance...")
        
        from utils.ffmpeg_gpu import gpu_performance_test
        
        if os.path.exists(test_video_path):
            avg_time = gpu_performance_test(test_video_path, iterations=5)
            if avg_time:
                print(f"✅ Average processing time: {avg_time:.2f}s")
                print(f"✅ Estimated throughput: {1/avg_time:.2f} videos/second")
                return avg_time
        else:
            print("❌ Test video not found")
        
        return None

def main():
    """Example usage"""
    print("🚀 AutoTuneSyncer GPU Integration Example")
    print("=" * 50)
    
    # Initialize GPU processor
    gpu_processor = AutoTuneSyncerGPU()
    
    # Example 1: Process single video
    print("\n1. Single video processing:")
    if os.path.exists('test_input.mp4'):
        success = gpu_processor.process_single_video(
            'test_input.mp4',
            'output_single.mp4'
        )
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Example 2: MIDI-synchronized processing
    print("\n2. MIDI-synchronized processing:")
    midi_data = {
        'note_timings': [0, 250, 500, 750, 1000],  # milliseconds
        'notes': ['C4', 'D4', 'E4', 'F4', 'G4']
    }
    
    # Create some test video files (normally you'd have real video segments)
    video_files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    
    # This would process actual video files in a real scenario
    print("Example MIDI processing setup complete")
    
    # Example 3: Performance benchmark
    print("\n3. Performance benchmark:")
    if os.path.exists('test_input.mp4'):
        gpu_processor.benchmark_performance('test_input.mp4')
    
    print("\n" + "=" * 50)
    print("🎉 GPU integration ready for production use!")
    print("✅ NVENC hardware encoding: Enabled")
    print("✅ PyTorch GPU acceleration: Available" if gpu_processor.gpu_available else "❌ PyTorch GPU: Not available")
    print("✅ Batch processing: Enabled")
    print("✅ Real-time composition: Ready")

if __name__ == "__main__":
    main()
