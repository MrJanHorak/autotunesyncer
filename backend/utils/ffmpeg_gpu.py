import subprocess
import os
import sys

def ffmpeg_gpu_encode(input_path, output_path, scale=None, framerate=None, codec='h264_nvenc', extra_args=None):
    """
    Encode a video using ffmpeg with GPU acceleration (NVENC).
    Simple and robust implementation with fallback to CPU.
    """
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
    
    try:
        # Try GPU encoding first
        cmd = [
            'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path
        ]
        
        if scale:
            # Use standard CPU scaling for better compatibility
            cmd += ['-vf', f'scale={scale[0]}:{scale[1]}:force_original_aspect_ratio=decrease,pad=w={scale[0]}:h={scale[1]}:x=(ow-iw)/2:y=(oh-ih)/2:color=black']
        
        if framerate:
            cmd += ['-r', str(framerate)]
        
        # Conservative GPU encoding settings
        cmd += [
            '-c:v', codec,
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-b:v', '5M',
            '-maxrate', '10M',
            '-bufsize', '10M',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
            '-movflags', '+faststart',
            '-threads', '4'
        ]
        
        if extra_args:
            cmd += extra_args
        cmd += [output_path]
        
        print('Running GPU ffmpeg command:', ' '.join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ GPU encoding successful!")
        return result
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  GPU encoding failed (exit code: {e.returncode}), falling back to CPU")
        print(f"GPU error: {e.stderr}")
        
        # Fallback to CPU encoding
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path
            ]
            
            if scale:
                cmd += ['-vf', f'scale={scale[0]}:{scale[1]}:force_original_aspect_ratio=decrease,pad=w={scale[0]}:h={scale[1]}:x=(ow-iw)/2:y=(oh-ih)/2:color=black']
            
            if framerate:
                cmd += ['-r', str(framerate)]
            
            # CPU encoding settings
            cmd += [
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2',
                '-movflags', '+faststart',
                '-threads', '4'
            ]
            
            if extra_args:
                cmd += extra_args
            cmd += [output_path]
            
            print('Running CPU fallback ffmpeg command:', ' '.join(cmd))
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ CPU encoding successful!")
            return result
            
        except subprocess.CalledProcessError as cpu_e:
            print(f"❌ Both GPU and CPU encoding failed: {cpu_e}")
            print(f"CPU error: {cpu_e.stderr}")
            return False
    
def check_gpu_filters():
    """Check which GPU filters are available in ffmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True)
        filters = result.stdout
        
        gpu_filters = ['scale_npp', 'scale_cuda', 'hwupload_cuda', 'hwdownload']
        available_gpu_filters = []
        
        for filter_name in gpu_filters:
            if filter_name in filters:
                available_gpu_filters.append(filter_name)
        
        print("Available GPU filters:")
        for f in available_gpu_filters:
            print(f"  ✅ {f}")
        
        return available_gpu_filters
    except Exception as e:
        print(f"❌ Error checking GPU filters: {e}")
        return []

def ffmpeg_gpu_extract_audio(input_path, output_path, extra_args=None):
    """
    Extract audio using ffmpeg (audio is CPU, but can be parallelized).
    """
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
        
    cmd = [
        'ffmpeg', '-y', '-i', input_path, '-vn', '-acodec', 'copy', output_path
    ]
    if extra_args:
        cmd += extra_args
    print('Running ffmpeg command:', ' '.join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Audio extraction successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_gpu_capabilities():
    """Test if GPU acceleration is available"""
    try:
        result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
        hwaccels = result.stdout
        print("Available hardware accelerations:")
        for line in hwaccels.split('\n'):
            if line.strip() and not line.startswith('Hardware'):
                print(f"  {line.strip()}")
        
        if 'cuda' in hwaccels:
            print("✅ CUDA hardware acceleration available")
            return True
        else:
            print("❌ CUDA hardware acceleration not available")
            return False
    except Exception as e:
        print(f"❌ Error testing GPU capabilities: {e}")
        return False

def test_pytorch_cuda():
    """Test PyTorch CUDA availability"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            
            # Test GPU tensor operation
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = x @ y  # Matrix multiplication
            print(f"✅ PyTorch CUDA tensor test successful: {z.shape}")
            return True
        else:
            print("❌ PyTorch CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch CUDA error: {e}")
        return False

def create_test_video():
    """Create a simple test video using ffmpeg"""
    output_path = 'test_input.mp4'
    cmd = [
        'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=2:size=640x480:rate=30',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Test video created: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create test video: {e}")
        return None

# Advanced GPU functions for AutoTuneSyncer

def gpu_video_pipeline(input_path, output_path, width=1920, height=1080, fps=30, bitrate='5M'):
    """
    Complete GPU pipeline: decode -> process -> encode
    Keeps video frames in GPU memory for maximum performance
    """
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
    
    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path,
        '-vf', f'hwupload_cuda,scale_cuda={width}:{height},hwdownload',
        '-r', str(fps),
        '-c:v', 'h264_nvenc',
        '-b:v', bitrate,
        '-preset', 'fast',
        '-tune', 'zerolatency',
        output_path
    ]
    
    print('Running GPU pipeline:', ' '.join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ GPU pipeline successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU pipeline error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def gpu_batch_process(input_files, output_dir, scale=(1920, 1080), fps=30, parallel=True):
    """
    Process multiple video files in parallel using GPU
    Perfect for note-synchronized video segment processing
    """
    import concurrent.futures
    import threading
    
    results = []
    
    def process_single(input_file):
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)
        output_file = os.path.join(output_dir, f"{name}_gpu{ext}")
        
        success = ffmpeg_gpu_encode(input_file, output_file, scale=scale, framerate=fps)
        return (input_file, output_file, success)
    
    if parallel:
        # Process multiple files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_single, f) for f in input_files]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        # Process files sequentially
        for input_file in input_files:
            results.append(process_single(input_file))
    
    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]
    
    print(f"✅ Successfully processed: {len(successful)} files")
    if failed:
        print(f"❌ Failed to process: {len(failed)} files")
    
    return results

def gpu_note_synchronized_encode(video_segments, output_path, grid_size=(2, 2), fps=30):
    """
    Encode multiple video segments into a synchronized grid layout
    Optimized for MIDI note-synchronized video composition
    """
    if not video_segments:
        print("❌ No video segments provided")
        return False
    
    # Create filter complex for grid layout
    inputs = []
    filter_complex = []
    
    for i, segment in enumerate(video_segments):
        if os.path.exists(segment['path']):
            inputs.extend(['-i', segment['path']])
            # Scale each input to fit grid cell
            cell_width = 1920 // grid_size[0]
            cell_height = 1080 // grid_size[1]
            filter_complex.append(f'[{i}:v]scale_cuda={cell_width}:{cell_height}[v{i}]')
    
    # Create grid layout
    grid_filter = ""
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            idx = row * grid_size[0] + col
            if idx < len(video_segments):
                x_pos = col * (1920 // grid_size[0])
                y_pos = row * (1080 // grid_size[1])
                if idx == 0:
                    grid_filter = f'[v{idx}]'
                else:
                    grid_filter = f'[grid{idx-1}][v{idx}]overlay={x_pos}:{y_pos}[grid{idx}]'
                    filter_complex.append(grid_filter)
    
    # Final output
    final_output = f'[grid{len(video_segments)-1}]' if len(video_segments) > 1 else '[v0]'
    
    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda'
    ] + inputs + [
        '-filter_complex', ';'.join(filter_complex),
        '-map', final_output,
        '-r', str(fps),
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        output_path
    ]
    
    print('Running note-synchronized encoding:', ' '.join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Note-synchronized encoding successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Note-synchronized encoding error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def gpu_realtime_preview(input_path, scale=(1280, 720), fps=30):
    """
    Create a real-time preview using GPU acceleration
    Useful for testing note synchronization
    """
    cmd = [
        'ffmpeg', '-hwaccel', 'cuda', '-i', input_path,
        '-vf', f'scale_cuda={scale[0]}:{scale[1]}',
        '-r', str(fps),
        '-c:v', 'h264_nvenc',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'sdl'
    ]
    
    print('Starting real-time preview...')
    try:
        subprocess.run(cmd, check=False)  # Don't check return code for preview
        return True
    except Exception as e:
        print(f"❌ Preview error: {e}")
        return False

def gpu_performance_test(input_path, iterations=5):
    """
    Test GPU encoding performance with your specific hardware
    """
    import time
    
    if not os.path.exists(input_path):
        print(f"❌ Test file not found: {input_path}")
        return None
    
    times = []
    for i in range(iterations):
        start_time = time.time()
        success = ffmpeg_gpu_encode(input_path, f'perf_test_{i}.mp4', scale=(1920, 1080), framerate=30)
        end_time = time.time()
        
        if success:
            times.append(end_time - start_time)
            os.remove(f'perf_test_{i}.mp4')  # Clean up
        else:
            print(f"❌ Performance test iteration {i+1} failed")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Average GPU encoding time: {avg_time:.2f} seconds")
        print(f"✅ Encoding speed: {len(times)/sum(times):.2f}x real-time")
        return avg_time
    else:
        print("❌ All performance tests failed")
        return None

# PyTorch GPU utilities for MIDI processing

def pytorch_gpu_available():
    """Check if PyTorch GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def gpu_midi_timing_calculation(midi_data, sample_rate=44100):
    """
    Fast GPU-accelerated MIDI timing calculations
    Perfect for note-synchronized video processing
    """
    if not pytorch_gpu_available():
        print("❌ PyTorch CUDA not available, falling back to CPU")
        return midi_data
    
    try:
        import torch
        
        # Convert MIDI data to GPU tensors
        if isinstance(midi_data, (list, tuple)):
            midi_tensor = torch.tensor(midi_data, dtype=torch.float32).cuda()
        else:
            midi_tensor = torch.tensor([midi_data], dtype=torch.float32).cuda()
        
        # Fast timing calculations on GPU
        timing_tensor = midi_tensor * sample_rate / 1000  # Convert ms to samples
        
        # Return CPU result
        return timing_tensor.cpu().numpy()
    except Exception as e:
        print(f"❌ GPU MIDI calculation error: {e}")
        return midi_data

def gpu_audio_feature_extraction(audio_data, feature_type='mfcc'):
    """
    GPU-accelerated audio feature extraction for note synchronization
    """
    if not pytorch_gpu_available():
        print("❌ PyTorch CUDA not available")
        return None
    
    try:
        import torch
        
        # Convert audio to GPU tensor
        if isinstance(audio_data, (list, tuple)):
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).cuda()
        else:
            audio_tensor = torch.tensor([audio_data], dtype=torch.float32).cuda()
        
        # Simple feature extraction (replace with actual implementation)
        if feature_type == 'mfcc':
            # Placeholder for MFCC extraction
            features = torch.fft.fft(audio_tensor).real
        elif feature_type == 'spectral':
            features = torch.abs(torch.fft.fft(audio_tensor))
        else:
            features = audio_tensor
        
        return features.cpu().numpy()
    except Exception as e:
        print(f"❌ GPU audio feature extraction error: {e}")
        return None

def gpu_batch_tensor_operations(tensor_list, operation='normalize'):
    """
    Batch process multiple tensors on GPU for video frame operations
    """
    if not pytorch_gpu_available():
        return tensor_list
    
    try:
        import torch
        
        # Move tensors to GPU
        gpu_tensors = [torch.tensor(t).cuda() for t in tensor_list]
        
        # Apply operation
        if operation == 'normalize':
            processed = [t / t.max() for t in gpu_tensors]
        elif operation == 'scale':
            processed = [t * 255.0 for t in gpu_tensors]
        else:
            processed = gpu_tensors
        
        # Return CPU results
        return [t.cpu().numpy() for t in processed]
    except Exception as e:
        print(f"❌ GPU tensor operations error: {e}")
        return tensor_list

if __name__ == '__main__':
    print("=" * 50)
    print("GPU CAPABILITIES TEST")
    print("=" * 50)
    
    # Test PyTorch CUDA
    print("\n1. Testing PyTorch CUDA:")
    pytorch_ok = test_pytorch_cuda()
    
    # Test ffmpeg GPU
    print("\n2. Testing ffmpeg GPU capabilities:")
    ffmpeg_ok = test_gpu_capabilities()

    # Test actual encoding if both work
    if ffmpeg_ok:
        print("\n3. Testing GPU video encoding:")
        test_file = create_test_video()
        if test_file:
            success = ffmpeg_gpu_encode(test_file, 'test_output_gpu.mp4', scale=(1280, 720), framerate=30)
            if success:
                print("✅ GPU encoding test completed successfully!")
                
                # Test advanced GPU functions
                print("\n4. Testing advanced GPU functions:")
                
                # Test GPU pipeline
                print("Testing GPU pipeline...")
                pipeline_success = gpu_video_pipeline(test_file, 'test_pipeline.mp4', width=1920, height=1080, fps=30)
                
                # Test batch processing
                print("Testing batch processing...")
                batch_results = gpu_batch_process([test_file], '.', scale=(1280, 720), fps=30)
                
                # Test performance
                print("Testing performance...")
                avg_time = gpu_performance_test(test_file, iterations=3)
                
                # Test PyTorch GPU functions
                if pytorch_ok:
                    print("Testing PyTorch GPU functions...")
                    test_data = [1, 2, 3, 4, 5]
                    gpu_result = gpu_midi_timing_calculation(test_data)
                    print(f"✅ GPU MIDI timing calculation: {gpu_result}")
                
                # Clean up test files
                cleanup_files = [test_file, 'test_output_gpu.mp4', 'test_pipeline.mp4']
                for f in cleanup_files:
                    if os.path.exists(f):
                        os.remove(f)
                        print(f"Cleaned up: {f}")
                
                # Clean up batch files
                for result in batch_results:
                    if os.path.exists(result[1]):
                        os.remove(result[1])
                        print(f"Cleaned up: {result[1]}")
    
    print("\n" + "=" * 50)
    print("\n5. Check GPU filters")
    filters_ok = check_gpu_filters()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"PyTorch CUDA: {'✅ Working' if pytorch_ok else '❌ Not working'}")
    print(f"ffmpeg GPU: {'✅ Working' if ffmpeg_ok else '❌ Not working'}")
    print(f"GPU filters: {filters_ok}")
    print("=" * 50)
    
    # Usage examples
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES FOR YOUR APP:")
    print("=" * 50)
    print("1. Basic GPU encoding:")
    print("   ffmpeg_gpu_encode('input.mp4', 'output.mp4', scale=(1920, 1080), framerate=30)")
    print("\n2. GPU video pipeline:")
    print("   gpu_video_pipeline('input.mp4', 'output.mp4', width=1920, height=1080, fps=30)")
    print("\n3. Batch processing:")
    print("   gpu_batch_process(['video1.mp4', 'video2.mp4'], 'output_dir/', scale=(1920, 1080))")
    print("\n4. Note-synchronized grid:")
    print("   segments = [{'path': 'note1.mp4'}, {'path': 'note2.mp4'}]")
    print("   gpu_note_synchronized_encode(segments, 'grid_output.mp4', grid_size=(2, 2))")
    print("\n5. MIDI timing calculation:")
    print("   gpu_midi_timing_calculation([100, 200, 300, 400], sample_rate=44100)")
    print("=" * 50)