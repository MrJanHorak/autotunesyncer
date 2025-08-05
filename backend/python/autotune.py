import sys
import os
import warnings
import shutil
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import threading
import psutil

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import configured GPU state with error handling
try:
    from gpu_setup import gpu_available, tf, is_gpu_available, get_tensorflow
    logging.info("GPU setup module loaded successfully")
except ImportError:
    logging.warning("GPU setup not available, using CPU fallback")
    gpu_available = False
    tf = None
    def is_gpu_available():
        return False
    def get_tensorflow():
        return None
except Exception as e:
    logging.warning(f"Error loading GPU setup: {e}")
    gpu_available = False
    tf = None
    def is_gpu_available():
        return False
    def get_tensorflow():
        return None

# Rest of imports
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

# Optional dependencies with fallbacks
try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    print("Warning: CREPE not available, using librosa for pitch detection")
    CREPE_AVAILABLE = False
    crepe = None

try:
    import pyrubberband as pyrb
    PYRUBBERBAND_AVAILABLE = True
except ImportError:
    print("Warning: pyrubberband not available, using librosa for pitch shifting")
    PYRUBBERBAND_AVAILABLE = False
    pyrb = None

# Enhanced logging for parallel processing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Thread-safe processing tracking
autotune_lock = threading.RLock()
processing_stats = {
    'total_frames': 0,
    'gpu_frames': 0,
    'cpu_frames': 0,
    'processing_time': 0.0
}

# Enhanced versions of existing functions
def setup_gpu():
    """Verify GPU setup"""
    if not gpu_available or tf is None:
        return False
    try:
        if tf is not None:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
            logging.info("GPU test computation successful")
            return True
    except Exception as e:
        logging.warning(f"GPU test failed: {e}")
        return False

class ParallelAutotuneProcessor:
    """Enhanced autotune processor with parallel processing capabilities"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, psutil.cpu_count(logical=False))
        self.gpu_available = setup_gpu()
        
    def process_audio_segments_parallel(self, audio_segments: List[Tuple[np.ndarray, float]], 
                                      target_pitches: List[float], sr: int) -> List[np.ndarray]:
        """Process multiple audio segments in parallel"""
        start_time = time.perf_counter()
        
        if len(audio_segments) <= 1:
            # Single segment, no need for parallelization
            if audio_segments:
                result = [autotune_audio_enhanced(audio_segments[0][0], target_pitches[0], sr)]
            else:
                result = []
        else:
            # Parallel processing for multiple segments
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_segment = {}
                
                for i, (segment, start_time_seg) in enumerate(audio_segments):
                    target_pitch = target_pitches[i] if i < len(target_pitches) else target_pitches[0]
                    
                    future = executor.submit(
                        autotune_audio_enhanced,
                        segment,
                        target_pitch,
                        sr
                    )
                    future_to_segment[future] = i
                
                # Collect results in order
                results = [None] * len(audio_segments)
                for future in as_completed(future_to_segment):
                    idx = future_to_segment[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logging.error(f"Autotune processing failed for segment {idx}: {e}")
                        # Fallback to original audio
                        results[idx] = audio_segments[idx][0]
                
                result = results
        
        total_time = time.perf_counter() - start_time
        with autotune_lock:
            processing_stats['processing_time'] += total_time
        
        logging.info(f"Parallel autotune processing completed in {total_time:.2f}s for {len(audio_segments)} segments")
        return result

# Global processor instance
parallel_processor = ParallelAutotuneProcessor()

def autotune_audio_enhanced(audio_data: np.ndarray, target_pitch: float, sr: int) -> np.ndarray:
    """Enhanced autotune processing with GPU acceleration and fallback"""
    with autotune_lock:
        processing_stats['total_frames'] += 1
    
    try:
        # Try GPU-accelerated processing first
        if gpu_available and tf is not None:
            try:
                with autotune_lock:
                    processing_stats['gpu_frames'] += 1
                return autotune_audio_gpu(audio_data, target_pitch, sr)
            except Exception as e:
                logging.warning(f"GPU autotune failed, falling back to CPU: {e}")
        
        # CPU fallback
        with autotune_lock:
            processing_stats['cpu_frames'] += 1
        return autotune_audio_cpu(audio_data, target_pitch, sr)
        
    except Exception as e:
        logging.error(f"Autotune processing failed: {e}")
        return audio_data  # Return original audio on failure

def autotune_audio_gpu(audio_data: np.ndarray, target_pitch: float, sr: int) -> np.ndarray:
    """GPU-accelerated autotune processing"""
    if not gpu_available or tf is None:
        raise RuntimeError("GPU not available")
    
    if not CREPE_AVAILABLE:
        raise RuntimeError("CREPE not available for GPU processing")
    
    try:
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Use GPU for pitch detection
        with tf.device('/GPU:0'):
            time_vals, frequency, confidence, _ = crepe.predict(
                audio_data,
                sr,
                step_size=10,  # ms
                viterbi=True,
                model_capacity='full',
                verbose=0
            )
        
        # Process pitch correction
        return apply_pitch_correction(audio_data, frequency, target_pitch, sr, confidence)
        
    except Exception as e:
        logging.error(f"GPU autotune processing failed: {e}")
        raise

def autotune_audio_cpu(audio_data: np.ndarray, target_pitch: float, sr: int) -> np.ndarray:
    """CPU fallback autotune processing"""
    try:
        # Use librosa for CPU-based pitch detection
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Pitch detection using librosa
        frequency, _, confidence = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048
        )
        
        # Fill NaN values with 0
        frequency = np.nan_to_num(frequency, nan=0.0)
        confidence = np.nan_to_num(confidence, nan=0.0)
        
        return apply_pitch_correction(audio_data, frequency, target_pitch, sr, confidence)
        
    except Exception as e:
        logging.error(f"CPU autotune processing failed: {e}")
        return audio_data

def apply_pitch_correction(audio_data: np.ndarray, detected_pitch: np.ndarray, 
                         target_pitch: float, sr: int, confidence: np.ndarray) -> np.ndarray:
    """Apply pitch correction using detected pitch data"""
    try:
        # Calculate pitch shift ratios
        valid_pitch_mask = (detected_pitch > 0) & (confidence > 0.5)
        
        if not np.any(valid_pitch_mask):
            return audio_data  # No valid pitch detected
        
        # Calculate semitone shift
        pitch_ratios = np.ones_like(detected_pitch)
        pitch_ratios[valid_pitch_mask] = target_pitch / detected_pitch[valid_pitch_mask]
        
        # Smooth pitch ratios to avoid artifacts
        pitch_ratios = signal.medfilt(pitch_ratios, kernel_size=5)
        
        # Apply pitch shifting using librosa
        shifted_audio = librosa.effects.pitch_shift(
            audio_data, 
            sr=sr, 
            n_steps=12 * np.log2(np.mean(pitch_ratios[valid_pitch_mask]))
        )
        
        return shifted_audio
        
    except Exception as e:
        logging.error(f"Pitch correction failed: {e}")
        return audio_data

def get_autotune_stats() -> Dict:
    """Get autotune processing statistics"""
    with autotune_lock:
        return processing_stats.copy()

def check_rubberband():
    """Check if rubberband-cli is available"""
    return shutil.which('rubberband') is not None

def get_pitch_crepe(audio, sr, hop_length=512):
    """Get precise pitch using CREPE"""
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    if gpu_available and tf is not None:
        try:
            print("Using GPU for CREPE")
            print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")
            
            # Import CREPE here to ensure it uses our GPU configuration
            import crepe
            
            # Use GPU without additional configuration
            with tf.device('/GPU:0'):
                print("Running CREPE prediction on GPU...")
                result = crepe.predict(
                    audio,
                    sr,
                    step_size=hop_length/sr*1000,
                    viterbi=True,
                    model_capacity='full',
                    verbose=0                )
                print("CREPE GPU processing completed")
                return result
        except Exception as e:
            print(f"GPU CREPE failed: {e}")
            print("Falling back to CPU")
    
    print("Using CPU for CREPE")
    # Ensure we're not using GPU when falling back
    if tf is not None:
        with tf.device('/CPU:0'):
            import crepe
            result = crepe.predict(
                audio,
                sr,
                step_size=hop_length/sr*1000,
                viterbi=True,
                model_capacity='medium',
                verbose=0
            )
    else:
        # TensorFlow not available, use basic CREPE
        import crepe
        result = crepe.predict(
            audio,
            sr,
            step_size=hop_length/sr*1000,
            viterbi=True,
            model_capacity='medium',
            verbose=1
        )
    return result

def librosa_pitch_detect(audio, sr):
    """Fallback pitch detection using librosa"""
    print("Using librosa pitch detection")
    frequency = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048
    )[0]
    confidence = np.array([1.0 if f is not None else 0.0 for f in frequency])
    frequency = np.array([f if f is not None else 0.0 for f in frequency])
    time = librosa.times_like(frequency)
    return time, frequency, confidence, None

def autotune_frame(frame, current_pitch, target_pitch, sr):
    """
    Autotune a single frame using multiple methods
    """
    if current_pitch == 0 or target_pitch == 0:
        return frame
    
    # Calculate pitch shift in semitones
    shift = 12 * np.log2(target_pitch / current_pitch)
    
    # Try methods in order of preference
    try:
        if check_rubberband() and PYRUBBERBAND_AVAILABLE:
            return pyrb.pitch_shift(frame, sr, shift)
    except Exception as e:
        print(f"Rubberband failed: {e}, trying librosa")
        
    try:
        return librosa.effects.pitch_shift(
            frame,
            sr=sr,
            n_steps=shift,
            bins_per_octave=12
        )
    except Exception as e:
        print(f"Librosa pitch shift failed: {e}, using basic resampling")
        
    # Fallback to basic resampling
    rate = current_pitch / target_pitch
    shifted = signal.resample(frame, int(len(frame) * rate))
    if len(shifted) < len(frame):
        shifted = np.pad(shifted, (0, len(frame) - len(shifted)))
    else:
        shifted = shifted[:len(frame)]
    return shifted

def smooth_pitch_curve(frequency, confidence, smoothing_window=15):
    """
    Smooth the pitch curve to avoid jarring transitions
    """
    valid_pitch = frequency[confidence > 0.5]
    if len(valid_pitch) < 4:  # Need at least 4 points for cubic interpolation
        # Fall back to linear interpolation or return original
        if len(valid_pitch) > 1:
            time_valid = np.arange(len(valid_pitch))
            time_full = np.arange(len(frequency))
            f = interp1d(time_valid, valid_pitch, kind='linear',
                        fill_value=(valid_pitch[0], valid_pitch[-1]),
                        bounds_error=False)
            return f(time_full)
        return frequency

    try:
        # Try cubic interpolation first
        time_valid = np.arange(len(valid_pitch))
        f = interp1d(time_valid, valid_pitch, kind='cubic',
                     fill_value=(valid_pitch[0], valid_pitch[-1]),
                     bounds_error=False)
    except ValueError:
        # Fall back to linear interpolation
        f = interp1d(time_valid, valid_pitch, kind='linear',
                     fill_value=(valid_pitch[0], valid_pitch[-1]),
                     bounds_error=False)

    # Generate smooth curve
    time_full = np.arange(len(frequency))
    smooth_pitch = f(time_full)

    # Simple moving average instead of convolution for more stability
    window_size = min(smoothing_window, len(smooth_pitch) // 2)
    if window_size > 2:
        cumsum = np.cumsum(np.insert(smooth_pitch, 0, 0))
        smooth_pitch = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    return smooth_pitch

def validate_audio(audio, sr):
    """
    Validate and clean audio input
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Empty audio data")
    
    # Convert to float32
    audio = audio.astype(np.float32)
    
    # Ensure correct shape (samples, channels)
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=1)
    elif len(audio.shape) == 2:
        # If channels are first dimension, transpose
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T
    
    # Verify shape
    if len(audio.shape) != 2:
        raise ValueError(f"Invalid audio shape: {audio.shape}")
    if audio.shape[1] > audio.shape[0]:
        raise ValueError(f"Suspicious audio shape (might need transpose): {audio.shape}")
    
    return audio

def process_audio(audio, sr, target_midi_note=60):
    """
    Process audio with GPU acceleration when available
    """
    if gpu_available and tf is not None:
        print("\nGPU Diagnostics:")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU devices available: {tf.config.list_physical_devices('GPU')}")
        print(f"CUDA available: {tf.test.is_built_with_cuda()}")
        print(f"CUDA path: {os.environ.get('CUDA_PATH', 'Not set')}")
        
        # Simple GPU test
        try:
            with tf.device('/GPU:0'):
                x = tf.random.normal([1000, 1000])
                tf.matmul(x, x)
            print("GPU compute test successful")
        except Exception as e:
            print(f"GPU compute test failed: {e}")
    
    # Validate and fix audio shape
    audio = validate_audio(audio, sr)
    
    # Convert to mono for processing
    mono_audio = np.mean(audio, axis=1)
    original_length = len(mono_audio)
    
    print(f"Processing {original_length} samples at {sr}Hz")
    
    # Verify we have enough samples
    if original_length < sr * 0.1:  # Less than 0.1 seconds
        raise ValueError(f"Audio too short: {original_length} samples")
    
    # Get pitch
    time, frequency, confidence, _ = get_pitch_crepe(mono_audio, sr)
    
    # Remove NaN values before smoothing
    frequency = np.nan_to_num(frequency, nan=0.0)
    confidence = np.nan_to_num(confidence, nan=0.0)
    
    # Smooth pitch curve with better handling of edge cases
    print("Smoothing pitch curve...")
    smooth_frequency = smooth_pitch_curve(frequency, confidence)
    
    # Convert MIDI note to frequency
    target_freq = 440.0 * (2.0 ** ((target_midi_note - 69) / 12.0))
    print(f"Target MIDI note {target_midi_note} = {target_freq:.2f} Hz")
    
    # Process in overlapping frames with larger sizes
    frame_length = 4096
    hop_length = frame_length // 4  # 75% overlap
    
    # Add padding to ensure we process the entire signal
    pad_length = frame_length
    padded_audio = np.pad(mono_audio, (pad_length, pad_length), mode='reflect')
    output = np.zeros(len(padded_audio))
    
    print("\nProcessing audio frames...")
    for i in range(0, len(padded_audio) - frame_length, hop_length):
        frame = padded_audio[i:i+frame_length]
        frame_index = max(0, min(i//hop_length, len(smooth_frequency)-1))
        frame_pitch = smooth_frequency[frame_index]
        
        if frame_pitch > 0:  # Only process if we detected a pitch
            shifted_frame = autotune_frame(frame, frame_pitch, target_freq, sr)
            # Apply window without mixing original
            window = signal.windows.hann(frame_length)
            shifted_frame = shifted_frame * window
            output[i:i+frame_length] += shifted_frame

    # Normalize output
    output = output / np.max(np.abs(output))
    
    # Remove padding and ensure exact length match
    output = output[pad_length:pad_length + original_length]
    
    # Ensure output is clean
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
    output = np.clip(output, -1.0, 1.0) * 0.98
    
    # Ensure stereo output has correct shape (samples, channels)
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        stereo_output = np.column_stack((output, output))
    else:
        stereo_output = np.expand_dims(output, axis=1)
    
    print(f"Final output shape: {stereo_output.shape}")
    print(f"Output audio range: {np.min(stereo_output):.2f} to {np.max(stereo_output):.2f}")
    
    return stereo_output, smooth_frequency, frequency

def main():
    # Set proper encoding for Windows
    import sys
    import io
    
    # Force UTF-8 encoding for stdout/stderr
    if sys.platform.startswith('win'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        # Set environment variable for Python subprocess encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Verify GPU configuration at start
        print(f"GPU available: {gpu_available}")
        if gpu_available and tf is not None:        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        elif gpu_available:
            print("GPU marked available but TensorFlow not loaded properly")
        else:
            print("Using CPU processing")
        
        if len(sys.argv) not in [3, 4]:
            print("Usage: python autotune.py input_path output_path [target_midi_note]")
            sys.exit(1)

        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
        # Optional target MIDI note (defaults to Middle C)
        target_midi_note = 60  # Middle C
        if len(sys.argv) == 4:
            try:
                target_midi_note = int(sys.argv[3])
                print(f"Target MIDI note: {target_midi_note}")
            except ValueError:
                print(f"Warning: Invalid MIDI note '{sys.argv[3]}', using default (60)")
        else:
            print(f"Using default target: MIDI note {target_midi_note} (Middle C)")
        
        # Load audio with proper encoding
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        print(f"Processing file: {input_path}")
        
        # Load audio with proper shape handling
        try:
            audio, sr = sf.read(input_path)
            print(f"Initial audio shape: {audio.shape}")
            print(f"Sample rate: {sr} Hz")
            
        except Exception as e:
            print(f"Soundfile loading failed: {e}, trying alternative methods")
            try:
                sr, audio = wavfile.read(input_path)
                audio = audio.astype(np.float32)
                if audio.dtype.kind == 'i':
                    audio = audio / np.iinfo(audio.dtype).max
            except Exception as e:
                print(f"Wavfile loading failed: {e}, trying librosa")
                audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Validate audio shape
        audio = validate_audio(audio, sr)
        
        print(f"Loaded audio format: {audio.dtype}, shape: {audio.shape}")
          # Additional shape validation  
        if len(audio.shape) != 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        
        print("\nInput Audio Statistics:")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {audio.shape[0]/sr:.2f} seconds")
        print(f"Samples: {audio.shape[0]}")
        print(f"Channels: {audio.shape[1]}")
          # Process audio
        processed, smooth_pitch, raw_pitch = process_audio(audio, sr, target_midi_note)
        
        # Calculate average pitches with NaN handling
        valid_pitch = raw_pitch[raw_pitch > 0]
        valid_smooth = smooth_pitch[smooth_pitch > 0]
        
        print("\nPitch Statistics:")
        if len(valid_pitch) > 0:
            print(f"Original average pitch: {np.nanmean(valid_pitch):.2f} Hz")
        if len(valid_smooth) > 0:
            print(f"Smoothed average pitch: {np.nanmean(valid_smooth):.2f} Hz")
        
        # Calculate target frequency from MIDI note  
        target_freq = 440.0 * (2.0 ** ((target_midi_note - 69) / 12.0))
        print(f"Target pitch: MIDI note {target_midi_note} = {target_freq:.2f} Hz")
        
        # Save with specific format and ensure correct shape
        print(f"\nSaving to: {output_path}")
        sf.write(
            output_path,
            processed,
            sr,
            subtype='PCM_16',
            format='WAV'
        )
        
        print("Processing completed successfully!")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Critical error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)