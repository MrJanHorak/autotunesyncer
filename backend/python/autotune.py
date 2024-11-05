import sys
import os
import numpy as np
import crepe
import pytsmod as tsm
import librosa
import soundfile as sf
import pyrubberband as pyrb
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 encoding
if sys.platform.startswith('win'):
    import locale
    if sys.version_info[0] < 3:
        sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout)
    else:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# Set environment variables for encoding and tensorflow
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_pitch_crepe(audio, sr, hop_length=512):  # Increased from 128
    """
    Get precise pitch using CREPE (deep learning pitch tracker)
    """
    try:
        # Ensure audio is mono and 1D
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=0)
        
        # Ensure minimum length for resampling
        min_samples = sr // 100  # At least 10ms of audio
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        # Convert to float32 for TensorFlow
        audio = audio.astype(np.float32)
        
        # CREPE now uses simpler API
        time, frequency, confidence, activation = crepe.predict(
            audio,
            sr,
            step_size=hop_length/sr*1000,  # Convert to milliseconds
            viterbi=True
        )
        
        # Relaxed confidence threshold
        frequency[confidence < 0.5] = 0  # Changed from 0.7
        return frequency, confidence, time
    except Exception as e:
        print(f"CREPE error: {str(e)}")
        raise

def autotune_frame(frame, current_pitch, target_pitch, sr):
    """
    Autotune a single frame using both pytsmod and pyrubberband
    """
    if current_pitch == 0 or target_pitch == 0:
        return frame
    
    # Calculate pitch shift in semitones
    shift = 12 * np.log2(target_pitch / current_pitch)
    
    try:
        # Try pytsmod first (better quality)
        shifted = tsm.pitch_shift(frame, shift)
    except:
        try:
            # Fallback to pyrubberband
            shifted = pyrb.pitch_shift(frame, sr, shift)
        except:
            # Last resort: basic resampling
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
    
    # Force correct shape (samples, channels)
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=1)
    elif len(audio.shape) == 2:
        if audio.shape[0] > audio.shape[1]:  # More samples than channels
            audio = audio.T
    
    # Verify shape is sensible
    if audio.shape[1] > 2:  # If more than 2 channels, transpose
        if audio.shape[0] == 2:
            audio = audio.T
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
    
    return audio

def process_audio(audio, sr):
    """
    Process audio with precise pitch detection and correction
    """
    try:
        # Validate and fix audio shape first
        audio = validate_audio(audio, sr)
        print(f"Audio shape after validation: {audio.shape}")
        
        # Convert to mono for processing
        mono_audio = np.mean(audio, axis=1)
        
        original_length = len(mono_audio)
        print(f"Audio shape before processing: {audio.shape}")
        print(f"Mono audio shape: {mono_audio.shape}")
        
        # Validate input audio
        audio = validate_audio(audio, sr)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            mono_audio = np.mean(audio, axis=1)
        else:
            mono_audio = audio
        
        original_length = len(mono_audio)
        print(f"Audio range: {np.min(mono_audio):.2f} to {np.max(mono_audio):.2f}")
        
        # Get pitch using CREPE with more conservative settings
        print("Detecting pitch...")
        try:
            import tensorflow  # Check if tensorflow is available
            frequency, confidence, time = get_pitch_crepe(mono_audio, sr, hop_length=1024)
        except ImportError:
            print("CREPE (tensorflow) not available, using basic pitch detection")
            frequency = librosa.pyin(
                mono_audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=2048
            )[0]
            confidence = np.array([1.0 if f is not None else 0.0 for f in frequency])
            frequency = np.array([f if f is not None else 0.0 for f in frequency])
            time = librosa.times_like(frequency)

        if np.all(frequency == 0) or np.all(np.isnan(frequency)):
            raise ValueError("No valid pitch detected in audio")
        
        # Smooth pitch curve
        print("Smoothing pitch curve...")
        smooth_frequency = smooth_pitch_curve(frequency, confidence)
        
        # Target frequency (Middle C)
        target_freq = 261.63
        
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
                shifted_frame = 0.8 * shifted_frame + 0.2 * frame
                window = signal.windows.hann(frame_length)
                shifted_frame = shifted_frame * window
                output[i:i+frame_length] += shifted_frame
        
        # Remove padding and ensure exact length match
        output = output[pad_length:pad_length + original_length]
        
        # Ensure output is clean
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        output = np.clip(output, -1.0, 1.0) * 0.98
        
        # Create stereo output if input was stereo
        if len(audio.shape) > 1:
            stereo_output = np.vstack((output, output)).T
        else:
            stereo_output = output
        
        print(f"Output audio range: {np.min(stereo_output):.2f} to {np.max(stereo_output):.2f}")
        return stereo_output, smooth_frequency, frequency
        
    except Exception as e:
        print(f"Error in audio processing: {str(e)}")
        raise

def main():
    try:
        if len(sys.argv) != 3:
            print("Usage: python autotune.py input_path output_path")
            sys.exit(1)

        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
        # Load audio with proper encoding
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        print(f"Processing file: {input_path}")
        
        # Load audio with proper shape handling
        try:
            audio, sr = sf.read(input_path)
            print(f"Initial audio shape: {audio.shape}")
            
            # Ensure consistent shape (samples, channels)
            if len(audio.shape) == 1:
                audio = np.expand_dims(audio, axis=1)
            if len(audio.shape) == 2 and audio.shape[1] != 2:
                audio = audio.T
                
            print(f"Normalized audio shape: {audio.shape}")
            
        except Exception as e:
            print(f"Soundfile loading failed: {e}, trying alternative methods")
            try:
                sr, audio = wavfile.read(input_path)
                audio = audio.astype(np.float32)
                if audio.dtype.kind == 'i':
                    audio = audio / np.iinfo(audio.dtype).max
                if len(audio.shape) == 1:
                    audio = np.expand_dims(audio, axis=0)
                elif len(audio.shape) == 2:
                    audio = audio.T
            except Exception as e:
                print(f"Wavfile loading failed: {e}, trying librosa")
                audio, sr = librosa.load(input_path, sr=None, mono=False)
                if len(audio.shape) == 1:
                    audio = np.expand_dims(audio, axis=0)
        
        print(f"Loaded audio format: {audio.dtype}, shape: {audio.shape}")
        
        # Additional shape validation
        if len(audio.shape) != 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        
        print("\nInput Audio Statistics:")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {audio.shape[1]/sr:.2f} seconds")
        print(f"Channels: {audio.shape[0]}")
        print(f"Samples: {audio.shape[1]}")
        
        # Process audio
        processed, smooth_pitch, raw_pitch = process_audio(audio, sr)
        
        # Calculate average pitches
        valid_pitch = raw_pitch[raw_pitch > 0]
        valid_smooth = smooth_pitch[smooth_pitch > 0]
        
        print("\nPitch Statistics:")
        print(f"Original average pitch: {np.mean(valid_pitch):.2f} Hz")
        print(f"Smoothed average pitch: {np.mean(valid_smooth):.2f} Hz")
        print(f"Target pitch (Middle C): 261.63 Hz")
        
        # Save with specific format
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
    main()