import sys
import librosa
import numpy as np
import soundfile as sf

def autotune_to_middle_c(input_audio_path, output_audio_path):
    # Load the audio file with original sampling rate
    y, sr = librosa.load(input_audio_path, sr=None)

    # Estimate the pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[np.nonzero(pitches)]
    if len(pitch_values) == 0:
        print("Could not detect pitch.")
        sys.exit(1)
    original_pitch = np.median(pitch_values)

    # Calculate the number of semitones to shift to reach Middle C (261.63 Hz)
    target_pitch = 261.63  # Middle C
    semitones = 12 * np.log2(target_pitch / original_pitch)
    print(f"Original pitch: {original_pitch:.2f} Hz")
    print(f"Semitones to shift: {semitones:.2f}")

    # Shift the pitch using librosa
    autotuned_audio = librosa.effects.pitch_shift(y, sr, n_steps=semitones)

    # Save the autotuned audio
    sf.write(output_audio_path, autotuned_audio, sr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python autotune.py <input_audio_path> <output_audio_path>")
        sys.exit(1)

    input_audio_path = sys.argv[1]
    output_audio_path = sys.argv[2]
    autotune_to_middle_c(input_audio_path, output_audio_path)