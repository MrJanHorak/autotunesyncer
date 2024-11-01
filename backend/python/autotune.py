import sys
from pydub import AudioSegment
from pydub.effects import speedup

def shift_pitch(audio_segment, semitones):
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * (2.0 ** (semitones / 12.0)))
    }).set_frame_rate(audio_segment.frame_rate)

def autotune_to_middle_c(input_audio_path, output_audio_path):
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)

    # Calculate the pitch shift to middle C (261.63 Hz)
    original_pitch = 440.0  # Assuming the original pitch is A4 (440 Hz)
    middle_c = 261.63
    semitones = 12 * (middle_c / original_pitch)

    # Shift the pitch
    autotuned_audio = shift_pitch(audio, semitones)

    # Export the autotuned audio
    autotuned_audio.export(output_audio_path, format="wav")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python autotune.py <input_audio_path> <output_audio_path>")
        sys.exit(1)

    input_audio_path = sys.argv[1]
    output_audio_path = sys.argv[2]
    autotune_to_middle_c(input_audio_path, output_audio_path)