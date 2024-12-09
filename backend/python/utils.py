# utils.py
def normalize_instrument_name(name):
    """Match frontend's normalizeInstrumentName"""
    return name.lower().replace(' ', '_')

def midi_to_note(midi_num):
    """Convert MIDI note number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes[midi_num % 12]
    octave = (midi_num // 12) - 1
    return f"{note_name}{octave}"