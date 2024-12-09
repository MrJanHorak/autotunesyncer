# drum_utils.py
def is_drum_kit(instrument):
    """Check if instrument is a drum kit based on name or channel 10 (9 in zero-based)"""
    drum_keywords = ['standard kit', 'drum kit', 'drums', 'percussion']
    name = instrument.get('name', '').lower()
    channel = instrument.get('channel', 0)
    
    return (
        any(keyword in name for keyword in drum_keywords) or
        'drum' in name or
        channel == 9
    )

def get_drum_groups(track):
    """Match frontend's DRUM_GROUPS logic"""
    drum_groups = {
        'kick': [35, 36],
        'snare': [38, 40],
        'hihat': [42, 44, 46],
        'cymbal': [49, 51, 52, 55, 57],
        'tom': [41, 43, 45, 47, 48, 50]
    }
    groups = set()
    for note in track.get('notes', []):
        for group, midi_numbers in drum_groups.items():
            if note['midi'] in midi_numbers:
                groups.add(group)
    return groups