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
        'kick': [35, 36],  # Acoustic Bass Drum, Bass Drum 1
        'snare': [38, 40, 37],  # Acoustic Snare, Electric Snare, Side Stick
        'hihat': [42, 44, 46],  # Closed HH, Pedal HH, Open HH
        'cymbal': [49, 51, 52, 55, 57, 53, 59],  # Crash 1, Ride 1, Chinese, Splash, Crash 2, Ride Bell, Ride 2
        'tom': [41, 43, 45, 47, 48, 50],  # Low Floor, High Floor, Low Tom, Low-Mid Tom, Hi-Mid Tom, High Tom
        'percussion': [39, 54, 56, 58, 60, 61, 62, 63, 64],  # Hand Clap, Tambourine, Cowbell, Vibraslap, etc
        'effects': [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]  # Various percussion effects
    }
    groups = set()
    for note in track.get('notes', []):
        for group, midi_numbers in drum_groups.items():
            if note['midi'] in midi_numbers:
                groups.add(group)
    return groups


# placing this here for reference
drum_names = {
    # Basic Drum Kit (35-49)
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    49: "Crash Cymbal 1",
    
    # Toms and Cymbals (50-59)
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    58: "Vibraslap",
    59: "Ride Cymbal 2",
    
    # Latin Percussion (60-69)
    60: "High Bongo",
    61: "Low Bongo",
    62: "Mute High Conga",
    63: "Open High Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    
    # Additional Percussion (70-81)
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "High Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle"
}