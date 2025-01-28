import logging

DRUM_NOTES = {
  27: 'Laser',
  28: 'Whip',
  29: 'Scratch Push',
  30: 'Scratch Pull',
  31: 'Stick Click',
  32: 'Metronome Click',
  34: 'Metronome Bell',
  35: 'Bass Drum',
  36: 'Kick Drum',
  37: 'Snare Cross Stick',
  38: 'Snare Drum',
  39: 'Hand Clap',
  40: 'Electric Snare Drum',
  41: 'Floor Tom 2',
  42: 'Hi-Hat Closed',
  43: 'Floor Tom 1',
  44: 'Hi-Hat Foot',
  45: 'Low Tom',
  46: 'Hi-Hat Open',
  47: 'Low-Mid Tom',
  48: 'High-Mid Tom',
  49: 'Crash Cymbal',
  50: 'High Tom',
  51: 'Ride Cymbal',
  52: 'China Cymbal',
  53: 'Ride Bell',
  54: 'Tambourine',
  55: 'Splash cymbal',
  56: 'Cowbell',
  57: 'Crash Cymbal 2',
  58: 'Vibraslap',
  59: 'Ride Cymbal 2',
  60: 'High Bongo',
  61: 'Low Bongo',
  62: 'Conga Dead Stroke',
  63: 'Conga',
  64: 'Tumba',
  65: 'High Timbale',
  66: 'Low Timbale',
  67: 'High Agogo',
  68: 'Low Agogo',
  69: 'Cabasa',
  70: 'Maracas',
  71: 'Whistle Short',
  72: 'Whistle Long',
  73: 'Guiro Short',
  74: 'Guiro Long',
  75: 'Claves',
  76: 'High Woodblock',
  77: 'Low Woodblock',
  78: 'Cuica High',
  79: 'Cuica Low',
  80: 'Triangle Mute',
  81: 'Triangle Open',
  82: 'Shaker',
  83: 'Sleigh Bell',
  84: 'Bell Tree',
  85: 'Castanets',
  86: 'Surdu Dead Stroke',
  87: 'Surdu',
  91: 'Snare Drum Rod',
  92: 'Ocean Drum',
  93: 'Snare Drum Brush'
}

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

def get_drum_groups(input_data):
    """Get drum groups from either a track or single MIDI note"""
    # Add debug logging
    logging.info(f"get_drum_groups called with: {input_data}")
    
    # Handle single MIDI note
    if isinstance(input_data, (int, float)):
        drum_name = DRUM_NOTES.get(int(input_data))
        logging.info(f"MIDI note {input_data} maps to drum: {drum_name}")
        return drum_name
        
    # Handle track dictionary
    if isinstance(input_data, dict):
        if not input_data.get('notes'):
            return set()
            
        drum_groups = set()
        for note in input_data['notes']:
            midi_note = note.get('midi')
            if midi_note in DRUM_NOTES:
                drum_name = DRUM_NOTES[midi_note]
                logging.info(f"MIDI note {midi_note} maps to drum: {drum_name}")
                drum_groups.add(drum_name)
        return drum_groups
        
    return None

def get_drum_name(midi_note):
    """Get drum name from MIDI note number"""
    return DRUM_NOTES.get(midi_note)

# Replace the old group-based function with exact drum names
def process_drum_track(track):
    """Process a drum track and return a dict of drum names and their notes"""
    drum_notes = {}
    
    if not track.get('notes'):
        return drum_notes
        
    for note in track['notes']:
        drum_name = get_drum_name(note['midi'])
        if drum_name not in drum_notes:
            drum_notes[drum_name] = []
        drum_notes[drum_name].append(note)
        
    return drum_notes