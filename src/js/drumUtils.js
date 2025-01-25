// // Standard MIDI drum mapping (GM Standard)
// export const DRUM_NOTE_MAP = {
//   35: 'Acoustic Bass Drum',
//   36: 'Bass Drum',
//   37: 'Side Stick',
//   38: 'Acoustic Snare',
//   39: 'Hand Clap',
//   40: 'Electric Snare',
//   41: 'Low Floor Tom',
//   42: 'Closed Hi Hat',
//   43: 'High Floor Tom',
//   44: 'Pedal Hi-Hat',
//   45: 'Low Tom',
//   46: 'Open Hi-Hat',
//   47: 'Low-Mid Tom',
//   48: 'Hi Mid Tom',
//   49: 'Crash Cymbal 1',
//   50: 'High Tom',
//   51: 'Ride Cymbal 1',
//   52: 'Chinese Cymbal',
//   53: 'Ride Bell',
//   54: 'Tambourine',
//   55: 'Splash Cymbal',
//   56: 'Cowbell',
//   57: 'Crash Cymbal 2',
//   58: 'Vibraslap',
//   59: 'Ride Cymbal 2',
//   60: 'Hi Bongo',
//   61: 'Low Bongo',
//   62: 'Mute Hi Conga',
//   63: 'Open Hi Conga',
//   64: 'Low Conga',
//   65: 'High Timbale',
//   66: 'Low Timbale',
//   67: 'High Agogo',
//   68: 'Low Agogo',
//   69: 'Cabasa',
//   70: 'Maracas',
//   71: 'Short Whistle',
//   72: 'Long Whistle',
//   73: 'Short Guiro',
//   74: 'Long Guiro',
//   75: 'Claves',
//   76: 'Hi Wood Block',
//   77: 'Low Wood Block',
//   78: 'Mute Cuica',
//   79: 'Open Cuica',
//   80: 'Mute Triangle',
//   81: 'Open Triangle'
// };

// Group similar drum instruments
// export const DRUM_GROUPS = {
//   'kick': [35, 36], // Bass drums
//   'snare': [38, 40], // Snares
//   'hihat': [42, 44, 46], // Hi-hats
//   'tom': [41, 43, 45, 47, 48, 50], // Toms
//   'cymbal': [49, 51, 52, 53, 55, 57, 59], // Cymbals
//   'percussion': [54, 56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81] // Other percussion
// };

export const isDrumTrack = (track) => {
  // Check if it's channel 10 (9 in zero-based) or has "drum" in the name
  return track.channel === 9 || 
         track.instrument?.family?.toLowerCase().includes('drum') ||
         track.instrument?.name?.toLowerCase().includes('drum');
};

// export const getNoteGroup = (midiNote) => {
//   for (const [group, notes] of Object.entries(DRUM_GROUPS)) {
//     if (notes.includes(midiNote)) {
//       return group;
//     }
//   }
//   return 'other';
// };
export const DRUM_NOTES = {
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
};

export const getDrumName = (midiNote) => {
  return DRUM_NOTES[midiNote] || `Unknown Drum (${midiNote})`;
};

// Update existing getNoteGroup to use specific names
export const getNoteGroup = (midiNote) => {
  return getDrumName(midiNote);
};