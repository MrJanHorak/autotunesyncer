import pkg from '@tonejs/midi';
const { Midi } = pkg;

const DRUM_NOTES = {
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
  93: 'Snare Drum Brush',
};

export const isDrumTrack = (track) => {
  return track.channel === 9 || 
         track.instrument?.family?.toLowerCase().includes('drum') ||
         track.instrument?.name?.toLowerCase().includes('drum');
};

export const getDrumName = (midiNote) => {
  return DRUM_NOTES[midiNote] || `Unknown Drum (${midiNote})`;
};

export const normalizeInstrumentName = (name) => {
  return name.toLowerCase().replace(/\s+/g, '_');
};

export const parseMidiFile = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  return new Midi(arrayBuffer);
};

export const getNoteGroup = (midiNote) => {
  return getDrumName(midiNote);
};

export const extractInstruments = (midi) => {
  const instruments = new Set();
  midi.tracks.forEach((track) => {
    track.notes.forEach((note) => {
      instruments.add(note.instrument.number);
    });
  });
  return Array.from(instruments);
};

export const createInstrumentTrackMap = (midi) => {
  const instrumentTrackMap = {};
  midi.tracks.forEach((track, index) => {
    track.notes.forEach((note) => {
      instrumentTrackMap[note.instrument.number] = index;
    });
  });
  return instrumentTrackMap;
};
