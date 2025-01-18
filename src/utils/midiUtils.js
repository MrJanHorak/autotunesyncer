import { Midi } from '@tonejs/midi';

// MIDI note ranges for drum groups
const DRUM_GROUPS = {
  kick: [35, 36],
  snare: [38, 40],
  hihat: [42, 44, 46],
  tom: [41, 43, 45, 47, 48, 50],
  cymbal: [49, 51, 52, 53, 54, 55, 57, 59],
  percussion: [56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
};

export const getNoteGroup = (midiNote) => {
  for (const [group, notes] of Object.entries(DRUM_GROUPS)) {
    if (notes.includes(midiNote)) {
      return group;
    }
  }
  return 'other';
};

export const isDrumTrack = (track) => {
  return track.channel === 9 || track.instrument?.family === 'drums';
};

export const normalizeInstrumentName = (name) => {
  return name.toLowerCase().replace(/\s+/g, '_');
};

export const parseMidiFile = async (file) => {
  const arrayBuffer = await file.arrayBuffer();
  return new Midi(arrayBuffer);
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

export const calculateLongestNotes = (midiData) => {
  const longestNotes = {};

  midiData.tracks.forEach((track) => {
    if (!track.notes || track.notes.length === 0) return;

    if (isDrumTrack(track)) {
      // Process drum tracks
      track.notes.forEach((note) => {
        const group = getNoteGroup(note.midi);
        const drumKey = `drum_${group}`;
        longestNotes[drumKey] = Math.max(
          longestNotes[drumKey] || 0,
          note.duration
        );
      });
    } else {
      // Process melodic tracks
      const normalizedName = normalizeInstrumentName(track.instrument.name);
      track.notes.forEach((note) => {
        longestNotes[normalizedName] = Math.max(
          longestNotes[normalizedName] || 0,
          note.duration
        );
      });
    }
  });

  return longestNotes;
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

export { DRUM_GROUPS };
