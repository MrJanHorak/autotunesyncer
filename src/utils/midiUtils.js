import { Midi } from '@tonejs/midi';
import { isDrumTrack, getDrumName } from '../js/drumUtils';

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
      // Process drum tracks using exact drum names
      track.notes.forEach((note) => {
        const drumName = getDrumName(note.midi);
        longestNotes[drumName] = Math.max(
          longestNotes[drumName] || 0,
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