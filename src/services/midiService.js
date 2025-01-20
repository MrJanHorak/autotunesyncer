import { parseMidiFile, extractInstruments, createInstrumentTrackMap, calculateLongestNotes } from '../utils/midiUtils';

export const processMidiFile = async (file) => {
  const midi = await parseMidiFile(file);
  return {
    midiData: midi,
    instruments: extractInstruments(midi),
    longestNotes: calculateLongestNotes(midi),
    trackMap: createInstrumentTrackMap(midi)
  };
};