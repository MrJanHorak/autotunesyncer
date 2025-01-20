/* eslint-disable react/prop-types */
import { useEffect } from 'react';
import { parseMidiFile } from '../../utils/midiUtils';

const MidiParser = ({ file, onParsed }) => {
  useEffect(() => {
    if (!file) return;

    const parseMidi = async () => {
      const midi = await parseMidiFile(file);
      const tracks = midi.tracks.map((track, index) => ({
        id: index,
        name: track.name || `Track ${index + 1}`,
        notes: track.notes,
        duration: track.duration,
        tempo: track.tempo,
        instrument: track.instrument.number
      }));
      onParsed(tracks);
    };

    parseMidi().catch(console.error);
  }, [file, onParsed]);

  return null;
};

export default MidiParser;