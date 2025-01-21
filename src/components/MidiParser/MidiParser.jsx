/* eslint-disable react/prop-types */
import { useEffect } from 'react';
import { parseMidiFile } from '../../utils/midiUtils';

const KEY_SIGNATURES = {
  0: 'C',
  1: 'G',
  2: 'D',
  3: 'A',
  4: 'E',
  5: 'B',
  6: 'F#',
  7: 'C#',
  '-1': 'F',
  '-2': 'Bb',
  '-3': 'Eb',
  '-4': 'Ab',
  '-5': 'Db',
  '-6': 'Gb',
  '-7': 'Cb',
};

const MidiParser = ({ file, onParsed }) => {
  useEffect(() => {
    if (!file) return;

    const parseMidi = async () => {
      const midi = await parseMidiFile(file);

      const calculateDuration = () => {
        console.log('calculating duration...');
        let lastTime = 0;

        // Find last note end time across all tracks
        midi.tracks.forEach((track) => {
          if (!track.notes) return;

          track.notes.forEach((note) => {
            const noteEndTime = note.time + note.duration;
            lastTime = Math.max(lastTime, noteEndTime);
          });
        });

        console.log('Last tick time:', lastTime);

        // Get tempo (microseconds per quarter note)
        const tempo = midi.header.tempos[0]?.bpm || 120;
        const ppq = midi.header.ppq;

        console.log('Tempo:', tempo, 'PPQ:', ppq);

        // Convert ticks to seconds
        const secondsPerBeat = 60 / tempo;
        const seconds = (lastTime / ppq) * secondsPerBeat;

        console.log('Calculated duration:', seconds);

        return seconds;
      };

      const duration = calculateDuration();
      console.log('Duration:', duration);
      const keySignature = midi.header.keySignatures[0];
      const key = keySignature
        ? {
            note: KEY_SIGNATURES[keySignature.key] || 'Unknown',
            scale: keySignature.scale === 0 ? 'major' : 'minor',
            sharpsFlats: keySignature.key,
          }
        : null;

      // Get total note count across all tracks
      const totalNotes = midi.tracks.reduce(
        (sum, track) => sum + (track.notes?.length || 0),
        0
      );

      const midiInfo = {
        // Track info
        tracks: midi.tracks.map((track, index) => ({
          id: index,
          name: track.name || `Track ${index + 1}`,
          notes: track.notes,
          duration: track.duration,
          tempo: track.tempo,
          instrument: track.instrument.number,
        })),
        // File metadata
        header: {
          format: midi.header.format,
          ticksPerBeat: midi.header.ppq,
          timeSignature: `${midi.header.timeSignatures[0]?.timeSignature[0]}/${midi.header.timeSignatures[0]?.timeSignature[1]}`,
          keySignature: midi.header.keySignatures[0]?.key,
          key: key ? `${key.note} ${key.scale}` : 'Unknown',
          tempo: midi.header.tempos[0]?.bpm || 120,
        },
        // Summary data
        summary: {
          totalTracks: midi.tracks.length,
          totalNotes: totalNotes,
          duration: duration,
          name: file.name,
          size: file.size,
        },
      };

      onParsed(midiInfo);
    };

    parseMidi().catch(console.error);
  }, [file, onParsed]);

  return null;
};

export default MidiParser;
