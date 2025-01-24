/* eslint-disable react/prop-types */
import { useEffect, useCallback, useMemo } from 'react';
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

  const lastProcessedFile = useMemo(() => ({
    fingerprint: null
  }), []);

  const calculateDuration = useCallback((midi) => {
    console.log('calculating duration...');
    let lastTime = 0;

    midi.tracks.forEach((track) => {
      if (!track.notes) return;
      track.notes.forEach((note) => {
        const noteEndTime = note.time + note.duration;
        lastTime = Math.max(lastTime, noteEndTime);
      });
    });

    const tempo = midi.header.tempos[0]?.bpm || 120;
    const ppq = midi.header.ppq;
    const secondsPerBeat = 60 / tempo;
    return (lastTime / ppq) * secondsPerBeat;
  }, []);

  const parseMidi = useCallback(async () => {
    const midi = await parseMidiFile(file);
    
    const duration = calculateDuration(midi);
    const keySignature = midi.header.keySignatures[0];
    const key = keySignature ? {
      note: KEY_SIGNATURES[keySignature.key] || 'Unknown',
      scale: keySignature.scale === 0 ? 'major' : 'minor',
      sharpsFlats: keySignature.key,
    } : null;

    const totalNotes = midi.tracks.reduce(
      (sum, track) => sum + (track.notes?.length || 0),
      0
    );

    const midiInfo = {
      tracks: midi.tracks,
      duration,
      header: {
        format: midi.header.format,
        timeSignature: `${midi.header.timeSignatures[0]?.timeSignature[0]}/${midi.header.timeSignatures[0]?.timeSignature[1]}`,
        key: key ? `${key.note} ${key.scale}` : 'Unknown',
        tempo: midi.header.tempos[0]?.bpm || 120
      },
      summary: {
        name: file.name,
        totalTracks: midi.tracks.length,
        totalNotes
      }
    };
    onParsed(midiInfo);
  }, [file, onParsed, calculateDuration]);

   useEffect(() => {
    if (!file) return;
    const fileFingerprint = `${file.name}-${file.lastModified}`;

    if (lastProcessedFile.fingerprint !== fileFingerprint) {
      lastProcessedFile.fingerprint = fileFingerprint;
      parseMidi().catch(console.error);
    }
  }, [file, parseMidi, lastProcessedFile]);

  return null;
};

export default MidiParser;