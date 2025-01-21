import { useState, useEffect, useCallback } from 'react';
import {
  parseMidiFile,
  extractInstruments,
  createInstrumentTrackMap,
  calculateLongestNotes,
  normalizeInstrumentName,
} from '../utils/midiUtils';
import { isDrumTrack, getNoteGroup } from '../utils/midiUtils';

export const useMidiProcessing = (midiFile) => {
  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [instruments, setInstruments] = useState([]);
  const [instrumentTrackMap, setInstrumentTrackMap] = useState({});
  const [longestNotes, setLongestNotes] = useState({});

  const onMidiProcessed = useCallback((midiData) => {
    try {
      setParsedMidiData(midiData);

      // Use Map to track unique instruments and their combined notes
      const instrumentMap = new Map();
      const trackMapping = new Map();
      const noteDurations = {};

      midiData.tracks.forEach((track, index) => {
        if (!track.notes || track.notes.length === 0) return;

        if (isDrumTrack(track)) {
          // Handle drum tracks - group by drum type
          const drumGroups = new Set();
          track.notes.forEach((note) => {
            const group = getNoteGroup(note.midi);
            drumGroups.add(group);

            // Track longest note duration for each drum group
            const drumKey = `drum_${group}`;
            const noteDuration = note.duration;
            noteDurations[drumKey] = Math.max(
              noteDurations[drumKey] || 0,
              noteDuration
            );
          });

          // Add unique drum instruments
          drumGroups.forEach((group) => {
            const drumKey = `drum_${group}`;
            if (!instrumentMap.has(drumKey)) {
              instrumentMap.set(drumKey, {
                isDrum: true,
                group: group,
                family: 'percussion',
                number: track.channel,
              });
            }
            // Keep track of first occurrence for mapping
            if (!trackMapping.has(drumKey)) {
              trackMapping.set(drumKey, index);
            }
          });
        } else {
          // Handle melodic instruments
          const normalizedName = normalizeInstrumentName(track.instrument.name);
          if (!instrumentMap.has(normalizedName) && track.instrument.name) {
            instrumentMap.set(normalizedName, {
              isDrum: false,
              name: track.instrument.name,
              family: track.instrument.family,
              number: track.channel,
            });

            // Track longest note duration
            track.notes.forEach((note) => {
              noteDurations[normalizedName] = Math.max(
                noteDurations[normalizedName] || 0,
                note.duration
              );
            });

            // Keep track of first occurrence for mapping
            trackMapping.set(normalizedName, index);
          }
        }
      });

      // Convert Map to array of unique instruments
      const instrumentArray = Array.from(instrumentMap.values());

      setInstruments(instrumentArray);
      setInstrumentTrackMap(trackMapping);
      setLongestNotes(noteDurations);
    } catch (error) {
      console.error('Error processing MIDI data:', error);
    }
  }, []);

  useEffect(() => {
    if (midiFile) {
      const processMidi = async () => {
        const midi = await parseMidiFile(midiFile);
        setParsedMidiData(midi);
        setInstruments(extractInstruments(midi));
        setInstrumentTrackMap(createInstrumentTrackMap(midi));
        setLongestNotes(calculateLongestNotes(midi));
      };

      processMidi().catch(console.error);
    }
  }, [midiFile]);

  return {
    parsedMidiData,
    instruments,
    instrumentTrackMap,
    longestNotes,
    onMidiProcessed,
  };
};
