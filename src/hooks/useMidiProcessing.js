import { useState, useEffect, useCallback } from 'react';
import {
  parseMidiFile,
  extractInstruments,
  createInstrumentTrackMap,
  calculateLongestNotes,
  normalizeInstrumentName
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

      // Extract all instruments including drum groups
      const instrumentSet = new Set();
      const trackMapping = new Map();
      const noteDurations = {};

      midiData.tracks.forEach((track, index) => {
        if (track.notes && track.notes.length > 0) {
          if (isDrumTrack(track)) {
            // Handle drum tracks
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

            // Add individual drum instruments
            drumGroups.forEach((group) => {
              const drumInstrument = {
                isDrum: true,
                group: group,
                family: 'percussion',
                number: track.channel,
              };
              instrumentSet.add(JSON.stringify(drumInstrument));
              trackMapping.set(`drum_${group}`, index);
            });
          } else {
            // Handle melodic instruments
            const instrument = {
              isDrum: false,
              name: track.instrument.name,
              family: track.instrument.family,
              number: track.channel,
            };
            instrumentSet.add(JSON.stringify(instrument));

            // Track longest note duration for melodic instruments
            const normalizedName = normalizeInstrumentName(
              track.instrument.name
            );
            track.notes.forEach((note) => {
              noteDurations[normalizedName] = Math.max(
                noteDurations[normalizedName] || 0,
                note.duration
              );
            });

            trackMapping.set(
              normalizeInstrumentName(track.instrument.name),
              index
            );
          }
        }
      });

      // Convert instrument set back to array
      const instrumentArray = Array.from(instrumentSet).map((inst) =>
        JSON.parse(inst)
      );

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
