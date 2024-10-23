import {useEffect } from 'react';

const MidiParser = ({ file, onParsed }) => {
  useEffect(() => {
    if (!file) return;

    const parseMidiFile = async () => {
      const arrayBuffer = await file.arrayBuffer();
      const midiData = new Uint8Array(arrayBuffer);
      
      // Initialize track data structure
      let tracks = [];
      let currentTrack = null;
      let time = 0;
      
      // Parse MIDI header
      const format = (midiData[8] << 8) + midiData[9];
      const trackCount = (midiData[10] << 8) + midiData[11];
      const timeDivision = (midiData[12] << 8) + midiData[13];
      
      let position = 14; // Start after header

      // Parse each track
      for (let trackIndex = 0; trackIndex < trackCount; trackIndex++) {
        currentTrack = {
          id: trackIndex,
          name: `Track ${trackIndex + 1}`,
          notes: [],
          duration: 0,
          tempo: 120,
          instrument: 0
        };

        // Skip "MTrk" and track length
        position += 8;

        while (position < midiData.length) {
          // Parse delta time
          let deltaTime = 0;
          while (midiData[position] & 0x80) {
            deltaTime = (deltaTime << 7) + (midiData[position] & 0x7F);
            position++;
          }
          deltaTime = (deltaTime << 7) + (midiData[position]);
          position++;
          
          time += deltaTime;

          // Parse event
          const eventType = midiData[position];
          position++;

          if (eventType === 0xFF) { // Meta event
            const metaType = midiData[position];
            position++;
            const length = midiData[position];
            position++;

            if (metaType === 0x03) { // Track name
              currentTrack.name = new TextDecoder().decode(
                midiData.slice(position, position + length)
              );
            } else if (metaType === 0x51) { // Tempo
              const tempo = (midiData[position] << 16) + 
                          (midiData[position + 1] << 8) + 
                          midiData[position + 2];
              currentTrack.tempo = Math.round(60000000 / tempo);
            }

            position += length;
          } else if (eventType >= 0x80 && eventType <= 0xEF) { // MIDI event
            const noteNumber = midiData[position];
            const velocity = midiData[position + 1];
            position += 2;

            if ((eventType & 0xF0) === 0x90 && velocity > 0) { // Note on
              currentTrack.notes.push({
                note: noteNumber,
                time: time,
                velocity: velocity
              });
            }
          }
        }

        currentTrack.duration = time / timeDivision * (60 / currentTrack.tempo);
        tracks.push(currentTrack);
      }

      onParsed(tracks);
    };

    parseMidiFile().catch(console.error);
  }, [file, onParsed]);

  return null;
};

export default MidiParser;