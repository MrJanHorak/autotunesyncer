import fs from 'fs';
import midiParser from 'midi-file-parser';

const drumMap = {
  35: 'Acoustic Bass Drum',
  36: 'Bass Drum 1',
  37: 'Side Stick',
  38: 'Acoustic Snare',
  39: 'Hand Clap',
  40: 'Electric Snare',
  41: 'Low Floor Tom',
  42: 'Closed Hi-Hat',
  43: 'High Floor Tom',
  44: 'Pedal Hi-Hat',
  45: 'Low Tom',
  46: 'Open Hi-Hat',
  47: 'Low-Mid Tom',
  48: 'Hi-Mid Tom',
  49: 'Crash Cymbal 1',
  50: 'High Tom',
  51: 'Ride Cymbal 1',
  52: 'Chinese Cymbal',
  53: 'Ride Bell',
  54: 'Tambourine',
  55: 'Splash Cymbal',
  56: 'Cowbell',
  57: 'Crash Cymbal 2',
  58: 'Vibraslap',
  59: 'Ride Cymbal 2',
  60: 'Hi Bongo',
  61: 'Low Bongo',
  62: 'Mute Hi Conga',
  63: 'Open Hi Conga',
  64: 'Low Conga',
  65: 'High Timbale',
  66: 'Low Timbale',
  67: 'High Agogo',
  68: 'Low Agogo',
  69: 'Cabasa',
  70: 'Maracas',
  71: 'Short Whistle',
  72: 'Long Whistle',
  73: 'Short Guiro',
  74: 'Long Guiro',
  75: 'Claves',
  76: 'Hi Wood Block',
  77: 'Low Wood Block',
  78: 'Mute Cuica',
  79: 'Open Cuica',
  80: 'Mute Triangle',
  81: 'Open Triangle'
};

export const uploadMidi = (req, res) => {
  console.log('uploading midi file');
  console.log(req.file);

  const midiFileBuffer = fs.readFileSync(req.file.path);
  const midiFileString = midiFileBuffer.toString('binary');
  console.log('going to parse midi file');
  const midiData = midiParser(midiFileString);
  console.log('midi file parsed');

  const trackInfo = midiData.tracks.map((track, index) => {
    const trackNameEvent = track.find(
      (event) => event.type === 'meta' && event.subtype === 'trackName'
    );
    const trackName = trackNameEvent ? trackNameEvent.text : `Track ${index + 1}`;

    const instruments = new Set();
    const drumInstruments = new Set();
    const metaEvents = [];
    const channels = new Set();
    track.forEach((event) => {
      if (event.type === 'midi' && event.subtype === 'programChange') {
        instruments.add(event.programNumber);
        channels.add(event.channel);
      }
      if (event.type === 'meta') {
        metaEvents.push(event);
      }
      if (event.type === 'midi' && (event.subtype === 'noteOn' || event.subtype === 'noteOff')) {
        channels.add(event.channel);
        if (event.channel === 9) { // Channel 10 in MIDI is 9 (0-indexed)
          const drumName = drumMap[event.noteNumber];
          if (drumName) {
            drumInstruments.add(drumName);
          }
        }
      }
    });

    return {
      trackNumber: index + 1,
      trackName: trackName,
      instruments: Array.from(instruments),
      drumInstruments: Array.from(drumInstruments),
      metaEvents: metaEvents,
      channels: Array.from(channels)
    };
  });

  trackInfo.forEach((track) => {
    console.log(`Track ${track.trackNumber}: ${track.trackName}`);
    console.log(`Instruments: ${track.instruments.join(', ')}`);
    console.log(`Drum Instruments: ${track.drumInstruments.join(', ')}`);
    console.log(`Channels: ${track.channels.join(', ')}`);
    console.log('Meta Events:', track.metaEvents);
  });

  res.json({
    message: 'MIDI file uploaded and analyzed',
    header: midiData.header,
    tracks: trackInfo,
  });
};