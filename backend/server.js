import express from 'express';
import multer from 'multer';
import midiParser from 'midi-file-parser';
import ffmpeg from 'fluent-ffmpeg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// Define storage for MIDI files
const midiStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const midiDir = path.join(__dirname, 'uploads/midi');
    if (!fs.existsSync(midiDir)) {
      fs.mkdirSync(midiDir, { recursive: true });
    }
    cb(null, midiDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

// Define storage for video files
const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const videoDir = path.join(__dirname, 'uploads/videos');
    if (!fs.existsSync(videoDir)) {
      fs.mkdirSync(videoDir, { recursive: true });
    }
    cb(null, videoDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const midiUpload = multer({ storage: midiStorage });
const videoUpload = multer({ storage: videoStorage });

const extractAudio = (videoPath, outputAudioPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(outputAudioPath)
      .on('end', () => resolve(outputAudioPath))
      .on('error', (err) => reject(err))
      .run();
  });
};

const tuneToMidi = async (audioPath, midiPath) => {
  const midiData = midiParser(fs.readFileSync(midiPath));
  const notes = extractMelodyFromMidi(midiData); // Custom function to get MIDI melody

  await tunePitch.autoTune(audioPath, notes); // Tune audio to MIDI melody
  return tunedAudioPath;
};

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

app.post('/upload-midi', midiUpload.single('midi'), (req, res) => {
  console.log('uploading midi file');
  console.log(req.file);

  const midiFileBuffer = fs.readFileSync(req.file.path);
  const midiFileString = midiFileBuffer.toString('binary');
  console.log('going to parse midi file');
  const midiData = midiParser(midiFileString);
  console.log('midi file parsed');
  // Pretty-print the entire parsed MIDI data

  // Extract track names, instruments, and meta events
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

  // Log track information
  trackInfo.forEach((track) => {
    console.log(`Track ${track.trackNumber}: ${track.trackName}`);
    console.log(`Instruments: ${track.instruments.join(', ')}`);
    console.log(`Drum Instruments: ${track.drumInstruments.join(', ')}`);
    console.log(`Channels: ${track.channels.join(', ')}`);
    console.log('Meta Events:', track.metaEvents);
  });

  // Send detailed response
  res.json({
    message: 'MIDI file uploaded and analyzed',
    header: midiData.header,
    tracks: trackInfo,
  });
});

app.post('/api/upload-video', videoUpload.single('video'), async (req, res) => {
  console.log('Uploading video:', req.file);
  
  try {
    const videoPath = req.file.path;
    console.log('Video uploaded:', videoPath);
    const audioPath = path.join('uploads/videos', 'extracted-audio.wav');
    const tunedAudioPath = path.join('uploads/wav', 'tuned-audio.wav');
    const midiPath = path.join('uploads/midi', 'melody.mid'); // Path to MIDI file

    // Extract audio from video
    await extractAudio(videoPath, audioPath);

    // Tune the audio to the MIDI file's melody
    await tuneToMidi(audioPath, midiPath);

    // Send the tuned audio back to the client
    res.sendFile(tunedAudioPath);
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to process audio.' });
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});