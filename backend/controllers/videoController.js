// import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
// import midiParser from 'midi-parser-js'; // Assuming midi-parser-js is the correct library
// import tunePitch from 'tune-pitch'; // Assuming tune-pitch is the correct library

const extractAudio = (videoPath, outputAudioPath) => {
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .output(outputAudioPath)
      .on('end', () => resolve(outputAudioPath))
      .on('error', (err) => reject(err))
      .run();
  });
};

// const extractMelodyFromMidi = (midiData) => {
//   // Implement the logic to extract melody from MIDI data
//   // This is a placeholder implementation
//   return midiData.tracks[0].notes.map(note => note.midi);
// };

// const tuneToMidi = async (audioPath, midiPath) => {
//   const midiData = midiParser(fs.readFileSync(midiPath));
//   const notes = extractMelodyFromMidi(midiData); // Custom function to get MIDI melody
//   const tunedAudioPath = path.join('uploads/wav', 'tuned-audio.wav'); // Define tunedAudioPath

//   await tunePitch.autoTune(audioPath, notes); // Tune audio to MIDI melody
//   return tunedAudioPath;
// };

export const uploadVideo = async (req, res) => {
  console.log('Uploading video:', req.file);

  try {
    const videoPath = req.file.path;
    console.log('Video uploaded:', videoPath);
    const audioPath = path.join('uploads/videos', 'extracted-audio.wav');
    const tunedAudioPath = path.join('uploads/wav', 'tuned-audio.wav');
    // const midiPath = path.join('uploads/midi', 'melody.mid'); // Path to MIDI file

    // Extract audio from video
    await extractAudio(videoPath, audioPath);

    // Tune the audio to the MIDI file's melody
    // await tuneToMidi(audioPath, midiPath);

    // Send the tuned audio back to the client
    res.sendFile(tunedAudioPath);
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: 'Failed to process audio.' });
  }
};
