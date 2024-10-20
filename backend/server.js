// server.js
const express = require('express');
const multer = require('multer');
const midiParser = require('midi-file-parser');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.post('/upload-midi', upload.single('midi'), (req, res) => {
  const midiFile = fs.readFileSync(req.file.path);
  const midiData = midiParser(midiFile);
  // Extract instrument data from midiData
  console.log(midiData);
  res.send('MIDI file uploaded and analyzed');
});

app.post('/upload-videos', upload.fields([{ name: 'piano' }, { name: 'drums' }]), (req, res) => {
  const videoUrls = [];
  // Process and synchronize video clips using ffmpeg
  console.log(req.files);
  for (const [instrument, file] of Object.entries(req.files)) {
    const outputPath = path.join(__dirname, 'processed', `${instrument}.mp4`);
    ffmpeg(file[0].path)
      .output(outputPath)
      .on('end', () => {
        videoUrls.push(outputPath);
        if (videoUrls.length === Object.keys(req.files).length) {
          res.json({ videoUrls });
        }
      })
      .run();
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});