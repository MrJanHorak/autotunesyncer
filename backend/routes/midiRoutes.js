import express from 'express';
import multer from 'multer';
import { uploadMidi } from '../controllers/midiController.js';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// added for Python processing in new approach to process videos
import { runPythonProcessor } from '../js/pythonBridge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Define storage for MIDI files
const midiStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const midiDir = path.join(__dirname, '../uploads/midi');
    if (!fs.existsSync(midiDir)) {
      fs.mkdirSync(midiDir, { recursive: true });
    }
    cb(null, midiDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const midiUpload = multer({ storage: midiStorage });

router.post('/upload-midi', midiUpload.single('midi'), uploadMidi);


// Add new route for processing videos
router.post('/process-videos', async (req, res) => {
  try {
    const { midiTracks, videos } = req.body;
    
    // Validate input
    if (!midiTracks || !videos) {
      return res.status(400).json({ error: 'Missing required data' });
    }

    // Process videos
    const processedVideos = await runPythonProcessor(midiTracks, videos);

    res.json({ 
      success: true,
      videos: processedVideos
    });

  } catch (error) {
    console.error('Video processing error:', error);
    res.status(500).json({ 
      error: 'Failed to process videos',
      details: error.message 
    });
  }
});

export default router;