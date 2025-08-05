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

// Define enhanced storage for MIDI files
const midiStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const midiDir = path.join(__dirname, '../uploads/midi');
    if (!fs.existsSync(midiDir)) {
      fs.mkdirSync(midiDir, { recursive: true });
    }
    cb(null, midiDir);
  },
  filename: (req, file, cb) => {
    // Sanitize filename and add timestamp
    const sanitizedName = file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    cb(null, `${Date.now()}-${sanitizedName}`);
  },
});

// Optimized MIDI upload configuration
const midiUpload = multer({
  storage: midiStorage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB for MIDI files (generous for complex compositions)
    fieldSize: 50 * 1024 * 1024,
    fields: 10,
    files: 5,
  },
  fileFilter: (req, file, cb) => {
    // Accept MIDI files
    if (
      file.mimetype === 'audio/midi' ||
      file.mimetype === 'audio/x-midi' ||
      file.originalname.match(/\.(mid|midi)$/i)
    ) {
      cb(null, true);
    } else {
      cb(new Error('Only MIDI files are allowed'), false);
    }
  },
});

// Enhanced route with error handling
router.post(
  '/upload-midi',
  (req, res, next) => {
    console.log('MIDI upload request received');

    midiUpload.single('midi')(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('MIDI upload error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            error: 'MIDI file too large',
            details: 'Maximum MIDI file size is 50MB',
            code: err.code,
          });
        }
        return res.status(400).json({
          error: `Upload error: ${err.message}`,
          code: err.code,
        });
      } else if (err) {
        console.error('MIDI server error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
          suggestion: 'Please ensure you are uploading a valid MIDI file',
        });
      }

      if (!req.file) {
        return res.status(400).json({
          error: 'No MIDI file uploaded',
          suggestion: 'Please select a MIDI file to upload',
        });
      }

      console.log('MIDI file uploaded successfully:', {
        filename: req.file.originalname,
        size: req.file.size,
        path: req.file.path,
      });
      next();
    });
  },
  uploadMidi
);

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
      videos: processedVideos,
    });
  } catch (error) {
    console.error('Video processing error:', error);
    res.status(500).json({
      error: 'Failed to process videos',
      details: error.message,
    });
  }
});

export default router;
