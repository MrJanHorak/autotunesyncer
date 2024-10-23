import express from 'express';
import multer from 'multer';
import { uploadMidi } from '../controllers/midiController.js';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

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

export default router;