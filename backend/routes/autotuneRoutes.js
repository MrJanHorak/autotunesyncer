import express from 'express';
import multer from 'multer';
import { autotuneVideo } from '../controllers/autotuneController.js';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Define storage for video files
const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const videoDir = path.resolve(__dirname, '../uploads/videos');
    if (!fs.existsSync(videoDir)) {
      fs.mkdirSync(videoDir, { recursive: true });
    }
    cb(null, videoDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const videoUpload = multer({ storage: videoStorage });

// Ensure the video directory exists at server startup
const videoDir = path.resolve(__dirname, '../uploads/videos');
if (!fs.existsSync(videoDir)) {
  fs.mkdirSync(videoDir, { recursive: true });
}

// Route for autotuning video
router.post('/', videoUpload.single('video'), (req, res, next) => {
  console.log('Uploading video...');
  if (!req.file) {
    return res.status(400).send({ error: 'No video file uploaded' });
  }
  console.log('Video uploaded:', req.file);
  next();
}, autotuneVideo);

export default router;