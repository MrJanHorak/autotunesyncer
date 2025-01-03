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

router.post('/autotune-video', videoUpload.single('video'), autotuneVideo);

export default router;