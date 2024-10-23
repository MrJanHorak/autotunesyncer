import express from 'express';
import multer from 'multer';
import { uploadVideo } from '../controllers/videoController.js';
import path from 'path';
import fs from 'fs';

const router = express.Router();

// Define storage for video files
const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const __dirname = path.dirname(new URL(import.meta.url).pathname);
    const videoDir = path.join(__dirname, '../uploads/videos');
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

router.post('/upload-video', videoUpload.single('video'), uploadVideo);

export default router;