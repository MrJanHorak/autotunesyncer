import express from 'express';
import multer from 'multer';
import { autotuneVideo } from '../controllers/autotuneController.js';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Define enhanced storage for video files
const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const videoDir = path.resolve(__dirname, '../uploads/videos');
    if (!fs.existsSync(videoDir)) {
      fs.mkdirSync(videoDir, { recursive: true });
    }
    cb(null, videoDir);
  },
  filename: (req, file, cb) => {
    // Sanitize filename and add timestamp
    const sanitizedName = file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    cb(null, `${Date.now()}-${sanitizedName}`);
  },
});

// Optimized video upload configuration
const videoUpload = multer({
  storage: videoStorage,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1GB per file
    fieldSize: 1000 * 1024 * 1024,
    fields: 10,
    files: 5,
  },
  fileFilter: (req, file, cb) => {
    // Accept video files
    if (
      file.mimetype.startsWith('video/') ||
      file.originalname.match(/\.(mp4|avi|mov|wmv|flv|webm|mkv)$/i)
    ) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are allowed'), false);
    }
  },
});

// Enhanced route with error handling
router.post(
  '/autotune-video',
  (req, res, next) => {
    console.log('Video autotune request received');

    videoUpload.single('video')(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Video upload error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            error: 'Video file too large',
            details: 'Maximum video file size is 1GB',
            code: err.code,
          });
        }
        return res.status(400).json({
          error: `Upload error: ${err.message}`,
          code: err.code,
        });
      } else if (err) {
        console.error('Video server error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
          suggestion: 'Please ensure you are uploading a valid video file',
        });
      }

      if (!req.file) {
        return res.status(400).json({
          error: 'No video file uploaded',
          suggestion: 'Please select a video file to upload',
        });
      }

      console.log('Video uploaded successfully:', {
        filename: req.file.originalname,
        size: req.file.size,
        path: req.file.path,
      });
      next();
    });
  },
  autotuneVideo
);

export default router;
