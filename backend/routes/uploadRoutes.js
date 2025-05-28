import express from 'express';
import multer from 'multer';
import { handleUpload } from '../controllers/uploadController.js';

const router = express.Router();

const storage = multer.memoryStorage();

// Optimized multer configuration with larger limits and better error handling
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1GB limit for better user experience
    fieldSize: 1000 * 1024 * 1024,
    fields: 50,
    files: 20,
  },
  fileFilter: (req, file, cb) => {
    // Accept video files and common formats
    if (
      file.mimetype.startsWith('video/') ||
      file.mimetype.startsWith('audio/') ||
      file.originalname.match(/\.(mp4|avi|mov|wmv|flv|webm|mkv|mp3|wav|aac)$/i)
    ) {
      cb(null, true);
    } else {
      cb(new Error('Only video and audio files are allowed'), false);
    }
  },
}).single('video');

router.post(
  '/',
  (req, res, next) => {
    upload(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Upload multer error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            error: 'File too large',
            details: 'Maximum file size is 1GB',
            code: err.code,
          });
        }
        return res.status(400).json({
          error: `Upload error: ${err.message}`,
          code: err.code,
        });
      } else if (err) {
        console.error('Upload server error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
          suggestion: 'Please check file format and try again',
        });
      }

      console.log('Upload successful:', {
        filename: req.file?.originalname,
        size: req.file?.size,
        mimetype: req.file?.mimetype,
      });
      next();
    });
  },
  handleUpload
);

export default router;
