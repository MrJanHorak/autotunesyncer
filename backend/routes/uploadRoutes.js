import express from 'express';
import multer from 'multer';
import { authenticateToken } from '../middleware/auth.js';
import { requireProjectOwnership } from '../middleware/projectOwnership.js';
import { handleUpload } from '../controllers/uploadController.js';

const router = express.Router();

const storage = multer.memoryStorage();

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 1000 * 1024 * 1024,
    fieldSize: 1000 * 1024 * 1024,
    fields: 50,
    files: 20,
  },
  fileFilter: (req, file, cb) => {
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
  // Auth + project ownership run before multer so req.project is ready
  authenticateToken,
  requireProjectOwnership,
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
        return res.status(400).json({ error: `Upload error: ${err.message}`, code: err.code });
      } else if (err) {
        console.error('Upload server error:', err);
        return res.status(500).json({ error: `Server error: ${err.message}` });
      }
      console.log('Upload successful:', {
        filename: req.file?.originalname,
        size: req.file?.size,
        mimetype: req.file?.mimetype,
        project: req.project?.id,
      });
      next();
    });
  },
  handleUpload
);

export default router;

