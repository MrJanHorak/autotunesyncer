import express from 'express';
import multer from 'multer';
import { authenticateToken } from '../middleware/auth.js';
import { requireProjectOwnership } from '../middleware/projectOwnership.js';
import { handlePrecache } from '../controllers/precacheController.js';

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 1000 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('video/') || file.originalname.match(/\.(mp4|webm|mov)$/i)) {
      cb(null, true);
    } else {
      cb(new Error('Only video files are accepted for pre-caching'), false);
    }
  },
}).single('video');

router.post(
  '/',
  authenticateToken,
  requireProjectOwnership,
  (req, res, next) => {
    upload(req, res, (err) => {
      if (err) return res.status(400).json({ error: err.message });
      next();
    });
  },
  handlePrecache
);

export default router;
