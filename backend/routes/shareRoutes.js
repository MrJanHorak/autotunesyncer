import express from 'express';
import multer from 'multer';
import { authenticateToken } from '../middleware/auth.js';
import { shareComposition } from '../controllers/shareController.js';

const router = express.Router();

const videoUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 1000 * 1024 * 1024 },
}).single('video');

router.use(authenticateToken);

router.post('/', (req, res, next) => videoUpload(req, res, (err) => {
  if (err) return res.status(400).json({ error: err.message });
  next();
}), shareComposition);

export default router;
