import express from 'express';
import multer from 'multer';
import { composeVideo } from '../controllers/compositionController.js';

const router = express.Router();

// Configure multer for handling multipart/form-data
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
    files: 10 // Maximum 10 files
  }
}).any(); // Accept any field names

router.post('/', (req, res, next) => {
  upload(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(400).json({ error: `Upload error: ${err.message}` });
    } else if (err) {
      return res.status(500).json({ error: `Server error: ${err.message}` });
    }
    next();
  });
}, composeVideo);

export default router;