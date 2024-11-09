import express from 'express';
import multer from 'multer';
import { composeVideo } from '../controllers/compositionController.js';

const router = express.Router();

// Configure multer with significantly increased limits
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 500 * 1024 * 1024, // 500MB per file
    files: 20, // Allow up to 20 files
    fieldSize: 500 * 1024 * 1024 // 500MB field size
  }
}).any();

router.post('/', (req, res, next) => {
  console.log('Composition request received');
  
  upload(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      console.error('Multer error:', err);
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
          error: 'File too large',
          details: err.message,
          field: err.field
        });
      }
      return res.status(400).json({
        error: `Upload error: ${err.message}`,
        code: err.code,
        field: err.field
      });
    } else if (err) {
      console.error('Unknown error:', err);
      return res.status(500).json({
        error: `Server error: ${err.message}`
      });
    }

    console.log('Files received:', {
      count: req.files?.length,
      sizes: req.files?.map(f => ({ name: f.fieldname, size: f.size }))
    });
    next();
  });
}, composeVideo);

export default router;