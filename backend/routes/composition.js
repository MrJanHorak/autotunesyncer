import express from 'express';
import multer from 'multer';
import {
  composeVideo,
  getCompositionStatus,
  downloadComposition,
  streamComposition,
} from '../controllers/compositionController.js';

const router = express.Router();

// Configure multer with significantly increased limits for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 500 * 1024 * 1024, // 500MB per file
    files: 20, // Allow up to 20 files
    fieldSize: 500 * 1024 * 1024, // 500MB field size
  },
}).any();

// Enhanced multer configuration for optimized processing
const optimizedUpload = multer({
  storage: storage,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1GB per file for optimized processing
    files: 30, // Allow more files
    fieldSize: 1000 * 1024 * 1024,
  },
}).any();

// Main composition endpoint - now returns job ID for background processing
router.post(
  '/',
  (req, res, next) => {
    console.log('Composition request received');

    optimizedUpload(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Multer error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            error: 'File too large',
            details: err.message,
            field: err.field,
            maxSize: '1GB per file',
          });
        }
        return res.status(400).json({
          error: `Upload error: ${err.message}`,
          code: err.code,
          field: err.field,
        });
      } else if (err) {
        console.error('Unknown error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
        });
      }

      console.log('Files received for optimized processing:', {
        count: req.files?.length,
        totalSize: req.files?.reduce((sum, f) => sum + f.size, 0),
        sizes: req.files?.map((f) => ({ name: f.fieldname, size: f.size })),
      });
      next();
    });
  },
  composeVideo
);

// New endpoint: Check job status and progress
router.get('/status/:jobId', getCompositionStatus);

// New endpoint: Download completed composition
router.get('/download/:jobId', downloadComposition);

// New endpoint: Real-time streaming composition with progress updates
router.post(
  '/stream',
  (req, res, next) => {
    console.log('Streaming composition request received');

    optimizedUpload(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Streaming upload error:', err);
        return res.status(413).json({
          error: 'File too large for streaming',
          details: err.message,
          suggestion: 'Use regular composition endpoint for large files',
        });
      } else if (err) {
        console.error('Streaming upload error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
        });
      }
      next();
    });
  },
  streamComposition
);

// Legacy endpoint for backward compatibility
router.post(
  '/legacy',
  (req, res, next) => {
    console.log('Legacy composition request received');

    upload(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Legacy upload error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            error: 'File too large',
            details: err.message,
            field: err.field,
          });
        }
        return res.status(400).json({
          error: `Upload error: ${err.message}`,
          code: err.code,
          field: err.field,
        });
      } else if (err) {
        console.error('Legacy upload error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
        });
      }

      console.log('Legacy files received:', {
        count: req.files?.length,
        sizes: req.files?.map((f) => ({ name: f.fieldname, size: f.size })),
      });
      next();
    });
  },
  composeVideo
);

export default router;
