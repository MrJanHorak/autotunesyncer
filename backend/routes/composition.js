import express from 'express';

const router = express.Router();

/**
 * DEPRECATED — This Bull-queue composition path is not used by the frontend.
 * The active hot path is POST /api/process-videos (backend/routes/processVideos.js).
 * All routes below return 410 Gone with a migration note.
 */

const deprecatedResponse = (res) =>
  res.status(410).json({
    error: 'This endpoint is deprecated',
    replacement: 'POST /api/process-videos',
    message: 'Use /api/process-videos for all composition requests.',
  });

// Main composition endpoint
router.post('/', (req, res) => deprecatedResponse(res));

// Status / download / stream / legacy — all deprecated
router.get('/status/:jobId', (req, res) => deprecatedResponse(res));
router.get('/download/:jobId', (req, res) => deprecatedResponse(res));
router.post('/stream', (req, res) => deprecatedResponse(res));
router.post('/legacy', (req, res) => deprecatedResponse(res));

export default router;
