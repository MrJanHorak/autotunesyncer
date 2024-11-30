import express from 'express';
import { runPythonProcessor } from '../js/pythonBridge.js';

const router = express.Router();

router.post('/', async (req, res) => {
  try {
    const { midiTracks, videos } = req.body;
    
    // Validate input
    if (!midiTracks || !videos) {
      return res.status(400).json({ error: 'Missing required data' });
    }

    // Process videos
    const processedVideos = await runPythonProcessor(midiTracks, videos);

    res.json({ 
      success: true,
      videos: processedVideos
    });

  } catch (error) {
    console.error('Video processing error:', error);
    res.status(500).json({ 
      error: 'Failed to process videos',
      details: error.message 
    });
  }
});

export default router;  