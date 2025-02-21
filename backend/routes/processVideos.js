import express from 'express';
import multer from 'multer';
import { runPythonProcessor, preprocessVideo } from '../js/pythonBridge.js';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { v4 as uuidv4 } from 'uuid';

const router = express.Router();

// Ensure uploads directory exists
const uploadsDir = 'uploads';
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// Configure multer
const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (req, file, cb) => {
    // Keep original filename but make it unique
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  },
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    console.log('Received file:', file);
    if (file.fieldname === 'midiData' || file.fieldname === 'videos') {
      cb(null, true);
    } else {
      cb(new Error('Unexpected field'));
    }
  },
});

router.post('/', upload.any(), async (req, res) => {
  try {
    console.log('Received request files:', req.files);

    if (!req.files || req.files.length === 0) {
      throw new Error('No files were uploaded');
    }

    // Find MIDI file
    const midiFile = req.files.find((f) => f.fieldname === 'midiData');
    if (!midiFile) {
      throw new Error('MIDI data not found in upload');
    }

    // Read and parse MIDI data
    const midiDataString = fs.readFileSync(midiFile.path, 'utf8');
    const midiData = JSON.parse(midiDataString);

    // Process video files
    const videos = {};
    const videoFiles = req.files.filter((f) => f.fieldname === 'videos');

    // Get grid dimensions from MIDI data for sizing
    const gridArrangement = midiData.gridArrangement;
    const maxRow = Math.max(
      ...Object.values(gridArrangement).map((pos) => pos.row)
    );
    const maxCol = Math.max(
      ...Object.values(gridArrangement).map((pos) => pos.column)
    );
    const targetWidth = Math.floor(1920 / (maxCol + 1));
    const targetHeight = Math.floor(1080 / (maxRow + 1));

    // Process each video
    for (const file of videoFiles) {
      const instrumentName = path.parse(file.originalname).name;
      const originalPath = file.path;
      const processedPath = path.join(
        uploadsDir, 
        `processed_${path.basename(file.path)}`
      );

      try {
        // Preprocess video with correct dimensions
        await preprocessVideo(
          originalPath, 
          processedPath, 
          `${targetWidth}x${targetHeight}`
        );
        
        // videos[instrumentName] = processedPath;
        videos[instrumentName] = path.resolve(processedPath);
        console.log(`Processed ${instrumentName}: ${processedPath}`);
        
        // Cleanup original
        fs.unlinkSync(originalPath);
      } catch (err) {
        console.error(`Error preprocessing ${instrumentName}:`, err);
        throw err;
      }
    }

    if (videoFiles.length === 0) {
      throw new Error('No video files found in upload');
    }

    console.log('Final video mapping:', videos);

    // Create temp config file
    const config = {
      tracks: {
        tracks: midiData.tracks,
        header: midiData.header,
        gridArrangement: midiData.gridArrangement, // Add this line
      },
      videos: videos,
    };
    console.log('Grid arrangement in config:', config.tracks.gridArrangement);
    const configPath = path.join(os.tmpdir(), `video-config-${uuidv4()}.json`);
    fs.writeFileSync(configPath, JSON.stringify(config));

    // Call Python processor with config file path
    const result = await runPythonProcessor(configPath);

    // Cleanup
    fs.unlinkSync(configPath);

    res.json({
      success: true,
      result: result,
    });

  } catch (error) {
    console.error('Video processing error:', error);

    // Cleanup temporary files on error
    if (req.files) {
      req.files.forEach((file) => {
        try {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
          }
        } catch (err) {
          console.error(`Error cleaning up file ${file.path}:`, err);
        }
      });
    }

    res.status(500).json({
      error: 'Failed to process videos',
      details: error.message,
    });
  }
});

export default router;
