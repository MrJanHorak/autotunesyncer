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
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Enhanced multer configuration for better performance
const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (req, file, cb) => {
    // Keep original filename but make it unique and sanitized
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const sanitizedName = file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    cb(null, `${uniqueSuffix}-${sanitizedName}`);
  },
});

// Optimized upload configuration for video processing
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1GB per file
    fieldSize: 1000 * 1024 * 1024,
    fields: 50,
    files: 30, // Allow more files for multi-track compositions
  },
  fileFilter: (req, file, cb) => {
    console.log('Processing file:', {
      fieldname: file.fieldname,
      originalname: file.originalname,
      mimetype: file.mimetype,
    });

    if (file.fieldname === 'midiData') {
      // Accept JSON files for MIDI data
      cb(null, true);
    } else if (file.fieldname === 'videos') {
      // Accept video files
      if (
        file.mimetype.startsWith('video/') ||
        file.originalname.match(/\.(mp4|avi|mov|wmv|flv|webm|mkv)$/i)
      ) {
        cb(null, true);
      } else {
        cb(new Error(`Invalid video file format: ${file.originalname}`), false);
      }
    } else {
      console.warn(`Unexpected field: ${file.fieldname}`);
      cb(new Error(`Unexpected field: ${file.fieldname}`), false);
    }
  },
});

router.post(
  '/',
  (req, res, next) => {
    console.log('Video processing request received');

    upload.any()(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Video processing upload error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            error: 'File too large',
            details: 'Maximum file size is 1GB per file',
            code: err.code,
          });
        }
        return res.status(400).json({
          error: `Upload error: ${err.message}`,
          code: err.code,
        });
      } else if (err) {
        console.error('Video processing server error:', err);
        return res.status(500).json({
          error: `Server error: ${err.message}`,
          suggestion: 'Please check your files and try again',
        });
      }

      console.log('Video processing files received:', {
        count: req.files?.length,
        totalSize: req.files?.reduce((sum, f) => sum + f.size, 0),
        files: req.files?.map((f) => ({
          fieldname: f.fieldname,
          originalname: f.originalname,
          size: f.size,
        })),
      });
      next();
    });
  },
  async (req, res) => {
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
      const configPath = path.join(
        os.tmpdir(),
        `video-config-${uuidv4()}.json`
      );
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
  }
);

export default router;
