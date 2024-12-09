import express from 'express';
import multer from 'multer';
import { runPythonProcessor } from '../js/pythonBridge.js';
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
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  }
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
  }
});

// const upload = multer({ 
//   dest: 'uploads/',
//   limits: {
//     fileSize: 50 * 1024 * 1024 // 50MB limit
//   }
// });

router.post('/', upload.any(), async (req, res) => {
  try {
    console.log('Received request files:', req.files);

    if (!req.files || req.files.length === 0) {
      throw new Error('No files were uploaded');
    }

    // Find MIDI file
    const midiFile = req.files.find(f => f.fieldname === 'midiData');
    if (!midiFile) {
      throw new Error('MIDI data not found in upload');
    }

    // Read and parse MIDI data
    const midiDataString = fs.readFileSync(midiFile.path, 'utf8');
    const midiData = JSON.parse(midiDataString);

    // Process video files
    const videos = {};
    const videoFiles = req.files.filter(f => f.fieldname === 'videos');
    
    if (videoFiles.length === 0) {
      throw new Error('No video files found in upload');
    }

    // videoFiles.forEach(file => {
    //   const instrumentName = path.parse(file.originalname).name;
    //   videos[instrumentName] = file.path;
    //   console.log(`Processed video for ${instrumentName}: ${file.path}`);
    // });

    videoFiles.forEach(file => {
      const instrumentName = path.parse(file.originalname).name;
      // Store absolute path
      videos[instrumentName] = path.resolve(file.path);
      console.log(`Processed video for ${instrumentName}: ${file.path}`);
    });

    console.log('Final video mapping:', videos);

        // Create temp config file
        const config = {
          tracks: {
            tracks: midiData.tracks, 
            header: midiData.header
          },
          videos: videos
        };
        
        const configPath = path.join(os.tmpdir(), `video-config-${uuidv4()}.json`);
        fs.writeFileSync(configPath, JSON.stringify(config));
    
        // Call Python processor with config file path
        const result = await runPythonProcessor(configPath);
        
        // Cleanup
        fs.unlinkSync(configPath);
        
        res.json({ 
          success: true,
          result: result
        });
    

    // // Process videos
    // const processedVideos = await runPythonProcessor(midiData, videos);

    // // Cleanup temporary files
    // req.files.forEach(file => {
    //   try {
    //     if (fs.existsSync(file.path)) {
    //       fs.unlinkSync(file.path);
    //     }
    //   } catch (err) {
    //     console.error(`Error cleaning up file ${file.path}:`, err);
    //   }
    // });

    // res.json({ 
    //   success: true,
    //   videos: processedVideos
    // });

  } catch (error) {
    console.error('Video processing error:', error);
    
    // Cleanup temporary files on error
    if (req.files) {
      req.files.forEach(file => {
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
      details: error.message 
    });
  }
});

export default router;