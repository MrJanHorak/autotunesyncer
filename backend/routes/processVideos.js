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
      const targetHeight = Math.floor(1080 / (maxRow + 1)); // Process each video
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

          // Structure data to match Python processor expectations
          videos[instrumentName] = {
            path: path.resolve(processedPath),
            isDrum: instrumentName.toLowerCase().includes('drum'),
            notes: [], // Will be populated from MIDI data below
            layout: { x: 0, y: 0, width: targetWidth, height: targetHeight },
          };
          console.log(`Processed ${instrumentName}: ${processedPath}`);

          // Cleanup original
          fs.unlinkSync(originalPath);
        } catch (err) {
          console.error(`Error preprocessing ${instrumentName}:`, err);
          throw err;
        }
      }

      // NOW POPULATE NOTES FROM MIDI DATA
      console.log('\n=== Mapping MIDI Notes to Videos ===');

      // Helper function to normalize instrument names
      const normalizeInstrumentName = (name) => {
        return name.toLowerCase().replace(/\s+/g, '_');
      };

      // Helper function to check if track is drum track
      const isDrumTrack = (track) => {
        return (
          track.channel === 9 ||
          track.instrument?.name?.toLowerCase().includes('drum') ||
          track.instrument?.family?.toLowerCase().includes('drum')
        );
      }; // Helper function to get drum name from MIDI note
      const getDrumName = (midiNote) => {
        const DRUM_NOTES = {
          27: 'Laser',
          28: 'Whip',
          29: 'Scratch Push',
          30: 'Scratch Pull',
          31: 'Stick Click',
          32: 'Metronome Click',
          34: 'Metronome Bell',
          35: 'Bass Drum',
          36: 'Kick Drum',
          37: 'Snare Cross Stick',
          38: 'Snare Drum',
          39: 'Hand Clap',
          40: 'Electric Snare Drum',
          41: 'Floor Tom 2',
          42: 'Hi-Hat Closed',
          43: 'Floor Tom 1',
          44: 'Hi-Hat Foot',
          45: 'Low Tom',
          46: 'Hi-Hat Open',
          47: 'Low-Mid Tom',
          48: 'High-Mid Tom',
          49: 'Crash Cymbal',
          50: 'High Tom',
          51: 'Ride Cymbal',
          52: 'China Cymbal',
          53: 'Ride Bell',
          54: 'Tambourine',
          55: 'Splash cymbal',
          56: 'Cowbell',
          57: 'Crash Cymbal 2',
          58: 'Vibraslap',
          59: 'Ride Cymbal 2',
          60: 'High Bongo',
          61: 'Low Bongo',
          62: 'Conga Dead Stroke',
          63: 'Conga',
          64: 'Tumba',
          65: 'High Timbale',
          66: 'Low Timbale',
          67: 'High Agogo',
          68: 'Low Agogo',
          69: 'Cabasa',
          70: 'Maracas',
          71: 'Whistle Short',
          72: 'Whistle Long',
          73: 'Guiro Short',
          74: 'Guiro Long',
          75: 'Claves',
          76: 'High Woodblock',
          77: 'Low Woodblock',
          78: 'Cuica High',
          79: 'Cuica Low',
          80: 'Triangle Mute',
          81: 'Triangle Open',
          82: 'Shaker',
          83: 'Sleigh Bell',
          84: 'Bell Tree',
          85: 'Castanets',
          86: 'Surdu Dead Stroke',
          87: 'Surdu',
          91: 'Snare Drum Rod',
          92: 'Ocean Drum',
          93: 'Snare Drum Brush',
        };
        return DRUM_NOTES[midiNote] || `Drum_${midiNote}`;
      }; // Map MIDI notes to video files
      midiData.tracks.forEach((track, trackIndex) => {
        if (!track.notes || track.notes.length === 0) {
          console.log(`Track ${trackIndex}: No notes found`);
          return;
        }

        console.log(
          `Processing track ${trackIndex}: ${track.instrument?.name} (${track.notes.length} notes)`
        );

        if (isDrumTrack(track)) {
          // Handle drum tracks
          track.notes.forEach((note) => {
            const drumName = getDrumName(note.midi);
            const drumKey = `drum_${drumName
              .toLowerCase()
              .replace(/\s+/g, '_')}`;

            // Find video key that ends with the drum pattern
            const matchingVideoKey = Object.keys(videos).find(
              (key) => key.includes(drumKey) || key.endsWith(drumKey)
            );

            if (matchingVideoKey) {
              videos[matchingVideoKey].notes.push({
                midi: note.midi,
                time: note.time,
                duration: note.duration,
                velocity: note.velocity || 0.8,
              });
              console.log(
                `  Mapped drum note ${note.midi} (${drumName}) to ${matchingVideoKey}`
              );
            } else {
              console.log(`  No video found for drum: ${drumKey}`);
            }
          });
        } else {
          // Handle melodic instruments
          const normalizedName = normalizeInstrumentName(track.instrument.name);

          // Find video key that ends with the instrument name (handling timestamp prefixes)
          const matchingVideoKey = Object.keys(videos).find((key) => {
            // Extract instrument part from filename (after last dash)
            const keyParts = key.split('-');
            const instrumentPart = keyParts[keyParts.length - 1];
            return (
              instrumentPart === normalizedName || key.includes(normalizedName)
            );
          });

          if (matchingVideoKey) {
            track.notes.forEach((note) => {
              videos[matchingVideoKey].notes.push({
                midi: note.midi,
                time: note.time,
                duration: note.duration,
                velocity: note.velocity || 0.8,
              });
            });
            console.log(
              `  Mapped ${track.notes.length} notes to ${matchingVideoKey}`
            );
          } else {
            console.log(`  No video found for instrument: ${normalizedName}`);
            console.log(
              `  Available video keys: ${Object.keys(videos)
                .slice(0, 3)
                .join(', ')}...`
            );
          }
        }
      });

      // Log final note counts
      console.log('\n=== Final Note Mapping Results ===');
      Object.entries(videos).forEach(([key, video]) => {
        console.log(`${key}: ${video.notes.length} notes mapped`);
      });

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
      fs.writeFileSync(configPath, JSON.stringify(config)); // Call Python processor with config file path
      const result = await runPythonProcessor(configPath);

      // Move output file to permanent location
      const timestamp = Date.now();
      const permanentOutputPath = path.join(
        uploadsDir,
        `final_output_${timestamp}.mp4`
      );

      if (result.outputPath && fs.existsSync(result.outputPath)) {
        fs.copyFileSync(result.outputPath, permanentOutputPath);
        console.log(`Output moved to: ${permanentOutputPath}`);

        // Clean up temp output
        try {
          fs.unlinkSync(result.outputPath);
        } catch (e) {
          console.warn('Failed to cleanup temp output:', e.message);
        }
      }

      // Cleanup config file
      fs.unlinkSync(configPath);

      res.json({
        success: true,
        result: result,
        outputPath: permanentOutputPath,
        message: `Video composition completed successfully! Output saved to: ${permanentOutputPath}`,
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
