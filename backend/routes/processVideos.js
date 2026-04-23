import express from 'express';
import multer from 'multer';
import { runPythonProcessor, preprocessVideo } from '../js/pythonBridge.js';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { v4 as uuidv4 } from 'uuid';
import { authenticateToken } from '../middleware/auth.js';
import { requireProjectOwnership } from '../middleware/projectOwnership.js';

const router = express.Router();

// Dynamic multer storage: uses project-scoped dir when auth + project middleware run first
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = req.project?.uploadsDir || 'uploads';
    fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const sanitizedName = file.originalname.replace(/[^a-zA-Z0-9.-]/g, '_');
    cb(null, `${uniqueSuffix}-${sanitizedName}`);
  },
});

// ── Job store ──────────────────────────────────────────────────────────────
// Keyed by jobId. Fields: status, progress, outputPath, error, createdAt, completedAt, userId, projectId
const jobs = new Map();
// Never expire queued/processing jobs by age alone.
const JOB_COMPLETED_TTL_MS = 15 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if (
      ['done', 'failed'].includes(job.status) &&
      job.completedAt &&
      now - job.completedAt > JOB_COMPLETED_TTL_MS
    ) {
      if (job.outputPath) {
        try { fs.unlinkSync(job.outputPath); } catch { /* already removed */ }
      }
      jobs.delete(id);
    }
  }
}, 5 * 60 * 1000).unref();

const isFiniteNumber = (value) => Number.isFinite(Number(value));

const normalizeVelocity = (value) => {
  if (!isFiniteNumber(value)) return 100;
  const v = Number(value);
  // Handle normalized 0..1 velocity from some MIDI parsers.
  if (v > 0 && v <= 1) {
    return Math.max(1, Math.min(127, Math.round(v * 127)));
  }
  return Math.max(1, Math.min(127, Math.round(v)));
};

const validateComposeInputs = (midiData, videoFiles) => {
  if (!midiData || typeof midiData !== 'object') {
    return 'Invalid MIDI payload';
  }

  if (!Array.isArray(midiData.tracks) || midiData.tracks.length === 0) {
    return 'MIDI payload must include at least one track';
  }

  if (
    !midiData.gridArrangement ||
    typeof midiData.gridArrangement !== 'object'
  ) {
    return 'Grid arrangement is required';
  }

  const positions = Object.values(midiData.gridArrangement);
  if (positions.length === 0) {
    return 'Grid arrangement is empty';
  }

  const invalidPosition = positions.find(
    (pos) =>
      !pos ||
      !isFiniteNumber(pos.row) ||
      !isFiniteNumber(pos.column) ||
      Number(pos.row) < 0 ||
      Number(pos.column) < 0,
  );
  if (invalidPosition) {
    return 'Grid arrangement contains invalid row/column positions';
  }

  if (!Array.isArray(videoFiles) || videoFiles.length === 0) {
    return 'At least one video file is required';
  }

  const hasNotes = midiData.tracks.some(
    (track) => Array.isArray(track.notes) && track.notes.length > 0,
  );
  if (!hasNotes) {
    return 'MIDI payload does not contain note events';
  }

  return null;
};

// Optimized upload configuration for video processing
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1GB per file
    fieldSize: 1000 * 1024 * 1024,
    fields: 50,
    files: 100, // Support large compositions with many instruments
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

// ── Background composition job ─────────────────────────────────────────────
async function runCompositionJob(jobId, files, isPreview, jobUploadsDir) {
  const updateJob = (patch) => jobs.set(jobId, { ...jobs.get(jobId), ...patch });
  const tempFiles = []; // paths to clean up on failure

  try {
    updateJob({ status: 'processing', progress: 0 });

    // ── 1. Parse MIDI data ────────────────────────────────────────────────
    const midiFile = files.find((f) => f.fieldname === 'midiData');
    if (!midiFile) throw new Error('MIDI data not found in upload');
    tempFiles.push(midiFile.path);

    const midiDataString = fs.readFileSync(midiFile.path, 'utf8');
    let midiData;
    try {
      midiData = JSON.parse(midiDataString);
    } catch {
      throw new Error('Invalid midiData JSON payload');
    }

    const videoFiles = files.filter((f) => f.fieldname === 'videos');
    const validationError = validateComposeInputs(midiData, videoFiles);
    if (validationError) throw new Error(validationError);

    updateJob({ progress: 5 });

    // ── 2. Compute cell size ──────────────────────────────────────────────
    const gridArrangement = midiData.gridArrangement;
    const maxRow = Math.max(
      ...Object.values(gridArrangement).map((pos) => pos.row),
    );
    const maxCol = Math.max(
      ...Object.values(gridArrangement).map((pos) => pos.column),
    );
    const totalWidth = isPreview ? 640 : 1920;
    const totalHeight = isPreview ? 360 : 1080;
    const targetWidth = Math.floor(totalWidth / (maxCol + 1));
    const targetHeight = Math.floor(totalHeight / (maxRow + 1));
    console.log(
      `[Job ${jobId}] Cell size: ${targetWidth}x${targetHeight} (${isPreview ? 'PREVIEW' : 'PRODUCTION'})`,
    );

    // ── 3. Preprocess videos ──────────────────────────────────────────────
    const videos = {};
    const totalVideos = videoFiles.length;
    let processedCount = 0;

    for (const file of videoFiles) {
      const instrumentName = path.parse(file.originalname).name;
      const originalPath = file.path;
      tempFiles.push(originalPath);

      const fileBuffer = fs.readFileSync(originalPath);
      const cacheKey = crypto
        .createHash('sha1')
        .update(fileBuffer)
        .update(`${targetWidth}x${targetHeight}:${isPreview ? 'preview' : 'prod'}`)
        .digest('hex')
        .slice(0, 16);
      const processedPath = path.join(
        jobUploadsDir,
        `processed_${cacheKey}-${instrumentName}.mp4`,
      );

      if (fs.existsSync(processedPath)) {
        console.log(`[Job ${jobId}] Cache hit: ${instrumentName}`);
      } else {
        await preprocessVideo(
          originalPath,
          processedPath,
          `${targetWidth}x${targetHeight}`,
          {
            performanceMode: true,
            quality: isPreview ? 'low' : 'high',
            memoryLimit: 4,
          },
        );
      }

      videos[instrumentName] = {
        path: path.resolve(processedPath),
        isDrum: instrumentName.toLowerCase().includes('drum'),
        notes: [],
        layout: { x: 0, y: 0, width: targetWidth, height: targetHeight },
      };

      // Clean up the raw upload once preprocessed
      try { fs.unlinkSync(originalPath); } catch { /* ignore */ }
      // Remove from tempFiles since it's already gone
      const idx = tempFiles.indexOf(originalPath);
      if (idx !== -1) tempFiles.splice(idx, 1);

      processedCount++;
      updateJob({ progress: 5 + Math.round((processedCount / totalVideos) * 40) }); // 5→45%
    }

    // ── 4. Map MIDI notes to videos ───────────────────────────────────────
    const normalizeInstrumentName = (name) =>
      name.toLowerCase().replace(/\s+/g, '_');

    const isDrumTrack = (track) =>
      track.channel === 9 ||
      track.instrument?.name?.toLowerCase().includes('drum') ||
      track.instrument?.family?.toLowerCase().includes('drum');

    const getDrumName = (midiNote) => {
      const DRUM_NOTES = {
        27: 'Laser', 28: 'Whip', 29: 'Scratch Push', 30: 'Scratch Pull',
        31: 'Stick Click', 32: 'Metronome Click', 34: 'Metronome Bell',
        35: 'Bass Drum', 36: 'Kick Drum', 37: 'Snare Cross Stick',
        38: 'Snare Drum', 39: 'Hand Clap', 40: 'Electric Snare Drum',
        41: 'Floor Tom 2', 42: 'Hi-Hat Closed', 43: 'Floor Tom 1',
        44: 'Hi-Hat Foot', 45: 'Low Tom', 46: 'Hi-Hat Open',
        47: 'Low-Mid Tom', 48: 'High-Mid Tom', 49: 'Crash Cymbal',
        50: 'High Tom', 51: 'Ride Cymbal', 52: 'China Cymbal',
        53: 'Ride Bell', 54: 'Tambourine', 55: 'Splash cymbal',
        56: 'Cowbell', 57: 'Crash Cymbal 2', 58: 'Vibraslap',
        59: 'Ride Cymbal 2', 60: 'High Bongo', 61: 'Low Bongo',
        62: 'Conga Dead Stroke', 63: 'Conga', 64: 'Tumba',
        65: 'High Timbale', 66: 'Low Timbale', 67: 'High Agogo',
        68: 'Low Agogo', 69: 'Cabasa', 70: 'Maracas',
        71: 'Whistle Short', 72: 'Whistle Long', 73: 'Guiro Short',
        74: 'Guiro Long', 75: 'Claves', 76: 'High Woodblock',
        77: 'Low Woodblock', 78: 'Cuica High', 79: 'Cuica Low',
        80: 'Triangle Mute', 81: 'Triangle Open', 82: 'Shaker',
        83: 'Sleigh Bell', 84: 'Bell Tree', 85: 'Castanets',
        86: 'Surdu Dead Stroke', 87: 'Surdu', 91: 'Snare Drum Rod',
        92: 'Ocean Drum', 93: 'Snare Drum Brush',
      };
      return DRUM_NOTES[midiNote] || `Drum_${midiNote}`;
    };

    midiData.tracks.forEach((track, trackIndex) => {
      if (!track.notes || track.notes.length === 0) return;
      console.log(
        `[Job ${jobId}] Track ${trackIndex}: ${track.instrument?.name} (${track.notes.length} notes)`,
      );

      if (isDrumTrack(track)) {
        track.notes.forEach((note) => {
          const drumName = getDrumName(note.midi);
          const drumKey = `drum_${drumName.toLowerCase().replace(/\s+/g, '_')}`;
          const matchingKey = Object.keys(videos).find(
            (k) => k.includes(drumKey) || k.endsWith(drumKey),
          );
          if (matchingKey) {
            videos[matchingKey].notes.push({
              midi: note.midi,
              time: note.time,
              duration: note.duration,
              velocity: normalizeVelocity(note.velocity),
            });
          }
        });
      } else {
        const normalizedName = normalizeInstrumentName(track.instrument.name);
        const matchingKey = Object.keys(videos).find((k) => {
          const parts = k.split('-');
          return (
            parts[parts.length - 1] === normalizedName ||
            k.includes(normalizedName)
          );
        });
        if (matchingKey) {
          track.notes.forEach((note) => {
            videos[matchingKey].notes.push({
              midi: note.midi,
              time: note.time,
              duration: note.duration,
              velocity: normalizeVelocity(note.velocity),
            });
          });
        }
      }
    });

    const totalMappedNotes = Object.values(videos).reduce(
      (n, v) => n + (Array.isArray(v.notes) ? v.notes.length : 0),
      0,
    );
    if (totalMappedNotes === 0) {
      throw new Error(
        'No MIDI notes were mapped to uploaded videos. Check instrument naming.',
      );
    }

    updateJob({ progress: 50 });

    // ── 5. Build config & run Python ─────────────────────────────────────
    const trackVolumes = midiData.trackVolumes || {};
    const config = {
      tracks: midiData.tracks,
      header: midiData.header,
      gridArrangement: midiData.gridArrangement,
      trackVolumes,
      compositionStyle: midiData.compositionStyle || {},
      clipStyles: midiData.clipStyles || {},
      videos,
      preview: isPreview,
    };
    const configPath = path.join(os.tmpdir(), `video-config-${jobId}.json`);
    fs.writeFileSync(configPath, JSON.stringify(config));
    tempFiles.push(configPath);

    const result = await runPythonProcessor(configPath, {
      onProgress: (pct) =>
        updateJob({ progress: 50 + Math.round(pct * 0.4) }), // 50→90%
    });

    // Clean up config
    try { fs.unlinkSync(configPath); } catch { /* ignore */ }
    const cfgIdx = tempFiles.indexOf(configPath);
    if (cfgIdx !== -1) tempFiles.splice(cfgIdx, 1);

    // ── 6. Move output to permanent location ─────────────────────────────
    const permanentOutputPath = path.join(
      jobUploadsDir,
      `final_output_${jobId}.mp4`,
    );
    if (result.outputPath && fs.existsSync(result.outputPath)) {
      fs.copyFileSync(result.outputPath, permanentOutputPath);
      try { fs.unlinkSync(result.outputPath); } catch { /* ignore */ }
    }
    if (!fs.existsSync(permanentOutputPath)) {
      throw new Error('Composed video file not found after processing');
    }

    updateJob({
      status: 'done',
      progress: 100,
      outputPath: permanentOutputPath,
      completedAt: Date.now(),
    });
    console.log(`[Job ${jobId}] ✅ Done: ${permanentOutputPath}`);
  } catch (err) {
    console.error(`[Job ${jobId}] ❌ Failed:`, err.message);
    updateJob({ status: 'failed', error: err.message, completedAt: Date.now() });
  } finally {
    // Clean up any remaining temp files
    for (const f of tempFiles) {
      try { if (fs.existsSync(f)) fs.unlinkSync(f); } catch { /* ignore */ }
    }
  }
}

router.post(
  '/',
  authenticateToken,
  requireProjectOwnership,
  (req, res, next) => {
    console.log('Video processing request received');
    upload.any()(req, res, (err) => {
      if (err instanceof multer.MulterError) {
        console.error('Upload error:', err);
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({ error: 'File too large', details: 'Maximum file size is 1GB per file' });
        }
        return res.status(400).json({ error: `Upload error: ${err.message}` });
      } else if (err) {
        return res.status(500).json({ error: `Server error: ${err.message}` });
      }
      next();
    });
  },
  (req, res) => {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files were uploaded' });
    }
    const hasMidi = req.files.some((f) => f.fieldname === 'midiData');
    if (!hasMidi) {
      return res.status(400).json({ error: 'MIDI data not found in upload' });
    }
    const hasVideo = req.files.some((f) => f.fieldname === 'videos');
    if (!hasVideo) {
      return res.status(400).json({ error: 'No video files found in upload' });
    }

    const isPreview = req.body.preview === 'true' || req.body.preview === true;
    const jobId = uuidv4();
    const jobUploadsDir = req.project.uploadsDir;

    jobs.set(jobId, {
      status: 'queued',
      progress: 0,
      outputPath: null,
      error: null,
      createdAt: Date.now(),
      completedAt: null,
      userId: req.user.id,
      projectId: req.project.id,
    });

    res.status(202).json({ jobId });

    void runCompositionJob(jobId, req.files, isPreview, jobUploadsDir).catch((err) => {
      console.error(`[Job ${jobId}] Unhandled error:`, err);
      const job = jobs.get(jobId);
      if (job && job.status !== 'failed') {
        jobs.set(jobId, { ...job, status: 'failed', error: err.message, completedAt: Date.now() });
      }
    });
  },
);

// GET /status/:jobId — poll for job progress (auth required)
router.get('/status/:jobId', authenticateToken, (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  if (job.userId && job.userId !== req.user.id) {
    return res.status(403).json({ error: 'Access denied' });
  }
  res.json({ status: job.status, progress: job.progress, error: job.error || null });
});

// GET /result/:jobId — download the finished video (auth required)
router.get('/result/:jobId', authenticateToken, (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ error: 'Job not found' });
  if (job.userId && job.userId !== req.user.id) {
    return res.status(403).json({ error: 'Access denied' });
  }
  if (job.status === 'failed') return res.status(422).json({ error: job.error || 'Composition failed' });
  if (job.status !== 'done') return res.status(409).json({ status: job.status, message: 'Job not finished yet' });
  if (!job.outputPath || !fs.existsSync(job.outputPath)) {
    return res.status(410).json({ error: 'Output file is no longer available' });
  }

  const stats = fs.statSync(job.outputPath);
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Length', stats.size);
  res.setHeader('Content-Disposition', 'attachment; filename="composition.mp4"');

  const readStream = fs.createReadStream(job.outputPath);
  readStream.on('error', (streamErr) => {
    console.error('Stream error:', streamErr);
    if (!res.headersSent) res.status(500).json({ error: 'Failed to stream video' });
  });
  // Clean up only after the response is fully sent (res 'finish')
  res.on('finish', () => {
    try { fs.unlinkSync(job.outputPath); } catch { /* ignore */ }
    jobs.delete(req.params.jobId);
  });
  readStream.pipe(res);
});

export default router;
