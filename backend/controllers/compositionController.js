/* eslint-disable no-unused-vars */
import ffmpeg from 'fluent-ffmpeg';
import pkg from '@tonejs/midi';
import { Buffer } from 'buffer';
const { Midi } = pkg;
import { existsSync, mkdirSync, writeFileSync, rmSync, renameSync } from 'fs';
import { join, dirname } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { rm } from 'fs/promises';
import process from 'process';
import {
  isDrumTrack,
  DRUM_NOTE_MAP,
  getNoteGroup,
} from '../utils/drumUtils.js';
import { createReadStream, statSync } from 'fs';

// Import performance optimization services
import cacheService from '../services/cacheService.js';
import {
  addVideoCompositionJob,
  getJobStatus,
} from '../services/queueService.js';
import videoProcessor from '../utils/videoProcessor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMP_DIR = join(__dirname, '../temp');
const UPLOADS_DIR = join(__dirname, '../uploads');
const GPU_MEMORY_LIMIT = '2GB';

// Enhanced health monitoring integration
const healthMonitorPath = join(__dirname, '../utils/health_monitor.py');
let healthMonitor = null;
let healthMonitoringSession = null;

// Initialize health monitor if available
async function initializeHealthMonitor() {
  try {
    if (existsSync(healthMonitorPath)) {
      console.log('Initializing health monitoring system...');
      return true;
    } else {
      console.warn('Health monitor script not found at:', healthMonitorPath);
      return false;
    }
  } catch (error) {
    console.warn('Health monitor not available:', error.message);
    return false;
  }
}

// Start health monitoring session for a specific processing session
async function startHealthMonitoringSession(sessionId, totalVideos = 0) {
  try {
    if (!(await initializeHealthMonitor())) {
      return null;
    }

    healthMonitor = spawn(
      'python',
      [
        healthMonitorPath,
        '--session-id',
        sessionId,
        '--duration',
        '600', // 10 minutes max
        '--interval',
        '3',
      ],
      {
        stdio: ['pipe', 'pipe', 'pipe'],
      }
    );

    healthMonitoringSession = sessionId;

    healthMonitor.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output && !output.includes('INFO')) {
        console.log(`Health Monitor [${sessionId}]: ${output}`);
      }
    });

    healthMonitor.stderr.on('data', (data) => {
      const error = data.toString().trim();
      if (error && !error.includes('WARNING')) {
        console.warn(`Health Monitor Warning [${sessionId}]: ${error}`);
      }
    });

    healthMonitor.on('close', (code) => {
      if (code !== 0) {
        console.warn(`Health monitor process exited with code ${code}`);
      }
      healthMonitoringSession = null;
    });

    console.log(`Health monitoring started for session: ${sessionId}`);
    return healthMonitor;
  } catch (error) {
    console.warn('Failed to start health monitoring:', error.message);
    return null;
  }
}

// Stop health monitoring session
async function stopHealthMonitoringSession() {
  if (healthMonitor && healthMonitoringSession) {
    try {
      console.log(
        `Stopping health monitoring for session: ${healthMonitoringSession}`
      );
      healthMonitor.kill('SIGTERM');

      // Give it time to cleanup
      setTimeout(() => {
        if (healthMonitor && !healthMonitor.killed) {
          healthMonitor.kill('SIGKILL');
        }
      }, 5000);

      healthMonitor = null;
      healthMonitoringSession = null;
    } catch (error) {
      console.warn('Error stopping health monitor:', error.message);
    }
  }
}

// Ensure directories exist with proper error handling
// function ensureDirectoryExists(dir) {
//   try {
//     if (!existsSync(dir)) {
//       mkdirSync(dir, { recursive: true });
//     }
//     return true;
//   } catch (err) {
//     console.error(`Failed to create directory ${dir}:`, err);
//     return false;
//   }
// }

// function ticksToSeconds(ticks, midi) {
//   const bpm = midi.header.tempos[0]?.bpm || 120;
//   const ppq = midi.header.ppq;
//   return (ticks / ppq) * (60 / bpm);
// }

// Add this helper function at the top
const normalizeInstrumentName = (name) => {
  console.log('Normalizing instrument name:', name);
  return name.toLowerCase().replace(/\s+/g, '_');
};

const processDrumTrack = (track, index) => {
  console.log(`\nProcessing drum track ${index}:`, track.instrument.name);
  const drumNotes = {};
  track.notes.forEach((note) => {
    console.log(`\nProcessing drum note:`, note);
    const drumName = DRUM_NOTE_MAP[note.midi];
    console.log('Drum name:', drumName);
    if (drumName) {
      const key = drumName.toLowerCase().replace(/\s+/g, '_');
      if (!drumNotes[key]) {
        drumNotes[key] = [];
      }
      drumNotes[key].push(note);
    }
    console.log('Drum notes:', drumNotes);
  });
  return drumNotes;
};

// Enhanced processVideoWithPython function with performance monitoring
async function processVideoWithPython(
  midiData,
  processedFiles,
  outputPath,
  onProgress = null
) {
  console.log('Processing video with Python:', {
    outputPath,
    midiDataKeys: Object.keys(midiData),
    processedFilesCount: Object.keys(processedFiles).length,
  });

  return new Promise((resolve, reject) => {
    try {
      const startTime = Date.now();
      const videoFilesForPython = {};

      // Process files with memory monitoring
      Object.entries(processedFiles).forEach(([key, value]) => {
        console.log(`Processing video file: ${key}`);

        if (value.path && existsSync(value.path)) {
          videoFilesForPython[key] = {
            path: value.path.replace(/\\/g, '/'),
            isDrum: value.isDrum || false,
            drumName: value.drumName,
            notes: value.notes.map((note) => ({
              time: note.time,
              duration: note.duration,
              velocity: note.velocity || 1.0,
              midi: note.midi,
            })),
            layout: value.layout || {
              x: 0,
              y: 0,
              width: 960,
              height: 720,
            },
          };
        } else if (value.video) {
          // Handle video buffer data
          videoFilesForPython[key] = {
            videoData: value.video,
            isDrum: value.isDrum || false,
            drumName: value.drumName,
            notes: value.notes.map((note) => ({
              time: note.time,
              duration: note.duration,
              velocity: note.velocity || 1.0,
              midi: note.midi,
            })),
            layout: value.layout || {
              x: 0,
              y: 0,
              width: 960,
              height: 720,
            },
          };
        }
      });

      // Enhanced JSON file preparation
      const tempDir = dirname(outputPath);
      if (!existsSync(tempDir)) {
        mkdirSync(tempDir, { recursive: true });
      }

      const midiJsonPath = join(tempDir, 'midi_data.json');
      const videoFilesJsonPath = join(tempDir, 'video_files.json');

      console.log('Enhanced MIDI data processing:', {
        trackCount: Object.keys(midiData.tracks || {}).length,
        hasGridArrangement: !!midiData.gridArrangement,
        duration: midiData.duration,
      });

      // Enhanced data structure for Python processing
      const enhancedMidiData = {
        ...midiData,
        gridArrangement:
          midiData.gridArrangement ||
          calculateOptimalGridLayout(Object.keys(videoFilesForPython)),
        processingMetadata: {
          totalTracks: Object.keys(videoFilesForPython).length,
          timestamp: Date.now(),
          version: '2.0',
        },
      };

      writeFileSync(midiJsonPath, JSON.stringify(enhancedMidiData, null, 2));
      writeFileSync(
        videoFilesJsonPath,
        JSON.stringify(videoFilesForPython, null, 2)
      );

      // Enhanced Python process with performance monitoring
      const pythonArgs = [
        join(__dirname, '../utils/video_processor.py'),
        midiJsonPath,
        videoFilesJsonPath,
        outputPath,
        '--performance-mode', // New flag for optimized processing
        '--memory-limit',
        '4GB',
        '--parallel-tracks',
        Math.min(4, Object.keys(videoFilesForPython).length).toString(),
      ];

      console.log('Starting enhanced Python process:', pythonArgs);
      const pythonProcess = spawn('python', pythonArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONPATH: join(__dirname, '../utils'),
          OPENCV_LOG_LEVEL: 'ERROR', // Reduce OpenCV logging
          TF_CPP_MIN_LOG_LEVEL: '2', // Reduce TensorFlow logging if used
        },
      });

      let stdoutBuffer = '';
      let stderrBuffer = '';
      let lastProgressUpdate = 0;

      pythonProcess.stdout.on('data', (data) => {
        stdoutBuffer += data.toString();
        const lines = stdoutBuffer.split('\n');
        stdoutBuffer = lines.pop(); // Keep incomplete line

        lines.forEach((line) => {
          if (line.trim()) {
            console.log(`Python stdout: ${line}`);

            // Parse progress updates
            if (line.includes('PROGRESS:') && onProgress) {
              try {
                const progressMatch = line.match(/PROGRESS:(\d+)/);
                if (progressMatch) {
                  const progress = parseInt(progressMatch[1]);
                  if (progress !== lastProgressUpdate) {
                    onProgress({
                      progress,
                      message: `Processing video composition: ${progress}%`,
                      timestamp: Date.now(),
                    });
                    lastProgressUpdate = progress;
                  }
                }
              } catch (error) {
                console.error('Error parsing progress:', error);
              }
            }
          }
        });
      });

      pythonProcess.stderr.on('data', (data) => {
        stderrBuffer += data.toString();
        const lines = stderrBuffer.split('\n');
        stderrBuffer = lines.pop();

        lines.forEach((line) => {
          if (line.trim() && !line.includes('WARNING')) {
            console.error(`Python stderr: ${line}`);
          }
        });
      });

      // Enhanced timeout handling
      const timeout = setTimeout(() => {
        console.error('Python process timeout - killing process');
        pythonProcess.kill('SIGKILL');
        reject(new Error('Video processing timeout after 5 minutes'));
      }, 5 * 60 * 1000); // 5 minute timeout

      pythonProcess.on('close', async (code) => {
        clearTimeout(timeout);
        const processingTime = Date.now() - startTime;
        console.log(
          `Python process completed in ${processingTime}ms with code ${code}`
        );

        if (code === 0) {
          // Verify output with enhanced validation
          try {
            if (!existsSync(outputPath)) {
              throw new Error('Output file was not created');
            }

            const stats = statSync(outputPath);
            if (stats.size === 0) {
              throw new Error('Output file is empty');
            }

            // Enhanced audio/video validation
            await validateOutputVideo(outputPath);

            console.log(
              `✅ Video composition successful: ${stats.size} bytes in ${processingTime}ms`
            );

            // Clean up temp files
            try {
              rmSync(midiJsonPath, { force: true });
              rmSync(videoFilesJsonPath, { force: true });
            } catch (cleanupError) {
              console.warn(
                'Warning: Could not clean up temp files:',
                cleanupError.message
              );
            }

            resolve({
              outputPath,
              fileSize: stats.size,
              processingTime,
              performance: {
                processingTimeMs: processingTime,
                outputSizeBytes: stats.size,
                tracksProcessed: Object.keys(videoFilesForPython).length,
              },
            });
          } catch (validationError) {
            console.error('Output validation failed:', validationError);
            reject(validationError);
          }
        } else {
          const errorMessage = `Python process failed with code ${code}`;
          console.error(errorMessage);
          console.error('Final stderr buffer:', stderrBuffer);
          reject(new Error(`${errorMessage}\nDetails: ${stderrBuffer}`));
        }
      });

      pythonProcess.on('error', (error) => {
        clearTimeout(timeout);
        console.error('Python process error:', error);
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    } catch (error) {
      console.error('processVideoWithPython setup error:', error);
      reject(error);
    }
  });
}

// Helper function to calculate optimal grid layout
function calculateOptimalGridLayout(trackKeys) {
  const trackCount = trackKeys.length;

  if (trackCount === 1) {
    return { layout: 'single', segments: 1 };
  } else if (trackCount === 2) {
    return { layout: 'split', segments: 2 };
  } else if (trackCount <= 4) {
    return { layout: 'quad', segments: 4 };
  } else {
    const cols = Math.ceil(Math.sqrt(trackCount));
    const rows = Math.ceil(trackCount / cols);
    return {
      layout: 'grid',
      segments: trackCount,
      cols,
      rows,
      dimensions: { width: 960 / cols, height: 720 / rows },
    };
  }
}

// Enhanced video validation function
async function validateOutputVideo(outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(outputPath, (err, metadata) => {
      if (err) {
        reject(new Error(`Video validation failed: ${err.message}`));
        return;
      }

      try {
        const videoStream = metadata.streams.find(
          (s) => s.codec_type === 'video'
        );
        const audioStream = metadata.streams.find(
          (s) => s.codec_type === 'audio'
        );

        if (!videoStream) {
          throw new Error('Output video has no video stream');
        }

        if (!audioStream) {
          console.warn('⚠️ Output video has no audio stream');
        }

        // Check video properties
        if (videoStream.width <= 0 || videoStream.height <= 0) {
          throw new Error('Invalid video dimensions');
        }

        if (metadata.format.duration <= 0) {
          throw new Error('Invalid video duration');
        }

        console.log('✅ Video validation passed:', {
          duration: metadata.format.duration,
          videoCodec: videoStream.codec_name,
          audioCodec: audioStream?.codec_name || 'none',
          dimensions: `${videoStream.width}x${videoStream.height}`,
          fileSize: metadata.format.size,
        });

        resolve(metadata);
      } catch (validationError) {
        reject(validationError);
      }
    });
  });
}

// Add this helper function at the top
// function debugMidiTrack(track, index) {
//   return {
//     index,
//     name: track.instrument.name,
//     channel: track.channel,
//     notes: track.notes.length,
//     isDrum: isDrumTrack(track),
//     firstNote: track.notes[0]
//       ? {
//           midi: track.notes[0].midi,
//           channel: track.notes[0].channel,
//         }
//       : null,
//   };
// }

// Add this helper function for layout calculation
// function calculateLayoutConfig(activeTrackCount) {
//   if (activeTrackCount === 1) {
//     return {
//       segments: [
//         {
//           x: 0,
//           y: 0,
//           width: 960,
//           height: 720,
//           scale: 'scale=960:720',
//         },
//       ],
//     };
//   } else if (activeTrackCount === 2) {
//     return {
//       segments: [
//         { x: 0, y: 0, width: 480, height: 720, scale: 'scale=480:720' },
//         { x: 480, y: 0, width: 480, height: 720, scale: 'scale=480:720' },
//       ],
//     };
//   } else {
//     const cols = Math.ceil(Math.sqrt(activeTrackCount));
//     const rows = Math.ceil(activeTrackCount / cols);
//     const segmentWidth = 960 / cols;
//     const segmentHeight = 720 / rows;

//     return {
//       segments: Array.from({ length: activeTrackCount }, (_, index) => ({
//         x: (index % cols) * segmentWidth,
//         y: Math.floor(index / cols) * segmentHeight,
//         width: segmentWidth,
//         height: segmentHeight,
//         scale: `scale=${segmentWidth}:${segmentHeight}`,
//       })),
//     };
//   }
// }

export const composeVideo = async (req, res) => {
  const startTime = Date.now();
  const sessionId = uuidv4();

  // Performance tracking
  const performanceMetrics = {
    startTime,
    sessionId,
    steps: {},
  };

  try {
    console.log('Processing composition request', {
      sessionId,
      filesCount: req.files?.length,
      bodyKeys: Object.keys(req.body || {}),
    });

    // Step 1: Check cache for existing composition
    performanceMetrics.steps.cacheCheck = Date.now();
    const { midi, videoFiles } = req.body;
    const compositionKey = `composition:${Buffer.from(midi)
      .toString('base64')
      .substring(0, 32)}`;

    const cachedResult = await cacheService.get(compositionKey);
    if (cachedResult) {
      console.log('Returning cached composition result');
      const cachedPath = cachedResult.outputPath;
      if (existsSync(cachedPath)) {
        return streamVideoFile(res, cachedPath, () => {
          // Cleanup function for cached result
          setTimeout(
            () =>
              cleanupTempDirectory(dirname(cachedPath)).catch(console.error),
            1000
          );
        });
      }
    }

    // Step 2: Parse and validate MIDI data
    performanceMetrics.steps.midiParsing = Date.now();
    const midiData = new Midi(Buffer.from(midi));

    console.log('MIDI analysis:', {
      format: midiData.header.format,
      duration: midiData.duration,
      trackCount: midiData.tracks.length,
      tracks: midiData.tracks.map((t, i) => ({
        index: i,
        name: t.instrument?.name,
        channel: t.channel,
        noteCount: t.notes?.length,
      })),
    });

    // Step 3: Process tracks in parallel
    performanceMetrics.steps.trackProcessing = Date.now();
    const { processedTracks, processedDrums } = await processTracksInParallel(
      midiData,
      videoFiles,
      sessionId,
      performanceMetrics
    );

    // Step 4: Create job for background video composition
    performanceMetrics.steps.jobCreation = Date.now();
    const jobData = {
      midiData: midiData.toJSON(),
      processedTracks,
      processedDrums,
      sessionId,
      outputPath: join(TEMP_DIR, `output_${sessionId}.mp4`),
      performanceMetrics,
    };

    // Add job to queue for background processing
    const job = await addVideoCompositionJob(jobData);
    console.log(`Created background job ${job.id} for session ${sessionId}`);

    // Return job ID for client to track progress
    res.json({
      jobId: job.id,
      sessionId,
      message: 'Video composition started',
      estimatedTime: estimateProcessingTime(
        midiData,
        Object.keys(processedTracks).length + Object.keys(processedDrums).length
      ),
    });
  } catch (error) {
    console.error('Composition error:', error);

    // Log performance metrics even on error
    performanceMetrics.error = error.message;
    performanceMetrics.totalTime = Date.now() - startTime;
    await cacheService.logPerformanceMetrics('composition', performanceMetrics);

    if (!res.headersSent) {
      res.status(500).json({
        error: error.message,
        sessionId,
        troubleshooting: getTroubleshootingInfo(error),
      });
    }

    // Cleanup on error
    setTimeout(() => {
      cleanupTempDirectory(join(TEMP_DIR, sessionId)).catch(console.error);
    }, 1000);
  }
};

// New optimized function to process tracks in parallel
async function processTracksInParallel(
  midiData,
  videoFiles,
  sessionId,
  performanceMetrics
) {
  const processedTracks = {};
  const processedDrums = {};

  // Create processing promises for each track
  const trackPromises = midiData.tracks.map(async (track, index) => {
    if (isDrumTrack(track)) {
      console.log(`Processing drum track ${index}:`, track.instrument.name);
      const drumGroups = processDrumTrack(track, index);

      // Process drum groups in parallel
      const drumPromises = Object.entries(drumGroups).map(
        async ([drumName, notes]) => {
          const videoKey = `drum_${drumName}`;
          if (videoFiles[videoKey]) {
            // Check cache first
            const cacheKey = `drum:${videoKey}:${sessionId}`;
            let processedVideo = await cacheService.get(cacheKey);

            if (!processedVideo) {
              processedVideo = {
                notes,
                video: videoFiles[videoKey],
                index,
                processedAt: Date.now(),
              };

              // Cache for 1 hour
              await cacheService.set(cacheKey, processedVideo, 3600);
            }

            processedDrums[videoKey] = processedVideo;
          }
        }
      );

      await Promise.all(drumPromises);
    } else {
      const trackKey = normalizeInstrumentName(track.instrument.name);
      console.log(`Processing melodic track ${index}:`, track.instrument.name);

      if (videoFiles[trackKey]) {
        // Check cache first
        const cacheKey = `track:${trackKey}:${sessionId}`;
        let processedTrack = await cacheService.get(cacheKey);

        if (!processedTrack) {
          processedTrack = {
            notes: track.notes,
            video: videoFiles[trackKey],
            index,
            processedAt: Date.now(),
          };

          // Cache for 1 hour
          await cacheService.set(cacheKey, processedTrack, 3600);
        }

        processedTracks[trackKey] = processedTrack;
      }
    }
  });

  // Wait for all tracks to be processed
  await Promise.all(trackPromises);
  performanceMetrics.steps.trackProcessingComplete = Date.now();

  return { processedTracks, processedDrums };
}

// New function to handle video file streaming
function streamVideoFile(res, outputPath, cleanup) {
  try {
    const stat = statSync(outputPath);
    const fileSize = stat.size;
    const head = {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
      'Content-Disposition': 'attachment; filename="composed-video.mp4"',
      'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
      ETag: `"${stat.mtime.getTime()}-${fileSize}"`,
    };

    res.writeHead(200, head);
    const readStream = createReadStream(outputPath);

    readStream.on('error', (error) => {
      console.error('Stream error:', error);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to stream video' });
      }
      cleanup();
    });

    readStream.on('end', () => {
      console.log('Stream completed');
      cleanup();
    });

    readStream.pipe(res);
  } catch (error) {
    console.error('Error streaming video file:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to stream video' });
    }
    cleanup();
  }
}

// New function to estimate processing time
function estimateProcessingTime(midiData, trackCount) {
  const baseDuration = midiData.duration; // in seconds
  const complexityFactor = trackCount * 0.5; // Each track adds 0.5 seconds processing time
  const baseProcessingTime = 10; // Base 10 seconds

  return Math.ceil(baseProcessingTime + baseDuration * 0.1 + complexityFactor);
}

// New function to provide troubleshooting information
function getTroubleshootingInfo(error) {
  const errorMessage = error.message.toLowerCase();

  if (errorMessage.includes('midi')) {
    return {
      category: 'MIDI Processing',
      suggestions: [
        'Ensure MIDI file is valid and properly formatted',
        'Check that MIDI tracks have valid instrument names',
        'Verify MIDI file size is within limits',
      ],
    };
  } else if (errorMessage.includes('video')) {
    return {
      category: 'Video Processing',
      suggestions: [
        'Ensure video files are in supported formats (MP4, AVI, MOV)',
        'Check video file sizes are within limits',
        'Verify sufficient disk space for processing',
      ],
    };
  } else if (errorMessage.includes('memory') || errorMessage.includes('heap')) {
    return {
      category: 'Memory',
      suggestions: [
        'Try reducing video file sizes',
        'Process fewer tracks simultaneously',
        'Restart the application if memory usage is high',
      ],
    };
  }

  return {
    category: 'General',
    suggestions: [
      'Check server logs for detailed error information',
      'Ensure all required dependencies are installed',
      'Verify sufficient system resources are available',
    ],
  };
}

// Modify the composeVideo function to use the Python processor
// export const composeVideo = async (req, res) => {
//   console.log('Received composition request');
//   console.log(
//     'Files:',
//     req.files?.map((f) => ({
//       fieldname: f.fieldname,
//       size: f.size,
//       mimetype: f.mimetype,
//     }))
//   );

//   if (!req.files || req.files.length === 0) {
//     console.error('No files received');
//     return res.status(400).json({ error: 'No files were uploaded' });
//   }

//   const midiFile = req.files.find((file) => file.fieldname === 'midiData');
//   if (!midiFile) {
//     console.error('No MIDI data found');
//     return res.status(400).json({ error: 'MIDI data is missing' });
//   }

//   const videoFiles = req.files.filter((file) =>
//     file.fieldname.startsWith('videos[')
//   );
//   if (videoFiles.length === 0) {
//     console.error('No video files found');
//     return res.status(400).json({ error: 'No video files were uploaded' });
//   }

//   const sessionId = uuidv4();
//   const sessionDir = join(TEMP_DIR, sessionId);

//   if (
//     !ensureDirectoryExists(TEMP_DIR) ||
//     !ensureDirectoryExists(UPLOADS_DIR) ||
//     !ensureDirectoryExists(sessionDir)
//   ) {
//     console.error('Failed to create required directories');
//     return res
//       .status(500)
//       .json({ error: 'Failed to create required directories' });
//   }

//   try {
//     const files = req.files;
//     if (!files || files.length === 0) {
//       throw new Error('No files were uploaded');
//     }

//     console.log(
//       'Files received:',
//       files.map((f) => ({
//         fieldname: f.fieldname,
//         mimetype: f.mimetype,
//         size: f.size,
//       }))
//     );

//     // Process MIDI file
//     const midiFile = files.find((file) => file.fieldname === 'midiData');
//     if (!midiFile) {
//       throw new Error('MIDI data is missing');
//     }

//     // Parse MIDI data
//     let midi;
//     try {
//       if (midiFile.mimetype === 'application/json') {
//         const midiJson = JSON.parse(midiFile.buffer.toString());
//         console.log('Raw MIDI JSON:', {
//           tracks: midiJson.tracks.map((t, i) => ({
//             index: i,
//             name: t.instrument?.name,
//             channel: t.channel,
//             noteCount: t.notes?.length,
//           })),
//         });

//         midi = new Midi();

//         // Add tracks from JSON
//         midiJson.tracks?.forEach((track) => {
//           const newTrack = midi.addTrack();
//           track.notes?.forEach((note) => {
//             newTrack.addNote({
//               midi: note.midi,
//               time: note.time,
//               duration: note.duration,
//               velocity: note.velocity || 0.8,
//               channel: track.channel, // Make sure channel is preserved
//             });
//           });
//           if (track.instrument) {
//             newTrack.instrument.name = track.instrument.name;
//             newTrack.channel = track.channel; // Set channel explicitly
//           }
//         });

//         console.log(
//           'Parsed MIDI tracks detail:',
//           midi.tracks.map((track, i) => debugMidiTrack(track, i))
//         );
//       } else {
//         midi = new Midi(midiFile.buffer);
//       }
//     } catch (error) {
//       console.error('Error parsing MIDI data:', error);
//       throw new Error('Invalid MIDI data format');
//     }

//     // After parsing, add drum track validation
//     const midiDrumTracks = midi.tracks.filter(isDrumTrack);
//     console.log('\nDrum Track Analysis:');
//     console.log('Total drum tracks found:', midiDrumTracks.length);
//     midiDrumTracks.forEach((track, i) => {
//       console.log(`\nDrum Track ${i}:`, {
//         name: track.instrument.name,
//         channel: track.channel,
//         noteCount: track.notes.length,
//         notes: track.notes.slice(0, 3).map((n) => ({
//           midi: n.midi,
//           time: n.time,
//           channel: n.channel,
//         })),
//       });
//     });

//     // Save and process video files
//     const videoFiles = {};
//     const videos = files.filter((file) => file.fieldname.startsWith('videos'));

//     if (videos.length === 0) {
//       throw new Error('No video files were uploaded');
//     }

//     // Create a map of normalized instrument names to their videos
//     videos.forEach((video) => {
//       const instrumentMatch = video.fieldname.match(/\[(.*?)\]/);
//       if (instrumentMatch) {
//         const normalizedName = normalizeInstrumentName(instrumentMatch[1]);
//         videoFiles[normalizedName] = video;
//       }
//     });

//     console.log(
//       'Available video files for instruments:',
//       Object.keys(videoFiles)
//     );
//     console.log(
//       'MIDI tracks:',
//       midi.tracks.map((track) => track.instrument.name)
//     );

//     // Process each track
//     const processedFiles = new Map();
//     console.log('\n=== Starting Track Processing ===');

//     // First, validate drum videos against MIDI tracks
//     const drumTracksNeeded = new Set();
//     midi.tracks.forEach((track, index) => {
//       if (isDrumTrack(track)) {
//         console.log(`\nAnalyzing drum track ${index}:`, track.instrument.name);
//         track.notes.forEach((note) => {
//           const group = getNoteGroup(note.midi);
//           drumTracksNeeded.add(group);
//           console.log(
//             `Required drum group: ${group} for MIDI note: ${note.midi}`
//           );
//         });
//       }
//     });

//     // Check available drum videos
//     const availableDrumVideos = Object.keys(videoFiles).filter((name) =>
//       name.startsWith('drum_')
//     );
//     console.log('\nDrum videos validation:');
//     console.log('Required drum groups:', Array.from(drumTracksNeeded));
//     console.log('Available drum videos:', availableDrumVideos);

//     // Validate drum video coverage
//     drumTracksNeeded.forEach((group) => {
//       const drumVideoKey = `drum_${group.toLowerCase().replace(/\s+/g, '_')}`;
//       if (!videoFiles[drumVideoKey]) {
//         console.log(`⚠️ Missing drum video for group: ${group}`);
//       }
//     });

//     // Filter out tracks with no notes first
//     const activeTracksData = midi.tracks
//       .filter((track) => track.notes && track.notes.length > 0)
//       .map((track, index) => ({
//         track,
//         index,
//         normalizedName: normalizeInstrumentName(track.instrument.name),
//       }));

//     console.log(`Active tracks (with notes): ${activeTracksData.length}`);

//     // Calculate layout based on number of active tracks
//     const layoutConfig = calculateLayoutConfig(activeTracksData.length);
//     console.log('Layout configuration:', layoutConfig);

//     const trackPromises = activeTracksData.map(
//       async ({ track, index, normalizedName }, layoutIndex) => {
//         console.log(`\nProcessing track ${index}:`, {
//           name: track.instrument.name,
//           notes: track.notes.length,
//           channel: track.channel,
//         });

//         // Skip if no notes
//         if (!track.notes || track.notes.length === 0) {
//           console.log(`Skipping empty track ${index}`);
//           return null;
//         }

//         if (isDrumTrack(track)) {
//           // Use imported function
//           console.log(`Processing drum track ${index}:`, track.instrument.name);
//           const drumNotes = processDrumTrack(track, index);

//           // Process each drum group
//           for (const [drumName, notes] of Object.entries(drumNotes)) {
//             const drumVideoKey = `drum_${drumName
//               .toLowerCase()
//               .replace(/\s+/g, '_')}`;
//             const videoFile = videoFiles[drumVideoKey]; // Use the map instead of find

//             if (videoFile) {
//               console.log(
//                 `Processing drum group: ${drumName} with ${notes.length} notes`
//               );
//               const uniqueTrackId = `${drumVideoKey}_track${index}`;
//               const tempVideoPath = join(
//                 sessionDir,
//                 `${uniqueTrackId}_temp.mp4`
//               );
//               const finalVideoPath = join(sessionDir, `${uniqueTrackId}.mp4`);

//               try {
//                 writeFileSync(tempVideoPath, videoFile.buffer);
//                 await convertVideoFormat(tempVideoPath, finalVideoPath);
//                 await rm(tempVideoPath);

//                 processedFiles[drumVideoKey] = {
//                   path: finalVideoPath,
//                   track: index,
//                   type: 'drum',
//                   drumName: drumName,
//                   notes: notes,
//                 };
//               } catch (error) {
//                 console.error(
//                   `Error processing drum video ${drumName}:`,
//                   error
//                 );
//                 throw error;
//               }
//             } else {
//               console.log(
//                 `Missing video file for drum: ${drumName} (key: ${drumVideoKey})`
//               );
//             }
//           }
//         } else {
//           // Handle melodic tracks
//           const normalizedName = normalizeInstrumentName(track.instrument.name);
//           const uniqueTrackId = `${normalizedName}_track${index}`;

//           // Find corresponding video file
//           const videoFile = videoFiles[normalizedName];
//           if (!videoFile) {
//             console.log(
//               `No video found for instrument: ${track.instrument.name} (normalized: ${normalizedName})`
//             );
//             return null;
//           }

//           const tempVideoPath = join(sessionDir, `${uniqueTrackId}_temp.mp4`);
//           const finalVideoPath = join(sessionDir, `${uniqueTrackId}.mp4`);

//           try {
//             writeFileSync(tempVideoPath, videoFile.buffer);
//             await convertVideoFormat(tempVideoPath, finalVideoPath);
//             await rm(tempVideoPath);

//             const notes = track.notes.map((note) => ({
//               midi: note.midi,
//               time: ticksToSeconds(note.ticks, midi),
//               duration: Math.max(ticksToSeconds(note.durationTicks, midi), 0.1),
//               velocity: note.velocity || 0.8,
//             }));

//             processedFiles.set(uniqueTrackId, {
//               path: finalVideoPath,
//               isDrum: false,
//               notes: notes,
//               trackIndex: index,
//               layout: layoutConfig.segments[layoutIndex],
//             });

//             console.log(
//               `Processed ${uniqueTrackId} with ${notes.length} notes`
//             );
//             return uniqueTrackId;
//           } catch (error) {
//             console.error(
//               `Error processing melodic track ${uniqueTrackId}:`,
//               error
//             );
//             return null;
//           }
//         }
//       }
//     );

//     await Promise.all(trackPromises);

//     // Log final processed files before sending to Python
//     console.log('\n=== Final Processed Files ===');
//     console.log('Total processed tracks:', processedFiles.size);
//     processedFiles.forEach((value, key) => {
//       console.log(`\nTrack: ${key}`);
//       console.log(`- Is drum: ${value.isDrum}`);
//       console.log(`- Note count: ${value.notes.length}`);
//       console.log(`- File exists: ${existsSync(value.path)}`);
//       console.log(`- First 3 notes:`, value.notes.slice(0, 3));
//     });

//     const outputPath = join(sessionDir, 'output.mp4');
//     const midiData = midi.toJSON();
//     const tracksDict = {};
//     midiData.tracks.forEach((track, index) => {
//       tracksDict[index] = {
//         ...track,
//         index: index,
//         notes: track.notes || [],
//       };
//     });
//     midiData.duration = midi.duration;

//     // Add verification before Python processing
//     console.log('\n=== Pre-Python Processing Verification ===');
//     const processedDrumTracks = Array.from(processedFiles.entries())
//       .filter(([key, value]) => value.isDrum)
//       .map(([key, value]) => ({
//         key,
//         drumName: value.drumName,
//         noteCount: value.notes.length,
//       }));

//     console.log('Drum tracks being sent to Python:', processedDrumTracks);

//     // Wait for any pending file operations to complete
//     await new Promise((resolve) => setTimeout(resolve, 1000));

//     // Pass the Map directly to processVideoWithPython
//     await processVideoWithPython(
//       {
//         ...midiData,
//         tracks: tracksDict,
//       },
//       processedFiles,
//       outputPath
//     );

//     // Instead of sendFile, use streaming
//     if (existsSync(outputPath)) {
//       const stat = statSync(outputPath);
//       const fileSize = stat.size;
//       const head = {
//         'Content-Length': fileSize,
//         'Content-Type': 'video/mp4',
//         'Content-Disposition': 'attachment; filename="composed-video.mp4"',
//       };

//       res.writeHead(200, head);
//       const readStream = createReadStream(outputPath);

//       // Handle stream events
//       readStream.on('error', (error) => {
//         console.error('Stream error:', error);
//         if (!res.headersSent) {
//           res.status(500).json({ error: 'Failed to stream video' });
//         }
//         cleanup();
//       });

//       readStream.on('end', () => {
//         console.log('Stream completed');
//         cleanup();
//       });

//       // Pipe the file to response
//       readStream.pipe(res);
//     } else {
//       throw new Error('Output video file was not created');
//     }

//     // Cleanup function
//     function cleanup() {
//       setTimeout(() => {
//         cleanupTempDirectory(sessionDir).catch(console.error);
//       }, 1000);
//     }
//   } catch (error) {
//     console.error('Composition error:', error);
//     if (!res.headersSent) {
//       res.status(500).json({ error: error.message });
//     }
//     setTimeout(() => {
//       cleanupTempDirectory(sessionDir).catch(console.error);
//     }, 1000);
//   }
// };

async function cleanupTempDirectory(dirPath) {
  try {
    await rm(dirPath, { recursive: true, force: true });
    console.log(`Temp directory ${dirPath} cleaned up successfully.`);
  } catch (error) {
    console.error(`Error cleaning up temp directory ${dirPath}:`, error);
  }
}

// async function getOptimalEncodingSettings() {
//   try {
//     const result = await new Promise((resolve) => {
//       const ffmpegProcess = spawn('ffmpeg', ['-encoders']);
//       let output = '';
//       ffmpegProcess.stdout.on('data', (data) => (output += data.toString()));
//       ffmpegProcess.on('close', () => resolve(output));
//     });

//     if (result.includes('h264_nvenc')) {
//       return {
//         videoCodec: 'h264_nvenc',
//         preset: 'p2', // Faster preset
//         tune: 'fastdecode',
//         hwaccel: 'cuda',
//         pixelFormat: 'cuda',
//         maxGpuMemory: GPU_MEMORY_LIMIT,
//       };
//     }
//     return { videoCodec: 'libx264' }; // Fallback to CPU
//   } catch (error) {
//     return { videoCodec: 'libx264' };
//   }
// }

// async function convertVideoFormat(inputPath, outputPath) {
//   const maxRetries = 3;

//   for (let attempt = 1; attempt <= maxRetries; attempt++) {
//     try {
//       await new Promise((resolve, reject) => {
//         ffmpeg.ffprobe(inputPath, (err, metadata) => {
//           if (err) return reject(err);

//           const audioStream = metadata.streams.find(
//             (s) => s.codec_type === 'audio'
//           );
//           console.log(`[Attempt ${attempt}] Input audio stream:`, audioStream);

//           const command = ffmpeg(inputPath);

//           // Add specific input handling for different codecs
//           if (audioStream) {
//             switch (audioStream.codec_name) {
//               case 'opus':
//                 command.inputOptions(['-c:a', 'libopus']);
//                 break;
//               case 'vorbis':
//                 command.inputOptions(['-c:a', 'libvorbis']);
//                 break;
//               case 'aac':
//                 command.inputOptions(['-c:a', 'aac']);
//                 break;
//             }
//           }

//           command.outputOptions([
//             // Video settings
//             '-c:v',
//             'libx264',
//             '-preset',
//             'ultrafast',
//             '-pix_fmt',
//             'yuv420p',
//             '-vf',
//             'scale=960:720',
//             '-r',
//             '30',

//             // Audio settings
//             '-c:a',
//             'aac',
//             '-ar',
//             '44100',
//             '-ac',
//             '2',
//             '-b:a',
//             '192k',

//             // Audio filters for proper handling and normalization
//             '-af',
//             [
//               'aresample=44100:first_pts=0',
//               'aformat=sample_fmts=fltp:channel_layouts=stereo',
//               'pan=stereo|c0=c0|c1=c1',
//               'dynaudnorm=f=75:g=25:p=0.95:m=10',
//               'volume=1.5',
//             ].join(','),

//             // General settings
//             '-y',
//             '-movflags',
//             '+faststart',
//             '-map_metadata',
//             '-1',
//             '-map',
//             '0:v:0',
//             '-map',
//             '0:a:0?',

//             // Extended probe size for better format detection
//             '-analyzeduration',
//             '10000000',
//             '-probesize',
//             '10000000',
//           ]);

//           command
//             .on('start', (cmdline) =>
//               console.log(
//                 `[Attempt ${attempt}] Running FFmpeg command:`,
//                 cmdline
//               )
//             )
//             .on('stderr', (stderrLine) =>
//               console.log(`[Attempt ${attempt}] FFmpeg stderr:`, stderrLine)
//             )
//             .on('end', () => resolve())
//             .on('error', (err) => reject(err))
//             .save(outputPath);
//         });
//       });

//       return; // Success - exit retry loop
//     } catch (error) {
//       console.error(`Attempt ${attempt} failed:`, error);
//       if (attempt === maxRetries) {
//         throw new Error(
//           `Failed to convert video after ${maxRetries} attempts: ${error.message}`
//         );
//       }
//       await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
//     }
//   }
// }

// function isGeneralMidiDrumKit(instrumentName, channel) {
//   return (
//     channel === 9 ||
//     (instrumentName || '').toLowerCase().includes('drum') ||
//     (instrumentName || '').toLowerCase().includes('percussion')
//   );
// }

// Initialize controller

// New endpoint to check job status and progress
export const getCompositionStatus = async (req, res) => {
  try {
    const { jobId } = req.params;

    if (!jobId) {
      return res.status(400).json({ error: 'Job ID is required' });
    }

    // Get job status from queue service
    const jobStatus = await getJobStatus(jobId);

    if (!jobStatus) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Check cache for performance metrics
    const metricsKey = `job:metrics:${jobId}`;
    const metrics = await cacheService.get(metricsKey);

    res.json({
      jobId,
      status: jobStatus.status,
      progress: jobStatus.progress || 0,
      message: jobStatus.message || 'Processing...',
      estimatedTimeRemaining: jobStatus.estimatedTimeRemaining,
      metrics: metrics || null,
      ...(jobStatus.status === 'completed' && {
        downloadUrl: `/api/composition/download/${jobId}`,
        outputPath: jobStatus.outputPath,
      }),
      ...(jobStatus.status === 'failed' && {
        error: jobStatus.error,
        troubleshooting: getTroubleshootingInfo(new Error(jobStatus.error)),
      }),
    });
  } catch (error) {
    console.error('Error getting composition status:', error);
    res.status(500).json({
      error: 'Failed to get job status',
      details: error.message,
    });
  }
};

// New endpoint to download completed composition
export const downloadComposition = async (req, res) => {
  try {
    const { jobId } = req.params;

    if (!jobId) {
      return res.status(400).json({ error: 'Job ID is required' });
    }

    // Get job status to verify completion and get output path
    const jobStatus = await getJobStatus(jobId);

    if (!jobStatus) {
      return res.status(404).json({ error: 'Job not found' });
    }

    if (jobStatus.status !== 'completed') {
      return res.status(400).json({
        error: 'Job not completed yet',
        currentStatus: jobStatus.status,
        progress: jobStatus.progress,
      });
    }

    const outputPath = jobStatus.outputPath;

    if (!outputPath || !existsSync(outputPath)) {
      return res.status(404).json({ error: 'Composition file not found' });
    }

    // Stream the video file with optimizations
    streamVideoFile(res, outputPath, () => {
      // Cleanup function - remove file after streaming with delay
      setTimeout(async () => {
        try {
          await cleanupTempDirectory(dirname(outputPath));
          // Also clean up job cache
          await cacheService.del(`job:${jobId}`);
          await cacheService.del(`job:metrics:${jobId}`);
          console.log(`Cleaned up job ${jobId} after download`);
        } catch (error) {
          console.error(`Error cleaning up job ${jobId}:`, error);
        }
      }, 5000); // 5 second delay to ensure download completes
    });
  } catch (error) {
    console.error('Error downloading composition:', error);
    res.status(500).json({
      error: 'Failed to download composition',
      details: error.message,
    });
  }
};

// New endpoint for streaming video processing (for real-time feedback)
export const streamComposition = async (req, res) => {
  const sessionId = uuidv4();

  try {
    console.log('Starting streaming composition:', { sessionId });

    // Set headers for Server-Sent Events
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Cache-Control',
    });

    const { midi, videoFiles } = req.body;

    // Send initial progress
    res.write(
      `data: ${JSON.stringify({
        type: 'progress',
        step: 'parsing',
        progress: 10,
        message: 'Parsing MIDI data...',
      })}\n\n`
    );

    const midiData = new Midi(Buffer.from(midi));

    // Send parsing complete
    res.write(
      `data: ${JSON.stringify({
        type: 'progress',
        step: 'tracks',
        progress: 30,
        message: 'Processing tracks...',
      })}\n\n`
    );

    // Process tracks with progress updates
    const { processedTracks, processedDrums } = await processTracksWithProgress(
      midiData,
      videoFiles,
      sessionId,
      (progress) => {
        res.write(
          `data: ${JSON.stringify({
            type: 'progress',
            step: 'processing',
            progress: 30 + progress * 0.4, // 30-70%
            message: `Processing track ${progress.current} of ${progress.total}...`,
          })}\n\n`
        );
      }
    );

    // Send composition starting
    res.write(
      `data: ${JSON.stringify({
        type: 'progress',
        step: 'composition',
        progress: 70,
        message: 'Starting video composition...',
      })}\n\n`
    );

    // Create and monitor background job
    const jobData = {
      midiData: midiData.toJSON(),
      processedTracks,
      processedDrums,
      sessionId,
      outputPath: join(TEMP_DIR, `output_${sessionId}.mp4`),
    };

    const job = await addVideoCompositionJob(jobData);

    // Monitor job progress
    const progressInterval = setInterval(async () => {
      try {
        const status = await getJobStatus(job.id);
        if (status) {
          res.write(
            `data: ${JSON.stringify({
              type: 'progress',
              step: 'composition',
              progress: 70 + (status.progress || 0) * 0.3, // 70-100%
              message: status.message || 'Composing video...',
              jobId: job.id,
            })}\n\n`
          );

          if (status.status === 'completed') {
            res.write(
              `data: ${JSON.stringify({
                type: 'completed',
                progress: 100,
                message: 'Composition completed!',
                jobId: job.id,
                downloadUrl: `/api/composition/download/${job.id}`,
              })}\n\n`
            );
            clearInterval(progressInterval);
            res.end();
          } else if (status.status === 'failed') {
            res.write(
              `data: ${JSON.stringify({
                type: 'error',
                error: status.error || 'Composition failed',
                troubleshooting: getTroubleshootingInfo(
                  new Error(status.error)
                ),
              })}\n\n`
            );
            clearInterval(progressInterval);
            res.end();
          }
        }
      } catch (error) {
        console.error('Error monitoring job progress:', error);
        clearInterval(progressInterval);
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            error: 'Failed to monitor progress',
          })}\n\n`
        );
        res.end();
      }
    }, 2000); // Check every 2 seconds

    // Cleanup on client disconnect
    req.on('close', () => {
      clearInterval(progressInterval);
      console.log('Client disconnected, cleaning up streaming composition');
    });
  } catch (error) {
    console.error('Streaming composition error:', error);
    res.write(
      `data: ${JSON.stringify({
        type: 'error',
        error: error.message,
        troubleshooting: getTroubleshootingInfo(error),
      })}\n\n`
    );
    res.end();
  }
};

// Helper function to process tracks with progress callbacks
async function processTracksWithProgress(
  midiData,
  videoFiles,
  sessionId,
  onProgress
) {
  const processedTracks = {};
  const processedDrums = {};
  const totalTracks = midiData.tracks.length;
  let completedTracks = 0;

  const trackPromises = midiData.tracks.map(async (track, index) => {
    try {
      if (isDrumTrack(track)) {
        const drumGroups = processDrumTrack(track, index);
        const drumPromises = Object.entries(drumGroups).map(
          async ([drumName, notes]) => {
            const videoKey = `drum_${drumName}`;
            if (videoFiles[videoKey]) {
              const cacheKey = `drum:${videoKey}:${sessionId}`;
              let processedVideo = await cacheService.get(cacheKey);

              if (!processedVideo) {
                processedVideo = {
                  notes,
                  video: videoFiles[videoKey],
                  index,
                  processedAt: Date.now(),
                };
                await cacheService.set(cacheKey, processedVideo, 3600);
              }

              processedDrums[videoKey] = processedVideo;
            }
          }
        );

        await Promise.all(drumPromises);
      } else {
        const trackKey = normalizeInstrumentName(track.instrument.name);
        if (videoFiles[trackKey]) {
          const cacheKey = `track:${trackKey}:${sessionId}`;
          let processedTrack = await cacheService.get(cacheKey);

          if (!processedTrack) {
            processedTrack = {
              notes: track.notes,
              video: videoFiles[trackKey],
              index,
              processedAt: Date.now(),
            };
            await cacheService.set(cacheKey, processedTrack, 3600);
          }

          processedTracks[trackKey] = processedTrack;
        }
      }

      completedTracks++;
      onProgress({
        current: completedTracks,
        total: totalTracks,
        percentage: (completedTracks / totalTracks) * 100,
      });
    } catch (error) {
      console.error(`Error processing track ${index}:`, error);
      completedTracks++;
      onProgress({
        current: completedTracks,
        total: totalTracks,
        percentage: (completedTracks / totalTracks) * 100,
        error: `Failed to process track ${index}: ${error.message}`,
      });
    }
  });

  await Promise.all(trackPromises);
  return { processedTracks, processedDrums };
}
