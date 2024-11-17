/* eslint-disable no-unused-vars */
import ffmpeg from 'fluent-ffmpeg';
import pkg from '@tonejs/midi';
const { Midi } = pkg;
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  rmSync,
  renameSync,
} from 'fs';
import { join } from 'path';
import os from 'os';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { spawn } from 'child_process';
import { rm } from 'fs/promises';
import { isDrumTrack, DRUM_NOTE_MAP, getNoteGroup } from '../utils/drumUtils.js';
import { createReadStream, statSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMP_DIR = join(__dirname, '../temp');
const UPLOADS_DIR = join(__dirname, '../uploads');

const logging = console;

function addLavfiSupport(command) {
  const originalAvailableFormats = command.availableFormats;
  command.availableFormats = (cb) => {
    originalAvailableFormats.call(command, (err, formats) => {
      if (!err && formats) {
        formats.lavfi = {
          canDemux: true,
          canMux: true,
          description: 'Lavfi',
        };
      }
      cb(err, formats);
    });
  };
}

async function createBlackPNG(outputPath, duration) {
  console.log('Creating black PNG with duration:', duration);
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input('color=c=black:s=960x720:r=30')
      .inputOptions(['-f', 'lavfi'])
      .outputOptions([
        '-c:v',
        'libx264',
        '-t',
        String(duration || 1), // Ensure duration is passed and used
        '-pix_fmt',
        'yuv420p',
        '-y',
      ])
      .on('start', (commandLine) => {
        console.log('FFmpeg command (black frame):', commandLine);
      })
      .save(outputPath)
      .on('end', () => {
        console.log(
          'Black frame created successfully with duration:',
          duration
        );
        resolve();
      })
      .on('error', (err) => {
        console.error('Error creating black frame:', err.message);
        reject(err);
      });
  });
}

// Ensure directories exist with proper error handling
function ensureDirectoryExists(dir) {
  try {
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    return true;
  } catch (err) {
    console.error(`Failed to create directory ${dir}:`, err);
    return false;
  }
}

function ticksToSeconds(ticks, midi) {
  const bpm = midi.header.tempos[0]?.bpm || 120;
  const ppq = midi.header.ppq;
  return (ticks / ppq) * (60 / bpm);
}

// Helper function to convert MIDI note number to frequency ratio
function midiNoteToFrequencyRatio(targetMidiNote, sourceMidiNote = 60) {
  // Calculate frequency ratio based on semitone difference
  // Each semitone is a factor of 2^(1/12)
  const semitoneDifference = targetMidiNote - sourceMidiNote;
  return Math.pow(2, semitoneDifference / 12);
}

// Helper function to convert MIDI note number to note name
function midiNoteToName(midiNote) {
  const notes = [
    'C',
    'C#',
    'D',
    'D#',
    'E',
    'F',
    'F#',
    'G',
    'G#',
    'A',
    'A#',
    'B',
  ];
  const octave = Math.floor(midiNote / 12) - 1;
  const noteName = notes[midiNote % 12];
  return `${noteName}${octave}`;
}

async function processNoteSegment(videoPath, note, outputPath, baseNote = 60, isDrum = false) {
  return new Promise((resolve, reject) => {
    if (!existsSync(videoPath)) {
      return reject(new Error(`Input video file not found: ${videoPath}`));
    }

    const outputDir = dirname(outputPath);
    ensureDirectoryExists(outputDir);

    const command = ffmpeg(videoPath).setStartTime(0).duration(note.duration);

    // Add video filter chain
    command.videoFilters(['scale=320:240']);

    if (!isDrum) {
      // Only apply pitch shifting for non-drum tracks
      const pitchRatio = midiNoteToFrequencyRatio(note.midi, baseNote);
      command.audioFilters([
        'aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo',
        `rubberband=pitch=${pitchRatio}:tempo=1`,
        `volume=${note.velocity}`,
      ]);
    } else {
      // For drums, just apply volume adjustment
      command.audioFilters([
        'aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo',
        `volume=${note.velocity}`,
      ]);
    }

    // Set output options
    command.outputOptions([
      // Video codec settings
      '-c:v',
      'libx264',
      '-preset',
      'ultrafast',
      '-pix_fmt',
      'yuv420p',

      // Audio codec settings
      '-c:a',
      'aac',
      '-b:a',
      '192k',
      '-ar',
      '44100',
      '-ac',
      '2',

      // Force video duration to match note duration
      '-t',
      String(note.duration),

      // Overwrite output file if it exists
      '-y',
      '-filter:a:1',
      'volume=2.0', // Increase audio volume
    ]);

    // Add event handlers
    command
      .on('start', (commandLine) => {
        console.log('FFmpeg command:', commandLine);
      })
      .on('stderr', (stderrLine) => {
        console.log('FFmpeg stderr:', stderrLine);
      })
      .on('error', (err) => {
        console.error(`Error processing note segment:`, err);
        reject(err);
      })
      .on('end', () => {
        if (existsSync(outputPath)) {
          console.log(`Successfully processed note segment: ${outputPath}`);
          console.log(
            `Note details: ${midiNoteToName(note.midi)}, duration: ${
              note.duration
            }s, velocity: ${note.velocity}`
          );
          resolve();
        } else {
          reject(new Error(`Failed to create output file: ${outputPath}`));
        }
      });

    // Run the command
    command.save(outputPath);
  });
}

async function composeFinalVideo(tracks, outputPath, midiDuration) {
  if (!tracks || tracks.length === 0) {
    throw new Error('No tracks provided for final composition');
  }

  const outputDir = dirname(outputPath);
  const sanitizedOutputPath = outputPath.replace(/\\/g, '/');

  console.log('Output directory:', outputDir);
  console.log('Sanitized output path:', sanitizedOutputPath);
  console.log('MIDI duration:', midiDuration);

  // Temporary file paths
  const tempFiles = {
    blackFrame: join(outputDir, 'black_frame.mp4'),
    videoOverlay: join(outputDir, 'video_with_overlays_temp.mp4'),
    mergedAudio: join(outputDir, 'merged_audio_temp.mp4'),
  };

  // Ensure output directory exists and is writable
  try {
    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true });
    }
  } catch (error) {
    throw new Error(`Cannot access output directory: ${error.message}`);
  }

  // Sanitize all paths
  Object.keys(tempFiles).forEach((key) => {
    tempFiles[key] = tempFiles[key].replace(/\\/g, '/');
    console.log(`${key} path:`, tempFiles[key]);
  });

  const cleanup = (tempFiles) => {
    Object.values(tempFiles).forEach((file) => {
      if (existsSync(file)) {
        try {
          rmSync(file, { force: true });
          console.log(`Cleaned up temp file: ${file}`);
        } catch (error) {
          console.warn(`Failed to cleanup temp file ${file}:`, error.message);
        }
      }
    });
  };

  try {
    // Create black frame base
    await createBlackFrame(tempFiles.blackFrame, midiDuration);

    // Validate input tracks and segments
    const validSegments = tracks.reduce((acc, track, trackIndex) => {
      const trackSegments =
        track.segments?.filter((segment) => {
          const exists = existsSync(segment.path);
          if (!exists) {
            console.log(`Skipping non-existent segment: ${segment.path}`);
          }
          return exists;
        }) || [];

      trackSegments.forEach((segment) => {
        acc.push({
          ...segment,
          trackIndex,
          x: (trackIndex % 3) * 320,
          y: Math.floor(trackIndex / 3) * 240,
        });
      });

      return acc;
    }, []);

    console.log(`Found ${validSegments.length} valid segments`);

    if (validSegments.length === 0) {
      throw new Error('No valid video segments found in tracks');
    }

    // Create video with overlays using sequential processing
    // const totalTracks = midi.tracks.length;
    await createVideoOverlaysSequential(
      validSegments,
      tempFiles.blackFrame,
      tempFiles.videoOverlay,
      tracks.length
    );

    // Create merged audio
    await createMergedAudio(validSegments, tempFiles.mergedAudio);

    // Final merge
    await mergeFinalVideo(
      tempFiles.videoOverlay,
      tempFiles.mergedAudio,
      sanitizedOutputPath,
      midiDuration
    );

    // cleanup(tempFiles);
    return sanitizedOutputPath;
  } catch (error) {
    // cleanup(tempFiles);
    throw new Error(`Video composition failed: ${error.message}`);
  }
}

async function createBlackFrame(outputPath) {
  console.log('Creating black frame at:', outputPath);
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input('color=c=black:s=960x720:r=30')
      .inputOptions(['-f', 'lavfi'])
      .outputOptions([
        '-c:v',
        'libx264',
        '-t',
        '1',
        '-pix_fmt',
        'yuv420p',
        '-y',
        '-filter:a:1',
        'volume=2.0', // Increase audio volume
      ])
      .on('start', (commandLine) => {
        console.log('FFmpeg command (black frame):', commandLine);
      })
      .save(outputPath)
      .on('end', () => {
        console.log('Black frame created successfully');
        resolve();
      })
      .on('error', (err) => {
        console.error('Error creating black frame:', err.message);
        reject(err);
      });
  });
}

async function createVideoOverlaysSequential(
  segments,
  blackFramePath,
  outputPath,
  totalTracks
) {
  let currentInput = blackFramePath;
  console.log('Starting sequential video overlay process');

  // Sort segments by start time to ensure proper overlay order
  segments.sort((a, b) => a.startTime - b.startTime);

  // Get unique track indices that have segments (i.e., tracks with notes)
  const activeTrackIndices = [...new Set(segments.map(s => s.trackIndex))];
  const actualTotalTracks = activeTrackIndices.length;

  console.log(`Active tracks: ${actualTotalTracks}`);

  // Special layout handling based on number of active tracks
  let layoutConfig;
  if (actualTotalTracks === 1) {
    // Single track - fill the entire screen
    layoutConfig = {
      segments: [{
        x: 0,
        y: 0,
        width: 960,
        height: 720,
        // Remove the crop and just scale to fill
        scale: 'scale=960:720'
      }]
    };
    console.log('Using single track full screen layout');
  } else if (actualTotalTracks === 2) {
    // Two tracks - choose layout style (uncomment preferred style)
    
    // Style 1: Opposite corners
    layoutConfig = {
      segments: [
        { x: 0, y: 0, width: 480, height: 360, scale: 'crop=480:360:in_w/2-240:in_h/2-180' },
        { x: 480, y: 360, width: 480, height: 360, scale: 'crop=480:360:in_w/2-240:in_h/2-180' }
      ]
    };
    
    // Style 2: Split screen (alternative layout)
    // layoutConfig = {
    //   segments: [
    //     { x: 0, y: 0, width: 480, height: 720, scale: 'crop=480:720:in_w/2-240:0' },
    //     { x: 480, y: 0, width: 480, height: 720, scale: 'crop=480:720:in_w/2-240:0' }
    //   ]
    // };
  } else {
    // Default grid layout for 3+ tracks
    const cols = Math.ceil(Math.sqrt(actualTotalTracks));
    const rows = Math.ceil(actualTotalTracks / cols);
    const segmentWidth = 960 / cols;
    const segmentHeight = 720 / rows;
    layoutConfig = {
      segments: activeTrackIndices.map((_, index) => ({
        x: (index % cols) * segmentWidth,
        y: Math.floor(index / cols) * segmentHeight,
        width: segmentWidth,
        height: segmentHeight,
        scale: `scale=${segmentWidth}:${segmentHeight}`
      }))
    };
  }

  // Debug log the layout configuration
  console.log('Layout configuration:', JSON.stringify(layoutConfig, null, 2));

  // Create a mapping of original track indices to layout positions
  const trackLayoutMapping = {};
  activeTrackIndices.forEach((originalIndex, newIndex) => {
    trackLayoutMapping[originalIndex] = layoutConfig.segments[newIndex];
  });

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const tempOutput = `${outputPath}_temp_${i}.mp4`;
    const layout = trackLayoutMapping[segment.trackIndex];

    console.log(`Processing segment ${i + 1}/${segments.length}`);
    console.log(`Input: ${segment.path}`);
    console.log(`Layout: x=${layout.x}, y=${layout.y}, width=${layout.width}, height=${layout.height}`);
    console.log(`Timing: start=${segment.startTime}, duration=${segment.duration}`);

    await new Promise((resolve, reject) => {
      ffmpeg()
        .input(currentInput)
        .input(segment.path)
        .complexFilter([
          // Apply crop/scale filter based on layout configuration
          `[1:v]${layout.scale}[v1]`,
          `[0:v][v1]overlay=${layout.x}:${layout.y}:enable='between(t,${segment.startTime},${
            segment.startTime + segment.duration
          })'[out]`
        ])
        .outputOptions([
          '-map',
          '[out]',
          '-c:v',
          'libx264',
          '-preset',
          'ultrafast',
          '-pix_fmt',
          'yuv420p',
          '-y',
          '-filter:a:1',
          'volume=2.0'
        ])
        .on('start', (commandLine) => {
          console.log(`FFmpeg command (overlay ${i + 1}):`, commandLine);
        })
        .save(tempOutput)
        .on('end', () => {
          console.log(`Overlay ${i + 1} completed`);
          if (currentInput !== blackFramePath) {
            try {
              rmSync(currentInput);
              console.log(`Removed intermediate file: ${currentInput}`);
            } catch (err) {
              console.warn(`Failed to clean up intermediate file: ${currentInput}`, err);
            }
          }
          currentInput = tempOutput;
          resolve();
        })
        .on('error', (err) => {
          console.error(`Error in overlay ${i + 1}:`, err.message);
          reject(err);
        });
    });
  }

  // Rename final temp file to target output
  try {
    if (existsSync(outputPath)) {
      rmSync(outputPath);
    }
    renameSync(currentInput, outputPath);
    console.log(`Renamed ${currentInput} to ${outputPath}`);
  } catch (err) {
    throw new Error(`Failed to finalize video file: ${err.message}`);
  }
}

async function createMergedAudio(segments, outputPath) {
  console.log('Starting audio merge process');
  console.log('Number of segments:', segments.length);

  // Sort segments by start time
  segments.sort((a, b) => a.startTime - b.startTime);

  // Find the total duration needed
  const totalDuration = Math.max(
    ...segments.map((s) => s.startTime + s.duration)
  );
  const BATCH_SIZE = 5; // Process 5 segments at a time

  try {
    console.log('Total duration:', totalDuration);

    if (isNaN(totalDuration) || totalDuration <= 0) {
      throw new Error('Invalid total duration');
    }

    // Normalize output path
    const normalizedOutputPath = outputPath.replace(/\\/g, '/');
    let currentOutputPath = normalizedOutputPath;

    // Process segments in batches
    for (let i = 0; i < segments.length; i += BATCH_SIZE) {
      const batchSegments = segments.slice(i, i + BATCH_SIZE);

      if (batchSegments.length === 0) {
        console.log(`Skipping empty batch ${i}`);
        continue;
      }

      console.log(
        `Processing batch ${i} with ${batchSegments.length} segments`
      );

      const isFirstBatch = i === 0;
      const batchOutputPath = isFirstBatch
        ? normalizedOutputPath
        : join(dirname(normalizedOutputPath), `batch_${i}.mp4`).replace(
            /\\/g,
            '/'
          );

      await new Promise((resolve, reject) => {
        const command = ffmpeg();

        // For first batch, create silent base. For subsequent batches, use previous output
        if (isFirstBatch) {
          command
            .input('anullsrc')
            .inputOptions(['-f', 'lavfi', '-t', String(totalDuration)]);
        } else {
          command.input(currentOutputPath);
        }

        // Add batch segments with normalized paths
        batchSegments.forEach((segment) => {
          const normalizedPath = segment.path.replace(/\\/g, '/');
          console.log(`Adding input file: ${normalizedPath}`);
          command.input(normalizedPath);
        });

        // Create filter complex
        const filterComplex = [];

        // Format base audio
        filterComplex.push(
          '[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[base]'
        );

        // Process each segment
        batchSegments.forEach((segment, index) => {
          const inputIndex = index + 1;
          const delay = Math.max(1, Math.round(segment.startTime * 1000));

          // Add error checking for audio stream
          filterComplex.push(
            `[${inputIndex}:a]asetpts=PTS-STARTPTS,` +
              `aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,` +
              `adelay=${delay}|${delay}[delayed${index}]`
          );
        });

        // Create mix command with a final output label
        const mixInputs = ['[base]'];
        batchSegments.forEach((_, index) => {
          mixInputs.push(`[delayed${index}]`);
        });

        const filterString = `${mixInputs.join('')}amix=inputs=${
          mixInputs.length
        }:normalize=0[audio_out]`;
        filterComplex.push(filterString);

        console.log('Filter complex:', filterComplex.join(';'));

        // Ensure output path has correct extension
        const outputPathWithExt = batchOutputPath.toLowerCase().endsWith('.mp4')
          ? batchOutputPath
          : batchOutputPath.replace(/\.[^/.]+$/, '') + '.mp4';

        // Apply filter complex and output options
        command
          .complexFilter(filterComplex.join(';'))
          .outputOptions([
            '-map',
            '[audio_out]',
            '-c:a',
            'aac',
            '-b:a',
            '256k',
            '-f',
            'mp4',
            '-movflags',
            '+faststart',
            '-t',
            String(totalDuration),
            '-filter:a:1',
            'volume=2.0', // Increase audio volume
          ])
          .on('start', (commandLine) => {
            console.log(`FFmpeg command (batch ${i}):`, commandLine);
          })
          .on('error', (err) => {
            console.error(`Error in batch ${i}:`, err);
            reject(err);
          })
          .on('stderr', (stderrLine) => {
            console.log(`FFmpeg stderr: ${stderrLine}`);
          })
          .save(outputPathWithExt)
          .on('end', () => {
            // Clean up previous intermediate file if it exists
            if (!isFirstBatch && currentOutputPath !== normalizedOutputPath) {
              try {
                rmSync(currentOutputPath);
              } catch (err) {
                console.warn(
                  `Failed to clean up intermediate file: ${currentOutputPath}`,
                  err
                );
              }
            }
            currentOutputPath = outputPathWithExt;
            console.log(`Successfully processed batch ${i}`);
            resolve();
          });
      });
    }
  } catch (err) {
    console.error('Error in audio processing:', err);
    throw err;
  }
}

async function mergeFinalVideo(
  videoPath,
  audioPath,
  outputPath,
  totalDuration
) {
  console.log('Starting final merge process');
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(videoPath)
      .input(audioPath)
      .outputOptions([
        '-c:v',
        'copy',
        '-c:a',
        'aac',
        '-strict',
        'experimental',
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        '-t',
        String(totalDuration), // Set the total duration here
        '-y',
        '-filter:a:1',
        'volume=2.0', // Increase audio volume
      ])
      .on('start', (commandLine) => {
        console.log('FFmpeg command (final merge):', commandLine);
      })
      .save(outputPath)
      .on('end', () => {
        console.log('Final merge completed');
        resolve();
      })
      .on('error', (err) => {
        console.error('Error in final merge:', err.message);
        reject(err);
      });
  });
}

// Add this helper function at the top
const normalizeInstrumentName = (name) => {
  return name.toLowerCase().replace(/\s+/g, '_');
};

// Add this new function
async function processVideoWithPython(midiData, processedFiles, outputPath) {
  return new Promise((resolve, reject) => {
    try {
      console.log('Processing files type:', typeof processedFiles, processedFiles instanceof Map);
      
      // Convert the processed files to the format Python expects
      const videoFilesForPython = {};

      if (processedFiles instanceof Map) {
        processedFiles.forEach((value, key) => {
          videoFilesForPython[key] = {
            path: value.path,
            isDrum: value.isDrum,
            notes: value.notes.map(note => ({
              time: note.time,
              duration: note.duration,
              velocity: note.velocity || 1.0,
              midi: note.midi
            })),
            // Add layout information
            layout: {
              x: 0,
              y: 0,
              width: 960,
              height: 720,
              fullscreen: processedFiles.size === 1
            }
          };
        });
      } else {
        // Handle if somehow a plain object is passed
        Object.entries(processedFiles).forEach(([key, value]) => {
          videoFilesForPython[key] = {
            path: value.path,
            isDrum: value.isDrum,
            notes: value.notes.map(note => ({
              time: note.time,
              duration: note.duration,
              velocity: note.velocity || 1.0,
              midi: note.midi
            }))
          };
        });
      }

      // Create temporary JSON files for the data
      const midiJsonPath = join(TEMP_DIR, 'midi_data.json');
      const videoFilesJsonPath = join(TEMP_DIR, 'video_files.json');
      
      // Write the data to temporary files
      writeFileSync(midiJsonPath, JSON.stringify(midiData));
      writeFileSync(videoFilesJsonPath, JSON.stringify(videoFilesForPython));

      // Spawn Python process
      const pythonProcess = spawn('python', [
        join(__dirname, '../utils/video_processor.py'),
        midiJsonPath,
        videoFilesJsonPath,
        outputPath
      ]);
      
      // Handle process events
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
      });
      
      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Python process exited with code ${code}`));
        }
      });
    } catch (error) {
      console.error('Error preparing data:', error);
      reject(new Error(`Failed to prepare data for Python: ${error.message}`));
    }
  });
}

// Add this helper function near the top
function logDrumNoteDetails(note, group) {
  return {
    midi: note.midi,
    group,
    time: note.time,
    duration: note.duration,
    velocity: note.velocity
  };
}

// Add this helper function at the top
function debugMidiTrack(track, index) {
  return {
    index,
    name: track.instrument.name,
    channel: track.channel,
    notes: track.notes.length,
    isDrum: isDrumTrack(track),
    firstNote: track.notes[0] ? {
      midi: track.notes[0].midi,
      channel: track.notes[0].channel
    } : null
  };
}

// Add this helper function for layout calculation
function calculateLayoutConfig(activeTrackCount) {
  if (activeTrackCount === 1) {
    return {
      segments: [{
        x: 0,
        y: 0,
        width: 960,
        height: 720,
        scale: 'scale=960:720'
      }]
    };
  } else if (activeTrackCount === 2) {
    return {
      segments: [
        { x: 0, y: 0, width: 480, height: 720, scale: 'scale=480:720' },
        { x: 480, y: 0, width: 480, height: 720, scale: 'scale=480:720' }
      ]
    };
  } else {
    const cols = Math.ceil(Math.sqrt(activeTrackCount));
    const rows = Math.ceil(activeTrackCount / cols);
    const segmentWidth = 960 / cols;
    const segmentHeight = 720 / rows;
    
    return {
      segments: Array.from({ length: activeTrackCount }, (_, index) => ({
        x: (index % cols) * segmentWidth,
        y: Math.floor(index / cols) * segmentHeight,
        width: segmentWidth,
        height: segmentHeight,
        scale: `scale=${segmentWidth}:${segmentHeight}`
      }))
    };
  }
}

// Modify the composeVideo function to use the Python processor
export const composeVideo = async (req, res) => {
  console.log('Received composition request');
  console.log('Files:', req.files?.map(f => ({
    fieldname: f.fieldname,
    size: f.size,
    mimetype: f.mimetype
  })));
  
  if (!req.files || req.files.length === 0) {
    console.error('No files received');
    return res.status(400).json({ error: 'No files were uploaded' });
  }

  const midiFile = req.files.find(file => file.fieldname === 'midiData');
  if (!midiFile) {
    console.error('No MIDI data found');
    return res.status(400).json({ error: 'MIDI data is missing' });
  }

  const videoFiles = req.files.filter(file => file.fieldname.startsWith('videos['));
  if (videoFiles.length === 0) {
    console.error('No video files found');
    return res.status(400).json({ error: 'No video files were uploaded' });
  }

  const sessionId = uuidv4();
  const sessionDir = join(TEMP_DIR, sessionId);
  
  if (
    !ensureDirectoryExists(TEMP_DIR) ||
    !ensureDirectoryExists(UPLOADS_DIR) ||
    !ensureDirectoryExists(sessionDir)
  ) {
    logging.error('Failed to create required directories');
    return res
      .status(500)
      .json({ error: 'Failed to create required directories' });
  }

  try {
    const files = req.files;
    if (!files || files.length === 0) {
      throw new Error('No files were uploaded');
    }

    logging.debug('Files received:', files.map(f => ({
      fieldname: f.fieldname,
      mimetype: f.mimetype,
      size: f.size,
    })));

    // Process MIDI file
    const midiFile = files.find((file) => file.fieldname === 'midiData');
    if (!midiFile) {
      throw new Error('MIDI data is missing');
    }

    // Parse MIDI data
    let midi;
    try {
      if (midiFile.mimetype === 'application/json') {
        const midiJson = JSON.parse(midiFile.buffer.toString());
        console.log('Raw MIDI JSON:', {
          tracks: midiJson.tracks.map((t, i) => ({
            index: i,
            name: t.instrument?.name,
            channel: t.channel,
            noteCount: t.notes?.length
          }))
        });
        
        midi = new Midi();

        // Add tracks from JSON
        midiJson.tracks?.forEach((track) => {
          const newTrack = midi.addTrack();
          track.notes?.forEach((note) => {
            newTrack.addNote({
              midi: note.midi,
              time: note.time,
              duration: note.duration,
              velocity: note.velocity || 0.8,
              channel: track.channel // Make sure channel is preserved
            });
          });
          if (track.instrument) {
            newTrack.instrument.name = track.instrument.name;
            newTrack.channel = track.channel; // Set channel explicitly
          }
        });

        console.log('Parsed MIDI tracks detail:', 
          midi.tracks.map((track, i) => debugMidiTrack(track, i))
        );
      } else {
        midi = new Midi(midiFile.buffer);
      }
    } catch (error) {
      console.error('Error parsing MIDI data:', error);
      throw new Error('Invalid MIDI data format');
    }

    // After parsing, add drum track validation
    const midiDrumTracks = midi.tracks.filter(isDrumTrack);
    console.log('\nDrum Track Analysis:');
    console.log('Total drum tracks found:', midiDrumTracks.length);
    midiDrumTracks.forEach((track, i) => {
      console.log(`\nDrum Track ${i}:`, {
        name: track.instrument.name,
        channel: track.channel,
        noteCount: track.notes.length,
        notes: track.notes.slice(0, 3).map(n => ({
          midi: n.midi,
          time: n.time,
          channel: n.channel
        }))
      });
    });

    // Save and process video files
    const videoFiles = {};
    const videos = files.filter((file) => file.fieldname.startsWith('videos'));

    if (videos.length === 0) {
      throw new Error('No video files were uploaded');
    }

    // Create a map of normalized instrument names to their videos
    videos.forEach((video) => {
      const instrumentMatch = video.fieldname.match(/\[(.*?)\]/);
      if (instrumentMatch) {
        const normalizedName = normalizeInstrumentName(instrumentMatch[1]);
        videoFiles[normalizedName] = video;
      }
    });

    console.log('Available video files for instruments:', Object.keys(videoFiles));
    console.log('MIDI tracks:', midi.tracks.map(track => track.instrument.name));

    // Process each track
    const processedFiles = new Map();
    console.log('\n=== Starting Track Processing ===');
    
    // First, validate drum videos against MIDI tracks
    const drumTracksNeeded = new Set();
    midi.tracks.forEach((track, index) => {
      if (isDrumTrack(track)) {
        console.log(`\nAnalyzing drum track ${index}:`, track.instrument.name);
        track.notes.forEach(note => {
          const group = getNoteGroup(note.midi);
          drumTracksNeeded.add(group);
          console.log(`Required drum group: ${group} for MIDI note: ${note.midi}`);
        });
      }
    });

    // Check available drum videos
    const availableDrumVideos = Object.keys(videoFiles)
      .filter(name => name.startsWith('drum_'));
    console.log('\nDrum videos validation:');
    console.log('Required drum groups:', Array.from(drumTracksNeeded));
    console.log('Available drum videos:', availableDrumVideos);

    // Validate drum video coverage
    drumTracksNeeded.forEach(group => {
      const drumVideoKey = `drum_${group}`;
      if (!videoFiles[drumVideoKey]) {
        console.warn(`⚠️ Missing drum video for group: ${group}`);
      }
    });

    // Filter out tracks with no notes first
    const activeTracksData = midi.tracks
      .filter(track => track.notes && track.notes.length > 0)
      .map((track, index) => ({
        track,
        index,
        normalizedName: normalizeInstrumentName(track.instrument.name)
      }));
    
    console.log(`Active tracks (with notes): ${activeTracksData.length}`);

    // Calculate layout based on number of active tracks
    const layoutConfig = calculateLayoutConfig(activeTracksData.length);
    console.log('Layout configuration:', layoutConfig);

    const trackPromises = activeTracksData.map(async ({ track, index, normalizedName }, layoutIndex) => {
      console.log(`\nProcessing track ${index}:`, {
        name: track.instrument.name,
        notes: track.notes.length,
        channel: track.channel
      });

      // Skip if no notes
      if (!track.notes || track.notes.length === 0) {
        console.log(`Skipping empty track ${index}`);
        return null;
      }

      const isDrumTrack = isGeneralMidiDrumKit(track.instrument.name, track.channel);
      
      if (isDrumTrack) {
        // Existing drum track processing...
        const drumNoteGroups = {};
        track.notes.forEach(note => {
          const group = getNoteGroup(note.midi);
          if (!drumNoteGroups[group]) {
            drumNoteGroups[group] = [];
          }
          drumNoteGroups[group].push({
            midi: note.midi,
            time: ticksToSeconds(note.ticks, midi),
            duration: Math.max(ticksToSeconds(note.durationTicks, midi), 0.1),
            velocity: note.velocity || 0.8
          });
        });

        // Only process drum groups that have videos
            const processedGroups = [];
            const videoFiles = req.files.reduce((acc, file) => {
              if (file.fieldname.startsWith('videos[')) {
                const instrumentMatch = file.fieldname.match(/\[(.*?)\]/);
                if (instrumentMatch) {
                  const normalizedName = normalizeInstrumentName(instrumentMatch[1]);
                  acc[normalizedName] = file;
                }
              }
              return acc;
            }, {});
        
            for (const [group, notes] of Object.entries(drumNoteGroups)) {
              const drumVideoKey = `drum_${group}`;
              const videoFile = videoFiles[drumVideoKey];
          
          if (videoFile) {
            const uniqueTrackId = `${drumVideoKey}_track${index}`;
            const tempVideoPath = join(sessionDir, `${uniqueTrackId}_temp.mp4`);
            const finalVideoPath = join(sessionDir, `${uniqueTrackId}.mp4`);
            
            try {
              writeFileSync(tempVideoPath, videoFile.buffer);
              await convertVideoFormat(tempVideoPath, finalVideoPath);
              await rm(tempVideoPath);
              
              processedFiles.set(uniqueTrackId, {
                path: finalVideoPath,
                isDrum: true,
                notes: notes,
                trackIndex: index,
                drumGroup: group,
                layout: layoutConfig.segments[layoutIndex]
              });
              
              processedGroups.push(group);
              console.log(`Processed ${uniqueTrackId} with ${notes.length} notes`);
            } catch (error) {
              console.error(`Error processing drum group ${group}:`, error);
            }
          }
        }
        return processedGroups;
      } else {
        // Handle melodic tracks
        const normalizedName = normalizeInstrumentName(track.instrument.name);
        const uniqueTrackId = `${normalizedName}_track${index}`;
        
        // Find corresponding video file
        const videoFile = videoFiles[normalizedName];
        if (!videoFile) {
          console.log(`No video found for instrument: ${track.instrument.name} (normalized: ${normalizedName})`);
          return null;
        }

        const tempVideoPath = join(sessionDir, `${uniqueTrackId}_temp.mp4`);
        const finalVideoPath = join(sessionDir, `${uniqueTrackId}.mp4`);
        
        try {
          writeFileSync(tempVideoPath, videoFile.buffer);
          await convertVideoFormat(tempVideoPath, finalVideoPath);
          await rm(tempVideoPath);
          
          const notes = track.notes.map(note => ({
            midi: note.midi,
            time: ticksToSeconds(note.ticks, midi),
            duration: Math.max(ticksToSeconds(note.durationTicks, midi), 0.1),
            velocity: note.velocity || 0.8
          }));
          
          processedFiles.set(uniqueTrackId, {
            path: finalVideoPath,
            isDrum: false,
            notes: notes,
            trackIndex: index,
            layout: layoutConfig.segments[layoutIndex]
          });

          console.log(`Processed ${uniqueTrackId} with ${notes.length} notes`);
          return uniqueTrackId;
        } catch (error) {
          console.error(`Error processing melodic track ${uniqueTrackId}:`, error);
          return null;
        }
      }
    });

    await Promise.all(trackPromises);

    // Log final processed files before sending to Python
    console.log('\n=== Final Processed Files ===');
    console.log('Total processed tracks:', processedFiles.size);
    processedFiles.forEach((value, key) => {
      console.log(`\nTrack: ${key}`);
      console.log(`- Is drum: ${value.isDrum}`);
      console.log(`- Note count: ${value.notes.length}`);
      console.log(`- File exists: ${existsSync(value.path)}`);
      console.log(`- First 3 notes:`, value.notes.slice(0, 3));
    });

    const outputPath = join(sessionDir, 'output.mp4');
    const midiData = midi.toJSON();
    midiData.duration = midi.duration;

    // Add verification before Python processing
    console.log('\n=== Pre-Python Processing Verification ===');
    const processedDrumTracks = Array.from(processedFiles.entries())
      .filter(([key, value]) => value.isDrum);
    console.log('Drum tracks being sent to Python:', 
      processedDrumTracks.map(([key, value]) => ({
        key,
        noteCount: value.notes.length
      }))
    );

    // Wait for any pending file operations to complete
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Pass the Map directly to processVideoWithPython
    await processVideoWithPython(
      midiData,
      processedFiles,
      outputPath
    );

    // Instead of sendFile, use streaming
    if (existsSync(outputPath)) {
      const stat = statSync(outputPath);
      const fileSize = stat.size;
      const head = {
        'Content-Length': fileSize,
        'Content-Type': 'video/mp4',
        'Content-Disposition': 'attachment; filename="composed-video.mp4"'
      };

      res.writeHead(200, head);
      const readStream = createReadStream(outputPath);
      
      // Handle stream events
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

      // Pipe the file to response
      readStream.pipe(res);
    } else {
      throw new Error('Output video file was not created');
    }

    // Cleanup function
    function cleanup() {
      setTimeout(() => {
        cleanupTempDirectory(sessionDir).catch(console.error);
      }, 1000);
    }

  } catch (error) {
    logging.error('Composition error:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    }
    setTimeout(() => {
      cleanupTempDirectory(sessionDir).catch(console.error);
    }, 1000);
  }
};

async function cleanupTempDirectory(dirPath) {
  try {
    await rm(dirPath, { recursive: true, force: true });
    console.log(`Temp directory ${dirPath} cleaned up successfully.`);
  } catch (error) {
    console.error(`Error cleaning up temp directory ${dirPath}:`, error);
  }
}

// Add this helper function to detect general MIDI drum kits
async function getOptimalEncodingSettings() {
  try {
    // First check for CUDA capability
    const result = await new Promise((resolve) => {
      const ffmpegProcess = spawn('ffmpeg', ['-encoders']);
      let output = '';

      ffmpegProcess.stdout.on('data', (data) => output += data.toString());
      ffmpegProcess.stderr.on('data', (data) => output += data.toString());
      
      ffmpegProcess.on('close', () => resolve(output));
    });

    if (result.includes('h264_nvenc')) {
      console.log('NVENC encoder found, using GPU acceleration');
      return {
        videoCodec: 'h264_nvenc'
      };
    }
    
    console.log('NVENC not found, falling back to CPU encoding');
    return {
      videoCodec: 'libx264'
    };
  } catch (error) {
    console.warn('Error detecting encoders, falling back to CPU:', error);
    return {
      videoCodec: 'libx264'
    };
  }
}

async function convertVideoFormat(inputPath, outputPath) {
  const maxRetries = 3;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      await new Promise((resolve, reject) => {
        // Create FFmpeg command with more conservative settings
        const command = ffmpeg(inputPath)
          .outputOptions([
            // Video settings
            '-c:v', 'libx264',          // Use CPU encoding instead of NVENC
            '-preset', 'ultrafast',      // Faster encoding
            '-pix_fmt', 'yuv420p',      // Standard pixel format
            '-vf', 'scale=960:720',     // Target resolution
            '-r', '30',                 // Target framerate
            '-analyzeduration', '10000000',  // Increase analyze duration
            '-probesize', '10000000',    // Increase probe size
            
            // Audio settings
            '-c:a', 'aac',              
            '-ar', '48000',             
            '-ac', '2',                 
            '-b:a', '192k',             
            
            // General settings
            '-y',                       // Overwrite output
            '-movflags', '+faststart'   // Enable streaming
          ]);

        // Add event handlers
        command
          .on('start', (cmdline) => {
            console.log(`[Attempt ${attempt}] Running FFmpeg command:`, cmdline);
          })
          .on('progress', (progress) => {
            if (progress.percent) {
              console.log(`[Attempt ${attempt}] Processing: ${Math.floor(progress.percent)}% done`);
            }
          })
          .on('stderr', (stderrLine) => {
            console.log(`[Attempt ${attempt}] FFmpeg stderr: ${stderrLine}`);
          })
          .on('error', (err) => {
            console.error(`[Attempt ${attempt}] Conversion error:`, err);
            reject(err);
          })
          .on('end', () => {
            console.log(`[Attempt ${attempt}] Conversion completed successfully`);
            resolve();
          })
          .save(outputPath);
      });
      
      // If successful, break out of retry loop
      return;
      
    } catch (error) {
      console.error(`Attempt ${attempt} failed:`, error);
      
      if (attempt === maxRetries) {
        throw new Error(`Failed to convert video after ${maxRetries} attempts: ${error.message}`);
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
}

// Add this helper function to detect general MIDI drum kits
function isGeneralMidiDrumKit(instrumentName, channel) {
  // Check if it's on channel 10 (9 in zero-based) or has a drum kit name
  const drumKitNames = [
    'standard kit',
    'room kit',
    'power kit',
    'electronic kit',
    'tr-808',
    'jazz kit',
    'brush kit',
    'orchestra kit',
    'sound fx kit'
  ];
  
  return channel === 9 || drumKitNames.some(kit => 
    instrumentName.toLowerCase().includes(kit.toLowerCase())
  );
}

// Add these utility functions after the other helper functions
async function standardizeVideo(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions([
        // Video settings
        '-c:v', 'libx264',          // Use H.264 codec
        '-preset', 'ultrafast',      // Faster encoding
        '-pix_fmt', 'yuv420p',      // Standard pixel format
        '-r', '30',                 // 30fps
        '-vf', 'scale=960:720',     // Standardize resolution
        
        // Audio settings
        '-c:a', 'aac',              // AAC audio codec
        '-ar', '48000',             // Sample rate
        '-ac', '2',                 // Stereo audio
        '-b:a', '192k',             // Audio bitrate
        
        // Container settings
        '-movflags', '+faststart',   // Enable fast start for web playback
        '-y'                         // Overwrite output
      ])
      .on('start', (cmd) => console.log('Started standardizing video:', cmd))
      .on('progress', (progress) => {
        if (progress.percent) {
          console.log(`Standardizing progress: ${Math.floor(progress.percent)}%`);
        }
      })
      .on('end', () => resolve(outputPath))
      .on('error', (err) => reject(err))
      .save(outputPath);
  });
}

// Modify the video processing part in composeVideo function
const processVideoFile = async (file, sessionDir, trackInfo, midi) => {
  const { uniqueTrackId } = trackInfo;
  const tempPath = join(sessionDir, `${uniqueTrackId}_original.mp4`);
  const standardPath = join(sessionDir, `${uniqueTrackId}_standard.mp4`);

  try {
    // Save uploaded file
    writeFileSync(tempPath, file.buffer);
    
    // Standardize the video
    await standardizeVideo(tempPath, standardPath);
    
    // Cleanup original
    await rm(tempPath);
    
    return standardPath;
  } catch (error) {
    console.error(`Error processing video for ${uniqueTrackId}:`, error);
    throw error;
  }
};
