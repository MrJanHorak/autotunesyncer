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
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { spawn } from 'child_process';
import { rm } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMP_DIR = join(__dirname, '../temp');
const UPLOADS_DIR = join(__dirname, '../uploads');

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

async function processNoteSegment(videoPath, note, outputPath, baseNote = 60) {
  return new Promise((resolve, reject) => {
    if (!existsSync(videoPath)) {
      return reject(new Error(`Input video file not found: ${videoPath}`));
    }

    const outputDir = dirname(outputPath);
    ensureDirectoryExists(outputDir);

    // Calculate pitch shift ratio based on MIDI note numbers
    const pitchRatio = midiNoteToFrequencyRatio(note.midi, baseNote);

    console.log('Processing note segment with params:', {
      videoPath,
      note,
      outputPath,
      targetNote: midiNoteToName(note.midi),
      baseNote: midiNoteToName(baseNote),
      pitchRatio,
    });

    // Split the filter complex into separate video and audio chains
    const command = ffmpeg(videoPath).setStartTime(0).duration(note.duration);

    // Add video filter chain
    command.videoFilters(['scale=320:240']);

    // Add audio filter chain
    command.audioFilters([
      // Convert mono to stereo and set sample rate
      'aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo',
      // Apply pitch shift
      `rubberband=pitch=${pitchRatio}:tempo=1`,
      // Apply velocity as volume
      `volume=${note.velocity}`,
    ]);

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

  // Calculate the layout based on the number of tracks
  const cols = Math.ceil(Math.sqrt(totalTracks));
  const rows = Math.ceil(totalTracks / cols);
  const segmentWidth = 960 / cols;
  const segmentHeight = 720 / rows;

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const tempOutput = `${outputPath}_temp_${i}.mp4`;
    const trackIndex = segment.trackIndex;
    const x = (trackIndex % cols) * segmentWidth;
    const y = Math.floor(trackIndex / cols) * segmentHeight;

    console.log(`Processing segment ${i + 1}/${segments.length}`);
    console.log(`Input: ${segment.path}`);
    console.log(`Position: x=${x}, y=${y}`);
    console.log(
      `Timing: start=${segment.startTime}, duration=${segment.duration}`
    );

    await new Promise((resolve, reject) => {
      ffmpeg()
        .input(currentInput)
        .input(segment.path)
        .complexFilter([
          `[1:v]scale=${segmentWidth}:${segmentHeight},setpts=PTS-STARTPTS[v1]`,
          `[0:v][v1]overlay=${x}:${y}:enable='between(t,${segment.startTime},${
            segment.startTime + segment.duration
          })'[out]`,
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
          'volume=2.0',
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
              console.warn(
                `Failed to clean up intermediate file: ${currentInput}`,
                err
              );
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
async function processVideoWithPython(midiData, videoFiles, outputPath) {
  return new Promise((resolve, reject) => {
    // Create temporary JSON files for the data
    const midiJsonPath = join(TEMP_DIR, 'midi_data.json');
    const videoFilesJsonPath = join(TEMP_DIR, 'video_files.json');
    
    // Write the data to temporary files
    writeFileSync(midiJsonPath, JSON.stringify(midiData));
    writeFileSync(videoFilesJsonPath, JSON.stringify(videoFiles));
    
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
  });
}

// Modify the composeVideo function to use the Python processor
export const composeVideo = async (req, res) => {
  const sessionId = uuidv4();
  const sessionDir = join(TEMP_DIR, sessionId);
  
  // ... existing directory setup code ...

  if (
    !ensureDirectoryExists(TEMP_DIR) ||
    !ensureDirectoryExists(UPLOADS_DIR) ||
    !ensureDirectoryExists(sessionDir)
  ) {
    return res
      .status(500)
      .json({ error: 'Failed to create required directories' });
  }

  try {
    const files = req.files;
    if (!files || files.length === 0) {
      throw new Error('No files were uploaded');
    }

    console.log(
      'Files received:',
      files.map((f) => ({
        fieldname: f.fieldname,
        mimetype: f.mimetype,
        size: f.size,
      }))
    );

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
            });
          });
          if (track.instrument) {
            newTrack.instrument.name = track.instrument.name;
          }
        });

        console.log('Parsed MIDI from JSON:', {
          tracks: midi.tracks.length,
          duration: midi.duration,
        });
      } else {
        midi = new Midi(midiFile.buffer);
      }
    } catch (error) {
      console.error('Error parsing MIDI data:', error);
      throw new Error('Invalid MIDI data format');
    }

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
    const trackPromises = midi.tracks.map(async (track, trackIndex) => {
      if (track.notes.length === 0) return null;

      const normalizedInstrumentName = normalizeInstrumentName(track.instrument.name);
      const videoFile = videoFiles[normalizedInstrumentName];

      if (!videoFile) {
        console.log(`No video found for instrument: ${track.instrument.name}`);
        return null;
      }

      // Only process each unique instrument once
      if (!processedFiles.has(normalizedInstrumentName)) {
        const videoPath = join(sessionDir, `${normalizedInstrumentName}.mp4`);
        const webmPath = join(sessionDir, `${normalizedInstrumentName}.webm`);
        
        writeFileSync(videoPath, videoFile.buffer);
        await convertToWebM(videoPath, webmPath);
        
        processedFiles.set(normalizedInstrumentName, webmPath);
        videoFiles[normalizedInstrumentName] = webmPath;

        // Clean up MP4 file after successful conversion
        try {
          await deleteFileWithRetry(videoPath);
        } catch (err) {
          console.warn(`Failed to clean up MP4 file: ${videoPath}`, err);
        }
      }
    });

    await Promise.all(trackPromises);

    const outputPath = join(sessionDir, 'output.mp4');
    const midiData = midi.toJSON();
    midiData.duration = midi.duration;

    // Wait for any pending file operations to complete
    await new Promise(resolve => setTimeout(resolve, 1000));

    await processVideoWithPython(
      midiData,
      Object.fromEntries(processedFiles),
      outputPath
    );

    res.sendFile(outputPath, async (err) => {
      if (err) {
        console.error('Error sending file:', err);
      }
      // Delay cleanup to ensure files are not in use
      setTimeout(() => cleanupTempDirectory(sessionDir), 1000);
    });
  } catch (error) {
    console.error('Composition error:', error);
    res.status(500).json({ error: error.message });
    // Delay cleanup to ensure files are not in use
    setTimeout(() => cleanupTempDirectory(sessionDir), 1000);
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

async function convertToWebM(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .output(outputPath)
      .outputOptions([
        '-c:v', 'libvpx',
        '-c:a', 'libvorbis',
        '-b:v', '1M',
        '-deadline', 'realtime',
        '-cpu-used', '5',
        '-auto-alt-ref', '0',
        '-qmin', '0',
        '-qmax', '50'
      ])
      .on('end', () => {
        // Verify the converted file
        ffmpeg.ffprobe(outputPath, (err, metadata) => {
          if (err || !metadata) {
            reject(new Error(`Failed to verify WebM file: ${err?.message}`));
            return;
          }
          console.log(`Successfully converted and verified ${outputPath}`);
          resolve();
        });
      })
      .on('error', (err) => {
        console.error(`Error converting ${inputPath} to WebM:`, err);
        reject(err);
      })      .run();  });}
// Add this helper function
async function deleteFileWithRetry(filePath, retries = 3, delay = 1000) {
  for (let i = 0; i < retries; i++) {
    try {
      await rm(filePath);
      console.log(`Successfully deleted ${filePath}`);
      break;
    } catch (err) {
      if (i < retries - 1) {
        console.warn(`Retrying deletion of ${filePath} (${i + 1})`);
        await new Promise(res => setTimeout(res, delay));
      } else {
        console.error(`Failed to delete ${filePath}:`, err);
      }
    }
  }
}
