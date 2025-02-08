/* eslint-disable no-unused-vars */
import ffmpeg from 'fluent-ffmpeg';
import pkg from '@tonejs/midi';
import { Buffer } from 'buffer';
const { Midi } = pkg;
import { existsSync, mkdirSync, writeFileSync, rmSync, renameSync } from 'fs';
import { join } from 'path';
// import os from 'os';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { spawn } from 'child_process';
import { rm } from 'fs/promises';
import {
  isDrumTrack,
  DRUM_NOTE_MAP,
  getNoteGroup,
} from '../utils/drumUtils.js';
import { createReadStream, statSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMP_DIR = join(__dirname, '../temp');
const UPLOADS_DIR = join(__dirname, '../uploads');
// const encodingSettings = await getOptimalEncodingSettings();
const GPU_MEMORY_LIMIT = '2GB';

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

// Add this new function
async function processVideoWithPython(midiData, processedFiles, outputPath) {
  console.log('Processing video with Python:', {
    outputPath,
    midiData,
    processedFiles,
  });
  return new Promise((resolve, reject) => {
    try {
      const videoFilesForPython = {};

      processedFiles.forEach((value, key) => {
        console.log(`\nProcessing video file: ${key}`);
        console.log('File details:', value);
        if (existsSync(value.path)) {
          videoFilesForPython[key] = {
            path: value.path.replace(/\\/g, '/'),
            isDrum: value.isDrum,
            drumName: value.drumName, // Add drum name instead of group
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
          console.log('Added video file:', videoFilesForPython[key]);
        } else {
          console.log(`Video file not found: ${value.path}`);
        }
      });

      // Ensure all video files have audio streams before processing
      const midiJsonPath = join(TEMP_DIR, 'midi_data.json');
      const videoFilesJsonPath = join(TEMP_DIR, 'video_files.json');
      console.log('MIDI data:', midiData);
      console.log('Writing JSON files:', { midiJsonPath, videoFilesJsonPath });

      // Add detailed logging for grid arrangement
      console.log('\n=== Grid Arrangement Validation ===');
      console.log('Full MIDI data structure:', Object.keys(midiData));
      console.log('Grid arrangement:', midiData.gridArrangement);

      // Log the data being written to files
      const dataToWrite = {
        ...midiData,
        gridArrangement: midiData.gridArrangement, // Ensure grid arrangement is included
      };
      console.log('Writing MIDI data with arrangement:', dataToWrite);

      writeFileSync(midiJsonPath, JSON.stringify(dataToWrite));
      writeFileSync(videoFilesJsonPath, JSON.stringify(videoFilesForPython));

      // Spawn Python process with detailed logger
      const pythonProcess = spawn('python', [
        join(__dirname, '../utils/video_processor.py'),
        midiJsonPath,
        videoFilesJsonPath,
        outputPath,
      ]);
      console.log('Python process started:', pythonProcess.pid);
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          // Verify output video has audio
          ffmpeg.ffprobe(outputPath, (err, metadata) => {
            if (err) {
              console.error('Error verifying output video:', err);
              reject(err);
              return;
            }

            const hasAudio = metadata.streams.some(
              (s) => s.codec_type === 'audio'
            );
            if (!hasAudio) {
              console.error('Output video has no audio stream!');
              reject(new Error('Output video missing audio'));
              return;
            }

            resolve();
          });
        } else {
          reject(new Error(`Python process exited with code ${code}`));
        }
      });
    } catch (error) {
      reject(error);
    }
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
  const sessionId = uuidv4();

  try {
    console.log('Processing request body', {
      sessionId,
      filesCount: req.files?.length,
      bodyKeys: Object.keys(req.body || {}),
    });

    const { midi, videoFiles } = req.body;
    const midiData = new Midi(Buffer.from(midi));

    console.log('MIDI data:', {
      format: midiData.header.format,
      tracks: midiData.tracks.map((t, i) => ({
        index: i,
        name: t.instrument?.name,
        channel: t.channel,
        noteCount: t.notes?.length,
      })),
    });
    const processedTracks = {};
    const processedDrums = {};

    midiData.tracks.forEach((track, index) => {
      if (isDrumTrack(track)) {
        console.log(`\nAnalyzing drum track ${index}:`, track.instrument.name);
        const drumGroups = processDrumTrack(track, index);
        console.log(`\nDrum groups for track ${index}:`, drumGroups);
        Object.entries(drumGroups).forEach(([drumName, notes]) => {
          const videoKey = `drum_${drumName}`;
          console.log('Drum video key:', videoKey);
          if (videoFiles[videoKey]) {
            console.log('Drum video found:', videoFiles[videoKey]);
            processedDrums[videoKey] = {
              notes,
              video: videoFiles[videoKey],
              index,
            };
            console.log('Processed drums:', processedDrums);
          }
        });
      } else {
        const trackKey = normalizeInstrumentName(track.instrument.name);
        console.log(
          `\nProcessing melodic track ${index}:`,
          track.instrument.name
        );
        console.log('Normalized key:', trackKey);
        if (videoFiles[trackKey]) {
          processedTracks[trackKey] = {
            notes: track.notes,
            video: videoFiles[trackKey],
            index,
          };
        }
      }
    });

    const config = {
      tracks: processedTracks,
      drums: processedDrums,
      sessionId: uuidv4(),
    };

    writeFileSync(
      join(TEMP_DIR, `config_${config.sessionId}.json`),
      JSON.stringify(config, null, 2)
    );

    const outputPath = join(TEMP_DIR, `output_${config.sessionId}.mp4`);
    await processVideoWithPython(midiData, config, outputPath);

    const stat = statSync(outputPath);
    const fileSize = stat.size;
    const head = {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
      'Content-Disposition': 'attachment; filename="composed-video.mp4"',
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

    function cleanup() {
      setTimeout(() => {
        cleanupTempDirectory(TEMP_DIR).catch(console.error);
      }, 1000);
    }
  } catch (error) {
    console.error('Composition error:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: error.message });
    }
    setTimeout(() => {
      cleanupTempDirectory(TEMP_DIR).catch(console.error);
    }, 1000);
  }
};

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
