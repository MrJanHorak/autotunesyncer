/* eslint-disable no-unused-vars */
import ffmpeg from 'fluent-ffmpeg';
import pkg from '@tonejs/midi';
const { Midi } = pkg;
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  rmSync,
  copyFileSync,
  createWriteStream,
} from 'fs';
import { PNG } from 'pngjs';
import { Buffer } from 'buffer';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

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

async function createBlackPNG(outputPath, width = 960, height = 720) {
  return new Promise((resolve, reject) => {
    const png = new PNG({ width, height });
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (width * y + x) << 2;
        png.data[idx] = 0; // R
        png.data[idx + 1] = 0; // G
        png.data[idx + 2] = 0; // B
        png.data[idx + 3] = 255; // A
      }
    }
    png
      .pack()
      .pipe(createWriteStream(outputPath))
      .on('finish', resolve)
      .on('error', reject);
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

async function processNoteSegment(videoPath, note, outputPath) {
  return new Promise((resolve, reject) => {
    if (!existsSync(videoPath)) {
      return reject(new Error(`Input video file not found: ${videoPath}`));
    }

    const outputDir = dirname(outputPath);
    ensureDirectoryExists(outputDir);

    console.log('Processing note segment with params:', {
      videoPath,
      note,
      outputPath,
    });

    const command = ffmpeg(videoPath)
      .setStartTime(0)
      .duration(note.duration)
      .outputOptions([
        '-vf',
        `scale=320:240,setpts=${1 / note.duration}*PTS`,
        '-af',
        `volume=${note.velocity}`,
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-c:a',
        'aac',
      ]);

    command.on('start', (commandLine) => {
      console.log('FFmpeg command:', commandLine);
    });

    command.on('stderr', (stderrLine) => {
      console.log('FFmpeg stderr:', stderrLine);
    });

    command
      .save(outputPath)
      .on('end', () => {
        if (existsSync(outputPath)) {
          console.log(`Successfully processed note segment: ${outputPath}`);
          resolve();
        } else {
          reject(new Error(`Failed to create output file: ${outputPath}`));
        }
      })
      .on('error', (err) => {
        console.error(`Error processing note segment:`, err);
        reject(err);
      });
  });
}

// async function composeFinalVideo(tracks, outputPath) {
//   if (!tracks || tracks.length === 0) {
//     throw new Error('No tracks provided for final composition');
//   }

//   console.log('Starting final composition with tracks:', tracks);

//   return new Promise((resolve, reject) => {
//     const blackFramePath = join(dirname(outputPath), 'black_frame.mp4');

//     // Create a black video first with explicit video stream
//     return new Promise((resolveBlackFrame, rejectBlackFrame) => {
//       ffmpeg()
//         .addInput('nullsrc')
//         .inputOptions(['-f', 'lavfi', '-t', '1', '-s', '960x720'])
//         .outputOptions([
//           '-c:v',
//           '-tune',
//           'libx264',
//           'stillimage',
//           '-pix_fmt',
//           'yuv420p',
//           '-t',
//           '1',
//         ])
//         .save(blackFramePath)
//         .on('end', () => {
//           // Now start the main composition
//           const command = ffmpeg();

//           // Initialize filter complex parts
//           let filterComplex = '';
//           let overlayCount = 0;
//           let hasValidSegments = false;

//           // First pass to count valid segments and inputs
//           tracks.forEach((track) => {
//             if (!track.segments || track.segments.length === 0) return;
//             track.segments.forEach((segment) => {
//               if (existsSync(segment.path)) {
//                 hasValidSegments = true;
//                 overlayCount++;
//               }
//             });
//           });

//           if (!hasValidSegments) {
//             rmSync(blackFramePath);
//             return reject(new Error('No valid segments found for composition'));
//           }

//           // Use the black frame as base and loop it
//           command.input(blackFramePath).inputOptions(['-stream_loop', '-1']);

//           // Initialize with black background
//           filterComplex += '[0:v]scale=960:720,setsar=1:1[base];';

//           // Add all segment inputs and create overlay chain
//           let currentBase = 'base';
//           let inputIndex = 1; // Start from 1 since we used 0 for black frame

//           tracks.forEach((track, trackIndex) => {
//             if (!track.segments || track.segments.length === 0) return;

//             track.segments.forEach((segment, segmentIndex) => {
//               if (!existsSync(segment.path)) {
//                 console.warn(`Skipping missing segment: ${segment.path}`);
//                 return;
//               }

//               // Add input
//               command.input(segment.path);

//               const x = (trackIndex % 3) * 320;
//               const y = Math.floor(trackIndex / 3) * 240;

//               // Create video processing chain
//               filterComplex += `[${inputIndex}:v]setpts=PTS-STARTPTS+${segment.startTime}/TB[v${inputIndex}];`;
//               filterComplex += `[${currentBase}][v${inputIndex}]overlay=${x}:${y}:enable='between(t,${
//                 segment.startTime
//               },${segment.startTime + segment.duration})'[v${inputIndex}_out];`;

//               currentBase = `v${inputIndex}_out`;
//               inputIndex++;
//             });
//           });

//           // Handle audio mixing
//           let audioInputs = [];
//           for (let i = 1; i < inputIndex; i++) {
//             filterComplex += `[${i}:a]asetpts=PTS-STARTPTS+${
//               tracks[Math.floor((i - 1) / tracks[0].segments.length)].segments[
//                 (i - 1) % tracks[0].segments.length
//               ].startTime
//             }/TB[a${i}];`;
//             audioInputs.push(`[a${i}]`);
//           }

//           if (audioInputs.length > 0) {
//             filterComplex += `${audioInputs.join('')}amix=inputs=${
//               audioInputs.length
//             }:normalize=0[aout]`;
//           }

//           // Final output mapping
//           const outputs = [currentBase];
//           if (audioInputs.length > 0) {
//             outputs.push('aout');
//           }

//           console.log('Filter complex:', filterComplex);

//           command
//             .complexFilter(filterComplex, outputs)
//             .outputOptions([
//               '-map',
//               `[${currentBase}]`,
//               ...(audioInputs.length > 0 ? ['-map', '[aout]'] : []),
//               '-c:v',
//               'libx264',
//               '-preset',
//               'ultrafast',
//               '-c:a',
//               'aac',
//               '-shortest',
//             ])
//             .on('start', (commandLine) => {
//               console.log('FFmpeg final composition command:', commandLine);
//             })
//             .on('progress', (progress) => {
//               console.log('Processing: ', progress.percent, '% done');
//             })
//             .on('stderr', (stderrLine) => {
//               console.log('FFmpeg stderr:', stderrLine);
//             })
//             .save(outputPath)
//             .on('end', () => {
//               // Clean up the temporary black frame
//               try {
//                 rmSync(blackFramePath);
//               } catch (err) {
//                 console.warn('Failed to remove temporary black frame:', err);
//               }

//               if (existsSync(outputPath)) {
//                 console.log('Final video created successfully');
//                 resolve();
//               } else {
//                 reject(new Error('Output file was not created'));
//               }
//             })
//             .on('error', (err) => {
//               console.error('FFmpeg error:', err);
//               // Clean up on error
//               try {
//                 rmSync(blackFramePath);
//               } catch (cleanupErr) {
//                 console.warn(
//                   'Failed to remove temporary black frame:',
//                   cleanupErr
//                 );
//               }
//               reject(err);
//             });
//         })
//         .on('error', (err) => {
//           console.error('Error creating black frame:', err);
//           rejectBlackFrame(err);
//         });
//     });
//   });
// }

// async function composeFinalVideo(tracks, outputPath) {
//   if (!tracks || tracks.length === 0) {
//     throw new Error('No tracks provided for final composition');
//   }

//   // Ensure output directory exists
//   const outputDir = dirname(outputPath);
//   try {
//     if (!existsSync(outputDir)) {
//       mkdirSync(outputDir, { recursive: true });
//     }
    
//     // Test write permissions by creating a temp file
//     const testPath = join(outputDir, 'test.txt');
//     writeFileSync(testPath, 'test');
//     rmSync(testPath);
//   } catch (err) {
//     throw new Error(`Cannot write to output directory ${outputDir}: ${err.message}`);
//   }

//   console.log('Starting final composition with tracks:', tracks);
//   console.log('Output path:', outputPath);

//   return new Promise((resolve, reject) => {
//     const blackFramePath = join(outputDir, 'black_frame.mp4');

//     // Create a black video first
//     return new Promise((resolveBlackFrame, rejectBlackFrame) => {
//       ffmpeg()
//         .input('color=c=black:s=960x720:r=30')
//         .inputOptions(['-f', 'lavfi'])
//         .inputFormat('lavfi')
//         .outputOptions([
//           '-c:v', 'libx264',
//           '-t', '1',
//           '-pix_fmt', 'yuv420p'
//         ])
//         .save(blackFramePath)
//         .on('end', () => {
//           // Verify black frame was created
//           if (!existsSync(blackFramePath)) {
//             return rejectBlackFrame(new Error('Failed to create black frame video'));
//           }

//           // Now start the main composition
//           const command = ffmpeg();

//           // Initialize filter complex parts
//           let filterComplex = '';
//           let hasValidSegments = false;

//           // First pass to check segments
//           tracks.forEach((track) => {
//             console.log('Segment and Track:', track.segments);
//             if (!track.segments || track.segments.length === 0) return;
//             track.segments.forEach((segment) => {
//               if (existsSync(segment.path)) {
//                 hasValidSegments = true;
//               } else {
//                 console.warn(`Missing segment file: ${segment.path}`);
//               }
//             });
//           });

//           if (!hasValidSegments) {
//             rmSync(blackFramePath);
//             return reject(new Error('No valid segments found for composition'));
//           }

//           // Use the black frame as base and loop it
//           command.input(blackFramePath).inputOptions(['-stream_loop', '-1']);

//           // Initialize with black background
//           filterComplex += '[0:v]scale=960:720,setsar=1:1[base];';

//           let currentBase = 'base';
//           let inputIndex = 1;
//           const audioInputs = [];

//           // Build filter complex
//           tracks.forEach((track, trackIndex) => {
//             if (!track.segments || track.segments.length === 0) return;

//             track.segments.forEach((segment, segmentIndex) => {
//               if (!existsSync(segment.path)) {
//                 console.warn(`Skipping missing segment: ${segment.path}`);
//                 return;
//               }

//               // Add input
//               command.input(segment.path);

//               const x = (trackIndex % 3) * 320;
//               const y = Math.floor(trackIndex / 3) * 240;

//               // Create unique labels
//               const videoLabel = `v${inputIndex}`;
//               const overlayLabel = `overlay${inputIndex}`;

//               // Video processing chain
//               filterComplex += `[${inputIndex}:v]setpts=PTS-STARTPTS+${segment.startTime}/TB[${videoLabel}];`;
//               filterComplex += `[${currentBase}][${videoLabel}]overlay=${x}:${y}:enable='between(t,${segment.startTime},${segment.startTime + segment.duration})'[${overlayLabel}];`;

//               // Audio processing
//               filterComplex += `[${inputIndex}:a]asetpts=PTS-STARTPTS+${segment.startTime}/TB[a${inputIndex}];`;
//               audioInputs.push(`[a${inputIndex}]`);

//               currentBase = overlayLabel;
//               inputIndex++;
//             });
//           });

//           // Remove trailing semicolon
//           filterComplex = filterComplex.slice(0, -1);

//           // Add audio mixing if needed
//           if (audioInputs.length > 0) {
//             filterComplex += `;${audioInputs.join('')}amix=inputs=${audioInputs.length}:normalize=0[aout]`;
//           }

//           console.log('Filter complex:', filterComplex);

//           // Final command setup
//           const finalCommand = command
//             .complexFilter(filterComplex, [currentBase, 'aout'])
//             .outputOptions([
//               '-map', `[${currentBase}]`,
//               '-map', '[aout]',
//               '-c:v', 'libx264',
//               '-preset', 'ultrafast',
//               '-c:a', 'aac',
//               '-shortest'
//             ]);

//           // Log the full command for debugging
//           finalCommand.on('start', (commandLine) => {
//             console.log('FFmpeg command:', commandLine);
//           });

//           // Monitor progress
//           finalCommand.on('progress', (progress) => {
//             console.log('Processing:', progress.percent, '% done');
//           });

//           // Detailed error logging
//           finalCommand.on('stderr', (stderrLine) => {
//             console.log('FFmpeg stderr:', stderrLine);
//           });

//           // Save the output
//           finalCommand
//             .save(outputPath)
//             .on('end', () => {
//               // Verify output was created
//               if (!existsSync(outputPath)) {
//                 reject(new Error('Output file was not created'));
//                 return;
//               }

//               try {
//                 rmSync(blackFramePath);
//               } catch (err) {
//                 console.warn('Failed to remove temporary black frame:', err);
//               }

//               console.log('Final video created successfully at:', outputPath);
//               resolve();
//             })
//             .on('error', (err) => {
//               console.error('FFmpeg error:', err);
//               try {
//                 rmSync(blackFramePath);
//               } catch (cleanupErr) {
//                 console.warn('Failed to remove temporary black frame:', cleanupErr);
//               }
//               reject(err);
//             });
//         })
//         .on('error', (err) => {
//           console.error('Error creating black frame:', err);
//           rejectBlackFrame(err);
//         });
//     });
//   });
// }

// async function composeFinalVideo(tracks, outputPath) {
//   if (!tracks || tracks.length === 0) {
//     throw new Error('No tracks provided for final composition');
//   }

//   const outputDir = dirname(outputPath);
//   if (!existsSync(outputDir)) {
//     mkdirSync(outputDir, { recursive: true });
//   }

//   // Create a black frame as the base video
//   const blackFramePath = join(outputDir, 'black_frame.mp4');
//   await new Promise((resolve, reject) => {
//     ffmpeg()
//       .input('color=c=black:s=960x720:r=30')
//       .inputOptions(['-f', 'lavfi'])
//       .outputOptions([
//         '-c:v', 'libx264',
//         '-t', '1',
//         '-pix_fmt', 'yuv420p'
//       ])
//       .save(blackFramePath)
//       .on('end', resolve)
//       .on('error', reject);
//   });

//   const videoOutputPath = join(outputDir, 'video_with_overlays.mp4');
//   const audioOutputPath = join(outputDir, 'merged_audio.mp3');

//   try {
//     // Step 1: Create video with overlays
//     await new Promise((resolve, reject) => {
//       const command = ffmpeg();
//       let filterComplex = '[0:v]scale=960:720,setsar=1:1[base];';

//       let inputIndex = 1;
//       tracks.forEach((track, trackIndex) => {
//         if (track.segments && track.segments.length > 0) {
//           track.segments.forEach((segment) => {
//             if (existsSync(segment.path)) {
//               const x = (trackIndex % 3) * 320;
//               const y = Math.floor(trackIndex / 3) * 240;

//               command.input(segment.path);
//               filterComplex += `[${inputIndex}:v]setpts=PTS-STARTPTS+${segment.startTime}/TB[v${inputIndex}];[base][v${inputIndex}]overlay=${x}:${y}:enable='between(t,${segment.startTime},${segment.startTime + segment.duration})'[base];`;
//               inputIndex++;
//             }
//           });
//         }
//       });

//       filterComplex = filterComplex.slice(0, -1); // Remove trailing semicolon

//       command.input(blackFramePath)
//         .complexFilter(filterComplex, 'base')
//         .outputOptions([
//           '-map', '[base]',
//           '-c:v', 'libx264',
//           '-preset', 'ultrafast',
//           '-shortest'
//         ])
//         .save(videoOutputPath)
//         .on('end', resolve)
//         .on('error', reject);
//     });

//     // Step 2: Create merged audio track
//     await new Promise((resolve, reject) => {
//       const command = ffmpeg();
//       let audioInputs = [];
//       let inputIndex = 0;

//       tracks.forEach((track) => {
//         if (track.segments && track.segments.length > 0) {
//           track.segments.forEach((segment) => {
//             if (existsSync(segment.path)) {
//               command.input(segment.path);
//               audioInputs.push(`[${inputIndex}:a]`);
//               inputIndex++;
//             }
//           });
//         }
//       });

//       const audioFilterComplex = `${audioInputs.join('')}amix=inputs=${audioInputs.length}:normalize=0[aout]`;

//       command.complexFilter(audioFilterComplex, 'aout')
//         .outputOptions(['-map', '[aout]', '-c:a', 'aac'])
//         .save(audioOutputPath)
//         .on('end', resolve)
//         .on('error', reject);
//     });

//     // Step 3: Merge video with overlays and audio
//     return new Promise((resolve, reject) => {
//       ffmpeg()
//         .input(videoOutputPath)
//         .input(audioOutputPath)
//         .outputOptions(['-c:v', 'copy', '-c:a', 'aac', '-shortest'])
//         .save(outputPath)
//         .on('end', () => {
//           rmSync(blackFramePath);
//           rmSync(videoOutputPath);
//           rmSync(audioOutputPath);
//           console.log('Final video created successfully at:', outputPath);
//           resolve();
//         })
//         .on('error', (err) => {
//           console.error('Final merge error:', err);
//           rmSync(blackFramePath);
//           rmSync(videoOutputPath);
//           rmSync(audioOutputPath);
//           reject(err);
//         });
//     });
//   } catch (error) {
//     console.error('Error during video composition:', error);
//     throw error;
//   }
// }

async function composeFinalVideo(tracks, outputPath) {
  if (!tracks || tracks.length === 0) {
    throw new Error('No tracks provided for final composition');
  }

  const outputDir = dirname(outputPath);
  
  // Ensure output directory exists
  if (!existsSync(outputDir)) {
    console.log(`Creating output directory at: ${outputDir}`);
    mkdirSync(outputDir, { recursive: true });
  }

  // Create a black frame as the base video
  const blackFramePath = join(outputDir, 'black_frame.mp4');
  await new Promise((resolve, reject) => {
    ffmpeg()
      .input('color=c=black:s=960x720:r=30')
      .inputOptions(['-f', 'lavfi'])
      .outputOptions(['-c:v', 'libx264', '-t', '1', '-pix_fmt', 'yuv420p'])
      .save(blackFramePath)
      .on('end', () => {
        console.log(`Black frame created at: ${blackFramePath}`);
        resolve();
      })
      .on('error', (err) => {
        console.error(`Error creating black frame: ${err.message}`);
        reject(err);
      });
  });

  // const videoOutputPath = join(outputDir, 'video_with_overlays_temp.mp4'); // Simplified name for clarity
  // const audioOutputPath = join(outputDir, 'merged_audio_temp.mp3'); // Simplified name for clarity
  const videoOutputPath = join(outputDir, 'video_with_overlays_temp.mp4').replace(/\\/g, '/');
  const audioOutputPath = join(outputDir, 'merged_audio_temp.mp3').replace(/\\/g, '/');

  
  try {
    // Step 1: Create video with overlays
    await new Promise((resolve, reject) => {
      const command = ffmpeg();
      let filterComplex = '[0:v]scale=960:720,setsar=1:1[base];';
      let inputIndex = 1;

      tracks.forEach((track, trackIndex) => {
        if (track.segments && track.segments.length > 0) {
          track.segments.forEach((segment) => {
            if (existsSync(segment.path)) {
              const x = (trackIndex % 3) * 320;
              const y = Math.floor(trackIndex / 3) * 240;

              command.input(segment.path);
              filterComplex += `[${inputIndex}:v]setpts=PTS-STARTPTS+${segment.startTime}/TB[v${inputIndex}];[base][v${inputIndex}]overlay=${x}:${y}:enable='between(t,${segment.startTime},${segment.startTime + segment.duration})'[base];`;
              inputIndex++;
            }
          });
        }
      });

      filterComplex = filterComplex.slice(0, -1); // Remove trailing semicolon

      command
        .input(blackFramePath)
        .complexFilter(filterComplex, 'base')
        .outputOptions([
          '-map', '[base]',
          '-c:v', 'libx264',
          '-preset', 'ultrafast',
          '-y', // Overwrite if exists
        ])
        .save(videoOutputPath)
        .on('end', () => {
          console.log(`Video with overlays saved to: ${videoOutputPath}`);
          resolve();
        })
        .on('error', (err) => {
          console.error(`Error during video overlay creation: ${err.message}`);
          reject(err);
        });
    });

    // Step 2: Create merged audio track
    await new Promise((resolve, reject) => {
      const command = ffmpeg();
      let audioInputs = [];
      let inputIndex = 0;

      tracks.forEach((track) => {
        if (track.segments && track.segments.length > 0) {
          track.segments.forEach((segment) => {
            if (existsSync(segment.path)) {
              command.input(segment.path);
              audioInputs.push(`[${inputIndex}:a]`);
              inputIndex++;
            }
          });
        }
      });

      const audioFilterComplex = `${audioInputs.join('')}amix=inputs=${audioInputs.length}:normalize=0[aout]`;

      command.complexFilter(audioFilterComplex, 'aout')
        .outputOptions(['-map', '[aout]', '-c:a', 'aac'])
        .save(audioOutputPath)
        .on('end', () => {
          console.log(`Merged audio track saved to: ${audioOutputPath}`);
          resolve();
        })
        .on('error', (err) => {
          console.error(`Error during audio merging: ${err.message}`);
          reject(err);
        });
    });

    // Step 3: Merge video with overlays and audio
    return new Promise((resolve, reject) => {
      ffmpeg()
        .input(videoOutputPath)
        .input(audioOutputPath)
        .outputOptions(['-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y'])
        .save(outputPath)
        .on('end', () => {
          console.log('Final video created successfully at:', outputPath);
          // Cleanup temp files
          rmSync(blackFramePath, { force: true });
          rmSync(videoOutputPath, { force: true });
          rmSync(audioOutputPath, { force: true });
          resolve();
        })
        .on('error', (err) => {
          console.error('Error in final merge:', err.message);
          // Cleanup temp files
          rmSync(blackFramePath, { force: true });
          rmSync(videoOutputPath, { force: true });
          rmSync(audioOutputPath, { force: true });
          reject(err);
        });
    });
  } catch (error) {
    console.error('Error during video composition:', error.message);
    throw error;
  }
}

export const composeVideo = async (req, res) => {
  const sessionId = uuidv4();
  const sessionDir = join(TEMP_DIR, sessionId);

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

    for (const video of videos) {
      try {
        const instrument = video.fieldname.match(/\[(.*?)\]/)?.[1];
        if (!instrument) continue;

        const videoPath = join(sessionDir, `${instrument}.webm`);
        writeFileSync(videoPath, video.buffer);

        if (existsSync(videoPath)) {
          videoFiles[instrument] = videoPath;
          console.log(
            `Successfully saved video for instrument ${instrument} at ${videoPath}`
          );
        } else {
          throw new Error(`Failed to save video file for ${instrument}`);
        }
      } catch (err) {
        console.error(`Error saving video file:`, err);
        throw err;
      }
    }

    // Process each track
    const trackPromises = midi.tracks.map(async (track, trackIndex) => {
      console.log(`Processing track ${trackIndex}:`, {
        name: track.name,
        instrument: track.instrument.name,
        noteCount: track.notes.length,
      });

      if (track.notes.length === 0) return null;

      const instrument = track.instrument.name || `track${trackIndex}`;
      const videoPath = videoFiles[instrument];
      if (!videoPath) {
        console.log(`No video found for instrument: ${instrument}`);
        return null;
      }

      const segments = [];
      for (const [noteIndex, note] of track.notes.entries()) {
        const startTimeSeconds = ticksToSeconds(note.ticks, midi);
        const durationSeconds = ticksToSeconds(note.durationTicks, midi);

        console.log(`Processing note ${noteIndex}:`, {
          startTime: startTimeSeconds,
          duration: durationSeconds,
          velocity: note.velocity,
        });

        const segmentPath = join(
          sessionDir,
          `track${trackIndex}_note${noteIndex}.mp4`
        );

        try {
          await processNoteSegment(
            videoPath,
            {
              startTime: startTimeSeconds,
              duration: durationSeconds,
              velocity: note.velocity,
            },
            segmentPath
          );

          segments.push({
            path: segmentPath,
            startTime: startTimeSeconds,
            duration: durationSeconds,
          });
        } catch (error) {
          console.error(`Error processing note segment ${noteIndex}:`, error);
          // Continue with other notes even if one fails
        }
      }

      return segments.length > 0
        ? {
            trackIndex,
            instrument,
            segments,
          }
        : null;
    });

    const tracks = (await Promise.all(trackPromises)).filter(Boolean);

    if (tracks.length === 0) {
      throw new Error('No valid tracks to process');
    }

    // Create a black PNG image
    const blackPngPath = join(sessionDir, 'black.png');
    await createBlackPNG(blackPngPath);

    const outputPath = join(sessionDir, 'output.mp4');
    await composeFinalVideo(tracks, outputPath, blackPngPath);

    if (!existsSync(outputPath)) {
      throw new Error(
        'Final video composition failed - output file not created'
      );
    }

    const localCopyPath = join(UPLOADS_DIR, `output-${sessionId}.mp4`);
    copyFileSync(outputPath, localCopyPath);
    console.log('Local copy saved to:', localCopyPath);

    res.sendFile(outputPath, (err) => {
      if (err) {
        console.error('Error sending file:', err);
      }
      // Cleanup after successful send
      try {
        rmSync(sessionDir, { recursive: true, force: true });
      } catch (cleanupErr) {
        console.error('Error cleaning up temp files:', cleanupErr);
      }
    });
  } catch (error) {
    console.error('Composition error:', error);
    res.status(500).json({ error: error.message });
    // Cleanup on error
    try {
      rmSync(sessionDir, { recursive: true, force: true });
    } catch (cleanupErr) {
      console.error('Error cleaning up temp files after error:', cleanupErr);
    }
  }
};
