/* eslint-disable no-unused-vars */
import ffmpeg from 'fluent-ffmpeg';
import pkg from '@tonejs/midi';
const { Midi } = pkg;
import { existsSync, mkdirSync, writeFileSync, rmSync } from 'fs';
import { Buffer } from 'buffer';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMP_DIR = join(__dirname, '../temp');
if (!existsSync(TEMP_DIR)) mkdirSync(TEMP_DIR);

// Convert MIDI ticks to seconds
function ticksToSeconds(ticks, midi) {
  const bpm = midi.header.tempos[0]?.bpm || 120; // Default to 120 BPM if not specified
  const ppq = midi.header.ppq; // Pulses (ticks) per quarter note
  return (ticks / ppq) * (60 / bpm);
}

export const composeVideo = async (req, res) => {
  const sessionId = uuidv4();
  const sessionDir = join(TEMP_DIR, sessionId);
  mkdirSync(sessionDir);

  try {
    const files = req.files;
    console.log(
      'Files received:',
      files.map((f) => ({
        fieldname: f.fieldname,
        mimetype: f.mimetype,
        size: f.size,
      }))
    );

    const midiFile = files.find((file) => file.fieldname === 'midiData');
    if (!midiFile) {
      throw new Error('MIDI data is missing');
    }

    let midiBuffer;
    // Check if the MIDI data is JSON
    if (midiFile.mimetype === 'application/json') {
      try {
        const midiJson = JSON.parse(midiFile.buffer.toString());
        // Convert JSON to MIDI buffer using @tonejs/midi
        const midi = new Midi();
        // Add tracks from JSON
        midiJson.tracks?.forEach((track) => {
          const newTrack = midi.addTrack();
          track.notes?.forEach((note) => {
            newTrack.addNote({
              midi: note.midi,
              time: note.time,
              duration: note.duration,
              velocity: note.velocity,
            });
          });
          if (track.instrument) {
            newTrack.instrument.name = track.instrument.name;
          }
        });
        midiBuffer = Buffer.from(midi.toArray());
      } catch (error) {
        console.error('Error parsing MIDI JSON:', error);
        throw new Error('Invalid MIDI JSON format');
      }
    } else {
      midiBuffer = midiFile.buffer;
    }

    // Validate MIDI buffer
    if (
      midiBuffer.length < 14 ||
      midiBuffer.toString('ascii', 0, 4) !== 'MThd'
    ) {
      throw new Error('Invalid MIDI file format');
    }

    const midi = new Midi(midiBuffer);
    console.log('MIDI parsed successfully:', {
      tracks: midi.tracks.length,
      duration: midi.duration,
      ppq: midi.header.ppq,
      bpm: midi.header.tempos[0]?.bpm || 120,
    });

    // Save videos to temp directory
    const videoFiles = {};
    const videos = files.filter((file) => file.fieldname.startsWith('videos'));
    videos.forEach((video) => {
      const instrument = video.fieldname.match(/\[(.*?)\]/)[1];
      const videoPath = join(sessionDir, `${instrument}.webm`);
      writeFileSync(videoPath, video.buffer);
      videoFiles[instrument] = videoPath;
    });

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

      // Generate video segments for each note
      const noteSegments = await Promise.all(
        track.notes.map(async (note, noteIndex) => {
          const segmentPath = join(
            sessionDir,
            `track${trackIndex}_note${noteIndex}.mp4`
          );

          // Convert ticks to seconds
          const startTimeSeconds = ticksToSeconds(note.ticks, midi);
          const durationSeconds = ticksToSeconds(note.durationTicks, midi);

          console.log(`Note ${noteIndex}:`, {
            ticks: note.ticks,
            durationTicks: note.durationTicks,
            startTime: startTimeSeconds,
            duration: durationSeconds,
          });

          await processNoteSegment(
            videoPath,
            {
              startTime: startTimeSeconds,
              duration: durationSeconds,
              velocity: note.velocity,
            },
            segmentPath
          );

          return {
            path: segmentPath,
            startTime: startTimeSeconds,
            duration: durationSeconds,
          };
        })
      );

      return {
        trackIndex,
        instrument,
        segments: noteSegments.filter(Boolean),
      };
    });

    const tracks = (await Promise.all(trackPromises)).filter(Boolean);

    // Compose final video
    const outputPath = join(sessionDir, 'output.mp4');
    await composeFinalVideo(tracks, outputPath);

    // Stream the result to client
    res.sendFile(outputPath, () => {
      // Cleanup temp files
      rmSync(sessionDir, { recursive: true, force: true });
    });
  } catch (error) {
    console.error('Composition error:', error);
    res.status(500).json({ error: error.message });
    // Cleanup on error
    rmSync(sessionDir, { recursive: true, force: true });
  }
};

async function processNoteSegment(videoPath, note, outputPath) {
  return new Promise((resolve, reject) => {
    const command = ffmpeg(videoPath)
      .setStartTime(0)
      .duration(note.duration)
      .size('320x240')
      .outputOptions([
        `-filter:v "setpts=${1 / note.duration}*PTS"`, // Adjust video speed to match note duration
        '-c:v libx264',
        '-preset ultrafast',
        '-c:a aac',
        // Scale volume based on note velocity
        `-filter:a "volume=${note.velocity}"`,
      ]);

    // Add debug logging
    command.on('start', (commandLine) => {
      console.log('FFmpeg command:', commandLine);
    });

    command
      .save(outputPath)
      .on('end', () => {
        console.log(`Processed note segment: ${outputPath}`);
        resolve();
      })
      .on('error', (err) => {
        console.error(`Error processing note segment: ${err.message}`);
        reject(err);
      });
  });
}

async function composeFinalVideo(tracks, outputPath) {
  return new Promise((resolve, reject) => {
    const command = ffmpeg();

    // Create filter complex string for combining videos
    let filterComplex = '';
    let overlayCount = 0;

    tracks.forEach((track, trackIndex) => {
      track.segments.forEach((segment, segmentIndex) => {
        const inputIndex = overlayCount * 2; // Account for both video and audio inputs
        command.input(segment.path);

        // Position video based on track index (grid layout)
        const x = (trackIndex % 3) * 320;
        const y = Math.floor(trackIndex / 3) * 240;

        filterComplex += `[${inputIndex}:v]setpts=PTS-STARTPTS[v${overlayCount}];`;
        filterComplex += `[${inputIndex}:a]asetpts=PTS-STARTPTS[a${overlayCount}];`;

        if (overlayCount === 0) {
          filterComplex += `[v${overlayCount}]pad=960:720[base];`;
        } else {
          filterComplex += `[base][v${overlayCount}]overlay=${x}:${y}:enable='between(t,${
            segment.startTime
          },${segment.startTime + segment.duration})'[base];`;
        }

        overlayCount++;
      });
    });

    // Add audio mixing
    const audioInputs = Array.from(
      { length: overlayCount },
      (_, i) => `[a${i}]`
    ).join('');
    filterComplex += `${audioInputs}amix=inputs=${overlayCount}:normalize=0[aout]`;

    command
      .complexFilter(filterComplex, ['base', 'aout'])
      .outputOptions([
        '-map',
        '[base]',
        '-map',
        '[aout]',
        '-c:v',
        'libx264',
        '-preset',
        'ultrafast',
        '-c:a',
        'aac',
      ])
      .on('start', (commandLine) => {
        console.log('FFmpeg final composition command:', commandLine);
      })
      .on('progress', (progress) => {
        console.log('Processing: ', progress.percent, '% done');
      })
      .save(outputPath)
      .on('end', resolve)
      .on('error', (err) => {
        console.error('FFmpeg error:', err);
        reject(err);
      });
  });
}
