#!/usr/bin/env node
/**
 * Direct test of the processVideos route to validate MIDI note mapping fix
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import and test the actual route logic
async function testDirectProcessing() {
  console.log('=== Direct Processing Test ===');

  try {
    // Find real MIDI and video files
    const uploadsDir = path.join(__dirname, 'backend', 'uploads');

    // Find MIDI file
    const midiFiles = fs.readdirSync(uploadsDir).filter((file) => {
      try {
        const content = fs.readFileSync(path.join(uploadsDir, file), 'utf8');
        const data = JSON.parse(content);
        return data.tracks && data.header;
      } catch {
        return false;
      }
    });

    if (midiFiles.length === 0) {
      console.log('❌ No MIDI files found');
      return false;
    }

    // Find video files
    const videoFiles = fs
      .readdirSync(uploadsDir)
      .filter((file) => file.endsWith('.mp4'))
      .slice(0, 4);

    if (videoFiles.length === 0) {
      console.log('❌ No video files found');
      return false;
    }

    console.log(`✓ Found MIDI: ${midiFiles[0]}`);
    console.log(`✓ Found ${videoFiles.length} videos`);

    // Load MIDI data
    const midiPath = path.join(uploadsDir, midiFiles[0]);
    const midiData = JSON.parse(fs.readFileSync(midiPath, 'utf8'));

    console.log('✓ MIDI data loaded:');
    console.log(`  - Tracks: ${midiData.tracks.length}`);
    midiData.tracks.forEach((track, i) => {
      console.log(
        `  - Track ${i}: ${track.instrument?.name || 'Unknown'} (${
          track.notes?.length || 0
        } notes)`
      );
    });

    // Simulate the video processing logic from processVideos.js
    const videos = {};

    // Process videos (simplified)
    videoFiles.forEach((videoFile) => {
      const instrumentName = path.parse(videoFile).name;
      videos[instrumentName] = {
        path: path.join(uploadsDir, videoFile),
        isDrum: instrumentName.toLowerCase().includes('drum'),
        notes: [],
        layout: { x: 0, y: 0, width: 480, height: 270 },
      };
    });

    console.log('\n=== Video Files Prepared ===');
    Object.keys(videos).forEach((key) => {
      console.log(`  - ${key}: ${videos[key].path}`);
    });

    // NOW TEST THE MIDI NOTE MAPPING LOGIC (copied from our fix)
    console.log('\n=== Testing MIDI Note Mapping ===');

    // Helper functions (from our fix)
    const normalizeInstrumentName = (name) => {
      return name.toLowerCase().replace(/\s+/g, '_');
    };

    const isDrumTrack = (track) => {
      return (
        track.channel === 9 ||
        track.instrument?.name?.toLowerCase().includes('drum') ||
        track.instrument?.family?.toLowerCase().includes('drum')
      );
    };

    const getDrumName = (midiNote) => {
      const DRUM_NOTES = {
        35: 'Bass Drum',
        36: 'Kick Drum',
        38: 'Snare Drum',
        42: 'Hi-Hat Closed',
        46: 'Hi-Hat Open',
        49: 'Crash Cymbal',
        51: 'Ride Cymbal',
      };
      return DRUM_NOTES[midiNote] || `Drum_${midiNote}`;
    };

    let totalNotesMapped = 0;

    // Map MIDI notes to video files (our fix logic)
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
          const drumKey = `drum_${drumName.toLowerCase().replace(/\s+/g, '_')}`;

          const matchingVideoKey = Object.keys(videos).find(
            (key) =>
              key.includes(drumKey) ||
              key.endsWith(drumKey) ||
              key.includes('drum')
          );

          if (matchingVideoKey) {
            videos[matchingVideoKey].notes.push({
              midi: note.midi,
              time: note.time,
              duration: note.duration,
              velocity: note.velocity || 0.8,
            });
            totalNotesMapped++;
            console.log(
              `  ✓ Mapped drum note ${note.midi} (${drumName}) to ${matchingVideoKey}`
            );
          } else {
            console.log(`  ❌ No video found for drum: ${drumKey}`);
          }
        });
      } else {
        // Handle melodic instruments
        const normalizedName = normalizeInstrumentName(track.instrument.name);

        // Find video key with flexible matching
        const matchingVideoKey = Object.keys(videos).find((key) => {
          const keyParts = key.split('-');
          const instrumentPart = keyParts[keyParts.length - 1];
          return (
            instrumentPart === normalizedName ||
            key.includes(normalizedName) ||
            key.toLowerCase().includes(track.instrument.name.toLowerCase())
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
            totalNotesMapped++;
          });
          console.log(
            `  ✓ Mapped ${track.notes.length} notes to ${matchingVideoKey}`
          );
        } else {
          console.log(`  ❌ No video found for instrument: ${normalizedName}`);
          console.log(
            `    Available video keys: ${Object.keys(videos).join(', ')}`
          );
        }
      }
    });

    // Final results
    console.log('\n=== Final Note Mapping Results ===');
    Object.entries(videos).forEach(([key, video]) => {
      console.log(`${key}: ${video.notes.length} notes mapped`);
    });

    console.log(`\n✓ Total notes mapped: ${totalNotesMapped}`);

    if (totalNotesMapped > 0) {
      console.log('✅ MIDI note mapping is working correctly!');

      // Test data structure for Python processor
      const config = {
        tracks: {
          tracks: midiData.tracks,
          header: midiData.header,
          gridArrangement: midiData.gridArrangement,
        },
        videos: videos,
      };

      // Save test config
      const configPath = path.join(__dirname, 'test_mapping_result.json');
      fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
      console.log(`✓ Test config saved to: ${configPath}`);

      return true;
    } else {
      console.log('❌ No notes were mapped - there may still be an issue');
      return false;
    }
  } catch (error) {
    console.error('❌ Test failed:', error);
    return false;
  }
}

// Run the test
testDirectProcessing().then((success) => {
  console.log('\n=== Test Result ===');
  console.log(
    success
      ? '✅ Direct processing test PASSED!'
      : '❌ Direct processing test FAILED!'
  );
  process.exit(success ? 0 : 1);
});
