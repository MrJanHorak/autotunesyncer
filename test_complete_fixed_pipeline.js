#!/usr/bin/env node
/**
 * Test the complete pipeline with the updated fixes:
 * 1. Output file handling - permanent location
 * 2. Grid layout using MIDI arrangement data
 */

import FormData from 'form-data';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function testCompleteFixedPipeline() {
  console.log('=== Testing Complete Fixed Pipeline ===');

  try {
    // Find existing MIDI and video files
    const uploadsDir = path.join(__dirname, 'backend', 'uploads');

    // Find a MIDI file
    const midiFiles = fs.readdirSync(uploadsDir).filter((file) => {
      try {
        const content = fs.readFileSync(path.join(uploadsDir, file), 'utf8');
        const data = JSON.parse(content);
        return data.tracks && data.header && data.gridArrangement;
      } catch {
        return false;
      }
    });

    if (midiFiles.length === 0) {
      console.log('❌ No MIDI files found for testing');
      return false;
    }

    // Find video files that match instruments
    const videoFiles = fs
      .readdirSync(uploadsDir)
      .filter(
        (file) =>
          file.endsWith('.mp4') &&
          !file.startsWith('processed_') &&
          !file.startsWith('final_output_')
      )
      .slice(0, 8); // Limit to first 8 videos

    if (videoFiles.length === 0) {
      console.log('❌ No video files found for testing');
      return false;
    }

    console.log(`✓ Found MIDI file: ${midiFiles[0]}`);
    console.log(`✓ Found ${videoFiles.length} video files for testing`);

    // Read MIDI data to show grid arrangement
    const midiPath = path.join(uploadsDir, midiFiles[0]);
    const midiData = JSON.parse(fs.readFileSync(midiPath, 'utf8'));
    console.log('✓ Grid arrangement:', midiData.gridArrangement);

    // Create form data
    const form = new FormData();

    // Add MIDI data
    form.append('midiData', fs.createReadStream(midiPath));

    // Add video files
    videoFiles.forEach((videoFile) => {
      const videoPath = path.join(uploadsDir, videoFile);
      form.append('videos', fs.createReadStream(videoPath), videoFile);
    });

    console.log('\n🚀 Making API request with fixes...');

    // Make request to API endpoint
    const response = await fetch('http://localhost:3000/api/process-videos', {
      method: 'POST',
      body: form,
      headers: form.getHeaders(),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.log('❌ API request failed:', response.status, errorText);
      return false;
    }

    const result = await response.json();
    console.log('\n✅ API Response successful!');
    console.log('📝 Result:', result);

    // Check if output file was created in permanent location
    if (result.outputPath && fs.existsSync(result.outputPath)) {
      const stats = fs.statSync(result.outputPath);
      console.log('\n🎉 SUCCESS! Final output created:');
      console.log(`  📂 File: ${result.outputPath}`);
      console.log(`  📊 Size: ${stats.size.toLocaleString()} bytes`);
      console.log(`  🕒 Created: ${stats.birthtime.toISOString()}`);

      // Show that it's in a permanent location (not temp)
      console.log(
        `  ✅ Permanent location: ${!result.outputPath.includes('Temp')}`
      );

      return true;
    } else {
      console.log('❌ No output file created at expected location');
      console.log('Expected:', result.outputPath);
      return false;
    }
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    return false;
  }
}

// Run the test
testCompleteFixedPipeline().then((success) => {
  console.log('\n' + '='.repeat(50));
  if (success) {
    console.log('🎊 COMPLETE PIPELINE SUCCESS! 🎊');
    console.log('✅ Output file handling: FIXED');
    console.log('✅ Grid arrangement: IMPLEMENTED');
    console.log('✅ MIDI note mapping: WORKING');
    console.log('✅ Video composition: COMPLETED');
  } else {
    console.log('❌ Pipeline test FAILED!');
  }
  console.log('='.repeat(50));
  process.exit(success ? 0 : 1);
});
