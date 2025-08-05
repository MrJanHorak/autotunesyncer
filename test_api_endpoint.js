#!/usr/bin/env node
/**
 * Test the actual API endpoint to validate the complete pipeline
 * including MIDI note mapping and video composition.
 */

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const fetch = require('node-fetch');

// Start the server in a child process
const { spawn } = require('child_process');

async function testApiEndpoint() {
  console.log('=== API Endpoint Pipeline Test ===');

  try {
    // Check if server is already running or start it
    const serverProcess = spawn('node', ['backend/server.js'], {
      cwd: __dirname,
      stdio: 'pipe',
    });

    // Wait a moment for server to start
    await new Promise((resolve) => setTimeout(resolve, 3000));

    // Find existing MIDI and video files
    const uploadsDir = path.join(__dirname, 'backend', 'uploads');

    // Find a MIDI file
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
      console.log('❌ No MIDI files found for testing');
      return false;
    }

    // Find video files
    const videoFiles = fs
      .readdirSync(uploadsDir)
      .filter((file) => file.endsWith('.mp4'))
      .slice(0, 4); // Limit to first 4 videos

    if (videoFiles.length === 0) {
      console.log('❌ No video files found for testing');
      return false;
    }

    console.log(`✓ Found MIDI file: ${midiFiles[0]}`);
    console.log(`✓ Found ${videoFiles.length} video files`);

    // Create form data
    const form = new FormData();

    // Add MIDI data
    const midiPath = path.join(uploadsDir, midiFiles[0]);
    form.append('midiData', fs.createReadStream(midiPath));

    // Add video files
    videoFiles.forEach((videoFile) => {
      const videoPath = path.join(uploadsDir, videoFile);
      form.append('videos', fs.createReadStream(videoPath), videoFile);
    });

    console.log('🚀 Making API request...');

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
    console.log('✓ API Response:', result);

    // Check if output file was created
    if (result.outputPath && fs.existsSync(result.outputPath)) {
      console.log('✓ Output video created successfully!');
      console.log(`  File: ${result.outputPath}`);
      console.log(`  Size: ${fs.statSync(result.outputPath).size} bytes`);
      return true;
    } else {
      console.log('❌ No output file created');
      return false;
    }
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    return false;
  } finally {
    // Clean up server process
    if (serverProcess) {
      serverProcess.kill();
    }
  }
}

// Run the test
testApiEndpoint().then((success) => {
  console.log('\n=== Test Result ===');
  console.log(
    success ? '✅ Pipeline test PASSED!' : '❌ Pipeline test FAILED!'
  );
  process.exit(success ? 0 : 1);
});
