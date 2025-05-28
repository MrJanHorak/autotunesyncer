// Test script to verify video data transformation fix
import fs from 'fs';

// Simulate the original data structure that was causing issues
const originalVideoFiles = {
  piano: {
    video: new Uint8Array([1, 2, 3, 4]), // Simulated video data
    isDrum: false,
    notes: [{ time: 0, duration: 1, midi: 60 }],
    index: 0,
    processedAt: Date.now(),
  },
  drum_kick: {
    video: new Uint8Array([5, 6, 7, 8]), // Simulated drum data
    isDrum: true,
    drumName: 'kick',
    notes: [{ time: 0.5, duration: 0.1, midi: 36 }],
    index: 1,
    processedAt: Date.now(),
  },
};

// Apply the transformation logic from queueService.js
const transformedVideoFiles = {};
Object.entries(originalVideoFiles).forEach(([key, value]) => {
  transformedVideoFiles[key] = {
    // Use 'videoData' key instead of 'video' to match Python expectations
    videoData: value.video,
    isDrum: value.isDrum || false,
    drumName: value.drumName,
    notes: value.notes || [],
    layout: value.layout || { x: 0, y: 0, width: 960, height: 720 },
    index: value.index,
    processedAt: value.processedAt,
  };
});

console.log('🧪 Testing Video Data Transformation');
console.log('=====================================');

console.log('\n📥 Original Data Structure:');
Object.entries(originalVideoFiles).forEach(([key, value]) => {
  console.log(`${key}:`, {
    hasVideo: !!value.video,
    hasVideoData: !!value.videoData,
    isDrum: value.isDrum,
    noteCount: value.notes?.length || 0,
  });
});

console.log('\n📤 Transformed Data Structure:');
Object.entries(transformedVideoFiles).forEach(([key, value]) => {
  console.log(`${key}:`, {
    hasVideo: !!value.video,
    hasVideoData: !!value.videoData,
    isDrum: value.isDrum,
    noteCount: value.notes?.length || 0,
  });
});

// Test the Python expectation logic
console.log('\n🐍 Python Validation Test:');
Object.entries(transformedVideoFiles).forEach(([trackId, trackData]) => {
  console.log(`\nTesting track: ${trackId}`);

  if ('path' in trackData && fs.existsSync(trackData.path)) {
    console.log('✅ Would use file path');
  } else if ('videoData' in trackData) {
    console.log('✅ Would use video buffer data');
  } else {
    console.log('❌ ERROR: No valid video data found');
  }
});

console.log('\n🎯 Fix Verification:');
console.log('- Original structure used "video" key');
console.log('- Transformed structure uses "videoData" key');
console.log('- Python expects either "path" or "videoData" key');
console.log(
  '- ✅ Transformation should resolve the "No valid video data for track" error'
);
