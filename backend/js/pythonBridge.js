import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Hard timeouts — long enough for real compositions, short enough to prevent infinite hangs.
const COMPOSITION_TIMEOUT_MS = 15 * 60 * 1000;  // 15 min
const PREPROCESS_TIMEOUT_MS  =  5 * 60 * 1000;  // 5 min

/**
 * Kill a process and its entire child process tree.
 * On Windows `process.kill()` only terminates the direct Python parent; FFmpeg
 * grandchildren keep running.  `taskkill /T` terminates the full tree.
 */
function killProcessTree(proc) {
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { stdio: 'ignore' });
    } else {
      proc.kill('SIGKILL');
    }
  } catch (_) {
    // best-effort — process may already be gone
  }
}

export const runPythonProcessor = async (configPath, { onProgress } = {}) => {
  return new Promise((resolve, reject) => {
    let midiJsonPath, videoJsonPath, outputPath;

    const cleanup = () => {
      try { if (midiJsonPath && fs.existsSync(midiJsonPath)) fs.unlinkSync(midiJsonPath); } catch (_) {}
      try { if (videoJsonPath && fs.existsSync(videoJsonPath)) fs.unlinkSync(videoJsonPath); } catch (_) {}
    };

    try {
      // Read the config file
      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

      // Create separate JSON files for MIDI data and video files
      const tempDir = path.dirname(configPath);
      const baseName = path.basename(configPath, '.json');

      midiJsonPath = path.join(tempDir, `${baseName}-midi.json`);
      videoJsonPath = path.join(tempDir, `${baseName}-videos.json`);
      outputPath = path.join(tempDir, `${baseName}-output.mp4`);

      const midiData = {
        tracks: config.tracks || [],
        gridArrangement: config.gridArrangement || {},
        trackVolumes: config.trackVolumes || {},
        compositionStyle: config.compositionStyle || {},
        clipStyles: config.clipStyles || {},
      };

      // Add validation to ensure grid arrangement is not empty
      if (
        !config.gridArrangement ||
        Object.keys(config.gridArrangement).length === 0
      ) {
        console.error(
          'Python Bridge - Grid arrangement is empty or missing:',
          config.gridArrangement
        );
        reject(new Error('Grid arrangement is required but was not provided'));
        return;
      }

      console.log('Python Bridge - Grid arrangement being sent:', JSON.stringify(config.gridArrangement, null, 2));
      console.log('Python Bridge - MIDI data structure:', Object.keys(midiData));
      console.log('Python Bridge - Grid arrangement validation passed:', Object.keys(config.gridArrangement).length, 'positions');

      fs.writeFileSync(midiJsonPath, JSON.stringify(midiData));
      fs.writeFileSync(videoJsonPath, JSON.stringify(config.videos || {}));

      // Use the enhanced video processor
      const pythonScript = path.join(__dirname, '../utils/video_processor.py');
      const pythonArgs = [
        pythonScript,
        '--midi-json', midiJsonPath,
        '--video-files-json', videoJsonPath,
        '--output-path', outputPath,
        '--performance-mode',
        '--memory-limit', '4',
      ];

      if (config.preview === true) {
        console.log('Python Bridge - Adding --preview flag for faster processing');
        pythonArgs.push('--preview');
      }

      const pythonProcess = spawn('python', pythonArgs);
      let output = '';
      let errorOutput = '';
      let settled = false;

      const finish = (resolveFn, rejectFn, value, isError) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutHandle);
        cleanup();
        if (isError) rejectFn(value);
        else resolveFn(value);
      };

      // Hard timeout — kill the entire process tree (not just the Python parent,
      // which would leave FFmpeg grandchildren running on Windows).
      const timeoutHandle = setTimeout(() => {
        console.error(`Python Bridge - composition timed out after ${COMPOSITION_TIMEOUT_MS / 60000} minutes`);
        killProcessTree(pythonProcess);
        finish(resolve, reject,
          new Error(`Composition timed out after ${COMPOSITION_TIMEOUT_MS / 60000} minutes.\nLast output:\n${errorOutput.slice(-2000)}`),
          true
        );
      }, COMPOSITION_TIMEOUT_MS);

      pythonProcess.stdout.on('data', (data) => {
        const message = data.toString();
        console.log(`Python output: ${message}`);
        output += message;
        const progressMatch = message.match(/PROGRESS:(\d+)/);
        if (progressMatch && onProgress) {
          onProgress(parseInt(progressMatch[1], 10));
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        const message = data.toString();
        console.error(`Python error: ${message}`);
        errorOutput += message;
      });

      pythonProcess.once('error', (err) => {
        finish(resolve, reject,
          new Error(`Failed to spawn Python process: ${err.message}\n${errorOutput.slice(-2000)}`),
          true
        );
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          finish(resolve, reject,
            new Error(`Python process failed (code ${code})\n${errorOutput.slice(-2000)}`),
            true
          );
        } else {
          finish(resolve, reject,
            { success: true, outputPath, message: output.trim() },
            false
          );
        }
      });

    } catch (error) {
      cleanup();
      reject(new Error(`Failed to setup Python processor: ${error.message}`));
    }
  });
};

export const preprocessVideo = async (
  inputPath,
  outputPath,
  targetSize,
  options = {}
) => {
  return new Promise((resolve, reject) => {
    const args = [
      path.join(__dirname, '../python/preprocess_videos.py'),
      inputPath,
      outputPath,
      targetSize || '',
    ];

    if (options.performanceMode !== false) args.push('--performance-mode');
    if (options.parallelTracks)  args.push('--parallel-tracks', options.parallelTracks.toString());
    if (options.memoryLimit)     args.push('--memory-limit', options.memoryLimit.toString());
    if (options.quality)         args.push('--quality', options.quality);

    const pythonProcess = spawn('python', args);
    let stdout = '';
    let stderr = '';
    let settled = false;

    const finish = (value, isError) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutHandle);
      if (isError) reject(value);
      else resolve(value);
    };

    const timeoutHandle = setTimeout(() => {
      console.error(`Python Bridge - preprocessVideo timed out after ${PREPROCESS_TIMEOUT_MS / 60000} minutes`);
      killProcessTree(pythonProcess);
      finish(new Error(`Preprocessing timed out after ${PREPROCESS_TIMEOUT_MS / 60000} minutes`), true);
    }, PREPROCESS_TIMEOUT_MS);

    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString();
      stdout += message;
      if (message.includes('PROGRESS:')) {
        const progressMatch = message.match(/PROGRESS:(\d+)/);
        if (progressMatch && options.onProgress) {
          options.onProgress(parseInt(progressMatch[1]));
        }
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const message = data.toString();
      stderr += message;
      console.error(`Python preprocessing error: ${message}`);
    });

    pythonProcess.once('error', (err) => {
      finish(new Error(`Failed to spawn preprocess: ${err.message}`), true);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        finish(new Error(`Preprocessing failed (code ${code}): ${stderr}`), true);
      } else {
        try {
          finish(JSON.parse(stdout), false);
        } catch {
          finish({ success: true, output: outputPath }, false);
        }
      }
    });
  });
};

export const preprocessVideoBatch = async (
  videoList,
  outputDir,
  options = {}
) => {
  return new Promise((resolve, reject) => {
    const batchConfig = { videos: videoList, output_dir: outputDir };
    const tempConfigPath = path.join(__dirname, '../temp', `batch_config_${Date.now()}.json`);

    const cleanupConfig = () => {
      try { fs.unlinkSync(tempConfigPath); } catch (_) {}
    };

    try {
      fs.writeFileSync(tempConfigPath, JSON.stringify(batchConfig, null, 2));

      const args = [
        path.join(__dirname, '../python/preprocess_videos.py'),
        '--batch-file', tempConfigPath,
      ];

      if (options.performanceMode !== false) args.push('--performance-mode');
      if (options.parallelTracks)  args.push('--parallel-tracks', options.parallelTracks.toString());
      if (options.memoryLimit)     args.push('--memory-limit', options.memoryLimit.toString());
      if (options.quality)         args.push('--quality', options.quality);

      const pythonProcess = spawn('python', args);
      let stdout = '';
      let stderr = '';
      let settled = false;

      const finish = (value, isError) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutHandle);
        cleanupConfig();
        if (isError) reject(value);
        else resolve(value);
      };

      const timeoutHandle = setTimeout(() => {
        console.error(`Python Bridge - preprocessVideoBatch timed out after ${PREPROCESS_TIMEOUT_MS / 60000} minutes`);
        killProcessTree(pythonProcess);
        finish(new Error(`Batch preprocessing timed out after ${PREPROCESS_TIMEOUT_MS / 60000} minutes`), true);
      }, PREPROCESS_TIMEOUT_MS);

      pythonProcess.stdout.on('data', (data) => {
        const message = data.toString();
        stdout += message;
        if (message.includes('PROGRESS:')) {
          const progressMatch = message.match(/PROGRESS:(\d+)/);
          if (progressMatch && options.onProgress) {
            options.onProgress(parseInt(progressMatch[1]));
          }
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        const message = data.toString();
        stderr += message;
        console.error(`Python batch preprocessing error: ${message}`);
      });

      pythonProcess.once('error', (err) => {
        finish(new Error(`Failed to spawn batch preprocess: ${err.message}`), true);
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          finish(new Error(`Batch preprocessing failed (code ${code}): ${stderr}`), true);
        } else {
          try {
            finish(JSON.parse(stdout), false);
          } catch (e) {
            finish(new Error(`Failed to parse batch results: ${e.message}`), true);
          }
        }
      });

    } catch (error) {
      cleanupConfig();
      reject(new Error(`Failed to create batch configuration: ${error.message}`));
    }
  });
};
