import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const runPythonProcessor = async (configPath) => {
  return new Promise((resolve, reject) => {
    let midiJsonPath, videoJsonPath, outputPath;

    try {
      // Read the config file
      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

      // Create separate JSON files for MIDI data and video files
      const tempDir = path.dirname(configPath);
      const baseName = path.basename(configPath, '.json');

      midiJsonPath = path.join(tempDir, `${baseName}-midi.json`);
      videoJsonPath = path.join(tempDir, `${baseName}-videos.json`);
      outputPath = path.join(tempDir, `${baseName}-output.mp4`); // Write MIDI data - ensure proper format with tracks wrapper and grid arrangement
      const midiData = {
        tracks: config.tracks || [],
        gridArrangement: config.gridArrangement || {},
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

      console.log(
        'Python Bridge - Grid arrangement being sent:',
        JSON.stringify(config.gridArrangement, null, 2)
      );
      console.log(
        'Python Bridge - MIDI data structure:',
        Object.keys(midiData)
      );
      console.log(
        'Python Bridge - Grid arrangement validation passed:',
        Object.keys(config.gridArrangement).length,
        'positions'
      );

      fs.writeFileSync(midiJsonPath, JSON.stringify(midiData));

      // Write video files data - ensure proper format
      const videoFiles = config.videos || {};
      fs.writeFileSync(videoJsonPath, JSON.stringify(videoFiles));

      // Use the enhanced video processor
      const pythonScript = path.join(__dirname, '../utils/video_processor.py');
      const process = spawn('python', [
        pythonScript,
        '--midi-json',
        midiJsonPath,
        '--video-files-json',
        videoJsonPath,
        '--output-path',
        outputPath,
        '--performance-mode',
        '--memory-limit',
        '4',
      ]);

      let output = '';
      let errorOutput = '';

      process.stdout.on('data', (data) => {
        const message = data.toString();
        console.log(`Python output: ${message}`);
        output += message;
      });

      process.stderr.on('data', (data) => {
        const message = data.toString();
        console.error(`Python error: ${message}`);
        errorOutput += message;
      });

      process.on('close', (code) => {
        // Cleanup temporary files
        try {
          if (fs.existsSync(midiJsonPath)) fs.unlinkSync(midiJsonPath);
          if (fs.existsSync(videoJsonPath)) fs.unlinkSync(videoJsonPath);
        } catch (cleanupError) {
          console.warn('Cleanup error:', cleanupError.message);
        }

        if (code !== 0) {
          reject(
            new Error(
              `Python process failed with code ${code}\nError: ${errorOutput}`
            )
          );
        } else {
          try {
            // Return the output path instead of parsing JSON
            resolve({
              success: true,
              outputPath: outputPath,
              message: output.trim(),
            });
          } catch (e) {
            reject(
              new Error(
                `Failed to process Python output: ${e.message}\nOutput: ${output}`
              )
            );
          }
        }
      });
    } catch (error) {
      // Cleanup on error
      try {
        if (midiJsonPath && fs.existsSync(midiJsonPath))
          fs.unlinkSync(midiJsonPath);
        if (videoJsonPath && fs.existsSync(videoJsonPath))
          fs.unlinkSync(videoJsonPath);
      } catch (cleanupError) {
        console.warn('Cleanup error:', cleanupError.message);
      }
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

    // Add performance optimization flags
    if (options.performanceMode !== false) {
      args.push('--performance-mode');
    }

    if (options.parallelTracks) {
      args.push('--parallel-tracks', options.parallelTracks.toString());
    }

    if (options.memoryLimit) {
      args.push('--memory-limit', options.memoryLimit.toString());
    }

    if (options.quality) {
      args.push('--quality', options.quality);
    }

    const pythonProcess = spawn('python', args);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString();
      stdout += message;

      // Check for progress updates
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

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Preprocessing failed with code ${code}: ${stderr}`));
      } else {
        try {
          // Try to parse JSON output for enhanced results
          const result = JSON.parse(stdout);
          resolve(result);
        } catch {
          // Fallback to simple success for legacy compatibility
          resolve({ success: true, output: outputPath });
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
    // Create batch configuration file
    const batchConfig = {
      videos: videoList,
      output_dir: outputDir,
    };

    const tempConfigPath = path.join(
      __dirname,
      '../temp',
      `batch_config_${Date.now()}.json`
    );

    try {
      // Write batch configuration
      fs.writeFileSync(tempConfigPath, JSON.stringify(batchConfig, null, 2));

      const args = [
        path.join(__dirname, '../python/preprocess_videos.py'),
        '--batch-file',
        tempConfigPath,
      ];

      // Add performance optimization flags
      if (options.performanceMode !== false) {
        args.push('--performance-mode');
      }

      if (options.parallelTracks) {
        args.push('--parallel-tracks', options.parallelTracks.toString());
      }

      if (options.memoryLimit) {
        args.push('--memory-limit', options.memoryLimit.toString());
      }

      if (options.quality) {
        args.push('--quality', options.quality);
      }

      const pythonProcess = spawn('python', args);

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        const message = data.toString();
        stdout += message;

        // Check for progress updates
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

      pythonProcess.on('close', (code) => {
        // Cleanup temp config file
        try {
          fs.unlinkSync(tempConfigPath);
        } catch (cleanupError) {
          console.warn(
            `Failed to cleanup temp config file: ${cleanupError.message}`
          );
        }

        if (code !== 0) {
          reject(
            new Error(`Batch preprocessing failed with code ${code}: ${stderr}`)
          );
        } else {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (e) {
            reject(
              new Error(
                `Failed to parse batch processing results: ${e.message}`
              )
            );
          }
        }
      });
    } catch (error) {
      reject(
        new Error(`Failed to create batch configuration: ${error.message}`)
      );
    }
  });
};
