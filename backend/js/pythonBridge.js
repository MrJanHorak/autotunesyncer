import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const runPythonProcessor = async (configPath) => {
  return new Promise((resolve, reject) => {
    // Change script path to compose_videos.py
    const pythonScript = path.join(__dirname, '../python/compose_videos.py');

    const process = spawn('python', [
      pythonScript,
      configPath, // Pass config path directly
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
      if (code !== 0) {
        reject(
          new Error(
            `Python process failed with code ${code}\nError: ${errorOutput}`
          )
        );
      } else {
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (e) {
          reject(
            new Error(
              `Failed to parse Python output: ${e.message}\nOutput: ${output}`
            )
          );
        }
      }
    });
  });
};

export const preprocessVideo = async (inputPath, outputPath, targetSize) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../python/preprocess_videos.py'),
      inputPath,
      outputPath,
      targetSize || ''
    ]);

    let stderr = '';
    pythonProcess.stderr.on('data', (data) => {
      stderr += data;
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python preprocessing error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Preprocessing failed: ${stderr}`));
      } else {
        resolve(outputPath);
      }
    });
  });
};
