// pythonBridge.js
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const runPythonProcessor = async (midiTracks, videoFiles) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../python/audio_processor.py'),
      JSON.stringify({ tracks: midiTracks, videos: videoFiles })
    ]);

    let dataString = '';

    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}`));
        return;
      }
      resolve(JSON.parse(dataString));
    });
  });
};