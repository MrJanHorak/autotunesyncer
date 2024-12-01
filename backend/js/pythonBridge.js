// // pythonBridge.js
// import { spawn } from 'child_process';
// import path from 'path';
// import { fileURLToPath } from 'url';

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// export const runPythonProcessor = async (midiTracks, videoFiles) => {
//   return new Promise((resolve, reject) => {
//     const pythonProcess = spawn('python', [
//       path.join(__dirname, '../python/audio_processor.py'),
//       JSON.stringify({ tracks: midiTracks, videos: videoFiles })
//     ]);

//     let dataString = '';

//     pythonProcess.stdout.on('data', (data) => {
//       dataString += data.toString();
//     });

//     pythonProcess.stderr.on('data', (data) => {
//       console.error(`Python Error: ${data}`);
//     });

//     pythonProcess.on('close', (code) => {
//       if (code !== 0) {
//         reject(new Error(`Process exited with code ${code}`));
//         return;
//       }
//       resolve(JSON.parse(dataString));
//     });
//   });
// };

// import { spawn } from 'child_process';
// import path from 'path';
// import { fileURLToPath } from 'url';

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// export async function runPythonProcessor(configPath) {
//   return new Promise((resolve, reject) => {
//     const pythonScript = path.join(
//       __dirname,
//       '..',
//       'python',
//       'audio_processor.py'
//     );
//     const process = spawn('python', [pythonScript, configPath]);

//     let output = '';

//     process.stdout.on('data', (data) => {
//       output += data;
//     });

//     process.stderr.on('data', (data) => {
//       console.error(`Python Error: ${data}`);
//     });

//     process.on('close', (code) => {
//       if (code !== 0) {
//         reject(new Error(`Python process exited with code ${code}`));
//       } else {
//         resolve(output);
//       }
//     });
//   });
// }

// import { spawn } from 'child_process';
// import path from 'path';
// import { fileURLToPath } from 'url';

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// export async function runPythonProcessor(midiData, videoMapping) {
//     return new Promise((resolve, reject) => {
//         // Use absolute path to Python script
//         const pythonScript = path.join(__dirname, '..', 'python', 'audio_processor.py');
        
//         // Create command line arguments as JSON strings
//         const midiArg = JSON.stringify(midiData);
//         const videoArg = JSON.stringify(videoMapping);
        
//         const process = spawn('python', [
//             pythonScript,
//             midiArg,
//             videoArg
//         ], {
//             // Increase buffer size for large arguments
//             maxBuffer: 1024 * 1024 * 10
//         });
// export async function runPythonProcessor(configPath) {
//   return new Promise((resolve, reject) => {

//     const pythonScript = path.join(__dirname, '..', 'python', 'audio_processor.py');
    
//     const process = spawn('python', [
//       pythonScript,
//       configPath
//     ]);

//     let output = '';

//         process.stdout.on('data', (data) => {
//             console.log(`Python output: ${data}`);
//         });

//         process.stderr.on('data', (data) => {
//             console.error(`Python error: ${data}`);
//         });

//         process.on('close', (code) => {
//             if (code !== 0) {
//                 reject(new Error(`Process exited with code ${code}`));
//             } else {
//                 resolve();
//             }
//         });
//     });
// }


import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function runPythonProcessor(configPath) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '..', 'python', 'audio_processor.py');
    
    // Read and verify config content
    const configContent = fs.readFileSync(configPath, 'utf8');
    console.log('Config file path:', configPath);
    console.log('Config content:', configContent);
    
    try {
      // Verify JSON is valid
      JSON.parse(configContent);
    } catch (e) {
      console.error('Invalid JSON in config:', e);
      reject(e);
      return;
    }

    const process = spawn('python', [
      pythonScript,
      `--config=${configPath}` // Pass as named argument
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
        reject(new Error(`Python process failed with code ${code}\nError: ${errorOutput}`));
      } else {
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${e.message}\nOutput: ${output}`));
        }
      }
    });
  });
}

// export async function runPythonProcessor(configPath) {
//   return new Promise((resolve, reject) => {
//     const pythonScript = path.join(__dirname, '..', 'python', 'audio_processor.py');
//     console.log('Config file contents:', fs.readFileSync(configPath, 'utf8'));
//     const process = spawn('python', [
//       pythonScript,
//       configPath
//     ]);

//     let output = '';
//     let errorOutput = '';

//     // Collect stdout data
//     process.stdout.on('data', (data) => {
//       const message = data.toString();
//       console.log(`Python output: ${message}`);
//       output += message;
//     });

//     // Collect stderr data
//     process.stderr.on('data', (data) => {
//       const message = data.toString();
//       console.error(`Python error: ${message}`);
//       errorOutput += message;
//     });

//     // Handle process completion
//     process.on('close', (code) => {
//       if (code !== 0) {
//         reject(new Error(`Python process failed with code ${code}\nError: ${errorOutput}`));
//       } else {
//         resolve({
//           success: true,
//           output: output,
//           configPath: configPath
//         });
//       }
//     });

//     // Handle process errors
//     process.on('error', (error) => {
//       reject(new Error(`Failed to start Python process: ${error.message}`));
//     });
//   });
// }
