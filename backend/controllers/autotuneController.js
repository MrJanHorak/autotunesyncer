import ffmpeg from 'fluent-ffmpeg';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper function for ffmpeg operations
const runFFmpeg = async (input, output, options = {}) => {
  return new Promise((resolve, reject) => {
    const command = ffmpeg(input);
    
    if (options.format) {
      command.toFormat(options.format);
    }
    
    if (options.outputOptions) {
      command.outputOptions(options.outputOptions);
    }
    
    if (options.input) {
      command.input(options.input);
    }
    
    command
      .output(output)
      .on('end', resolve)
      .on('error', reject)
      .run();
  });
};

// Helper function to run Python script
const runPythonScript = async (scriptPath, ...args) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, ...args]);

    pythonProcess.stdout.on('data', (data) => {
      console.log('Python output:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('Python error:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Python process exited with code ${code}`));
    });
  });
};

export const autotuneVideo = async (req, res) => {
  try {
    if (!req.file) {
      throw new Error('No video file provided');
    }

    // Create temporary directory with unique ID
    const tempDir = path.join(__dirname, '../temp', uuidv4());
    await fs.mkdir(tempDir, { recursive: true });

    const paths = {
      input: req.file.path,
      convertedInput: path.join(tempDir, 'converted-input.mp4'),
      inputAudio: path.join(tempDir, 'input-audio.wav'),
      autotunedAudio: path.join(tempDir, 'autotuned-audio.wav'),
      output: path.join(tempDir, 'autotuned-video.mp4')
    };

    // Convert input video to MP4 with H.264 codec
    await runFFmpeg(paths.input, paths.convertedInput, {
      outputOptions: [
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'aac'
      ]
    });

    // Extract audio
    console.log('Extracting audio...');
    await runFFmpeg(paths.convertedInput, paths.inputAudio, {
      format: 'wav'
    });

    // Run Python autotune script
    console.log('Autotuning audio...');
    await runPythonScript(
      path.join(__dirname, '../python/autotune.py'),
      paths.inputAudio,
      paths.autotunedAudio
    );

    // Combine video and autotuned audio
    console.log('Combining audio and video...');
    await runFFmpeg(paths.convertedInput, paths.output, {
      input: paths.autotunedAudio,
      outputOptions: [
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0'
      ]
    });

    // Read and send the processed video
    const processedVideo = await fs.readFile(paths.output);
    res.writeHead(200, {
      'Content-Type': 'video/mp4',
      'Content-Length': processedVideo.length
    });
    res.end(processedVideo);

    // Clean up
    await Promise.all([
      fs.rm(tempDir, { recursive: true }),
      fs.unlink(paths.input)
    ]).catch(error => {
      console.error('Cleanup error:', error);
      // Don't throw the error as the video was already sent to the client
    });

  } catch (error) {
    console.error('Error in autotuneVideo controller:', error);
    
    // Clean up any uploaded file if it exists
    if (req.file && req.file.path) {
      await fs.unlink(req.file.path).catch(cleanupError => {
        console.error('Error cleaning up uploaded file:', cleanupError);
      });
    }

    res.status(500).json({ 
      error: error.message || 'An error occurred while processing the video' 
    });
  }
};