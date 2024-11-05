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
    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
      console.log('Python output:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
      console.error('Python error:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(stdoutData);
      } else {
        reject(new Error(`Python process failed (code ${code}): ${stderrData}`));
      }
    });
  });
};

// Add debug logging helper
const logVideoInfo = async (filePath, label) => {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) {
        console.error(`Error probing ${label}:`, err);
        reject(err);
        return;
      }
      console.log(`\n${label} metadata:`, {
        format: metadata.format.format_name,
        duration: metadata.format.duration,
        size: metadata.format.size,
        streams: metadata.streams.map(s => ({
          codec_type: s.codec_type,
          codec_name: s.codec_name
        }))
      });
      resolve(metadata);
    });
  });
};

// Helper function to check if a Python module is installed
const checkPythonModule = async (moduleName) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', ['-c', `import ${moduleName}`]);
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(true);
      } else {
        reject(new Error(`Python module '${moduleName}' is not installed`));
      }
    });
  });
};

export const autotuneVideo = async (req, res) => {
  let tempDir = null;
  
  try {
    if (!req.file) {
      throw new Error('No video file provided');
    }

    // Check if numpy is installed
    await checkPythonModule('numpy');

    // Check if tensorflow is installed
    await checkPythonModule('tensorflow');

    // Check if librosa is installed
    await checkPythonModule('librosa');

    // Add check for 'crepe' module
    await checkPythonModule('crepe');

    // Check if pytsmod is installed
    await checkPythonModule('pytsmod');

    // Create temporary directory with unique ID
    tempDir = path.join(__dirname, '../temp', uuidv4());
    await fs.mkdir(tempDir, { recursive: true });

    const paths = {
      input: req.file.path,
      convertedInput: path.join(tempDir, 'converted-input.mp4'),
      inputAudio: path.join(tempDir, 'input-audio.wav'),
      autotunedAudio: path.join(tempDir, 'autotuned-audio.wav'),
      output: path.join(tempDir, 'autotuned-video.mp4')
    };

    // Log input video info
    await logVideoInfo(paths.input, 'Input video');

    // Convert input video to MP4 with H.264 codec
    await runFFmpeg(paths.input, paths.convertedInput, {
      outputOptions: [
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'aac',
        '-strict', 'experimental'
      ]
    });

    await logVideoInfo(paths.convertedInput, 'Converted input');

    // Extract audio with specific format
    console.log('Extracting audio...');
    await runFFmpeg(paths.convertedInput, paths.inputAudio, {
      format: 'wav',
      outputOptions: [
        '-ac', '2',  // force stereo
        '-ar', '44100',  // 44.1kHz sample rate
        '-acodec', 'pcm_s16le'  // ensure 16-bit PCM
      ]
    });

    // Run Python autotune script with better error handling
    console.log('Autotuning audio to middle C...');
    const pythonOutput = await runPythonScript(
      path.join(__dirname, '../python/autotune.py'),
      paths.inputAudio,
      paths.autotunedAudio
    );
    console.log('Python processing completed:', pythonOutput);

    // Verify autotuned audio
    await logVideoInfo(paths.autotunedAudio, 'Autotuned audio');

    // Combine video and autotuned audio with specific options
    console.log('Combining audio and video...');
    await runFFmpeg(paths.convertedInput, paths.output, {
      input: paths.autotunedAudio,
      outputOptions: [
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-ac', '2',  // ensure stereo output
        '-strict', 'experimental',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-async', '1'
      ]
    });

    // Verify final output
    await logVideoInfo(paths.output, 'Final output');

    // Read and send the processed video
    const processedVideo = await fs.readFile(paths.output);
    res.writeHead(200, {
      'Content-Type': 'video/mp4',
      'Content-Length': processedVideo.length,
      'Content-Disposition': 'inline'
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
    console.error('Detailed error in autotuneVideo:', error);
    
    // Clean up temp directory if it was created
    if (tempDir) {
      await fs.rm(tempDir, { recursive: true, force: true }).catch(console.error);
    }
    
    // Clean up any uploaded file if it exists
    if (req.file && req.file.path) {
      await fs.unlink(req.file.path).catch(cleanupError => {
        console.error('Error cleaning up uploaded file:', cleanupError);
      });
    }

    res.status(500).json({ 
      error: error.message || 'An error occurred while processing the video',
      details: error.stack
    });
  }
};