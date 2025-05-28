import ffmpeg from 'fluent-ffmpeg';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import crypto from 'crypto';
import os from 'os';
import cacheService from '../services/cacheService.js';

class VideoProcessor {
  constructor() {
    this.concurrentJobs = Math.min(4, os.cpus().length);
    this.activeJobs = new Set();
    this.jobQueue = [];
  }

  // Generate hash for video caching
  async generateVideoHash(videoBuffer, options = {}) {
    const hash = crypto.createHash('sha256');
    hash.update(videoBuffer);
    hash.update(JSON.stringify(options));
    return hash.digest('hex');
  }

  // Optimized FFmpeg command with hardware acceleration detection
  async getOptimalFFmpegSettings() {
    try {
      const hasNvidiaGPU = await this.detectHardwareAcceleration();

      if (hasNvidiaGPU) {
        return {
          inputOptions: ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'],
          outputOptions: [
            '-c:v',
            'h264_nvenc',
            '-preset',
            'p2',
            '-tune',
            'fastdecode',
            '-profile:v',
            'high',
            '-level',
            '4.1',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
          ],
        };
      } else {
        return {
          inputOptions: ['-threads', '0'],
          outputOptions: [
            '-c:v',
            'libx264',
            '-preset',
            'ultrafast',
            '-profile:v',
            'high',
            '-level',
            '4.1',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
          ],
        };
      }
    } catch {
      console.warn(
        'Hardware acceleration detection failed, using CPU fallback'
      );
      return this.getCPUFallbackSettings();
    }
  }

  getCPUFallbackSettings() {
    return {
      inputOptions: ['-threads', '0'],
      outputOptions: [
        '-c:v',
        'libx264',
        '-preset',
        'fast',
        '-crf',
        '23',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
      ],
    };
  }

  async detectHardwareAcceleration() {
    return new Promise((resolve) => {
      const ffmpegProcess = spawn('ffmpeg', ['-encoders']);
      let output = '';

      ffmpegProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      ffmpegProcess.on('close', () => {
        resolve(output.includes('h264_nvenc') || output.includes('h264_qsv'));
      });

      // Timeout after 5 seconds
      setTimeout(() => {
        ffmpegProcess.kill();
        resolve(false);
      }, 5000);
    });
  }

  // Batch process videos in parallel
  async processVideosBatch(videos, processFunction, maxConcurrency = 4) {
    const results = [];
    const semaphore = this.createSemaphore(maxConcurrency);

    const processVideo = async (video, index) => {
      await semaphore.acquire();
      try {
        const result = await processFunction(video, index);
        results[index] = result;
      } finally {
        semaphore.release();
      }
    };

    const promises = videos.map((video, index) => processVideo(video, index));
    await Promise.all(promises);

    return results;
  }

  createSemaphore(maxConcurrency) {
    let running = 0;
    const queue = [];

    const acquire = () => {
      return new Promise((resolve) => {
        if (running < maxConcurrency) {
          running++;
          resolve();
        } else {
          queue.push(resolve);
        }
      });
    };

    const release = () => {
      running--;
      if (queue.length > 0) {
        const next = queue.shift();
        running++;
        next();
      }
    };

    return { acquire, release };
  }

  // Stream video processing with progress tracking
  async processVideoStream(
    inputPath,
    outputPath,
    options = {},
    onProgress = null
  ) {
    const startTime = Date.now();
    const settings = await this.getOptimalFFmpegSettings();

    return new Promise((resolve, reject) => {
      const command = ffmpeg(inputPath);

      // Apply optimal settings
      if (settings.inputOptions) {
        command.inputOptions(settings.inputOptions);
      }

      command.outputOptions([
        ...settings.outputOptions,
        ...Object.entries(options)
          .map(([key, value]) => [key, value])
          .flat(),
      ]);

      // Progress tracking
      command.on('progress', (progress) => {
        if (onProgress) {
          onProgress({
            percent: progress.percent,
            currentTime: progress.timemark,
            fps: progress.currentFps,
          });
        }
      });

      command.on('start', (cmdline) => {
        console.log('FFmpeg command:', cmdline);
      });

      command.on('end', () => {
        const duration = Date.now() - startTime;
        console.log(`Video processing completed in ${duration}ms`);
        resolve(outputPath);
      });

      command.on('error', (err) => {
        console.error('FFmpeg error:', err);
        reject(err);
      });

      command.output(outputPath).run();
    });
  }

  // Memory-efficient video conversion
  async convertVideoFormat(inputPath, outputPath, targetFormat = 'mp4') {
    const videoHash = await this.generateVideoHash(
      await fs.readFile(inputPath),
      { format: targetFormat }
    );

    // Check cache first
    const cachedVideo = await cacheService.getCachedVideo(videoHash);
    if (cachedVideo) {
      await fs.writeFile(outputPath, cachedVideo);
      return outputPath;
    }

    // Process video
    const result = await this.processVideoStream(inputPath, outputPath, {
      '-c:a': 'aac',
      '-ar': '44100',
      '-ac': '2',
      '-b:a': '192k',
    });

    // Cache result
    const processedBuffer = await fs.readFile(result);
    await cacheService.cacheProcessedVideo(videoHash, processedBuffer);

    return result;
  }

  // Optimize video for web delivery
  async optimizeForWeb(inputPath, outputPath, quality = 'medium') {
    const qualitySettings = {
      low: { crf: 28, preset: 'faster', scale: '854:480' },
      medium: { crf: 23, preset: 'fast', scale: '1280:720' },
      high: { crf: 20, preset: 'slow', scale: '1920:1080' },
    };

    const settings = qualitySettings[quality] || qualitySettings.medium;

    return this.processVideoStream(inputPath, outputPath, {
      '-crf': settings.crf,
      '-preset': settings.preset,
      '-vf': `scale=${settings.scale}:force_original_aspect_ratio=decrease`,
      '-c:a': 'aac',
      '-b:a': '128k',
    });
  }

  // Extract audio with optimizations
  async extractAudio(videoPath, outputPath, format = 'wav') {
    const audioSettings = {
      wav: {
        '-c:a': 'pcm_s16le',
        '-ar': '44100',
        '-ac': '2',
      },
      mp3: {
        '-c:a': 'libmp3lame',
        '-ar': '44100',
        '-ac': '2',
        '-b:a': '192k',
      },
      aac: {
        '-c:a': 'aac',
        '-ar': '44100',
        '-ac': '2',
        '-b:a': '192k',
      },
    };

    const settings = audioSettings[format] || audioSettings.wav;

    return new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .outputOptions(Object.entries(settings).flat())
        .output(outputPath)
        .on('end', () => resolve(outputPath))
        .on('error', reject)
        .run();
    });
  }

  // Combine multiple videos with audio synchronization
  async combineVideos(videoPaths, outputPath, options = {}) {
    const command = ffmpeg();

    // Add all input videos
    videoPaths.forEach((videoPath) => {
      command.input(videoPath);
    });

    // Configure filter complex for combining
    const filterComplex = this.generateCombineFilter(
      videoPaths.length,
      options
    );
    command.complexFilter(filterComplex);

    // Output settings
    const settings = await this.getOptimalFFmpegSettings();
    command.outputOptions([
      ...settings.outputOptions,
      '-map',
      '[v]',
      '-map',
      '[a]',
    ]);

    return new Promise((resolve, reject) => {
      command
        .output(outputPath)
        .on('end', () => resolve(outputPath))
        .on('error', reject)
        .run();
    });
  }

  generateCombineFilter(videoCount, options = {}) {
    const { rows = 2, cols = 2 } = options;

    // Generate video scaling and positioning filters
    const videoFilters = [];
    const audioInputs = [];
    for (let i = 0; i < videoCount; i++) {
      videoFilters.push(`[${i}:v]scale=${1920 / cols}:${1080 / rows}[v${i}]`);
      audioInputs.push(`[${i}:a]`);
    }

    // Generate overlay filters
    let overlayChain = '[v0]';
    for (let i = 1; i < videoCount; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const x = col * (1920 / cols);
      const y = row * (1080 / rows);

      if (i === videoCount - 1) {
        overlayChain += `[v${i}]overlay=${x}:${y}[v]`;
      } else {
        overlayChain += `[v${i}]overlay=${x}:${y}[tmp${i}];[tmp${i}]`;
      }
    }

    // Audio mixing
    const audioMix = `${audioInputs.join(
      ''
    )}amix=inputs=${videoCount}:duration=longest[a]`;

    return `${videoFilters.join(';')};${overlayChain};${audioMix}`;
  }

  // Performance monitoring
  async getPerformanceMetrics() {
    const metrics = await cacheService.getMetrics('video-processing');

    if (metrics.length === 0) {
      return {
        averageProcessingTime: 0,
        totalJobs: 0,
        successRate: 0,
      };
    }

    const totalTime = metrics.reduce((sum, m) => sum + m.duration, 0);
    const averageTime = totalTime / metrics.length;

    return {
      averageProcessingTime: averageTime,
      totalJobs: metrics.length,
      successRate: 100, // Simplified - would need failure tracking
      metrics: metrics.slice(-10), // Last 10 metrics
    };
  }
}

export default new VideoProcessor();
