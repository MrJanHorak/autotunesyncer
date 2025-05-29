import Queue from 'bull';
import process from 'process';
import cacheService from './cacheService.js';

// Initialize queue with Redis connection
const videoProcessingQueue = new Queue('video processing', {
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
  },
  defaultJobOptions: {
    removeOnComplete: 10,
    removeOnFail: 5,
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
  },
});

// Process video composition jobs with enhanced error handling and progress tracking
videoProcessingQueue.process('compose', 4, async (job) => {
  const {
    midiData,
    processedTracks,
    processedDrums,
    sessionId,
    outputPath,
    performanceMetrics,
  } = job.data;

  try {
    console.log(`Starting composition job ${job.id} for session ${sessionId}`);

    // Update job progress with detailed status
    await job.progress(10, {
      stage: 'initialization',
      message: 'Preparing video composition...',
    });

    // Import the enhanced processor
    const { spawn } = await import('child_process');
    const { join } = await import('path');
    const { existsSync, writeFileSync } = await import('fs');

    // Prepare data for Python processor
    const tempDir = join(process.cwd(), 'backend', 'temp');
    const midiJsonPath = join(tempDir, `midi_${sessionId}.json`);
    const videoFilesJsonPath = join(tempDir, `videos_${sessionId}.json`); // Combine processed tracks and drums
    const allVideoFiles = { ...processedTracks, ...processedDrums };

    // Transform video files to match expected Python format
    const transformedVideoFiles = {};
    Object.entries(allVideoFiles).forEach(([key, value]) => {
      transformedVideoFiles[key] = {
        // Use 'videoData' key instead of 'video' to match Python expectations
        videoData: value.video,
        isDrum: value.isDrum || false,
        drumName: value.drumName,
        notes: value.notes || [],
        layout: value.layout || {
          x: 0,
          y: 0,
          width: 960,
          height: 720,
        },
        index: value.index,
        processedAt: value.processedAt,
      };
    });

    await job.progress(20, {
      stage: 'data_preparation',
      message: 'Writing processing data...',
    });

    // Write enhanced data files
    writeFileSync(
      midiJsonPath,
      JSON.stringify(
        {
          ...midiData,
          processingMetadata: {
            sessionId,
            timestamp: Date.now(),
            trackCount: Object.keys(transformedVideoFiles).length,
            performanceMetrics,
          },
        },
        null,
        2
      )
    );

    writeFileSync(
      videoFilesJsonPath,
      JSON.stringify(transformedVideoFiles, null, 2)
    );

    await job.progress(30, {
      stage: 'processing_start',
      message: 'Starting video processing...',
    }); // Launch enhanced Python processor
    const pythonArgs = [
      join(process.cwd(), 'backend', 'utils', 'video_processor.py'),
      '--midi-json',
      midiJsonPath,
      '--video-files-json',
      videoFilesJsonPath,
      '--output-path',
      outputPath,
      '--performance-mode',
      '--memory-limit',
      '4',
    ];

    const pythonProcess = spawn('python', pythonArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONPATH: join(process.cwd(), 'backend'),
      },
    });
    let lastProgress = 30;

    // Monitor Python process output for progress updates
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();

      // Parse progress updates
      const progressMatches = output.match(/PROGRESS:(\d+)/g);
      if (progressMatches) {
        const latestProgress = parseInt(
          progressMatches[progressMatches.length - 1].split(':')[1]
        );
        if (latestProgress > lastProgress) {
          lastProgress = latestProgress;
          const adjustedProgress = Math.min(95, 30 + latestProgress * 0.65); // Scale to 30-95%
          job
            .progress(adjustedProgress, {
              stage: 'video_processing',
              message: `Processing videos: ${latestProgress}%`,
              pythonProgress: latestProgress,
            })
            .catch(console.error);
        }
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const errorOutput = data.toString();
      if (!errorOutput.includes('WARNING') && !errorOutput.includes('INFO')) {
        console.error(`Python stderr: ${errorOutput}`);
      }
    });
    // Wait for Python process to complete
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pythonProcess.kill('SIGKILL');
        reject(new Error('Video processing timeout after 10 minutes'));
      }, 10 * 60 * 1000); // 10 minute timeout

      pythonProcess.on('close', (code) => {
        clearTimeout(timeout);
        if (code === 0) {
          resolve({ success: true, outputPath });
        } else {
          reject(new Error(`Python process failed with code ${code}`));
        }
      });

      pythonProcess.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    });

    await job.progress(95, {
      stage: 'validation',
      message: 'Validating output...',
    });

    // Validate output file
    if (!existsSync(outputPath)) {
      throw new Error('Output video file was not created');
    }

    const { statSync } = await import('fs');
    const stats = statSync(outputPath);
    if (stats.size === 0) {
      throw new Error('Output video file is empty');
    }

    await job.progress(100, {
      stage: 'completed',
      message: 'Video composition completed successfully',
      outputSize: stats.size,
    });

    // Cache result with extended metadata
    const cacheKey = `composition:${sessionId}`;
    await cacheService.set(
      cacheKey,
      {
        outputPath,
        sessionId,
        timestamp: Date.now(),
        fileSize: stats.size,
        trackCount: Object.keys(allVideoFiles).length,
        processingTime: Date.now() - job.timestamp,
      },
      7200
    ); // Cache for 2 hours

    // Store performance metrics
    const metricsKey = `job:metrics:${job.id}`;
    await cacheService.set(
      metricsKey,
      {
        ...performanceMetrics,
        completedAt: Date.now(),
        outputFileSize: stats.size,
        totalProcessingTime: Date.now() - job.timestamp,
      },
      3600
    ); // Cache metrics for 1 hour

    console.log(
      `✅ Composition job ${job.id} completed successfully: ${stats.size} bytes`
    );

    // Cleanup temp files
    try {
      const { rmSync } = await import('fs');
      rmSync(midiJsonPath, { force: true });
      rmSync(videoFilesJsonPath, { force: true });
    } catch (cleanupError) {
      console.warn('Could not clean up temp files:', cleanupError.message);
    }

    return {
      success: true,
      outputPath,
      fileSize: stats.size,
      sessionId,
      processingTime: Date.now() - job.timestamp,
    };
  } catch (error) {
    console.error(`❌ Composition job ${job.id} failed:`, error);

    // Store error details for troubleshooting
    const errorKey = `job:error:${job.id}`;
    await cacheService.set(
      errorKey,
      {
        error: error.message,
        timestamp: Date.now(),
        sessionId,
        jobData: job.data,
      },
      3600
    );

    throw error;
  }
});

// Process autotune jobs
videoProcessingQueue.process('autotune', 2, async (job) => {
  const { videoPath, outputPath } = job.data;

  try {
    await job.progress(20);

    // Import the autotune processor
    const { processAutotuneVideo } = await import(
      '../controllers/autotuneController.js'
    );

    const result = await processAutotuneVideo(videoPath, outputPath);

    await job.progress(100);

    return result;
  } catch (error) {
    console.error('Autotune job failed:', error);
    throw error;
  }
});

// Process MIDI analysis jobs
videoProcessingQueue.process('midi-analysis', 8, async (job) => {
  const { midiBuffer, fileName } = job.data;

  try {
    await job.progress(30);

    // Import MIDI processor
    const { analyzeMidiBuffer } = await import(
      '../controllers/midiController.js'
    );

    const result = await analyzeMidiBuffer(midiBuffer, fileName);

    await job.progress(100);

    return result;
  } catch (error) {
    console.error('MIDI analysis job failed:', error);
    throw error;
  }
});

// Queue monitoring and metrics
videoProcessingQueue.on('completed', async (job) => {
  const duration = Date.now() - job.timestamp;
  console.log(`Job ${job.id} completed in ${duration}ms`);

  // Cache performance metrics
  await cacheService.cacheMetrics(job.name, duration);
});

videoProcessingQueue.on('failed', (job, err) => {
  console.error(`Job ${job.id} failed:`, err.message);
});

videoProcessingQueue.on('stalled', (job) => {
  console.warn(`Job ${job.id} stalled`);
});

// Queue management functions
export const addVideoCompositionJob = async (
  midiData,
  videoFiles,
  outputPath,
  sessionId,
  priority = 0
) => {
  const job = await videoProcessingQueue.add(
    'compose',
    {
      midiData,
      videoFiles,
      outputPath,
      sessionId,
    },
    {
      priority,
      delay: 0,
    }
  );

  return job;
};

export const addAutotuneJob = async (videoPath, outputPath, priority = 0) => {
  const job = await videoProcessingQueue.add(
    'autotune',
    {
      videoPath,
      outputPath,
    },
    {
      priority,
    }
  );

  return job;
};

export const addMidiAnalysisJob = async (
  midiBuffer,
  fileName,
  priority = 10
) => {
  const job = await videoProcessingQueue.add(
    'midi-analysis',
    {
      midiBuffer,
      fileName,
    },
    {
      priority,
    }
  );

  return job;
};

export const getJobStatus = async (jobId) => {
  const job = await videoProcessingQueue.getJob(jobId);
  if (!job) return null;

  return {
    id: job.id,
    progress: job.progress(),
    state: await job.getState(),
    data: job.data,
    created: job.timestamp,
    processed: job.processedOn,
    finished: job.finishedOn,
  };
};

export const getQueueStats = async () => {
  const waiting = await videoProcessingQueue.getWaiting();
  const active = await videoProcessingQueue.getActive();
  const completed = await videoProcessingQueue.getCompleted();
  const failed = await videoProcessingQueue.getFailed();

  return {
    waiting: waiting.length,
    active: active.length,
    completed: completed.length,
    failed: failed.length,
  };
};

export default videoProcessingQueue;
