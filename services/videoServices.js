/* eslint-disable no-unused-vars */
const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

/**
 * Custom error class for video processing errors
 */
class VideoProcessingError extends Error {
  constructor(message, detail) {
    super(message);
    this.name = 'VideoProcessingError';
    this.detail = detail;
  }
}

/**
 * Configuration for video recording
 */
export const RECORDING_CONFIG = {
  defaultDuration: 5000,
  mimeType: 'video/mp4',
  videoBitsPerSecond: 2500000,
};

/**
 * Helper function to handle API responses
 */
async function handleApiResponse(response) {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new VideoProcessingError(
      'Request failed',
      errorData.message || response.statusText
    );
  }
  return response;
}

/**
 * Service for handling video uploads
 */
export const videoService = {
  async autotuneVideo(videoFile) {
    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch(`${API_BASE_URL}/autotune`, {
        method: 'POST',
        body: formData,
        // Add timeout and error handling options
        timeout: 30000,
        headers: {
          Accept: 'video/mp4',
        },
      });

      if (!response.ok) {
        throw new VideoProcessingError('Autotune failed', response.statusText);
      }

      const result = await handleApiResponse(response);
      const autotunedBlob = await result.blob();
      return new File([autotunedBlob], 'autotuned-video.mp4', {
        type: RECORDING_CONFIG.mimeType,
      });
    } catch (error) {
      console.error('Autotune error:', error);
      throw new VideoProcessingError('Autotune failed', error.message);
    }
  },

  async composeVideos(videoFiles, midiData, onProgress) {
    try {
      const formData = new FormData();

      // Add MIDI data
      const midiBlob = new Blob([JSON.stringify(midiData)], {
        type: 'application/json',
      });
      formData.append('midiData', midiBlob);

      // Add video files
      Object.entries(videoFiles).forEach(([instrument, blob]) => {
        if (!(blob instanceof Blob || blob instanceof File)) {
          throw new VideoProcessingError(
            'Invalid video format',
            `Invalid file for instrument: ${instrument}`
          );
        }
        formData.append(`videos[${instrument}]`, blob);
      });

      const response = await fetch(`${API_BASE_URL}/compose`, {
        method: 'POST',
        body: formData,
      });

      const result = await handleApiResponse(response);
      return result.blob();
    } catch (error) {
      throw new VideoProcessingError('Composition failed', error.message);
    }
  },
};

/**
 * Hook for managing video recording state
 */
export const useVideoRecorder = (options = {}) => {
  const {
    duration = RECORDING_CONFIG.defaultDuration,
    onRecordingStart,
    onRecordingStop,
    onProcessingStart,
    onProcessingComplete,
    onError,
  } = options;

  const recordVideo = async () => {
    let stream = null;
    let mediaRecorder = null;
    const chunks = [];

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      mediaRecorder = new MediaRecorder(stream, {
        mimeType: RECORDING_CONFIG.mimeType,
        videoBitsPerSecond: RECORDING_CONFIG.videoBitsPerSecond,
      });

      return new Promise((resolve, reject) => {
        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

        mediaRecorder.onerror = (error) => {
          reject(new VideoProcessingError('Recording failed', error.message));
        };

        mediaRecorder.onstop = async () => {
          try {
            onProcessingStart?.();

            const videoBlob = new Blob(chunks, {
              type: RECORDING_CONFIG.mimeType,
            });
            const videoFile = new File([videoBlob], 'recorded-video.mp4', {
              type: RECORDING_CONFIG.mimeType,
            });

            const autotunedVideo = await videoService.autotuneVideo(videoFile);

            onProcessingComplete?.();
            resolve({
              originalVideo: videoFile,
              autotunedVideo,
              originalUrl: URL.createObjectURL(videoBlob),
              autotunedUrl: URL.createObjectURL(autotunedVideo),
            });
          } catch (error) {
            onError?.(error);
            reject(error);
          } finally {
            if (stream) {
              stream.getTracks().forEach((track) => track.stop());
            }
          }
        };

        onRecordingStart?.();
        mediaRecorder.start();

        setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            onRecordingStop?.();
            mediaRecorder.stop();
          }
        }, duration);
      });
    } catch (error) {
      onError?.(error);
      throw new VideoProcessingError(
        'Recording initialization failed',
        error.message
      );
    }
  };

  return { recordVideo };
};
