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
  autotuneVideo: async (formData) => {
    try {
      console.log('Sending video data to server, size:', formData.get('video').size);
      
      const response = await fetch(`${API_BASE_URL}/autotune`, {
        method: 'POST',
        body: formData,
        headers: {
          // Remove Content-Type header to let browser set it with boundary
        },
      });
      
      if (!response.ok) {
        const error = await response.text();
        console.error('Server response:', error);
        throw new Error(`Failed to autotune video: ${response.statusText}`);
      }
      
      const blob = await response.blob();
      if (blob.size === 0) {
        throw new Error('Received empty response from server');
      }
      
      return blob;
    } catch (error) {
      console.error('Autotune error:', error);
      throw error;
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
      return await result.blob();
    } catch (error) {
      throw new VideoProcessingError('Composition failed', error.message);
    }
  },
};
