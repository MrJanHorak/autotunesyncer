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
      errorData.message || response.statusText,
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
      console.log(
        'Sending video data to server, size:',
        formData.get('video').size,
      );

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
};

// async composeVideos(videoFiles, midiData, onProgress) {
//   try {
//     const formData = new FormData();

//     // Add MIDI data
//     const midiBlob = new Blob([JSON.stringify(midiData)], {
//       type: 'application/json',
//     });
//     formData.append('midiData', midiBlob);

//     // Add video files
//     Object.entries(videoFiles).forEach(([instrument, blob]) => {
//       if (!(blob instanceof Blob || blob instanceof File)) {
//         throw new VideoProcessingError(
//           'Invalid video format',
//           `Invalid file for instrument: ${instrument}`
//         );
//       }
//       formData.append(`videos[${instrument}]`, blob);
//     });

//     const response = await fetch(`${API_BASE_URL}/compose`, {
//       method: 'POST',
//       body: formData,
//     });

//     const result = await handleApiResponse(response);
//     return await result.blob();
//   } catch (error) {
//     throw new VideoProcessingError('Composition failed', error.message);
//   }
// },

export const composeVideos = (formData, progressCallbacks = {}) => {
  if (!(formData instanceof FormData)) {
    return Promise.reject(new Error('Invalid compose request payload'));
  }

  const midiPart = formData.get('midiData');
  const videoParts = formData.getAll('videos');

  if (!midiPart) {
    return Promise.reject(new Error('Missing midiData in compose request'));
  }

  if (!videoParts || videoParts.length === 0) {
    return Promise.reject(new Error('Missing videos in compose request'));
  }

  const { onUploadProgress } = progressCallbacks;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/process-videos`);
    xhr.responseType = 'blob';

    if (onUploadProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded * 100) / event.total);
          onUploadProgress(pct);
        }
      };
      // Mark upload as complete once XHR fires the upload load event
      xhr.upload.onload = () => onUploadProgress(100);
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve({ data: xhr.response });
      } else {
        // Try to read error details from the blob response
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const errData = JSON.parse(reader.result);
            reject(
              new Error(
                errData.details ||
                  errData.error ||
                  `Server error ${xhr.status}`,
              ),
            );
          } catch {
            reject(new Error(`Server error ${xhr.status}`));
          }
        };
        reader.readAsText(xhr.response);
      }
    };

    xhr.onerror = () =>
      reject(new Error('Network error during video composition'));
    xhr.ontimeout = () => reject(new Error('Request timed out'));

    xhr.send(formData);
  });
};

/**
 * Start an async composition job. Returns a Promise that resolves with the
 * jobId string once the server has accepted the upload (HTTP 202).
 */
export const startCompositionJob = (formData, progressCallbacks = {}) => {
  if (!(formData instanceof FormData)) {
    return Promise.reject(new Error('Invalid compose request payload'));
  }

  const { onUploadProgress } = progressCallbacks;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/process-videos`);

    if (onUploadProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onUploadProgress(Math.round((event.loaded * 100) / event.total));
        }
      };
      xhr.upload.onload = () => onUploadProgress(100);
    }

    xhr.onload = () => {
      if (xhr.status === 202) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data.jobId);
        } catch {
          reject(new Error('Invalid server response: expected { jobId }'));
        }
      } else {
        try {
          const errData = JSON.parse(xhr.responseText);
          reject(new Error(errData.details || errData.error || `Server error ${xhr.status}`));
        } catch {
          reject(new Error(`Server error ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('Network error during video composition'));
    xhr.ontimeout = () => reject(new Error('Upload timed out'));
    xhr.send(formData);
  });
};

const POLL_INTERVAL_MS = 3000;
const POLL_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

/**
 * Poll a composition job until it completes, then download the result blob.
 * @param {string} jobId
 * @param {function} [onProgress] - called with progress 0-100
 * @returns {Promise<Blob>}
 */
export const pollCompositionJob = (jobId, onProgress) => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    const poll = () => {
      if (Date.now() - startTime > POLL_TIMEOUT_MS) {
        reject(new Error('Composition timed out after 30 minutes'));
        return;
      }

      fetch(`${API_BASE_URL}/process-videos/status/${jobId}`)
        .then((r) => {
          if (!r.ok) throw new Error(`Status check failed: ${r.statusText}`);
          return r.json();
        })
        .then(({ status, progress, error }) => {
          if (onProgress && typeof progress === 'number') onProgress(progress);

          if (status === 'done') {
            return fetch(`${API_BASE_URL}/process-videos/result/${jobId}`)
              .then((r) => {
                if (!r.ok) throw new Error('Failed to download composition result');
                return r.blob();
              })
              .then(resolve);
          } else if (status === 'failed') {
            reject(new Error(error || 'Composition failed on the server'));
          } else {
            setTimeout(poll, POLL_INTERVAL_MS);
          }
        })
        .catch(reject);
    };

    setTimeout(poll, POLL_INTERVAL_MS);
  });
};
