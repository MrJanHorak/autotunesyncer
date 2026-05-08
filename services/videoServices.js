/* eslint-disable no-unused-vars */
const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

/** Read the stored JWT from localStorage. */
function getToken() {
  return localStorage.getItem('auth_token');
}

/** Read the current project id from localStorage. */
function getProjectId() {
  try {
    const p = localStorage.getItem('current_project');
    return p ? JSON.parse(p).id : null;
  } catch {
    return null;
  }
}

/** Append ?projectId=<id> to a URL when a project is active. */
function withProjectId(url) {
  const projectId = getProjectId();
  if (!projectId) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}projectId=${projectId}`;
}

/** Return fetch-compatible headers with Authorization set. */
function authFetchHeaders(extra = {}) {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}`, ...extra } : extra;
}

/** Set Authorization + projectId query param on an open XMLHttpRequest. */
function prepareXhr(xhr, baseUrl) {
  const token = getToken();
  const url = withProjectId(baseUrl);
  if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
  return url;
}

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
    const xhrOld = new XMLHttpRequest();
    const xhrOldUrl = withProjectId(`${API_BASE_URL}/process-videos`);
    xhrOld.open('POST', xhrOldUrl);
    const tokenOld = getToken();
    if (tokenOld)
      xhrOld.setRequestHeader('Authorization', `Bearer ${tokenOld}`);
    xhrOld.responseType = 'blob';

    if (onUploadProgress) {
      xhrOld.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded * 100) / event.total);
          onUploadProgress(pct);
        }
      };
      xhrOld.upload.onload = () => onUploadProgress(100);
    }

    xhrOld.onload = () => {
      if (xhrOld.status >= 200 && xhrOld.status < 300) {
        resolve({ data: xhrOld.response });
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const errData = JSON.parse(reader.result);
            reject(
              new Error(
                errData.details ||
                  errData.error ||
                  `Server error ${xhrOld.status}`,
              ),
            );
          } catch {
            reject(new Error(`Server error ${xhrOld.status}`));
          }
        };
        reader.readAsText(xhrOld.response);
      }
    };

    xhrOld.onerror = () =>
      reject(new Error('Network error during video composition'));
    xhrOld.ontimeout = () => reject(new Error('Request timed out'));

    xhrOld.send(formData);
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

  const { onUploadProgress, signal } = progressCallbacks;

  return new Promise((resolve, reject) => {
    // Guard against double-settlement (abort + error can both fire)
    let settled = false;
    const settle = (fn, val) => {
      if (settled) return;
      settled = true;
      fn(val);
    };

    const xhr = new XMLHttpRequest();
    const xhrUrl = withProjectId(`${API_BASE_URL}/process-videos`);
    xhr.open('POST', xhrUrl);
    const token = getToken();
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);

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
          settle(resolve, data.jobId);
        } catch {
          settle(
            reject,
            new Error('Invalid server response: expected { jobId }'),
          );
        }
      } else {
        try {
          const errData = JSON.parse(xhr.responseText);
          settle(
            reject,
            new Error(
              errData.details || errData.error || `Server error ${xhr.status}`,
            ),
          );
        } catch {
          settle(reject, new Error(`Server error ${xhr.status}`));
        }
      }
    };

    xhr.onabort = () =>
      settle(reject, new DOMException('Upload cancelled', 'AbortError'));
    xhr.onerror = () =>
      settle(reject, new Error('Network error during video composition'));
    xhr.ontimeout = () => settle(reject, new Error('Upload timed out'));

    // Wire AbortSignal → XHR abort
    if (signal) {
      if (signal.aborted) {
        xhr.abort();
        return;
      }
      signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }

    xhr.send(formData);
  });
};

const POLL_INTERVAL_MS = 3000;
const DEFAULT_POLL_TIMEOUT_MS = 4 * 60 * 60 * 1000;
const POLL_TIMEOUT_MS = (() => {
  const configured = Number(import.meta.env.VITE_COMPOSITION_POLL_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0
    ? configured
    : DEFAULT_POLL_TIMEOUT_MS;
})();

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
        reject(
          new Error(
            `Composition timed out after ${Math.round(POLL_TIMEOUT_MS / 60000)} minutes`,
          ),
        );
        return;
      }

      fetch(withProjectId(`${API_BASE_URL}/process-videos/status/${jobId}`), {
        headers: authFetchHeaders(),
      })
        .then((r) => {
          if (!r.ok) throw new Error(`Status check failed: ${r.statusText}`);
          return r.json();
        })
        .then(({ status, progress, error }) => {
          if (onProgress && typeof progress === 'number') onProgress(progress);

          if (status === 'done') {
            return fetch(
              withProjectId(`${API_BASE_URL}/process-videos/result/${jobId}`),
              {
                headers: authFetchHeaders(),
              },
            )
              .then((r) => {
                if (!r.ok)
                  throw new Error('Failed to download composition result');
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

/**
 * Download the finished composition blob.
 * @param {string} jobId
 * @returns {Promise<Blob>}
 */
function downloadCompositionResult(jobId) {
  return fetch(
    withProjectId(`${API_BASE_URL}/process-videos/result/${jobId}`),
    { headers: authFetchHeaders() },
  ).then((r) => {
    if (!r.ok) throw new Error('Failed to download composition result');
    return r.blob();
  });
}

/**
 * Track a composition job via SSE (Server-Sent Events), falling back to polling
 * if the browser or network doesn't support streaming.
 *
 * Uses fetch + ReadableStream so the Authorization header is preserved.
 *
 * @param {string} jobId
 * @param {function} [onProgress] - called with progress 0-100
 * @param {AbortSignal} [signal] - optional AbortSignal for cancellation
 * @returns {Promise<Blob>}
 */
export const trackCompositionJob = (jobId, onProgress, signal) => {
  return new Promise((resolve, reject) => {
    // Fall back to polling if the browser lacks streaming support
    if (!window.ReadableStream || !window.TextDecoderStream) {
      return pollCompositionJob(jobId, onProgress).then(resolve).catch(reject);
    }

    let settled = false;

    const settle = (fn, ...args) => {
      if (settled) return;
      settled = true;
      fn(...args);
    };

    const url = withProjectId(
      `${API_BASE_URL}/process-videos/progress/${jobId}`,
    );

    fetch(url, { headers: authFetchHeaders(), signal })
      .then(async (r) => {
        if (!r.ok) throw new Error(`SSE endpoint error: ${r.statusText}`);
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });

          // Process all complete SSE messages (delimited by double newline)
          let boundary;
          while ((boundary = buf.indexOf('\n\n')) !== -1) {
            const chunk = buf.slice(0, boundary);
            buf = buf.slice(boundary + 2);
            let evt = '';
            let data = '';
            for (const line of chunk.split('\n')) {
              if (line.startsWith('event: ')) evt = line.slice(7).trim();
              else if (line.startsWith('data: ')) data = line.slice(6);
            }
            if (evt !== 'progress' || !data) continue;
            try {
              const { status, progress: pct, error } = JSON.parse(data);
              if (onProgress && typeof pct === 'number') onProgress(pct);
              if (status === 'done') {
                settle(
                  (res) =>
                    downloadCompositionResult(jobId).then(res).catch(reject),
                  resolve,
                );
                return; // stop reading
              } else if (status === 'failed') {
                settle(reject, new Error(error || 'Composition failed'));
                return;
              }
            } catch {
              /* malformed JSON — skip */
            }
          }
        }
      })
      .catch((err) => {
        if (settled) return;
        // Re-throw abort errors so the caller's catch/finally runs cleanly
        if (err?.name === 'AbortError' || signal?.aborted) {
          settle(
            reject,
            new DOMException('Composition cancelled', 'AbortError'),
          );
          return;
        }
        // SSE failed — fall back to polling
        console.warn(
          '[videoServices] SSE unavailable, falling back to poll:',
          err.message,
        );
        pollCompositionJob(jobId, onProgress).then(
          (blob) => settle(resolve, blob),
          (e) => settle(reject, e),
        );
      });
  });
};
