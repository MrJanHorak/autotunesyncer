export const CONFIG = {
  recording: {
    defaultDuration: 5000,
    mimeType: 'video/webm',
    videoBitsPerSecond: 2500000
  },
  api: {
    baseUrl: import.meta.env.VITE_API_URL,
    timeout: 30000
  }
};