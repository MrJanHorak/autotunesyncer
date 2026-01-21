export const CONFIG = {
  recording: {
    defaultDuration: 5000,
    mimeType: 'video/webm',
    videoBitsPerSecond: 2500000,
  },
  api: {
    baseUrl: import.meta.env.VITE_API_URL,
    timeout: 30000,
  },
  drums: {
    useSf2: true,
    sf2Path: '/soundfonts/Arachno SoundFont - Version 1.0.sf2',
    wasmBaseUrl: 'https://cdn.jsdelivr.net/npm/js-synthesizer@latest/dist',
    channel: 9,
    defaultVelocity: 110,
    defaultDurationMs: 800,
  },
};
