export const DEFAULT_RENDER_PRESET = 'landscape';

export const RENDER_PRESETS = {
  landscape: {
    id: 'landscape',
    label: 'Widescreen 16:9',
    shortLabel: 'Wide',
    description: 'TV, desktop, YouTube standard',
    aspectRatio: { width: 16, height: 9 },
    preview: { width: 640, height: 360 },
    export: { width: 1920, height: 1080 },
  },
  portrait: {
    id: 'portrait',
    label: 'Portrait 9:16',
    shortLabel: 'Phone',
    description: 'TikTok, Reels, Shorts, Stories',
    aspectRatio: { width: 9, height: 16 },
    preview: { width: 360, height: 640 },
    export: { width: 1080, height: 1920 },
  },
};

const hasOwn = (value) =>
  Object.prototype.hasOwnProperty.call(RENDER_PRESETS, value);

export const isRenderPreset = (value) =>
  typeof value === 'string' && hasOwn(value);

export const normalizeRenderPreset = (
  value,
  fallback = DEFAULT_RENDER_PRESET,
) => (isRenderPreset(value) ? value : fallback);

export const getRenderPresetConfig = (value) =>
  RENDER_PRESETS[normalizeRenderPreset(value)];

export const getRenderDimensions = (value, { preview = false } = {}) => {
  const preset = getRenderPresetConfig(value);
  return preview ? preset.preview : preset.export;
};

export const getRenderAspectRatio = (value) =>
  getRenderPresetConfig(value).aspectRatio;