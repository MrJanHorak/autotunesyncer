/**
 * Default styles for composition output.
 * compositionStyle = global settings (title, watermark, effects, etc.)
 * clipStyle = per-track settings (border, color grade, label, etc.)
 */

/** Font options available for all text overlays. */
export const FONT_OPTIONS = [
  { value: 'default',   label: 'Default (FFmpeg)' },
  { value: 'arial',     label: 'Arial' },
  { value: 'verdana',   label: 'Verdana' },
  { value: 'impact',    label: 'Impact' },
  { value: 'courier',   label: 'Courier New' },
  { value: 'times',     label: 'Times New Roman' },
  { value: 'georgia',   label: 'Georgia' },
  { value: 'trebuchet', label: 'Trebuchet MS' },
  { value: 'comic',     label: 'Comic Sans MS' },
];

export const DEFAULT_COMPOSITION_STYLE = {
  colorTheme: 'dark', // 'dark' | 'neon' | 'vintage' | 'cyberpunk' | 'minimal' | 'custom'
  backgroundColor: '#0a0a0f',

  // Title overlay
  titleEnabled: false,
  titleText: '',
  titleFontSize: 56,
  titleColor: '#ffffff',
  titleFont: 'default',
  titlePosition: 'top-center', // 'top-center' | 'bottom-center' | 'center'
  titleAnimated: true,

  // Tagline / lower-third
  taglineEnabled: false,
  taglineText: '',
  taglineFontSize: 24,
  taglineColor: '#cccccc',
  taglineFont: 'default',
  taglinePosition: 'bottom-center',

  // Watermark
  watermarkEnabled: false,
  watermarkText: '',
  watermarkFontSize: 18,
  watermarkColor: '#ffffff',
  watermarkFont: 'default',
  watermarkOpacity: 0.5,
  watermarkPosition: 'bottom-right', // 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'

  // Waveform bar
  waveformEnabled: false,
  waveformColor: '#00ff88',
  waveformHeight: 60,

  // Vignette
  vignetteEnabled: false,
  vignetteStrength: 0.5,

  // Glitch / VHS
  glitchEnabled: false,
  glitchIntensity: 'subtle', // 'subtle' | 'medium' | 'heavy'

  // Intro title card
  introCardEnabled: false,
  introCardDuration: 3,
  introCardBg: '#000000',
  introCardText: '',
  introCardSubtext: '',
  introCardTextColor: '#ffffff',
  introCardFont: 'default',
  introCardAnimated: true,
};

export const DEFAULT_CLIP_STYLE = {
  borderColor: '#7c3aed',
  borderWidth: 0, // 0 = no border

  // null = use composition background (seamlessly transparent);
  // set a hex color to give the clip its own distinct background.
  bgColor: null,

  roundedCorners: false,
  cornerRadius: 12,

  colorGrade: 'none', // 'none' | 'warm' | 'cool' | 'vintage' | 'cyberpunk' | 'bw' | 'vivid'

  labelEnabled: false,
  labelText: '', // empty = use instrument name
  labelColor: '#ffffff',
  labelFont: 'default',
  labelFontSize: 14,

  beatFlashEnabled: false,
  beatFlashColor: '#ffffff',
  beatFlashIntensity: 0.4, // 0–1 brightness boost

  fadeEnabled: false,
  fadeDuration: 0.15, // seconds for fade-in/out on note trigger

  transparentBg: false, // zoom-to-fill (crop) instead of letterbox
};

export const COLOR_THEMES = {
  dark: {
    backgroundColor: '#0a0a0f',
    clipDefaults: { borderColor: '#7c3aed', bgColor: '#1a1a2e', colorGrade: 'none' },
    titleColor: '#ffffff', taglineColor: '#cccccc', watermarkColor: '#ffffff',
  },
  neon: {
    backgroundColor: '#050510',
    clipDefaults: { borderColor: '#00ffff', bgColor: '#0a0a1a', colorGrade: 'vivid' },
    titleColor: '#00ffff', taglineColor: '#ff00ff', watermarkColor: '#00ffff',
  },
  vintage: {
    backgroundColor: '#1a0f00',
    clipDefaults: { borderColor: '#d4a044', bgColor: '#2a1500', colorGrade: 'vintage' },
    titleColor: '#f5deb3', taglineColor: '#d4a044', watermarkColor: '#d4a044',
  },
  cyberpunk: {
    backgroundColor: '#0d0221',
    clipDefaults: { borderColor: '#ff00ff', bgColor: '#0d0221', colorGrade: 'cyberpunk' },
    titleColor: '#ff00ff', taglineColor: '#00ffff', watermarkColor: '#ff00ff',
  },
  minimal: {
    backgroundColor: '#f5f5f5',
    clipDefaults: { borderColor: '#e0e0e0', bgColor: '#ffffff', colorGrade: 'none' },
    titleColor: '#111111', taglineColor: '#555555', watermarkColor: '#888888',
  },
};

export const COLOR_GRADE_LABELS = {
  none: 'None',
  warm: '🌅 Warm',
  cool: '❄️ Cool',
  vintage: '📷 Vintage',
  cyberpunk: '🌆 Cyberpunk',
  bw: '⬛ Black & White',
  vivid: '🌈 Vivid',
};
