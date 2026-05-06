/**
 * Default styles for composition output.
 * compositionStyle = global settings (title, watermark, effects, etc.)
 * clipStyle = per-track settings (border, color grade, label, etc.)
 */

/** Font options available for all text overlays. */
export const FONT_OPTIONS = [
  { value: 'default', label: 'Default (FFmpeg)' },
  { value: 'arial', label: 'Arial' },
  { value: 'verdana', label: 'Verdana' },
  { value: 'impact', label: 'Impact' },
  { value: 'courier', label: 'Courier New' },
  { value: 'times', label: 'Times New Roman' },
  { value: 'georgia', label: 'Georgia' },
  { value: 'trebuchet', label: 'Trebuchet MS' },
  { value: 'comic', label: 'Comic Sans MS' },
];

export const DEFAULT_COMPOSITION_STYLE = {
  colorTheme: 'dark', // 'dark' | 'neon' | 'vintage' | 'cyberpunk' | 'minimal' | 'custom'
  backgroundColor: '#0a0a0f',
  backgroundMode: 'color', // 'color' | 'image' | 'video'
  backgroundMedia: null,

  // Title overlay
  titleEnabled: false,
  titleText: '',
  titleFontSize: 56,
  titleColor: '#ffffff',
  titleFont: 'default',
  titlePosition: 'top-center', // 'top-center' | 'bottom-center' | 'center'
  titleSubtitleText: '',
  titleSubtitleFontSize: 24,
  titleSubtitleColor: '#d8d8e6',
  titleAnimated: true,
  titleAnimationPreset: 'fade', // 'fade' | 'scroll-up' | 'scroll-left' | 'bounce' | 'spin-soft' | 'blur-focus' | 'typewriter'
  titleAnimDuration: 0.7,
  titleAnimDelay: 0,
  titleAnimIntensity: 'medium', // 'low' | 'medium' | 'high'
  titleAnimDirection: 'left', // used by direction-aware presets
  titleAnimEasing: 'ease-out', // 'linear' | 'ease-out' | 'ease-in-out' | 'cubic-bezier(0.22, 1, 0.36, 1)'
  titleDuration: 4, // seconds to show before fade-out; set to 0 for permanent
  titleBackgroundEnabled: false,
  titleBackgroundMode: 'card', // 'card' | 'fullscreen'
  titleBackgroundColor: '#120b24',
  titleBackgroundOpacity: 0.82,
  // Title glow and shadow effects
  titleGlowEnabled: false,
  titleGlowColor: '#ffffff',
  titleGlowSize: 8,
  titleShadowEnabled: true,
  titleShadowSize: 2,
  titleShadowColor: '#000000',

  // Tagline / lower-third
  taglineEnabled: false,
  taglineText: '',
  taglineFontSize: 24,
  taglineColor: '#cccccc',
  taglineFont: 'default',
  taglinePosition: 'bottom-center', // 'bottom-left' | 'bottom-center' | 'bottom-right'
  taglineWidth: 72, // percentage (0-100)
  taglineVerticalOffset: 0, // px, positive moves bar upward
  taglineShape: 'rounded', // 'rounded' | 'pill' | 'square' | 'outline' | 'accent-left'
  taglineBackgroundEnabled: false,
  taglineBackgroundColor: '#0c1220',
  taglineBackgroundOpacity: 0.72,
  taglineAccentColor: '#ff4db8',
  // Tagline alignment and shadow
  taglineAlignment: 'center', // 'left' | 'center' | 'right'
  taglineShadowEnabled: true,
  taglineShadowSize: 2,
  taglineShadowColor: '#000000',
  // Tagline fade timing (NEW)
  taglineFadeInDuration: 0.5, // seconds to fade in
  taglineFadeOutDuration: 0.5, // seconds to fade out after duration

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

  // Opening transition
  transitionEnabled: false,
  transitionPreset: 'none', // 'none' | 'crossfade' | 'dip-black' | 'dip-white' | 'push-left' | 'push-right' | 'slide-left' | 'slide-right' | 'zoom' | 'zoom-in' | 'glitch-cut'
  transitionDuration: 0.6,
  transitionStrength: 'medium', // 'low' | 'medium' | 'high'
  transitionOn: 'start', // 'start' | 'section' | 'phrase' | 'manual-marker' | 'interval' | 'sections' | 'auto'
  transitionSectionInterval: 8, // repeat cadence in seconds when transitionOn='section' (legacy: 'interval'/'sections')
  transitionAutoCadenceSeconds: 8, // frontend-estimated cadence used when transitionOn='auto'
  transitionAutoReason: 'Sparse arrangement detected',
  transitionApplyAfterText: false,
  transitionApplyToIntroCard: false,

  // Beat-sync micro motion (phase 3 scaffold)
  beatSyncEnabled: false,
  beatSyncSensitivity: 'medium', // 'low' | 'medium' | 'high'
  beatSyncTargets: ['title', 'tagline'], // subset of: 'title' | 'tagline' | 'track-cells' | 'overlays'
  beatPulseMode: 'scale', // 'scale' | 'glow' | 'shake-lite'

  // Ending effect / outro
  outroEffectEnabled: false,
  outroEffectPreset: 'fade-black', // 'fade-black' | 'fade-white' | 'glitch-out' | 'zoom-out'
  outroEffectDuration: 1.2,
  outroEffectStrength: 'medium', // 'low' | 'medium' | 'high'

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

  // bgColorEnabled: user must explicitly toggle this on.
  // bgColor: the chosen color (preserved when toggled off so it can be restored).
  bgColorEnabled: false,
  bgColor: '#1a1a2e',

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

  transparentBg: false, // when true, idle gaps fall back to the global composition background
};

export const COLOR_THEMES = {
  dark: {
    backgroundColor: '#0a0a0f',
    clipDefaults: {
      borderColor: '#7c3aed',
      bgColor: '#1a1a2e',
      bgColorEnabled: false,
      colorGrade: 'none',
    },
    titleColor: '#ffffff',
    taglineColor: '#cccccc',
    watermarkColor: '#ffffff',
  },
  neon: {
    backgroundColor: '#050510',
    clipDefaults: {
      borderColor: '#00ffff',
      bgColor: '#0a0a1a',
      bgColorEnabled: false,
      colorGrade: 'vivid',
    },
    titleColor: '#00ffff',
    taglineColor: '#ff00ff',
    watermarkColor: '#00ffff',
  },
  vintage: {
    backgroundColor: '#1a0f00',
    clipDefaults: {
      borderColor: '#d4a044',
      bgColor: '#2a1500',
      bgColorEnabled: false,
      colorGrade: 'vintage',
    },
    titleColor: '#f5deb3',
    taglineColor: '#d4a044',
    watermarkColor: '#d4a044',
  },
  cyberpunk: {
    backgroundColor: '#0d0221',
    clipDefaults: {
      borderColor: '#ff00ff',
      bgColor: '#0d0221',
      bgColorEnabled: false,
      colorGrade: 'cyberpunk',
    },
    titleColor: '#ff00ff',
    taglineColor: '#00ffff',
    watermarkColor: '#ff00ff',
  },
  minimal: {
    backgroundColor: '#f5f5f5',
    clipDefaults: {
      borderColor: '#e0e0e0',
      bgColor: '#ffffff',
      bgColorEnabled: false,
      colorGrade: 'none',
    },
    titleColor: '#111111',
    taglineColor: '#555555',
    watermarkColor: '#888888',
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
