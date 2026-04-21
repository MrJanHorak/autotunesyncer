import { CONFIG } from '../../config.js';
import { Synthesizer } from 'js-synthesizer';

class Sf2DrumPlayer {
  constructor() {
    this.audioContext = null;
    this.synth = null;
    this.audioNode = null;
    this.loaded = false;
  }

  async loadFluidsynthWasm() {
    // If Fluidsynth is already loaded and initialized, skip.
    if (window.Module?.calledRun) return;
    const baseUrl = '/sf2';
    const preferred = 'libfluidsynth-2.4.6-with-libsndfile.js';
    const fallback = 'libfluidsynth-2.4.6.js';

    window.Module = window.Module || {};
    window.Module.locateFile = (path) => `${baseUrl}/${path}`;

    const loadScript = (src) =>
      new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = `${baseUrl}/${src}`;
        script.async = true;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(script);
      });

    try {
      await loadScript(preferred);
    } catch (_) {
      await loadScript(fallback);
    }

    // Let js-synthesizer bind and wait for WASM readiness
    await Synthesizer.waitForWasmInitialized();
  }

  async init() {
    if (this.audioContext) return;
    // Reuse Tone's AudioContext if present, else create a new one
    try {
      // Lazy require to avoid hard dependency
      const tone = await import('tone');
      this.audioContext =
        tone.Tone?.context?.rawContext ||
        new (window.AudioContext || window.webkitAudioContext)();
    } catch (_) {
      this.audioContext = new (
        window.AudioContext || window.webkitAudioContext
      )();
    }
  }

  async load(sf2Url = CONFIG.drums.sf2Path) {
    await this.init();
    if (this.loaded && this.synth) return true;

    // Ensure Fluidsynth WASM glue is loaded and ready before constructing synth
    await this.loadFluidsynthWasm();
    this.synth = new Synthesizer();
    this.synth.init(this.audioContext.sampleRate);

    // Create audio node first, then load SF2 according to docs
    this.audioNode = this.synth.createAudioNode(this.audioContext);
    this.audioNode.connect(this.audioContext.destination);

    const resp = await fetch(sf2Url);
    if (!resp.ok)
      throw new Error(`Failed to fetch SF2 at ${sf2Url}: ${resp.status}`);
    const sf2Data = await resp.arrayBuffer();
    await this.synth.loadSFont(sf2Data);

    // Mark drum channel explicitly
    try {
      this.synth.midiSetChannelType(CONFIG.drums.channel, true);
    } catch (_) {}

    this.loaded = true;
    return true;
  }

  isReady() {
    return !!this.loaded && !!this.synth;
  }

  async ensureReady() {
    if (!this.isReady()) {
      await this.load(CONFIG.drums.sf2Path);
    }
  }

  async play(
    midiNote,
    durationMs = CONFIG.drums.defaultDurationMs,
    velocity = CONFIG.drums.defaultVelocity,
  ) {
    await this.ensureReady();
    // Channel 9 (zero-based) is GM drums
    const channel = CONFIG.drums.channel;
    // Ensure context running
    if (this.audioContext.state !== 'running') {
      await this.audioContext.resume();
    }
    this.synth.midiNoteOn(
      channel,
      midiNote,
      Math.max(1, Math.min(127, velocity)),
    );
    setTimeout(() => {
      try {
        this.synth.midiNoteOff(channel, midiNote);
      } catch (e) {
        console.warn('[Sf2DrumPlayer] noteOff warning:', e);
      }
    }, durationMs);
  }
}

export const sf2DrumPlayer = new Sf2DrumPlayer();
