/* eslint-disable react/prop-types */
import { useRef, useState, useCallback, useEffect } from 'react';
import * as Tone from 'tone';
import Soundfont from 'soundfont-player';
import { CONFIG } from '../../../config.js';
import { sf2DrumPlayer } from '../../audio/sf2DrumPlayer.js';
import '../styles.css';

const SampleSoundButton = ({
  instrument,
  instrumentName,
  className,
  midiData,
  isDrumTrack,
}) => {
  const synthRef = useRef(null);
  const [isPlayingSample, setIsPlayingSample] = useState(false);
  const drumSoundfontRef = useRef(null);
  const melodicSoundfontRef = useRef(null);

  // Get MIDI program number for this instrument from the MIDI data
  const getMidiProgram = useCallback(() => {
    if (!midiData || !midiData.tracks || !instrument) return null;

    // For melodic instruments, match by channel number (more reliable than name)
    const channelToMatch =
      typeof instrument === 'object' ? instrument.number : null;

    if (channelToMatch !== null && channelToMatch !== undefined) {
      // Find track by channel
      const matchingTrack = midiData.tracks.find(
        (track) => track.channel === channelToMatch,
      );
      if (matchingTrack && matchingTrack.instrument) {
        return matchingTrack.instrument.number;
      }
    }

    // Fallback to name matching if channel matching fails
    const matchingTrack = midiData.tracks.find((track) => {
      const trackName = track.instrument?.name?.toLowerCase();
      const instName = (
        typeof instrument === 'object' ? instrument.name : instrument
      )?.toLowerCase();
      return trackName === instName;
    });

    if (matchingTrack && matchingTrack.instrument) {
      console.log(
        `[Instrument ${typeof instrument === 'object' ? instrument.name : instrument}] Matched via name, program: ${matchingTrack.instrument.number}`,
      );
      return matchingTrack.instrument.number;
    }

    console.warn(
      `[Instrument ${typeof instrument === 'object' ? instrument.name : instrument}] No MIDI program match found`,
    );
    return null;
  }, [midiData, instrument]);

  // Get the actual MIDI note for this drum from the MIDI data
  const getDrumMidiNote = useCallback(() => {
    const isPercussion = typeof instrument === 'object' && instrument.isDrum;
    if (!midiData || !midiData.tracks || !isPercussion) {
      console.log(
        '[getDrumMidiNote] Not a percussion instrument or no MIDI data',
      );
      return null;
    }

    const instObj =
      typeof instrument === 'object' ? instrument : { group: 'kick' };
    const drumGroup =
      instObj.isDrum && instObj.group ? instObj.group.toLowerCase() : 'kick';

    console.log(
      `[getDrumMidiNote] Drum group: "${drumGroup}", instrument:`,
      instrument,
    );

    // Find the drum track
    const drumTrack = midiData.tracks.find((track) => track.channel === 9);
    if (!drumTrack || !drumTrack.notes || drumTrack.notes.length === 0) {
      console.warn('[getDrumMidiNote] No drum track found on channel 9');
      return null;
    }

    console.log(
      `[getDrumMidiNote] Found drum track with ${drumTrack.notes.length} notes`,
    );

    // Map drum groups to MIDI note ranges - handle name variations
    const drumNoteRanges = {
      kick: [35, 36],
      'kick drum': [35, 36],
      'bass drum': [35, 36],
      snare: [38, 40],
      'snare drum': [38, 40],
      'electric snare drum': [40],
      'side stick': [37],
      hihat: [42, 44, 46],
      'hi-hat': [42, 44, 46],
      'hi-hat closed': [42],
      'hi-hat open': [46],
      'hi-hat pedal': [44],
      hh: [42, 44, 46],
      tom: [41, 43, 45, 47, 48, 50],
      'low tom': [45],
      'mid tom': [47, 48],
      'high-mid tom': [48],
      'high tom': [50],
      'floor tom': [41, 43],
      cymbal: [49, 51, 52, 55, 57, 59],
      crash: [49, 57],
      'crash cymbal': [49, 57],
      'crash cymbal 2': [57],
      ride: [51, 59],
      'ride cymbal': [51, 59],
      'ride cymbal 2': [59],
      'ride bell': [53],
      'china cymbal': [52],
      'china crash': [52],
      'splash cymbal': [55],
      splash: [55],
      bongo: [60, 61],
      'low bongo': [61],
      'high bongo': [60],
      conga: [62, 63, 64],
      'low conga': [64],
      'high conga': [62],
      'conga dead stroke': [62], // Mute high conga
      tumba: [64], // Tumba = low conga
      'high timbale': [65],
      'low timbale': [66],
      'high agogo': [67],
      'low agogo': [68],
      cabasa: [69],
      cowbell: [56],
      'cow bell': [56],
      'more cowbell': [56],
      claves: [75],
      whip: [58],
      laser: [81], // Laser sound - use open triangle (high metallic ping)
      'metronome click': [37], // Metronome - side stick
      'hand clap': [39],
      clap: [39],
      maracas: [70],
      shaker: [82], // Shaker
      tambourine: [54],
      surdu: [36],
      'snare cross stick': [37],
      'cross stick': [37],
      'floor tom 2': [41],
      'floor tom 1': [41],
      'low floor tom': [41],
      'high floor tom': [43],
      'hi-hat foot': [44],
      'hi-hat pedal': [44],
      triangle: [81, 80],
      'triangle open': [81],
      'triangle mute': [80],
      'low wood block': [77],
      'high wood block': [76],
      guiro: [73, 74],
    };

    const noteRange = drumNoteRanges[drumGroup];

    if (!noteRange) {
      console.warn(
        `[getDrumMidiNote] Unknown drum group "${drumGroup}", defaulting to snare`,
      );
      return 38; // Default to snare
    }

    console.log(`[getDrumMidiNote] Looking for notes in range:`, noteRange);

    // Find a note in the track that matches this drum group's MIDI note range
    const matchingNote = drumTrack.notes.find((note) =>
      noteRange.includes(note.midi),
    );

    const resultNote = matchingNote ? matchingNote.midi : noteRange[0];
    console.log(
      `[getDrumMidiNote] Returning MIDI note ${resultNote} for drum group "${drumGroup}"`,
    );

    return resultNote;
  }, [midiData, instrument]);

  // Detect the most common note from MIDI data to match the key
  const detectKeyFromMidi = useCallback(() => {
    if (!midiData || !midiData.tracks) return null;

    const noteCounts = {};

    // Count all notes in the MIDI file
    midiData.tracks.forEach((track) => {
      if (track.notes && Array.isArray(track.notes)) {
        track.notes.forEach((note) => {
          if (note.midi !== undefined) {
            const normalizedNote = note.midi % 12; // Get note class (0-11)
            noteCounts[normalizedNote] = (noteCounts[normalizedNote] || 0) + 1;
          }
        });
      }
    });

    // Find the most common note
    if (Object.keys(noteCounts).length === 0) return null;
    const mostCommonNote = Object.keys(noteCounts).reduce((a, b) =>
      noteCounts[a] > noteCounts[b] ? a : b,
    );

    return parseInt(mostCommonNote);
  }, [midiData]);

  // Map note class to note name in a comfortable octave for each instrument
  const getNoteFromKey = useCallback(
    (keyNote) => {
      const noteNames = [
        'C',
        'C#',
        'D',
        'D#',
        'E',
        'F',
        'F#',
        'G',
        'G#',
        'A',
        'A#',
        'B',
      ];
      const instName =
        typeof instrument === 'object' ? instrument.name : instrument;
      const instrumentNameStr = String(instName || '').toLowerCase();

      if (keyNote === null || keyNote === undefined) keyNote = 0; // Default to C
      const noteName = noteNames[keyNote % 12];

      // Choose octave based on instrument range
      let octave = 4; // Default
      if (instrumentNameStr.includes('bass')) octave = 2;
      else if (instrumentNameStr.includes('cello')) octave = 3;
      else if (
        instrumentNameStr.includes('violin') ||
        instrumentNameStr.includes('string')
      )
        octave = 4;
      else if (
        instrumentNameStr.includes('flute') ||
        instrumentNameStr.includes('wind')
      )
        octave = 5;
      else if (
        instrumentNameStr.includes('horn') ||
        instrumentNameStr.includes('brass')
      )
        octave = 4;
      else if (instrumentNameStr.includes('guitar')) octave = 3;

      return `${noteName}${octave}`;
    },
    [instrument],
  );

  // Play drum sounds using SF2 if configured; fallback to FluidR3_GM synth_drum
  const playDrumSound = useCallback(async () => {
    const midiNote = getDrumMidiNote();

    if (midiNote === null) {
      console.warn('[playDrumSound] No drum note found, skipping');
      return;
    }

    console.log(`[playDrumSound] Playing GM drum note ${midiNote}`);

    try {
      if (CONFIG.drums?.useSf2) {
        // Prefer SF2 (Arachno) playback for realistic kit pieces
        await sf2DrumPlayer.play(midiNote);
        return;
      }

      // Fallback: FluidR3_GM synth_drum
      if (!drumSoundfontRef.current) {
        const audioContext = Tone.context.rawContext;
        drumSoundfontRef.current = await Soundfont.instrument(
          audioContext,
          'synth_drum',
          {
            soundfont: 'FluidR3_GM',
          },
        );
        console.log(
          '[playDrumSound] Loaded drum kit synth_drum from FluidR3_GM',
        );
      }

      console.log(
        `[playDrumSound] Playing MIDI note ${midiNote} on loaded drum kit`,
      );
      drumSoundfontRef.current.play(midiNote, Tone.context.currentTime, {
        duration: 1.5,
        gain: 1.0,
      });
    } catch (error) {
      console.error(
        '[playDrumSound] Failed to load/play drum soundfont:',
        error,
      );
      // Last-chance fallback: try loading synth_drum if SF2 failed
      try {
        const audioContext = Tone.context.rawContext;
        if (!drumSoundfontRef.current) {
          drumSoundfontRef.current = await Soundfont.instrument(
            audioContext,
            'synth_drum',
            {
              soundfont: 'FluidR3_GM',
            },
          );
          console.log('[playDrumSound] Fallback loaded synth_drum');
        }
        drumSoundfontRef.current.play(midiNote, Tone.context.currentTime, {
          duration: 1.5,
          gain: 1.0,
        });
      } catch (fallbackErr) {
        console.warn(
          '[playDrumSound] Drum sound unavailable after fallback:',
          fallbackErr,
        );
      }
    }
  }, [getDrumMidiNote]);

  // Map MIDI program numbers to soundfont instrument names
  const getSoundfontInstrument = useCallback((midiProgram) => {
    // General MIDI program to soundfont instrument mapping
    const programMap = {
      // Piano (0-7)
      0: 'acoustic_grand_piano',
      1: 'bright_acoustic_piano',
      2: 'electric_grand_piano',
      3: 'honkytonk_piano',
      4: 'electric_piano_1',
      5: 'electric_piano_2',
      6: 'harpsichord',
      7: 'clavinet',
      // Chromatic Percussion (8-15)
      8: 'celesta',
      9: 'glockenspiel',
      10: 'music_box',
      11: 'vibraphone',
      12: 'marimba',
      13: 'xylophone',
      14: 'tubular_bells',
      15: 'dulcimer',
      // Organ (16-23)
      16: 'drawbar_organ',
      17: 'percussive_organ',
      18: 'rock_organ',
      19: 'church_organ',
      20: 'reed_organ',
      21: 'accordion',
      22: 'harmonica',
      23: 'tango_accordion',
      // Guitar (24-31)
      24: 'acoustic_guitar_nylon',
      25: 'acoustic_guitar_steel',
      26: 'electric_guitar_jazz',
      27: 'electric_guitar_clean',
      28: 'electric_guitar_muted',
      29: 'overdriven_guitar',
      30: 'distortion_guitar',
      31: 'guitar_harmonics',
      // Bass (32-39)
      32: 'acoustic_bass',
      33: 'electric_bass_finger',
      34: 'electric_bass_pick',
      35: 'fretless_bass',
      36: 'slap_bass_1',
      37: 'slap_bass_2',
      38: 'synth_bass_1',
      39: 'synth_bass_2',
      // Strings (40-47)
      40: 'violin',
      41: 'viola',
      42: 'cello',
      43: 'contrabass',
      44: 'tremolo_strings',
      45: 'pizzicato_strings',
      46: 'orchestral_harp',
      47: 'timpani',
      // Ensemble (48-55)
      48: 'string_ensemble_1',
      49: 'string_ensemble_2',
      50: 'synth_strings_1',
      51: 'synth_strings_2',
      52: 'choir_aahs',
      53: 'voice_oohs',
      54: 'synth_choir',
      55: 'orchestra_hit',
      // Brass (56-63)
      56: 'trumpet',
      57: 'trombone',
      58: 'tuba',
      59: 'muted_trumpet',
      60: 'french_horn',
      61: 'brass_section',
      62: 'synth_brass_1',
      63: 'synth_brass_2',
      // Reed (64-71)
      64: 'soprano_sax',
      65: 'alto_sax',
      66: 'tenor_sax',
      67: 'baritone_sax',
      68: 'oboe',
      69: 'english_horn',
      70: 'bassoon',
      71: 'clarinet',
      // Pipe (72-79)
      72: 'piccolo',
      73: 'flute',
      74: 'recorder',
      75: 'pan_flute',
      76: 'blown_bottle',
      77: 'shakuhachi',
      78: 'whistle',
      79: 'ocarina',
      // Synth Lead (80-87)
      80: 'lead_1_square',
      81: 'lead_2_sawtooth',
      82: 'lead_3_calliope',
      83: 'lead_4_chiff',
      84: 'lead_5_charang',
      85: 'lead_6_voice',
      86: 'lead_7_fifths',
      87: 'lead_8_bass__lead',
      // Synth Pad (88-95)
      88: 'pad_1_new_age',
      89: 'pad_2_warm',
      90: 'pad_3_polysynth',
      91: 'pad_4_choir',
      92: 'pad_5_bowed',
      93: 'pad_6_metallic',
      94: 'pad_7_halo',
      95: 'pad_8_sweep',
    };

    return programMap[midiProgram] || 'acoustic_grand_piano';
  }, []);

  // Create realistic instrument sounds based on MIDI program number
  const createSynth = useCallback(
    async (instrumentInput) => {
      const midiProgram = getMidiProgram();
      const instName =
        typeof instrumentInput === 'object'
          ? instrumentInput.name
          : instrumentInput;

      console.log(
        `[createSynth] Loading soundfont for "${instName}", program: ${midiProgram}, channel: ${typeof instrumentInput === 'object' ? instrumentInput.number : 'unknown'}`,
      );

      // Use soundfont if we have a valid MIDI program number
      if (midiProgram !== null && midiProgram !== undefined) {
        try {
          const soundfontName = getSoundfontInstrument(midiProgram);
          console.log(
            `[createSynth] Loading soundfont instrument: ${soundfontName}`,
          );

          const audioContext = Tone.context.rawContext;
          // Use FluidR3_GM for better compatibility (includes percussion instrument)
          const instrument = await Soundfont.instrument(
            audioContext,
            soundfontName,
            {
              soundfont: 'FluidR3_GM',
            },
          );

          console.log(`[createSynth] Soundfont loaded for ${soundfontName}`);
          return instrument;
        } catch (error) {
          console.warn(
            `[createSynth] Failed to load soundfont, falling back to synthesis:`,
            error,
          );
        }
      }

      // Fallback to Tone.js synthesis if soundfont fails
      console.log(`[createSynth] Using fallback synthesis`);
      let envelope = { attack: 0.1, decay: 0.2, sustain: 0.3, release: 0.5 };
      let oscillatorType = 'sine';
      let filter = null;

      // Use MIDI program number for more accurate instrument selection
      // General MIDI Program Numbers: 0-7 Piano, 8-15 Chromatic Percussion, etc.
      if (midiProgram !== null && midiProgram !== undefined) {
        if (midiProgram >= 0 && midiProgram <= 7) {
          // Piano family
          envelope = { attack: 0.005, decay: 0.4, sustain: 0, release: 1 };
          oscillatorType = 'triangle';
        } else if (midiProgram >= 8 && midiProgram <= 15) {
          // Chromatic Percussion (bells, glockenspiel, etc.)
          envelope = { attack: 0.001, decay: 0.3, sustain: 0.1, release: 0.8 };
          oscillatorType = 'sine';
        } else if (midiProgram >= 16 && midiProgram <= 23) {
          // Organ
          envelope = { attack: 0.05, decay: 0.1, sustain: 0.7, release: 0.2 };
          oscillatorType = 'square';
        } else if (midiProgram >= 24 && midiProgram <= 31) {
          // Guitar
          return new Tone.PluckSynth().toDestination();
        } else if (midiProgram >= 32 && midiProgram <= 39) {
          // Bass
          envelope = { attack: 0.01, decay: 0.2, sustain: 0.5, release: 0.3 };
          oscillatorType = 'sawtooth';
          filter = { frequency: 500, type: 'lowpass' };
        } else if (midiProgram >= 40 && midiProgram <= 47) {
          // Strings
          envelope = { attack: 0.3, decay: 0.1, sustain: 0.6, release: 0.5 };
          oscillatorType = 'sawtooth';
          filter = { frequency: 3000, type: 'lowpass' };
        } else if (midiProgram >= 48 && midiProgram <= 55) {
          // Ensemble (choir, voice)
          envelope = { attack: 0.25, decay: 0.1, sustain: 0.6, release: 0.4 };
          oscillatorType = 'triangle';
          filter = { frequency: 2500, type: 'lowpass' };
        } else if (midiProgram >= 56 && midiProgram <= 63) {
          // Brass
          envelope = { attack: 0.15, decay: 0.2, sustain: 0.4, release: 0.6 };
          oscillatorType = 'sawtooth';
          filter = { frequency: 2500, type: 'lowpass' };
        } else if (midiProgram >= 64 && midiProgram <= 71) {
          // Reed (sax, oboe, clarinet)
          envelope = { attack: 0.1, decay: 0.15, sustain: 0.5, release: 0.4 };
          oscillatorType = 'square';
          filter = { frequency: 2000, type: 'lowpass' };
        } else if (midiProgram >= 72 && midiProgram <= 79) {
          // Pipe (flute, recorder)
          envelope = { attack: 0.2, decay: 0.1, sustain: 0.4, release: 0.3 };
          oscillatorType = 'sine';
        } else if (midiProgram >= 80 && midiProgram <= 87) {
          // Synth Lead
          envelope = { attack: 0.05, decay: 0.2, sustain: 0.6, release: 0.3 };
          oscillatorType = 'sawtooth';
        } else if (midiProgram >= 88 && midiProgram <= 95) {
          // Synth Pad
          envelope = { attack: 0.5, decay: 0.2, sustain: 0.7, release: 1.0 };
          oscillatorType = 'triangle';
        } else if (midiProgram >= 96 && midiProgram <= 103) {
          // Synth Effects
          envelope = { attack: 0.1, decay: 0.3, sustain: 0.5, release: 0.5 };
          oscillatorType = 'square';
        } else if (midiProgram >= 104 && midiProgram <= 111) {
          // Ethnic instruments
          envelope = { attack: 0.05, decay: 0.3, sustain: 0.4, release: 0.4 };
          oscillatorType = 'triangle';
        } else if (midiProgram >= 112 && midiProgram <= 119) {
          // Percussive
          envelope = { attack: 0.001, decay: 0.3, sustain: 0, release: 0.1 };
          oscillatorType = 'triangle';
        } else {
          // Sound Effects (120-127)
          envelope = { attack: 0.1, decay: 0.5, sustain: 0.3, release: 0.5 };
          oscillatorType = 'sine';
        }
      } else {
        // Fallback to name-based detection
        if (name.includes('piano') || name.includes('grand')) {
          envelope = { attack: 0.005, decay: 0.4, sustain: 0, release: 1 };
          oscillatorType = 'sine';
        } else if (name.includes('violin') || name.includes('string')) {
          envelope = { attack: 0.3, decay: 0.1, sustain: 0.6, release: 0.5 };
          oscillatorType = 'sawtooth';
          filter = { frequency: 3000, type: 'lowpass' };
        } else if (name.includes('flute') || name.includes('wind')) {
          envelope = { attack: 0.2, decay: 0.1, sustain: 0.4, release: 0.3 };
          oscillatorType = 'sine';
        } else if (name.includes('horn') || name.includes('brass')) {
          envelope = { attack: 0.15, decay: 0.2, sustain: 0.4, release: 0.6 };
          oscillatorType = 'sawtooth';
          filter = { frequency: 2500, type: 'lowpass' };
        } else if (name.includes('guitar') || name.includes('electric')) {
          return new Tone.PluckSynth().toDestination();
        } else if (name.includes('organ') || name.includes('synth')) {
          envelope = { attack: 0.05, decay: 0.1, sustain: 0.7, release: 0.2 };
          oscillatorType = 'square';
        }
      }

      const synth = new Tone.Synth({
        oscillator: { type: oscillatorType },
        envelope,
      });

      // Add filter if needed for tonal shaping
      if (filter) {
        const filterNode = new Tone.Filter(filter);
        synth.connect(filterNode);
        filterNode.toDestination();
      } else {
        synth.toDestination();
      }

      return synth;
    },
    [getMidiProgram],
  );

  const playSampleSound = useCallback(async () => {
    const instName =
      typeof instrument === 'object' ? instrument.name : instrument;
    const isPercussion = typeof instrument === 'object' && instrument.isDrum;
    const isDrumChannel =
      typeof instrument === 'object' && instrument.number === 9;

    console.log(
      `[playSampleSound] Playing for ${instName}, isDrum: ${isPercussion}, isDrumChannel: ${isDrumChannel}`,
    );

    if (isPlayingSample) return;

    try {
      setIsPlayingSample(true);
      await Tone.start();

      if (Tone.context.state !== 'running') {
        await Tone.context.resume();
      }

      // If on drum channel OR marked as drum, play drum sound only
      if (isPercussion || isDrumChannel) {
        // Play drum sound (synchronous now)
        console.log(`[playSampleSound] Playing drum sound`);
        playDrumSound();
        setTimeout(() => setIsPlayingSample(false), 1500);
      } else {
        // Stop and dispose of existing synth
        if (synthRef.current) {
          // Only dispose if it's a Tone.js synth (has triggerRelease method)
          if (typeof synthRef.current.triggerRelease === 'function') {
            synthRef.current.triggerRelease();
          }
          // Only dispose if it's a Tone.js synth (soundfont instruments don't have dispose)
          if (typeof synthRef.current.dispose === 'function') {
            synthRef.current.dispose();
          }
          synthRef.current = null;
        }

        console.log(`[playSampleSound] Creating melodic synth`);
        synthRef.current = await createSynth(instrument);

        // Detect key from MIDI or use default
        const keyNote = detectKeyFromMidi();
        const note = getNoteFromKey(keyNote);

        console.log(`[playSampleSound] Playing note: ${note}`);

        // Check if it's a soundfont instrument (has .play method) or Tone.js synth
        if (
          synthRef.current.play &&
          typeof synthRef.current.play === 'function'
        ) {
          // Soundfont instrument - convert note name to MIDI number
          const noteToMidi = {
            C: 0,
            'C#': 1,
            D: 2,
            'D#': 3,
            E: 4,
            F: 5,
            'F#': 6,
            G: 7,
            'G#': 8,
            A: 9,
            'A#': 10,
            B: 11,
          };
          const noteName = note.slice(0, -1);
          const octave = parseInt(note.slice(-1));
          const midiNote = noteToMidi[noteName] + (octave + 1) * 12;

          console.log(
            `[playSampleSound] Playing soundfont note ${note} (MIDI ${midiNote})`,
          );
          synthRef.current.play(midiNote, Tone.context.currentTime, {
            duration: 1.5,
            gain: 1.0,
          });
        } else {
          // Tone.js synth
          const now = Tone.now();
          synthRef.current.triggerAttackRelease(note, '1.5s', now);
        }

        setTimeout(() => setIsPlayingSample(false), 1500);
      }
      setIsPlayingSample(false);
    } finally {
      setTimeout(() => {
        // Clean up synth after sound finishes
        if (
          synthRef.current &&
          typeof synthRef.current.dispose === 'function'
        ) {
          synthRef.current.dispose();
          synthRef.current = null;
        }
      }, 1600);
    }
  }, [
    instrument,
    isPlayingSample,
    isDrumTrack,
    createSynth,
    detectKeyFromMidi,
    getNoteFromKey,
    playDrumSound,
  ]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (synthRef.current && typeof synthRef.current.dispose === 'function' && !synthRef.current.disposed) {
        synthRef.current.dispose();
      }
      synthRef.current = null;
    };
  }, []);

  const displayName =
    typeof instrument === 'object'
      ? instrument.isDrum
        ? `Drum ${instrument.group}`
        : instrument.name
      : instrument;

  return (
    <button
      onClick={playSampleSound}
      className={`control-button ${className}`}
      disabled={isPlayingSample}
      title={`Play sample sound for ${displayName}`}
    >
      Play Sample Sound
    </button>
  );
};

export default SampleSoundButton;
