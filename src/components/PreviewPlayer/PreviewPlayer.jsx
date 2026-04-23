import { useEffect, useRef, useState, useCallback } from 'react';
import * as Tone from 'tone';
import PropTypes from 'prop-types';
import { isDrumTrack, getNoteGroup } from '../../js/drumUtils';

const PreviewPlayer = ({ midiData, videoFiles, volumes, instruments, muteStates = {}, soloTrack = null, onMeterUpdate, onPlayStateChange }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const samplersRef = useRef({});
  const channelsRef = useRef({});
  const metersRef = useRef({});
  const scheduledIdsRef = useRef([]);
  const rafRef = useRef(null);
  const isPlayingRef = useRef(false);

  // Throttle meter callback to ~15 Hz to avoid excessive re-renders
  const lastMeterCallRef = useRef(0);
  const emitMeterUpdate = useCallback(() => {
    if (!onMeterUpdate) return;
    const now = performance.now();
    if (now - lastMeterCallRef.current < 66) return; // ~15 Hz
    lastMeterCallRef.current = now;
    const levels = {};
    Object.entries(metersRef.current).forEach(([key, meter]) => {
      levels[key] = meter.getValue();
    });
    onMeterUpdate(levels);
  }, [onMeterUpdate]);

  const startMeterLoop = useCallback(() => {
    const tick = () => {
      if (!isPlayingRef.current) return;
      emitMeterUpdate();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [emitMeterUpdate]);

  const stopMeterLoop = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    // Zero-out all meters on stop
    if (onMeterUpdate) {
      const zeroed = {};
      Object.keys(metersRef.current).forEach((k) => { zeroed[k] = -Infinity; });
      onMeterUpdate(zeroed);
    }
  }, [onMeterUpdate]);

  // 1. Initialize Tone.js Instruments when videoFiles change
  useEffect(() => {
    const setupAudio = async () => {
      if (!videoFiles || Object.keys(videoFiles).length === 0) return;

      setIsLoaded(false);

      // Cleanup old nodes
      Object.values(samplersRef.current).forEach((s) => s.dispose());
      Object.values(channelsRef.current).forEach((c) => c.dispose());
      Object.values(metersRef.current).forEach((m) => m.dispose());
      // Revoke old blob URLs stored on samplers
      Object.values(samplersRef.current).forEach((s) => {
        if (s._blobUrl) URL.revokeObjectURL(s._blobUrl);
      });

      const newSamplers = {};
      const newChannels = {};
      const newMeters = {};

      for (const inst of instruments) {
        const key = inst.isDrum
          ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
          : inst.name.toLowerCase().replace(/\s+/g, '_');

        if (!videoFiles[key]) continue;

        const fileUrl = URL.createObjectURL(videoFiles[key]);
        const channel = new Tone.Channel(volumes[key] || 0, 0).toDestination();
        const meter = new Tone.Meter({ smoothing: 0.8 });
        channel.connect(meter);
        newChannels[key] = channel;
        newMeters[key] = meter;

        const sampler = new Tone.Sampler({
          urls: { C4: fileUrl },
          release: 1,
          onload: () => console.log(`Loaded sample for ${key}`),
          onerror: (err) => console.error(`Failed to load sample for ${key}:`, err),
        }).connect(channel);
        sampler._blobUrl = fileUrl;
        newSamplers[key] = sampler;
      }

      try {
        await Tone.loaded();
      } catch (e) {
        console.error('Error loading some samples:', e);
      }

      samplersRef.current = newSamplers;
      channelsRef.current = newChannels;
      metersRef.current = newMeters;
      setIsLoaded(true);
    };

    setupAudio();

    return () => {
      Object.values(samplersRef.current).forEach((s) => {
        if (s._blobUrl) URL.revokeObjectURL(s._blobUrl);
        s.dispose();
      });
      Object.values(channelsRef.current).forEach((c) => c.dispose());
      Object.values(metersRef.current).forEach((m) => m.dispose());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoFiles, instruments]);

  // 2. Update volumes/mute/solo in real-time
  useEffect(() => {
    const hasSolo = soloTrack !== null;
    Object.entries(channelsRef.current).forEach(([key, channel]) => {
      const isMuted = muteStates[key];
      const isSolo = key === soloTrack;
      if (isMuted || (hasSolo && !isSolo)) {
        channel.volume.rampTo(-Infinity, 0.05);
      } else {
        channel.volume.rampTo(volumes[key] ?? 0, 0.1);
      }
    });
  }, [volumes, muteStates, soloTrack]);

  // 3. Handle playback
  const togglePlayback = async () => {
    if (isPlaying) {
      Tone.Transport.stop();
      Tone.Transport.cancel();
      scheduledIdsRef.current = [];
      isPlayingRef.current = false;
      stopMeterLoop();
      setIsPlaying(false);
      onPlayStateChange?.(false);
      return;
    }

    await Tone.start();
    if (Tone.context.state !== 'running') {
      await Tone.context.resume();
    }

    Tone.Transport.cancel();
    scheduledIdsRef.current = [];

    // Use one shared base offset so all notes are scheduled from the same clock reference
    midiData.tracks.forEach((track) => {
      const drumTrack = isDrumTrack(track);

      if (drumTrack) {
        // Each note maps to a specific drum-group instrument key
        track.notes.forEach((note) => {
          const group = getNoteGroup(note.midi);
          const key = `drum_${group.toLowerCase().replace(/\s+/g, '_')}`;
          const sampler = samplersRef.current[key];
          if (!sampler?.loaded) return;

          const id = Tone.Transport.schedule((time) => {
            sampler.triggerAttackRelease('C4', note.duration, time, note.velocity);
          }, note.time);
          scheduledIdsRef.current.push(id);
        });
      } else {
        // Find matching melodic instrument by normalized name
        const normalizedTrackName = track.instrument?.name?.toLowerCase().replace(/\s+/g, '_');
        const sampler = samplersRef.current[normalizedTrackName];
        if (!sampler?.loaded) return;

        track.notes.forEach((note) => {
          const id = Tone.Transport.schedule((time) => {
            sampler.triggerAttackRelease(
              Tone.Frequency(note.midi, 'midi').toNote(),
              note.duration,
              time,
              note.velocity,
            );
          }, note.time);
          scheduledIdsRef.current.push(id);
        });
      }
    });

    // Auto-stop when MIDI ends
    const endTime = midiData.duration ?? 0;
    Tone.Transport.schedule(() => {
      Tone.Transport.stop();
      Tone.Transport.cancel();
      scheduledIdsRef.current = [];
      isPlayingRef.current = false;
      stopMeterLoop();
      setIsPlaying(false);
      onPlayStateChange?.(false);
    }, endTime + 0.1);

    isPlayingRef.current = true;
    Tone.Transport.start();
    startMeterLoop();
    setIsPlaying(true);
    onPlayStateChange?.(true);
  };

  return (
    <div className='preview-player'>
      <button
        onClick={togglePlayback}
        disabled={!isLoaded}
        className={`px-4 py-2 rounded ${isLoaded ? 'bg-green-500 text-white' : 'bg-gray-300'}`}
        style={{
          padding: '10px 20px',
          fontSize: '16px',
          fontWeight: 'bold',
          cursor: isLoaded ? 'pointer' : 'not-allowed',
          backgroundColor: isLoaded ? (isPlaying ? '#e74c3c' : '#2ecc71') : '#95a5a6',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          transition: 'background-color 0.3s',
        }}
      >
        {!isLoaded ? 'Loading Audio...' : isPlaying ? '⏹ Stop Preview' : '▶ Play Preview'}
      </button>
    </div>
  );
};

PreviewPlayer.propTypes = {
  midiData: PropTypes.object,
  videoFiles: PropTypes.object,
  volumes: PropTypes.object,
  instruments: PropTypes.array,
  muteStates: PropTypes.object,
  soloTrack: PropTypes.string,
  onMeterUpdate: PropTypes.func,
  onPlayStateChange: PropTypes.func,
};

export default PreviewPlayer;
