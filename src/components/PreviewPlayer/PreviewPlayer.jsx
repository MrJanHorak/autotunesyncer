import { useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import PropTypes from 'prop-types';
import { isDrumTrack, getNoteGroup } from '../../js/drumUtils';

const PreviewPlayer = ({ midiData, videoFiles, volumes, instruments, muteStates = {}, soloTrack = null }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const samplersRef = useRef({});
  const channelsRef = useRef({});
  // Track all scheduled event IDs so we can cancel cleanly
  const scheduledIdsRef = useRef([]);

  // 1. Initialize Tone.js Instruments when videoFiles change
  useEffect(() => {
    const setupAudio = async () => {
      if (!videoFiles || Object.keys(videoFiles).length === 0) return;

      setIsLoaded(false);

      // Cleanup old nodes
      Object.values(samplersRef.current).forEach((s) => s.dispose());
      Object.values(channelsRef.current).forEach((c) => c.dispose());

      const newSamplers = {};
      const newChannels = {};

      for (const inst of instruments) {
        const key = inst.isDrum
          ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
          : inst.name.toLowerCase().replace(/\s+/g, '_');

        if (!videoFiles[key]) continue;

        const fileUrl = URL.createObjectURL(videoFiles[key]);
        const channel = new Tone.Channel(volumes[key] || 0, 0).toDestination();
        newChannels[key] = channel;

        newSamplers[key] = new Tone.Sampler({
          urls: { C4: fileUrl },
          release: 1,
          onload: () => console.log(`Loaded sample for ${key}`),
          onerror: (err) => console.error(`Failed to load sample for ${key}:`, err),
        }).connect(channel);
      }

      try {
        await Tone.loaded();
      } catch (e) {
        console.error('Error loading some samples:', e);
      }

      samplersRef.current = newSamplers;
      channelsRef.current = newChannels;
      setIsLoaded(true);
    };

    setupAudio();

    return () => {
      Object.values(samplersRef.current).forEach((s) => s.dispose());
      Object.values(channelsRef.current).forEach((c) => c.dispose());
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
      setIsPlaying(false);
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
      setIsPlaying(false);
    }, endTime + 0.1);

    Tone.Transport.start();
    setIsPlaying(true);
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
};

export default PreviewPlayer;
