import { useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import PropTypes from 'prop-types';

const PreviewPlayer = ({ midiData, videoFiles, volumes, instruments }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const samplersRef = useRef({});
  const channelsRef = useRef({});

  // 1. Initialize Tone.js Instruments when videoFiles change
  useEffect(() => {
    const setupAudio = async () => {
      if (!videoFiles || Object.keys(videoFiles).length === 0) return;
      
      setIsLoaded(false);
      
      // Cleanup old nodes
      Object.values(samplersRef.current).forEach(s => s.dispose());
      Object.values(channelsRef.current).forEach(c => c.dispose());

      const newSamplers = {};
      const newChannels = {};

      // Create a channel and sampler for each instrument
      for (const inst of instruments) {
        const key = inst.isDrum 
          ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
          : inst.name.toLowerCase().replace(/\s+/g, '_');

        if (!videoFiles[key]) continue;

        // Create a URL for the blob
        const fileUrl = URL.createObjectURL(videoFiles[key]);

        // Create Channel (for volume control)
        const channel = new Tone.Channel(volumes[key] || 0, 0).toDestination();
        newChannels[key] = channel;

        // Create Sampler
        // We assume the recorded clip is C4 (middle C) for melodic instruments
        // For drums, pitch mapping is handled differently, but Sampler works for now
        newSamplers[key] = new Tone.Sampler({
          urls: { C4: fileUrl },
          release: 1,
          onload: () => {
            console.log(`Loaded sample for ${key}`);
          },
          onerror: (err) => {
            console.error(`Failed to load sample for ${key}:`, err);
          }
        }).connect(channel);
      }

      try {
        await Tone.loaded();
        console.log('All samples loaded');
        samplersRef.current = newSamplers;
        channelsRef.current = newChannels;
        setIsLoaded(true);
      } catch (e) {
        console.error('Error loading samples:', e);
        // Still set loaded to true so we can try to play what loaded? 
        // Or maybe just log it.
        // If Tone.loaded() fails, it means at least one buffer failed.
        // We should probably still allow playback of others.
        samplersRef.current = newSamplers;
        channelsRef.current = newChannels;
        setIsLoaded(true);
      }
    };

    setupAudio();
    
    return () => {
      // Cleanup
      Object.values(samplersRef.current).forEach(s => s.dispose());
      Object.values(channelsRef.current).forEach(c => c.dispose());
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoFiles, instruments]); // Re-run if files change

  // 2. Update Volumes in real-time
  useEffect(() => {
    Object.entries(volumes).forEach(([key, vol]) => {
      if (channelsRef.current[key]) {
        channelsRef.current[key].volume.rampTo(vol, 0.1);
      }
    });
  }, [volumes]);

  // 3. Handle Playback
  const togglePlayback = async () => {
    if (isPlaying) {
      Tone.Transport.stop();
      Tone.Transport.cancel(); // Clear scheduled events
      setIsPlaying(false);
    } else {
      await Tone.start();
      if (Tone.context.state !== 'running') {
        await Tone.context.resume();
      }
      
      // Schedule MIDI events
      midiData.tracks.forEach(track => {
        // Find which instrument this track belongs to
        const inst = instruments.find(i => i.number === track.instrument.number);
        if (!inst) return;

        const key = inst.isDrum 
          ? `drum_${inst.group.toLowerCase().replace(/\s+/g, '_')}`
          : inst.name.toLowerCase().replace(/\s+/g, '_');

        const sampler = samplersRef.current[key];
        if (!sampler) return;

        track.notes.forEach(note => {
          Tone.Transport.schedule((time) => {
            // Trigger the sample
            // For drums, we might just trigger 'C4' regardless of note, 
            // or map specific drum notes if you have multiple samples per drum track
            if (sampler.loaded) {
                sampler.triggerAttackRelease(
                inst.isDrum ? 'C4' : Tone.Frequency(note.midi, "midi").toNote(), 
                note.duration, 
                time, 
                note.velocity
                );
            }
          }, note.time);
        });
      });

      Tone.Transport.start();
      setIsPlaying(true);
    }
  };

  return (
    <div className="preview-player">
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
            transition: 'background-color 0.3s'
        }}
      >
        {!isLoaded ? 'Loading Audio...' : isPlaying ? 'Stop Preview' : '▶ Play Preview'}
      </button>
    </div>
  );
};

PreviewPlayer.propTypes = {
  midiData: PropTypes.object,
  videoFiles: PropTypes.object,
  volumes: PropTypes.object,
  instruments: PropTypes.array
};

export default PreviewPlayer;
