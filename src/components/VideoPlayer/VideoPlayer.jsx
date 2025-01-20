import { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import * as Tone from 'tone';

const VideoPlayer = ({ videoFiles, midiData }) => {
  const videoRefs = useRef({});
  const audioNodesRef = useRef({});
  const [isSetup, setIsSetup] = useState(false);

  // Cleanup function for audio nodes
  const cleanupAudioNodes = () => {
    Object.values(audioNodesRef.current).forEach(nodes => {
      if (nodes.source) {
        try {
          nodes.source.disconnect();
        } catch (e) {
          console.log('Error disconnecting source:', e);
        }
      }
      if (nodes.gain) {
        nodes.gain.dispose();
      }
      if (nodes.pitchShift) {
        nodes.pitchShift.dispose();
      }
    });
    audioNodesRef.current = {};
  };

  useEffect(() => {
    return () => {
      cleanupAudioNodes();
    };
  }, []);

  useEffect(() => {
    if (midiData && videoFiles && !isSetup) {
      setupPlayback(midiData, videoFiles);
    }
  }, [midiData, videoFiles, isSetup]);

  const getTrackNotes = (midiData, trackIndex) => {
    const track = midiData.tracks[trackIndex];
    return track?.notes || [];
  };

  const setupPlayback = async (midiData, videoFiles) => {
    try {
      // Clean up any existing audio nodes first
      cleanupAudioNodes();

      // Make sure Tone.js is started
      await Tone.start();
      
      // Process each video file
      for (const key of Object.keys(videoFiles)) {
        const [instrument, trackIndex] = key.split('-');
        const trackNotes = getTrackNotes(midiData, parseInt(trackIndex, 10));
        const videoBlob = videoFiles[key];

        console.log('Processing track:', {
          key,
          instrument,
          trackIndex,
          noteCount: trackNotes.length,
          hasVideoBlob: !!videoBlob
        });

        if (!videoBlob || !videoRefs.current[key]) {
          console.warn(`Missing video blob or element for key: ${key}`);
          continue;
        }

        const videoElement = videoRefs.current[key];
        videoElement.src = URL.createObjectURL(videoBlob);

        // Create audio processing chain
        try {
          const audioCtx = Tone.getContext().rawContext;
          const source = audioCtx.createMediaElementSource(videoElement);
          const gain = new Tone.Gain(0).toDestination();
          const pitchShift = new Tone.PitchShift().connect(gain);
          
          source.connect(pitchShift.input);
          
          audioNodesRef.current[key] = {
            source,
            gain,
            pitchShift
          };

          // Schedule notes
          if (trackNotes.length > 0) {
            trackNotes.forEach(note => {
              const startTime = Tone.Time(note.time).toSeconds();
              const endTime = startTime + note.duration;
              
              // Schedule gain automation
              gain.gain.cancelScheduledValues(startTime - 0.1);
              gain.gain.setValueAtTime(0, startTime - 0.1);
              gain.gain.linearRampToValueAtTime(1, startTime);
              gain.gain.setValueAtTime(1, endTime - 0.1);
              gain.gain.linearRampToValueAtTime(0, endTime);

              // Schedule pitch
              const midiNote = note.midi;
              const baseNote = 60; // Middle C
              const pitchDiff = midiNote - baseNote;
              pitchShift.pitch.setValueAtTime(pitchDiff, startTime);
            });
          }
        } catch (error) {
          console.error(`Error setting up audio for ${key}:`, error);
        }
      }

      setIsSetup(true);
    } catch (error) {
      console.error('Error in setupPlayback:', error);
    }
  };

  const handlePlayAll = async () => {
    try {
      await Tone.start();
      Tone.Transport.stop();
      Tone.Transport.position = 0;
      
      // Reset and play all videos
      Object.keys(videoRefs.current).forEach((key) => {
        const videoElement = videoRefs.current[key];
        if (videoElement) {
          videoElement.currentTime = 0;
          videoElement.play().catch(error => {
            console.error(`Error playing video ${key}:`, error);
          });
        }
      });

      Tone.Transport.start();
    } catch (error) {
      console.error('Error in handlePlayAll:', error);
    }
  };

  return (
    <div>
      <button 
        onClick={handlePlayAll}
        disabled={!isSetup}
      >
        Play All Videos
      </button>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
        {Object.keys(videoFiles).map((key) => (
          <video
            key={key}
            ref={(el) => (videoRefs.current[key] = el)}
            controls
            style={{ width: '300px', height: '200px' }}
          />
        ))}
      </div>
    </div>
  );
};

VideoPlayer.propTypes = {
  videoFiles: PropTypes.object.isRequired,
  midiData: PropTypes.object.isRequired,
};

export default VideoPlayer;