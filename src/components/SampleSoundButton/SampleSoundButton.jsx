/* eslint-disable react/prop-types */
import { useRef, useState, useCallback, useEffect } from 'react';
import * as Tone from 'tone';
import '../styles.css';

const SampleSoundButton = ({ instrument, className }) => {
  const synthRef = useRef(null);
  const [isPlayingSample, setIsPlayingSample] = useState(false);

  const playSampleSound = useCallback(async () => {
    const instrumentType = String(instrument || '').toLowerCase();
    if (instrumentType.includes('drum') || isPlayingSample) return;

    try {
      setIsPlayingSample(true);
      await Tone.start();
      
      if (!synthRef.current || synthRef.current.disposed) {
        synthRef.current = new Tone.Synth().toDestination();
      }

      if (Tone.context.state !== 'running') {
        await Tone.context.resume();
      }

      synthRef.current.triggerAttackRelease('C4', '1.5s');
    } catch (error) {
      console.error('Audio playback failed:', error);
      setIsPlayingSample(false);
    } finally {
      setTimeout(() => setIsPlayingSample(false), 1500);
    }
  }, [instrument, isPlayingSample]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (synthRef.current) {
        synthRef.current.dispose();
      }
    };
  }, []);

  return (
    <button 
      onClick={playSampleSound} 
      className={`control-button ${className}`}
      disabled={isPlayingSample}
    >
      Play Sample Sound
    </button>
  );
};

export default SampleSoundButton;
