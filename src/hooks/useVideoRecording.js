import { useState, useEffect } from 'react';
import { initializeAudioContext } from '../utils/audioUtils';
import { normalizeInstrumentName } from '../utils/midiUtils';

export const useVideoRecording = (instruments) => {
  const [videoFiles, setVideoFiles] = useState({});
  const [recordedVideosCount, setRecordedVideosCount] = useState(0);
  const [instrumentVideos, setInstrumentVideos] = useState({});
  const [isReadyToCompose, setIsReadyToCompose] = useState(false);
  const [audioContextStarted, setAudioContextStarted] = useState(false);
  const [isAudioContextReady, setIsAudioContextReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!instruments?.length) return;

    const requiredRecordings = instruments.map((instrument) => {
      const effectiveName = instrument.isDrum
        ? (instrument.group || instrument.name || '')
        : (instrument.name || '');
      return instrument.isDrum
        ? `drum_${effectiveName.toLowerCase().replace(/\s+/g, '_')}`
        : normalizeInstrumentName(effectiveName);
    });

    const hasAllRecordings = requiredRecordings.every(
      (instrumentName) => !!videoFiles[instrumentName]
    );
    setIsReadyToCompose(hasAllRecordings);
    setRecordedVideosCount(Object.keys(videoFiles).length);
  }, [instruments, videoFiles]);

  const startAudioContext = async () => {
    try {
      await initializeAudioContext(
        setAudioContextStarted,
        setIsAudioContextReady,
        setError
      );
    } catch (err) {
      setError(err.message);
    }
  };

  return {
    videoFiles,
    setVideoFiles,
    recordedVideosCount,
    setRecordedVideosCount,
    instrumentVideos,
    setInstrumentVideos,
    isReadyToCompose,
    setIsReadyToCompose,
    audioContextStarted,
    isAudioContextReady,
    error,
    startAudioContext,
  };
};
