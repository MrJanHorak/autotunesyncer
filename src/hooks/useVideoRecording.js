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

    const requiredRecordings = instruments.map(instrument => 
      instrument.isDrum ? `drum_${instrument.group}` : normalizeInstrumentName(instrument.name)
    );
    
    const hasAllRecordings = requiredRecordings.every(
      instrumentName => !!videoFiles[instrumentName]
    );

    setIsReadyToCompose(hasAllRecordings);
    setRecordedVideosCount(Object.keys(videoFiles).length);
    
    console.log('Required recordings:', requiredRecordings);
    console.log('Current videos:', Object.keys(videoFiles));
    console.log('Ready to compose:', hasAllRecordings);
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
    startAudioContext
  };
};