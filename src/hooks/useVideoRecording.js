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
      console.log(' required recordings Instrument:', instrument);
      if (instrument.isDrum) {
        console.log(' inside isDrum:', instrument.isDrum);
        instrument.name = instrument.group;
      }
      console.log('after and outside of isDrum Instrument:', instrument);
      return instrument.isDrum
        ? `drum_${instrument.name.toLowerCase().replace(/\s+/g, '_')}`
        : normalizeInstrumentName(instrument.name);
    });

    const hasAllRecordings = requiredRecordings.every(
      (instrumentName) => !!videoFiles[instrumentName]
    );
    console.log('hasAllRecordings:', hasAllRecordings);
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
    startAudioContext,
  };
};
