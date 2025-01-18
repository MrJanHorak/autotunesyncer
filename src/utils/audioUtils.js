import * as Tone from 'tone';

export const initializeAudioContext = async (setAudioContextStarted, setIsAudioContextReady, setError) => {
  try {
    await Tone.start();
    const context = Tone.context;
    await context.resume();
    setAudioContextStarted(true);
    setIsAudioContextReady(true);
    console.log('Audio context initialized successfully');
  } catch (error) {
    console.error('Failed to initialize audio context:', error);
    setError('Failed to initialize audio context: ' + error.message);
  }
};