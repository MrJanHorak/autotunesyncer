/* eslint-disable no-prototype-builtins */
/* eslint-disable no-unused-vars */
import { useEffect, useCallback } from 'react'; 

import { isDrumTrack, DRUM_GROUPS } from './js/drumUtils';
import InstrumentList from './components/InstrumentList/InstrumentList';

import { useMidiProcessing } from './hooks/useMidiProcessing'; 
import { useVideoRecording } from './hooks/useVideoRecording'; 

import MidiUploader from './components/MidiUploader/'; 
import RecordingSection from './components/RecordingSection/RecordingSection';
import AudioContextInitializer from './components/AudioContextInitializer/AudioContextInitializer';
import CompositionSection from './components/CompositionSection/CompositionSection';


// Add this helper function at the top
const normalizeInstrumentName = (name) => {
  return name.toLowerCase().replace(/\s+/g, '_');
};

// Add this helper function to extract drum instruments
const extractDrumInstruments = (track) => {
  if (!isDrumTrack(track)) return [];

  // Get unique MIDI notes from the track
  const uniqueNotes = new Set(track.notes.map((note) => note.midi));

  // Map notes to their drum groups
  const drumGroups = new Set();
  uniqueNotes.forEach((note) => {
    for (const [group, notes] of Object.entries(DRUM_GROUPS)) {
      if (notes.includes(note)) {
        drumGroups.add(group);
        break;
      }
    }
  });

  // Create instrument objects for each drum group
  return Array.from(drumGroups).map((group) => ({
    name: `drum_${group}`,
    family: 'drums',
    number: -1, // Use -1 to identify as drum instrument
    isDrum: true,
    group: group,
  }));
};

function App() {
  const {
    parsedMidiData,
    instruments,
    instrumentTrackMap,
    longestNotes,
    onMidiProcessed  
  } = useMidiProcessing();
  const {
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
  } = useVideoRecording(instruments);

  // Add handleRecordingComplete function
  const handleRecordingComplete = useCallback((blob, instrument) => {
    if (!(blob instanceof Blob)) {
      console.error('Invalid blob:', blob);
      return;
    }

    const key = instrument.isDrum
      ? `drum_${instrument.group}`
      : normalizeInstrumentName(instrument.name);

    console.log(
      'Recording complete for instrument:',
      key,
      'blob size:',
      blob.size
    );

    setVideoFiles((prev) => ({
      ...prev,
      [key]: blob,
    }));
  }, []);

  // Add handleVideoReady function
  const handleVideoReady = useCallback((videoUrl, instrument) => {
    const instrumentKey = instrument.isDrum
      ? `drum_${instrument.group}`
      : normalizeInstrumentName(instrument.name);

    setInstrumentVideos((prev) => ({
      ...prev,
      [instrumentKey]: videoUrl,
    }));
  }, []);

  // Add click handler to initialize audio context
  useEffect(() => {
    const handleClick = () => {
      if (!isAudioContextReady) {
        startAudioContext();
      }
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [isAudioContextReady, startAudioContext]);

  return (
    <div>
      <AudioContextInitializer
        audioContextStarted={audioContextStarted}
        onInitialize={startAudioContext}
      />
      <MidiUploader onMidiProcessed={onMidiProcessed} />
  
      {instruments.length > 0 && <InstrumentList instruments={instruments} />}
  
      <RecordingSection
        instruments={instruments}
        longestNotes={longestNotes}
        onRecordingComplete={handleRecordingComplete}
        onVideoReady={handleVideoReady}
        instrumentVideos={instrumentVideos}
      />
  
      {!isReadyToCompose && instruments.length > 0 && (
        <div className="mt-4 bg-yellow-100 p-4 rounded">
          <p>Recording Progress: {recordedVideosCount} / {instruments.length}</p>
        </div>
      )}
  
      {isReadyToCompose && instruments.length > 0 && (
        <CompositionSection
          videoFiles={videoFiles}
          midiData={parsedMidiData}
          instrumentTrackMap={instrumentTrackMap}
        />
      )}
    </div>
  );
}

export default App;
