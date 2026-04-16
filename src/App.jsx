/* eslint-disable no-unused-vars */
import { useEffect, useCallback, useState, useRef } from 'react';

import { isDrumTrack, DRUM_NOTES, getNoteGroup } from './js/drumUtils';
import InstrumentList from './components/InstrumentList/InstrumentList';

import { useMidiProcessing } from './hooks/useMidiProcessing';
import { useVideoRecording } from './hooks/useVideoRecording';

// Components
import MidiUploader from './components/MidiUploader/';
import MidiInfoDisplay from './components/MidiInfoDisplay/MidiInfoDisplay';
import RecordingSection from './components/RecordingSection/RecordingSection';
import AudioContextInitializer from './components/AudioContextInitializer/AudioContextInitializer';
import CompositionSection from './components/CompositionSection/CompositionSection';
import MidiParser from './components/MidiParser/MidiParser';
import ProgressBar from './components/ProgressBar/ProgressBar';
import Grid from './components/Grid/Grid';
import Mixer from './components/Mixer/Mixer';
import PreviewPlayer from './components/PreviewPlayer/PreviewPlayer';

import './App.css';

// Add this helper function at the top
const normalizeInstrumentName = (name) => {
  return name.toLowerCase().replace(/\s+/g, '_');
};

// // Add this helper function to extract drum instruments
// const extractDrumInstruments = (track) => {
//   if (!isDrumTrack(track)) return [];

//   // Get unique MIDI notes from the track
//   const uniqueNotes = new Set(track.notes.map((note) => note.midi));

//   // Map notes to their drum groups
//   const drumNames = new Set();
//   uniqueNotes.forEach((note) => {
//     const drumName = DRUM_NOTES[note];
//     if (drumName) {
//       drumNames.add(drumName);
//     }
//   });

//   // Create instrument objects for each drum group
//   return Array.from(drumNames).map((name) => ({
//     name: name.toLowerCase().replace(/\s+/g, '_'), // Normalize the name
//     family: 'drums',
//     number: -1,
//     isDrum: true,
//   }));
// };

function App() {
  const {
    // parsedMidiData,
    instruments,
    instrumentTrackMap,
    longestNotes,
    onMidiProcessed: processMidiData,
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

  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [midiFile, setMidiFile] = useState(null);
  const [gridArrangement, setGridArrangement] = useState({});
  const [trackVolumes, setTrackVolumes] = useState({});
  const [muteStates, setMuteStates] = useState({});
  const [soloTrack, setSoloTrack] = useState(null);
  const [activeLevels, setActiveLevels] = useState({});
  const lastMeterStateRef = useRef(0);

  // Track which instrument keys have already been queued for pre-caching
  // so we don't send duplicate requests on every re-render.
  const precachedKeysRef = useRef(new Set());

  // Fire-and-forget pre-cache request for one instrument's blob + MIDI notes.
  const triggerPrecache = useCallback((instrumentKey, blob, midiData) => {
    if (precachedKeysRef.current.has(instrumentKey)) return;
    precachedKeysRef.current.add(instrumentKey);

    // Collect unique MIDI notes for this instrument key
    const notes = new Set();
    midiData.tracks.forEach((track) => {
      if (isDrumTrack(track)) {
        // Match drum notes whose group maps to this key
        const expectedKey = `drum_${getNoteGroup(track.notes[0]?.midi ?? 0).toLowerCase().replace(/\s+/g, '_')}`;
        track.notes.forEach((note) => {
          const noteKey = `drum_${getNoteGroup(note.midi).toLowerCase().replace(/\s+/g, '_')}`;
          if (noteKey === instrumentKey) notes.add(note.midi);
        });
      } else {
        const trackKey = track.instrument?.name?.toLowerCase().replace(/\s+/g, '_');
        if (trackKey === instrumentKey) {
          track.notes.forEach((note) => notes.add(note.midi));
        }
      }
    });

    if (notes.size === 0) return;

    const formData = new FormData();
    formData.append('video', blob, `${instrumentKey}.mp4`);
    formData.append('midiNotes', JSON.stringify([...notes]));

    fetch('http://localhost:3000/api/autotune/precache', { method: 'POST', body: formData })
      .then((r) => {
        if (!r.ok) throw new Error(`precache HTTP ${r.status}`);
        console.log(`[precache] Queued ${instrumentKey} (${notes.size} notes)`);
      })
      .catch((err) => console.warn(`[precache] ${instrumentKey} failed:`, err));
  }, []);

  // Trigger precache whenever a new video is recorded AND MIDI is loaded,
  // or when MIDI loads after videos are already recorded.
  useEffect(() => {
    if (!parsedMidiData || Object.keys(videoFiles).length === 0) return;
    for (const [key, blob] of Object.entries(videoFiles)) {
      if (blob instanceof Blob) {
        triggerPrecache(key, blob, parsedMidiData);
      }
    }
  }, [parsedMidiData, videoFiles, triggerPrecache]);

  const handleVolumeChange = (trackKey, volume) => {
    setTrackVolumes((prev) => ({
      ...prev,
      [trackKey]: volume,
    }));
  };

  // Throttled meter update — called up to ~15 Hz from PreviewPlayer's rAF loop.
  // We gate state updates to ~10 Hz here to avoid excessive re-renders.
  const handleMeterUpdate = useCallback((levels) => {
    const now = Date.now();
    if (now - lastMeterStateRef.current < 100) return;
    lastMeterStateRef.current = now;
    setActiveLevels(levels);
  }, []);

  const handleMuteChange = (trackKey, isMuted) => {
    setMuteStates((prev) => ({ ...prev, [trackKey]: isMuted }));
  };

  const handleSoloChange = (trackKey) => {
    setSoloTrack((prev) => (prev === trackKey ? null : trackKey));
  };

  const handleMidiProcessed = (file) => {
    setMidiFile(file);
  };

  const handleParsedMidi = useCallback(
    (midiInfo) => {
      console.log('Parsed MIDI info:', midiInfo);
      setParsedMidiData(midiInfo);
      // Call MIDI processing hook with parsed data
      processMidiData(midiInfo);
    },
    [processMidiData],
  );

  // Add handleRecordingComplete function
  const handleRecordingComplete = useCallback((blob, instrument) => {
    if (!(blob instanceof Blob)) {
      console.error('Invalid blob:', blob);
      return;
    }
    console.log('Instrument: ', instrument);
    if (instrument.isDrum) {
      instrument.name = instrument.group;
    }
    console.log(
      'Handle Recording is complete Instrument after adding drum name:',
      instrument,
    );
    const key = instrument.isDrum
      ? `drum_${instrument.name.toLowerCase().replace(/\s+/g, '_')}`
      : normalizeInstrumentName(instrument.name);

    console.log(
      'Recording complete for instrument:',
      key,
      'blob size:',
      blob.size,
    );

    setVideoFiles((prev) => ({
      ...prev,
      [key]: blob,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Add handleVideoReady function
  const handleVideoReady = useCallback((videoUrl, instrument) => {
    instrument.isDrum ? (instrument.name = instrument.group) : instrument.name;
    const instrumentKey = instrument.isDrum
      ? `drum_${instrument.name}`
      : normalizeInstrumentName(instrument.name);

    setInstrumentVideos((prev) => ({
      ...prev,
      [instrumentKey]: videoUrl,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    <div className='app-container'>
      <AudioContextInitializer
        audioContextStarted={audioContextStarted}
        onInitialize={startAudioContext}
      />
      <MidiUploader onMidiProcessed={handleMidiProcessed} />

      {midiFile && <MidiParser file={midiFile} onParsed={handleParsedMidi} />}

      {parsedMidiData && (
        <>
          <MidiInfoDisplay midiData={parsedMidiData} />
          {instruments.length > 0 && (
            <InstrumentList instruments={instruments} />
          )}

          {instruments.length > 0 && (
            <div
              className='audio-control-section'
              style={{
                margin: '20px 0',
                padding: '20px',
                background: '#f5f5f5',
                borderRadius: '8px',
              }}
            >
              <Mixer
                instruments={instruments}
                volumes={trackVolumes}
                onVolumeChange={handleVolumeChange}
                muteStates={muteStates}
                soloTrack={soloTrack}
                onMuteChange={handleMuteChange}
                onSoloChange={handleSoloChange}
                activeLevels={activeLevels}
              />

              <div style={{ marginTop: '15px' }}>
                <PreviewPlayer
                  midiData={parsedMidiData}
                  videoFiles={videoFiles}
                  volumes={trackVolumes}
                  muteStates={muteStates}
                  soloTrack={soloTrack}
                  instruments={instruments}
                  onMeterUpdate={handleMeterUpdate}
                />
              </div>
            </div>
          )}

          <Grid
            midiData={parsedMidiData}
            onArrangementChange={setGridArrangement}
          />

          {!isReadyToCompose && instruments.length > 0 && (
            <ProgressBar
              current={recordedVideosCount}
              total={instruments.length}
            />
          )}

          <RecordingSection
            instruments={instruments}
            longestNotes={longestNotes}
            onRecordingComplete={handleRecordingComplete}
            onVideoReady={handleVideoReady}
            instrumentVideos={instrumentVideos}
            midiData={parsedMidiData}
          />
        </>
      )}

      {isReadyToCompose && instruments.length > 0 && (
        <CompositionSection
          videoFiles={videoFiles}
          midiData={parsedMidiData}
          instrumentTrackMap={instrumentTrackMap}
          gridArrangement={gridArrangement}
          trackVolumes={trackVolumes}
          muteStates={muteStates}
          soloTrack={soloTrack}
        />
      )}
    </div>
  );
}

export default App;
