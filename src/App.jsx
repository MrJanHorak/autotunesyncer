/* eslint-disable no-prototype-builtins */
/* eslint-disable no-unused-vars */
import { useState, useEffect, useCallback } from 'react'; // Add useCallback import
import Dropzone from 'react-dropzone';
import VideoRecorder from './components/videoRecorder'; // Updated import
import VideoComposer from './components/VideoComposer'; // Updated import
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';
import { isDrumTrack, DRUM_GROUPS } from './js/drumUtils';

// Add this helper function at the top
const normalizeInstrumentName = (name) => {
  return name.toLowerCase().replace(/\s+/g, '_');
};

// Add this helper function
const calculateLongestNotes = (midiData) => {
  const instrumentDurations = {};
  
  midiData.tracks.forEach(track => {
    if (isDrumTrack(track)) {
      // Group notes by drum type
      const drumGroups = new Map();
      track.notes.forEach(note => {
        for (const [group, notes] of Object.entries(DRUM_GROUPS)) {
          if (notes.includes(note.midi)) {
            if (!drumGroups.has(group)) {
              drumGroups.set(group, []);
            }
            drumGroups.get(group).push(note);
            break;
          }
        }
      });

      // Calculate longest duration for each drum group
      drumGroups.forEach((notes, group) => {
        const groupName = `drum_${group}`;
        let longestDuration = 0;
        notes.forEach(note => {
          longestDuration = Math.max(longestDuration, note.duration);
        });
        instrumentDurations[groupName] = longestDuration;
      });
    } else {
      // Handle regular instruments as before
      const instrumentName = track.instrument.name;
      let longestDuration = 0;
      
      track.notes.forEach(note => {
        longestDuration = Math.max(longestDuration, note.duration);
      });
      
      if (instrumentName in instrumentDurations) {
        instrumentDurations[instrumentName] = Math.max(
          instrumentDurations[instrumentName],
          longestDuration
        );
      } else {
        instrumentDurations[instrumentName] = longestDuration;
      }
    }
  });
  
  return instrumentDurations;
};

// Add this helper function to extract drum instruments
const extractDrumInstruments = (track) => {
  if (!isDrumTrack(track)) return [];
  
  // Get unique MIDI notes from the track
  const uniqueNotes = new Set(track.notes.map(note => note.midi));
  
  // Map notes to their drum groups
  const drumGroups = new Set();
  uniqueNotes.forEach(note => {
    for (const [group, notes] of Object.entries(DRUM_GROUPS)) {
      if (notes.includes(note)) {
        drumGroups.add(group);
        break;
      }
    }
  });

  // Create instrument objects for each drum group
  return Array.from(drumGroups).map(group => ({
    name: `drum_${group}`,
    family: 'drums',
    number: -1, // Use -1 to identify as drum instrument
    isDrum: true,
    group: group
  }));
};

const App = () => {
  const [midiFile, setMidiFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState({});
  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [instruments, setInstruments] = useState([]);
  const [recordedVideosCount, setRecordedVideosCount] = useState(0);
  const [audioContextStarted, setAudioContextStarted] = useState(false);
  const [instrumentTrackMap, setInstrumentTrackMap] = useState({});
  const [isReadyToCompose, setIsReadyToCompose] = useState(false); // New state
  const [error, setError] = useState(null); // Add error state
  const [longestNotes, setLongestNotes] = useState({});
  const [instrumentVideos, setInstrumentVideos] = useState({});
  const [readyForComposition, setReadyForComposition] = useState(false);
  const [isAudioContextReady, setIsAudioContextReady] = useState(false);

  // Add a useEffect to monitor video recording progress
  useEffect(() => {
    if (!instruments.length) return;
    
    const allInstrumentKeys = instruments.map(instrument => 
      instrument.isDrum ? `drum_${instrument.group}` : normalizeInstrumentName(instrument.name)
    );
    
    const recordedCount = Object.keys(videoFiles).length;
    const isComplete = recordedCount === allInstrumentKeys.length;

    // Batch state updates
    if (recordedCount !== recordedVideosCount) {
      setRecordedVideosCount(recordedCount);
    }
    
    if (isComplete !== readyForComposition) {
      setReadyForComposition(isComplete);
      setIsReadyToCompose(isComplete);
    }
  }, [instruments, videoFiles, recordedVideosCount, readyForComposition]);

  // Memoize checkCompositionReady
  const checkCompositionReady = useCallback((videos) => {
    if (!instruments.length) return false;
    
    const requiredInstruments = new Set(
      instruments.map(instrument => 
        instrument.isDrum ? `drum_${instrument.group}` : normalizeInstrumentName(instrument.name)
      )
    );

    return Array.from(requiredInstruments).every(inst => 
      videos.hasOwnProperty(inst)
    );
  }, [instruments]);

  // Update composition ready state when videos change
  useEffect(() => {
    const isReady = checkCompositionReady(videoFiles);
    if (isReady !== readyForComposition) {
      setReadyForComposition(isReady);
      setIsReadyToCompose(isReady);
    }
  }, [videoFiles, checkCompositionReady, readyForComposition]);

  // Add this effect to initialize audio context on first user interaction
  useEffect(() => {
    const initAudioContext = async () => {
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

    // Add click handler to document
    const handleClick = () => {
      if (!isAudioContextReady) {
        initAudioContext();
      }
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [isAudioContextReady]);

  // Update handleRecordingComplete to use the blob directly
  const handleRecordingComplete = useCallback((blob, instrument) => {
    if (!(blob instanceof Blob)) {
      console.error('Invalid blob:', blob);
      return;
    }
  
    const key = instrument.isDrum ? 
      `drum_${instrument.group}` : 
      normalizeInstrumentName(instrument.name);
  
    console.log('Recording complete for instrument:', key, 'blob size:', blob.size);
    
    // Update video files
    setVideoFiles(prev => {
      const newFiles = { ...prev, [key]: blob };
      console.log('Updated video files:', Object.keys(newFiles));
      return newFiles;
    });

    // Create object URL for preview
    const videoUrl = URL.createObjectURL(blob);
    setInstrumentVideos(prev => ({
      ...prev,
      [key]: videoUrl
    }));

    // Explicitly trigger composition check
    const updatedVideoCount = Object.keys(videoFiles).length + 1;
    setRecordedVideosCount(updatedVideoCount);
    
    if (updatedVideoCount === instruments.length) {
      setIsReadyToCompose(true);
      setReadyForComposition(true);
    }
  }, [instruments.length, videoFiles]);

  // Remove the separate monitoring effects and combine them into one
  useEffect(() => {
    if (!instruments.length) return;

    const videoCount = Object.keys(videoFiles).length;
    console.log('Current videos:', Object.keys(videoFiles));
    console.log('Video count:', videoCount, 'Required:', instruments.length);

    const isComplete = videoCount === instruments.length;
    setRecordedVideosCount(videoCount);
    setReadyForComposition(isComplete);
    setIsReadyToCompose(isComplete);
  }, [instruments, videoFiles]);

  const handleMidiUpload = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    
    try {
      // First, initialize audio context if not started
      if (!audioContextStarted) {
        // Create a temporary user interaction to start audio context
        const tempContext = new (window.AudioContext || window.webkitAudioContext)();
        await tempContext.resume();
        await Tone.start();
        setAudioContextStarted(true);
      }
  
      setMidiFile(file);
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const arrayBuffer = e.target.result;
        try {
          const midi = new Midi(arrayBuffer);
          setParsedMidiData(midi);
          
          // Extract all instruments including drum groups
          const instrumentSet = new Set();
          midi.tracks.forEach(track => {
            if (isDrumTrack(track)) {
              // Add individual drum instruments
              extractDrumInstruments(track).forEach(drumInst => {
                instrumentSet.add(JSON.stringify(drumInst));
              });
            } else {
              // Add regular instruments
              instrumentSet.add(JSON.stringify(track.instrument));
            }
          });
  
          const instrumentData = Array.from(instrumentSet).map(item => JSON.parse(item));
          setInstruments(instrumentData);
  
          // Calculate longest notes including drum groups
          const durations = calculateLongestNotes(midi);
          setLongestNotes(durations);
  
          // Create mapping including drum groups
          const trackMap = {};
          midi.tracks.forEach((track, index) => {
            if (isDrumTrack(track)) {
              // Map each drum group to this track
              extractDrumInstruments(track).forEach(drumInst => {
                trackMap[drumInst.name] = [index];
              });
            } else {
              const instrumentName = track.instrument.name;
              if (!trackMap[instrumentName]) {
                trackMap[instrumentName] = [];
              }
              trackMap[instrumentName].push(index);
            }
          });
          setInstrumentTrackMap(trackMap);
        } catch (error) {
          console.error('Error parsing MIDI file:', error);
          setError('Error parsing MIDI file: ' + error.message);
        }
      };
  
      reader.readAsArrayBuffer(file);
    } catch (error) {
      console.error('Error initializing audio:', error);
      setError('Error initializing audio: ' + error.message);
    }
  };

  const extractInstruments = (obj, instruments = new Set()) => {
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        if (key === 'instrument' && obj[key].name) {
          instruments.add(JSON.stringify(obj[key])); // Convert object to string to ensure uniqueness
        } else if (typeof obj[key] === 'object' && obj[key] !== null) {
          extractInstruments(obj[key], instruments);
        }
      }
    }
    return instruments;
  };

  // Memoize handleVideoReady to prevent unnecessary re-renders
  const handleVideoReady = useCallback((videoUrl, instrument) => {
    const instrumentKey = instrument.isDrum ? 
      `drum_${instrument.group}` : 
      normalizeInstrumentName(instrument.name);
    
    // Only update if the URL has changed
    setInstrumentVideos(prev => {
      if (prev[instrumentKey] === videoUrl) return prev;
      return {
        ...prev,
        [instrumentKey]: videoUrl
      };
    });
  }, []); // Empty dependency array since this function doesn't depend on any props or state

  // Add composition section render
  const renderCompositionSection = () => {
    if (!readyForComposition) {
      return (
        <div className="mt-4 p-4 bg-yellow-100 text-yellow-700 rounded">
          Please record or upload videos for all instruments to enable composition.
        </div>
      );
    }

    return (
      <div className="mt-4">
        <h2 className="text-xl font-bold mb-4">Video Composition</h2>
        <VideoComposer 
          videoFiles={videoFiles} 
          midiData={parsedMidiData} 
        />
      </div>
    );
  };

  return (
    <div>
      {/* Add a message to inform users about audio context */}
      {!audioContextStarted && (
        <div className="bg-yellow-100 p-4 rounded mb-4">
          Click anywhere on the page to initialize audio system
        </div>
      )}
      
      <h1>Upload MIDI File</h1>
      <Dropzone onDrop={handleMidiUpload}>
        {({ getRootProps, getInputProps }) => (
          <div {...getRootProps()}>
            <input {...getInputProps()} />
            <p>
              Drag &apos;n&apos; drop a MIDI file here, or click to select one
            </p>
          </div>
        )}
      </Dropzone>

      {instruments.length > 0 && (
        <div>
          <h2>Instruments</h2>
          <ul>
            {instruments.map((instrument, index) => (
              <li key={index}>
                {instrument.family} - {instrument.name} (Number:{' '}
                {instrument.number})
              </li>
            ))}
          </ul>
        </div>
      )}

      {instruments.map((instrument, index) => {
        const instrumentName = instrument.isDrum 
          ? `drum_${instrument.group}`
          : instrument.name;
        const minDuration = longestNotes[instrumentName] || 0;
        const recommendedDuration = Math.ceil(minDuration + 1);

        return (
          <div key={index} style={{ marginBottom: '20px' }}>
            <h3>
              {instrument.isDrum 
                ? `Drum - ${instrument.group.charAt(0).toUpperCase() + instrument.group.slice(1)}`
                : `${instrument.family} - ${instrument.name}`
              }
            </h3>
            <p>Minimum recording duration: {recommendedDuration} seconds</p>
            <VideoRecorder
              onRecordingComplete={(blob) => handleRecordingComplete(blob, instrument, index)}
              style={{ width: '300px', height: '200px' }}
              instrument={instrument}
              onVideoReady={(url) => handleVideoReady(url, instrument)}
              minDuration={recommendedDuration}
              currentVideo={instrumentVideos[instrumentName]}
              audioEnabled={audioContextStarted}
            />
          </div>
        );
      })}

      {isReadyToCompose && (
        <div className="mt-4 bg-green-100 p-4 rounded">
          <h2 className="text-xl font-bold mb-2">Ready to Compose</h2>
          <p>Recorded {recordedVideosCount} videos out of {instruments.length} instruments</p>
          <VideoComposer 
            videoFiles={videoFiles} 
            midiData={parsedMidiData}
            instrumentTrackMap={instrumentTrackMap} // Add this prop
          />
        </div>
      )}

      {!isReadyToCompose && instruments.length > 0 && (
        <div className="mt-4 bg-yellow-100 p-4 rounded">
          <p>Recording Progress: {recordedVideosCount} / {instruments.length}</p>
        </div>
      )}

      {/* Add composition section at the bottom */}
      {renderCompositionSection()}
    </div>
  );
}

export default App;
