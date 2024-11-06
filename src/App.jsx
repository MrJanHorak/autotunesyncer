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

function App() {
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

  const handleRecordingComplete = (blob, instrument, trackIndex) => {
    console.log('Recording complete:', instrument, blob);
    if (!(blob instanceof Blob)) {
      console.error('Invalid blob:', blob);
      return;
    }
  
    // Handle drum recordings
    const key = instrument.isDrum ? 
      `drum_${instrument.group}` : 
      normalizeInstrumentName(instrument.name);
  
    const formData = new FormData();
    formData.append(`videos[${key}]`, blob);
  
    setVideoFiles(prev => ({
      ...prev,
      [key]: blob
    }));
    setRecordedVideosCount(prev => prev + 1);
  };

  // Add useEffect to handle the state update
  useEffect(() => {
    if (recordedVideosCount + 1 === instruments.length) {
      setIsReadyToCompose(true);
    }
  }, [recordedVideosCount, instruments.length]);

  const handleMidiUpload = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setMidiFile(file);

    const reader = new FileReader();
    reader.onload = async (e) => {
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
      }
    };
    reader.readAsArrayBuffer(file);
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

  const startAudioContext = async () => {
    try {
      await Tone.start();
      setAudioContextStarted(true);
    } catch (err) {
      console.error('Failed to start audio context:', err);
      setError('Failed to initialize audio. Please try again.');
    }
  };

  const handleVideoReady = useCallback((url, instrument) => {
    const normalizedName = normalizeInstrumentName(instrument.name);
    setVideoFiles(prev => {
      // Only update if the URL is different
      if (prev[normalizedName] !== url) {
        return {
          ...prev,
          [normalizedName]: url
        };
      }
      return prev;
    });
    // Only increment count if we haven't recorded this instrument before
    setRecordedVideosCount(prev => {
      if (!videoFiles[normalizedName]) {
        return prev + 1;
      }
      return prev;
    });
  }, [videoFiles]);

  return (
    <div>
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
        const minDuration = longestNotes[instrument.name] || 0;
        const recommendedDuration = Math.ceil(minDuration + 1); // Add 1 second buffer
        
        return (
          <div key={index} style={{ marginBottom: '20px' }}>
            <h3>
              {instrument.isDrum ? (
                `Drum - ${instrument.group.charAt(0).toUpperCase() + instrument.group.slice(1)}`
              ) : (
                `${instrument.family} - ${instrument.name}`
              )}
            </h3>
            <p>
              Minimum recording duration: {recommendedDuration} seconds
            </p>
            <VideoRecorder
              onRecordingComplete={(blob) =>
                handleRecordingComplete(blob, instrument, index)
              }
              style={{ width: '300px', height: '200px' }}
              instrument={instrument}
              onVideoReady={(url) => handleVideoReady(url, instrument)}
              minDuration={recommendedDuration}
            />
          </div>
        );
      })}

      {console.log('Recorded Videos Count:', recordedVideosCount)}
      {console.log('Number of Tracks:', parsedMidiData?.tracks.length)}

      {isReadyToCompose && (
        <div>
          <VideoComposer videoFiles={videoFiles} midiData={parsedMidiData} />
        </div>
      )}

      {!audioContextStarted && (
        <div>
          <button onClick={startAudioContext}>Start Audio Context</button>
        </div>
      )}
    </div>
  );
}

export default App;
