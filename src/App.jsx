import { useState } from 'react';
import Dropzone from 'react-dropzone';
import VideoRecorder from './components/VideoRecorder';
import VideoComposer from './components/VideoComposer';  // Updated import
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';

function App() {
  const [midiFile, setMidiFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState({});
  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [instruments, setInstruments] = useState([]);
  const [recordedVideosCount, setRecordedVideosCount] = useState(0);
  const [audioContextStarted, setAudioContextStarted] = useState(false);
  const [instrumentTrackMap, setInstrumentTrackMap] = useState({});
  const [isReadyToCompose, setIsReadyToCompose] = useState(false);  // New state

  const handleRecordingComplete = (blob, instrument, trackIndex) => {
    console.log('Recording complete:', instrument, trackIndex, blob);
    if (!(blob instanceof Blob)) {
      console.error('Invalid blob:', blob);
      return;
    }

    // Duplicate the video blob for each track that uses the same instrument
    const newVideoFiles = { ...videoFiles };
    const tracks = instrumentTrackMap[instrument] || [];
    tracks.forEach((trackIdx) => {
      newVideoFiles[`${instrument}-${trackIdx}`] = blob;
    });

    setVideoFiles(newVideoFiles);
    setRecordedVideosCount((prevCount) => {
      const newCount = prevCount + 1;
      // Check if we've recorded all instruments
      if (newCount === instruments.length) {
        setIsReadyToCompose(true);
      }
      return newCount;
    });
  };

  const handleMidiUpload = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setMidiFile(file);

    const reader = new FileReader();
    reader.onload = async (e) => {
      const arrayBuffer = e.target.result;
      try {
        const midi = new Midi(arrayBuffer);
        setParsedMidiData(midi);
        const instrumentSet = extractInstruments(midi);
        const instrumentData = Array.from(instrumentSet).map((item) =>
          JSON.parse(item)
        );
        setInstruments(instrumentData);

        // Create a mapping of instruments to their respective tracks
        const trackMap = {};
        midi.tracks.forEach((track, index) => {
          const instrumentName = track.instrument.name;
          if (!trackMap[instrumentName]) {
            trackMap[instrumentName] = [];
          }
          trackMap[instrumentName].push(index);
        });
        setInstrumentTrackMap(trackMap);
      } catch (error) {
        console.error('Error parsing MIDI file:', error);
      }
    };
    reader.readAsArrayBuffer(file);
  };

  const extractInstruments = (midiData) => {
    const instruments = new Set();
    
    // Return empty set if no MIDI data
    if (!midiData) return instruments;
    
    // If we have tracks, process them directly
    if (midiData.tracks && Array.isArray(midiData.tracks)) {
      midiData.tracks.forEach((track, index) => {
        // Only add instrument if the track has notes and a valid instrument
        if (track?.notes?.length > 0 && track?.instrument?.name) {
          const instrumentData = {
            name: track.instrument.name,
            family: track.instrument.family,
            number: track.instrument.number,
            trackIndex: index
          };
          instruments.add(JSON.stringify(instrumentData));
        }
      });
      return instruments;
    }
    
    // Fallback recursive search if tracks aren't at top level
    const recursiveSearch = (obj) => {
      if (!obj || typeof obj !== 'object') return;
      
      // Check if current object is an instrument with notes
      if (obj.instrument?.name && obj.notes?.length > 0) {
        const instrumentData = {
          name: obj.instrument.name,
          family: obj.instrument.family,
          number: obj.instrument.number
        };
        instruments.add(JSON.stringify(instrumentData));
        return;
      }
      
      // Recursively search all object properties
      for (const key in obj) {
        if (obj.hasOwnProperty(key) && typeof obj[key] === 'object') {
          recursiveSearch(obj[key]);
        }
      }
    };
    
    recursiveSearch(midiData);
    return instruments;
  };

  const startAudioContext = async () => {
    await Tone.start();
    setAudioContextStarted(true);
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Upload MIDI File</h1>
      <Dropzone onDrop={handleMidiUpload}>
        {({ getRootProps, getInputProps }) => (
          <div 
            {...getRootProps()} 
            className="border-2 border-dashed border-gray-300 p-4 mb-4 cursor-pointer hover:border-gray-400"
          >
            <input {...getInputProps()} />
            <p>Drag &apos;n&apos; drop a MIDI file here, or click to select one</p>
          </div>
        )}
      </Dropzone>

      {instruments.length > 0 && (
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-2">Instruments</h2>
          <ul className="list-disc pl-5">
            {instruments.map((instrument, index) => (
              <li key={index}>
                {instrument.family} - {instrument.name} (Number:{' '}
                {instrument.number})
              </li>
            ))}
          </ul>
        </div>
      )}

      {instruments.length > 0 && (
        <div className="recordingHolder space-y-6">
          <h2 className="text-xl font-semibold">Record Videos for Instruments</h2>
          {instruments.map((instrument, index) => (
            <div key={index} className="p-4 border rounded">
              <h3 className="text-lg font-medium mb-2">
                {instrument.family} - {instrument.name}
              </h3>
              <VideoRecorder
                onRecordingComplete={(blob) =>
                  handleRecordingComplete(blob, instrument.name, index)
                }
                style={{ width: '320px', height: '240px' }}
                instrument={instrument.name}
                trackIndex={index}
              />
            </div>
          ))}
        </div>
      )}

      {isReadyToCompose && parsedMidiData && (
        <div className="mt-6">
          <VideoComposer 
            videoFiles={videoFiles} 
            midiData={parsedMidiData} 
          />
        </div>
      )}

      {!audioContextStarted && (
        <button 
          onClick={startAudioContext}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Start Audio Context
        </button>
      )}
    </div>
  );
}

export default App;