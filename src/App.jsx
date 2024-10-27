/* eslint-disable no-prototype-builtins */
/* eslint-disable no-unused-vars */
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

  const handleRecordingComplete = (blob, instrument) => {
    console.log('Recording complete:', instrument, blob);
    if (!(blob instanceof Blob)) {
      console.error('Invalid blob:', blob);
      return;
    }

    // Save the video blob for the instrument
    setVideoFiles((prev) => ({
      ...prev,
      [instrument]: blob,
    }));
    setRecordedVideosCount((prevCount) => prevCount + 1);

    // Check if all videos are recorded
    if (recordedVideosCount + 1 === instruments.length) {
      setIsReadyToCompose(true);
    }
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
        ); // Convert back to objects
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
    await Tone.start();
    setAudioContextStarted(true);
  };

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

      {instruments.length > 0 && (
        <div className='recordingHolder'>
          <h2>Record Videos for Instruments</h2>
          {instruments.map((instrument, index) => (
            <div key={index} style={{ marginBottom: '20px' }}>
              <h3>
                {instrument.family} - {instrument.name}
              </h3>
              <VideoRecorder
                onRecordingComplete={(blob) =>
                  handleRecordingComplete(blob, instrument.name)
                }
                style={{ width: '300px', height: '200px' }} // Custom styles to make the recorder smaller
                instrument={instrument.name}
              />
            </div>
          ))}
        </div>
      )}

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