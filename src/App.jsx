import { useState } from 'react';
import axios from 'axios';
import Dropzone from 'react-dropzone';
import ReactPlayer from 'react-player';
import VideoRecorder from './components/VideoRecorder';
import { Midi } from '@tonejs/midi';

function App() {
  const [midiFile, setMidiFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState({});
  const [videoUrls, setVideoUrls] = useState([]);
  const [videoBlob, setVideoBlob] = useState(null);
  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [instruments, setInstruments] = useState([]);

  const handleRecordingComplete = (blob) => {
    setVideoBlob(blob);
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
        const instrumentData = Array.from(instrumentSet).map(item => JSON.parse(item)); // Convert back to objects
        setInstruments(instrumentData);
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

  const uploadMidi = async () => {
    if (!midiFile) return;

    const formData = new FormData();
    formData.append('midi', midiFile);

    try {
      await axios.post('http://localhost:3000/upload-midi', formData);
    } catch (error) {
      console.error('Error uploading MIDI file:', error);
    }
  };

  const uploadVideos = async () => {
    const formData = new FormData();
    for (const [instrument, file] of Object.entries(videoFiles)) {
      formData.append(instrument, file);
    }

    try {
      const response = await axios.post('http://localhost:3000/upload-videos', formData);
      setVideoUrls(response.data.videoUrls);
    } catch (error) {
      console.error('Error uploading videos:', error);
    }
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
      <button onClick={uploadMidi}>Upload MIDI</button>

      {instruments.length > 0 && (
        <div>
          <h2>Instruments</h2>
          <ul>
            {instruments.map((instrument, index) => (
              <li key={index}>
                {instrument.family} - {instrument.name} (Number: {instrument.number})
              </li>
            ))}
          </ul>
        </div>
      )}

      <h1>Upload Video Clips</h1>
      <Dropzone
        onDrop={(acceptedFiles) => handleVideoUpload('piano', acceptedFiles)}
      >
        {({ getRootProps, getInputProps }) => (
          <div {...getRootProps()}>
            <input {...getInputProps()} />
            <p>
              Drag &apos;n&apos; drop a MIDI file here, or click to select one
            </p>
          </div>
        )}
      </Dropzone>
      <button onClick={uploadVideos}>Upload Videos</button>

      <VideoRecorder onRecordingComplete={handleRecordingComplete} />

      <h1>Video Playback</h1>
      {videoUrls.map((url, index) => (
        <ReactPlayer key={index} url={url} controls />
      ))}
    </div>
  );
}

export default App;