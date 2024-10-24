import { useState } from 'react';
import Dropzone from 'react-dropzone';
import VideoRecorder from './components/VideoRecorder';
import { Midi } from '@tonejs/midi';
import { YIN } from 'pitchfinder';
import * as Tone from 'tone';

function App() {
  const [midiFile, setMidiFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState({});
  const [parsedMidiData, setParsedMidiData] = useState(null);
  const [instruments, setInstruments] = useState([]);
  const [recordedVideosCount, setRecordedVideosCount] = useState(0);
  const [audioContextStarted, setAudioContextStarted] = useState(false);

  const handleRecordingComplete = (instrument, trackIndex, blob) => {
    console.log('Recording complete:', instrument, trackIndex, blob);
    setVideoFiles((prev) => ({
      ...prev,
      [`${instrument}-${trackIndex}`]: blob,
    }));
    setRecordedVideosCount((prevCount) => prevCount + 1);
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
        console.log('Parsed MIDI data:', midi);
        const instrumentSet = extractInstruments(midi);
        const instrumentData = Array.from(instrumentSet).map((item) =>
          JSON.parse(item)
        ); // Convert back to objects
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

  const getTrackNotes = (midiData, trackIndex) => {
    const track = midiData.tracks[trackIndex];
    console.log('Track:', track);
    console.log('Notes:', track.notes);
    return track.notes.map(note => ({
      time: note.time,
      midi: note.midi,
      duration: note.duration,
    }));
  };

  const startAudioContext = async () => {
    await Tone.start();
    setAudioContextStarted(true);
  };

  const schedulePlayback = (midiData, videoFiles) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const startTime = audioContext.currentTime;

    Object.keys(videoFiles).forEach(async (key) => {
      console.log('Scheduling playback for:', key);
      const [instrument, trackIndex] = key.split('-');
      console.log('Instrument:', instrument);
      console.log('Track index:', trackIndex);
      console.log('Video blob:', videoFiles[trackIndex]);
      console.log('key:', key);
      const trackNotes = getTrackNotes(midiData, parseInt(trackIndex, 10));
      const videoBlob = videoFiles[key];

      if (!videoBlob) {
        console.error(`No video blob found for key: ${key}`);
        return;
      }

      const videoElement = document.createElement('video');
      videoElement.src = URL.createObjectURL(videoBlob);
      videoElement.muted = true;

      const source = audioContext.createMediaElementSource(videoElement);
      const gainNode = audioContext.createGain();
      source.connect(gainNode).connect(audioContext.destination);

      trackNotes.forEach(note => {
        const noteStartTime = startTime + note.time;
        const noteEndTime = noteStartTime + note.duration;

        gainNode.gain.setValueAtTime(1, noteStartTime);
        gainNode.gain.setValueAtTime(0, noteEndTime);
      });

      videoElement.play();
    });
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
                  handleRecordingComplete(instrument.name, index, blob)
                }
                style={{ width: '300px', height: '200px' }} // Custom styles to make the recorder smaller
                instrument={instrument.name}
                trackIndex={index}
              />
            </div>
          ))}
        </div>
      )}

      {recordedVideosCount === parsedMidiData?.tracks.length && (
        <div>
          <button onClick={() => schedulePlayback(parsedMidiData, videoFiles)}>
            Play All Videos
          </button>
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