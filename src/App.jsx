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
  const [instrumentTrackMap, setInstrumentTrackMap] = useState({});

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
    setRecordedVideosCount((prevCount) => prevCount + tracks.length);
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

  const getTrackNotes = (midiData, trackIndex) => {
    const track = midiData.tracks[trackIndex];
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
      const [instrument, trackIndex] = key.split('-');
      const trackNotes = getTrackNotes(midiData, parseInt(trackIndex, 10));
      const videoBlob = videoFiles[key];

      if (!videoBlob) {
        console.error(`No video blob found for key: ${key}`);
        return;
      }

      console.log('Video blob:', videoBlob); // Debugging

      const videoElement = document.createElement('video');
      try {
        videoElement.src = URL.createObjectURL(videoBlob);
      } catch (error) {
        console.error('Failed to create object URL:', error);
        return;
      }
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
                  handleRecordingComplete(blob, instrument.name, index)
                }
                style={{ width: '300px', height: '200px' }} // Custom styles to make the recorder smaller
                instrument={instrument.name}
                trackIndex={index}
              />
            </div>
          ))}
        </div>
      )}

      {console.log('Recorded Videos Count:', recordedVideosCount)}
      {console.log('Number of Tracks:', parsedMidiData?.tracks.length)}

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