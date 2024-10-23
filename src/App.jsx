import { useState, useRef } from 'react';
import axios from 'axios';
import Dropzone from 'react-dropzone';
import ReactPlayer from 'react-player';
import VideoPlayback from './components/videoPlayback'; 

function App() {
  const [midiFile, setMidiFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState({});
  const [videoUrls, setVideoUrls] = useState([]);
  const [videoBlob, setVideoBlob] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null); // URL for the recorded video
  const [audioUrl, setAudioUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const handleStartRecording = async () => {
    // Reset videoUrl to switch back to live feed
    setVideoUrl(null);

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true,
    });
    streamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }

    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;

    let chunks = [];
    mediaRecorder.ondataavailable = (e) => {
      chunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const videoBlob = new Blob(chunks, { type: 'video/mp4' });
      setVideoBlob(videoBlob);
      const videoUrl = URL.createObjectURL(videoBlob);
      setVideoUrl(videoUrl); // Set the URL for the recorded video
      stream.getTracks().forEach((track) => track.stop()); // Stop all tracks
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleUpload = async () => {
    if (!videoBlob) return;

    const formData = new FormData();
    formData.append('video', new File([videoBlob], 'webcam-video.mp4'));

    try {
      const response = await fetch('http://localhost:3000/api/upload-video', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioUrl(audioUrl);
      } else {
        console.error('Failed to process video');
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleMidiUpload = (acceptedFiles) => {
    setMidiFile(acceptedFiles[0]);
  };

  const handleVideoUpload = (instrument, acceptedFiles) => {
    setVideoFiles({ ...videoFiles, [instrument]: acceptedFiles[0] });
  };

  const uploadMidi = async () => {
    const formData = new FormData();
    formData.append('midi', midiFile);
    await axios.post('http://localhost:3000/upload-midi', formData);
  };

  const uploadVideos = async () => {
    const formData = new FormData();
    for (const [instrument, file] of Object.entries(videoFiles)) {
      formData.append(instrument, file);
    }
    const response = await axios.post('/upload-videos', formData);
    setVideoUrls(response.data.videoUrls);
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

      <h1>Webcam Video to MIDI Audio</h1>
      <div style={{ width: '100%', position: 'relative' }}>
        {videoUrl ? (
          <VideoPlayback videoUrl={videoUrl} /> // Use the VideoPlayback component
        ) : (
          <video ref={videoRef} style={{ width: '100%' }}></video>
        )}
      </div>
      <button onClick={handleStartRecording} disabled={isRecording}>
        Start Recording
      </button>
      <button onClick={handleStopRecording} disabled={!isRecording}>
        Stop Recording
      </button>
      <button onClick={handleUpload} disabled={!videoBlob}>
        Upload Video
      </button>

      {audioUrl && (
        <div>
          <h2>Processed Audio</h2>
          <audio controls src={audioUrl}></audio>
        </div>
      )}

      <h1>Video Playback</h1>
      {videoUrls.map((url, index) => (
        <ReactPlayer key={index} url={url} controls />
      ))}
    </div>
  );
}

export default App;