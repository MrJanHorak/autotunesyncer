// App.js
import React, { useState } from 'react';
import axios from 'axios';
import Dropzone from 'react-dropzone';
import ReactPlayer from 'react-player';

function App() {
  const [midiFile, setMidiFile] = useState(null);
  const [videoFiles, setVideoFiles] = useState({});
  const [videoUrls, setVideoUrls] = useState([]);

  const handleMidiUpload = (acceptedFiles) => {
    setMidiFile(acceptedFiles[0]);
  };

  const handleVideoUpload = (instrument, acceptedFiles) => {
    setVideoFiles({ ...videoFiles, [instrument]: acceptedFiles[0] });
  };

  const uploadMidi = async () => {
    const formData = new FormData();
    formData.append('midi', midiFile);
    await axios.post('/upload-midi', formData);
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
            <p>Drag 'n' drop a MIDI file here, or click to select one</p>
          </div>
        )}
      </Dropzone>
      <button onClick={uploadMidi}>Upload MIDI</button>

      <h1>Upload Video Clips</h1>
      <Dropzone onDrop={(acceptedFiles) => handleVideoUpload('piano', acceptedFiles)}>
        {({ getRootProps, getInputProps }) => (
          <div {...getRootProps()}>
            <input {...getInputProps()} />
            <p>Drag 'n' drop a video file for piano here, or click to select one</p>
          </div>
        )}
      </Dropzone>
      <button onClick={uploadVideos}>Upload Videos</button>

      <h1>Video Playback</h1>
      {videoUrls.map((url, index) => (
        <ReactPlayer key={index} url={url} controls />
      ))}
    </div>
  );
}

export default App;