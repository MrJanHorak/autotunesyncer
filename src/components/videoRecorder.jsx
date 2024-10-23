import { useState, useRef } from 'react';
import VideoPlayback from './VideoPlayback';

function App() {
  const [videoBlob, setVideoBlob] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const originalVolumeRef = useRef(null);

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

    // Mute the audio output
    originalVolumeRef.current = videoRef.current.volume;
    videoRef.current.volume = 0;

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

      // Restore the original volume
      videoRef.current.volume = originalVolumeRef.current;
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

  return (
    <div>
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
    </div>
  );
}

export default App;