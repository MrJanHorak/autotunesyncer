/* eslint-disable react/prop-types */
import { useRef, useState, useEffect } from 'react';

const VideoRecorder = ({ onRecordingComplete, style, instrument, trackIndex }) => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]); // Use ref for chunks
  const [recordedVideoURL, setRecordedVideoURL] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (isRecording) {
      startRecording();
    }
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true; // Mute audio during recording
        videoRef.current.play();
      }

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data); // Use ref to store chunks
          console.log('Data available:', event.data.size); // Debugging
        } else {
          console.log('No data available'); // Debugging
        }
      };

      mediaRecorder.onstop = () => {
        setIsProcessing(true);
        console.log('Chunks:', chunksRef.current); // Debugging
        const blob = new Blob(chunksRef.current, { type: 'video/webm' }); // Change MIME type to webm
        const videoURL = URL.createObjectURL(blob);
        console.log('Generated video URL:', videoURL); // Debugging
        console.log('Blob size:', blob.size); // Debugging
        setRecordedVideoURL(videoURL);
        onRecordingComplete(blob, instrument, trackIndex); // Ensure blob is passed correctly
        chunksRef.current = []; // Clear chunks
        setIsRecording(false); // Reset recording state
        stopMediaStream();
        setIsProcessing(false);
      };

      mediaRecorder.start();
      console.log('MediaRecorder started:', mediaRecorder.state); // Debugging
    } catch (error) {
      console.error('Error accessing media devices.', error);
      setIsRecording(false); // Reset recording state if there's an error
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      console.log('MediaRecorder stopped:', mediaRecorderRef.current.state); // Debugging
    }
  };

  const stopMediaStream = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const handleReRecord = () => {
    setRecordedVideoURL(null);
    setIsRecording(false);
  };

  return (
    <div style={{ ...style, position: 'relative' }}>
      <video ref={videoRef} style={{ width: '100%', height: '100%' }}></video>
      {isRecording ? (
        <button onClick={stopRecording} style={{ position: 'absolute', bottom: '10px', left: '10px', zIndex: 10 }}>
          Stop Recording
        </button>
      ) : (
        <>
          <button onClick={() => setIsRecording(true)} style={{ position: 'absolute', bottom: '10px', left: '10px', zIndex: 10 }}>
            Start Recording
          </button>
          {recordedVideoURL && (
            <button onClick={handleReRecord} style={{ position: 'absolute', bottom: '10px', left: '120px', zIndex: 10 }}>
              Re-record
            </button>
          )}
        </>
      )}
      {isProcessing && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 20 }}>
          <div className="spinner"></div> {/* Add your spinner or activity indicator here */}
        </div>
      )}
      {!isRecording && recordedVideoURL && (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 5 }}>
          <video src={recordedVideoURL} controls style={{ width: '100%', height: '100%' }}></video>
        </div>
      )}
    </div>
  );
};

export default VideoRecorder;