/* eslint-disable react/prop-types */
import { useRef, useState, useEffect } from 'react';
import { handleRecord, uploadVideo } from '../js/handleRecordVideo';
import * as Tone from 'tone';

const VideoRecorder = ({ style, instrument }) => {
  const videoRef = useRef(null);
  const [recordedVideoURL, setRecordedVideoURL] = useState(null);
  const [autotunedVideoURL, setAutotunedVideoURL] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (isRecording) {
      startRecording();
    }
  }, [isRecording]);

  const startRecording = async () => {
    setIsProcessing(true);
    await handleRecord(setRecordedVideoURL, setAutotunedVideoURL);
    setIsProcessing(false);
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  const handleReRecord = () => {
    setRecordedVideoURL(null);
    setAutotunedVideoURL(null);
    setIsRecording(false);
  };

  const uploadAutotunedVideo = async () => {
    if (!autotunedVideoURL) return;

    const response = await fetch(autotunedVideoURL);
    const autotunedVideoBlob = await response.blob();
    const autotunedVideoFile = new File([autotunedVideoBlob], 'autotuned-video.mp4');

    await uploadVideo(autotunedVideoFile);
    alert('Autotuned video uploaded successfully!');
  };

  const playSampleSound = async () => {
    if (instrument.toLowerCase().includes('drum')) return;

    const synth = new Tone.Synth().toDestination();
    await Tone.start();
    synth.triggerAttackRelease('C4', '1.5s');
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
          <button onClick={playSampleSound} style={{ position: 'absolute', bottom: '10px', left: '230px', zIndex: 10 }}>
            Play Sample Sound
          </button>
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
      {!isRecording && autotunedVideoURL && (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 5 }}>
          <video src={autotunedVideoURL} controls style={{ width: '100%', height: '100%' }}></video>
          <button onClick={uploadAutotunedVideo} style={{ position: 'absolute', bottom: '10px', left: '10px', zIndex: 10 }}>
            Upload Autotuned Video
          </button>
        </div>
      )}
    </div>
  );
};

export default VideoRecorder;