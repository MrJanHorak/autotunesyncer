/* eslint-disable no-unused-vars */
// src/components/VideoRecorder.jsx
import { useRef, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Camera, RefreshCw, Upload, Music, X } from 'lucide-react';
import * as Tone from 'tone';
import { useVideoRecorder, videoService } from '../../services/videoServices';

const ErrorAlert = ({ message, onClose }) => (
  <div className='absolute top-4 left-4 right-4 z-50 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded flex justify-between items-center'>
    <span>{message}</span>
    <button onClick={onClose} className='ml-4'>
      <X className='h-4 w-4' />
    </button>
  </div>
);

ErrorAlert.propTypes = {
  message: PropTypes.string.isRequired,
  onClose: PropTypes.func.isRequired,
};

const VideoRecorder = ({ instrument, onVideoReady }) => {
  const videoRef = useRef(null);
  const [recordedVideoURL, setRecordedVideoURL] = useState(null);
  const [autotunedVideoURL, setAutotunedVideoURL] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [isAutotuning, setIsAutotuning] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [recordingInterval, setRecordingIntervalId] = useState(null);


  const { recordVideo } = useVideoRecorder({
    onRecordingStart: () => {
      setIsRecording(true);
      const intervalId = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
      setRecordingIntervalId(intervalId);
    },
    onRecordingStop: () => {
      setIsRecording(false);
      if (recordingInterval) {
        clearInterval(recordingInterval);
      }
      try {
        // Add any code that might throw an error here
      } finally {
        setIsProcessing(false);
        setIsAutotuning(false);
      }
    },
    onProcessingStart: () => setIsProcessing(true),
    onProcessingComplete: () => setIsProcessing(false),
    onError: (error) => setError(error.message),
  });

  useEffect(() => {
    const currentVideoRef = videoRef.current;
    return () => {
      // Cleanup function
      if (currentVideoRef?.srcObject) {
        currentVideoRef.srcObject.getTracks().forEach((track) => track.stop());
      }
      if (recordedVideoURL) URL.revokeObjectURL(recordedVideoURL);
      if (autotunedVideoURL) URL.revokeObjectURL(autotunedVideoURL);
    };
  }, [recordedVideoURL, autotunedVideoURL]);

  useEffect(() => {
    const currentVideoRef = videoRef.current;
    return () => {
      if (currentVideoRef?.srcObject) {
        currentVideoRef.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, [recordedVideoURL, autotunedVideoURL]);

  const stopRecording = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsRecording(false);
    setRecordingDuration(0);
  };

  const startRecording = async () => {
    try {
      setIsProcessing(true);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;
      videoRef.current.muted = true;
      await videoRef.current.play();

      const result = await recordVideo();
      setIsAutotuning(true);
      const autotunedResult = await videoService.autotuneVideo(result);
      setAutotunedVideoURL(URL.createObjectURL(autotunedResult));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
      setIsAutotuning(false);
    }

  } // Closing brace for startRecording function

  const handleReRecord = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    if (recordedVideoURL) {
      URL.revokeObjectURL(recordedVideoURL);
    }
    if (autotunedVideoURL) {
      URL.revokeObjectURL(autotunedVideoURL);
    }

    setRecordedVideoURL(null);
    setAutotunedVideoURL(null);
    setError(null);
    setRecordingDuration(0);
    setIsRecording(false);
  };

  const uploadAutotunedVideo = () => {
    if (autotunedVideoURL) {
      onVideoReady(autotunedVideoURL);
    }
  };

  const playSampleSound = async () => {
    if (instrument.toLowerCase().includes('drum')) return;

    try {
      const synth = new Tone.Synth().toDestination();
      await Tone.start();
      synth.triggerAttackRelease('C4', '1.5s');
    } catch (err) {
      console.error(err);
      setError('Failed to play sample sound');
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className='relative w-full h-full min-h-[480px] bg-gray-900 rounded-lg overflow-hidden'>
      <video ref={videoRef} className='w-full h-full object-cover' />

      {error && <ErrorAlert message={error} onClose={() => setError(null)} />}

      <div className='absolute bottom-4 left-4 right-4 flex justify-between items-center'>
        <div className='flex gap-2'>
          {isRecording ? (
            <button
              onClick={stopRecording}
              className='flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors'
            >
              <div className='w-3 h-3 rounded-full bg-white animate-pulse' />
              {formatDuration(recordingDuration)}
            </button>
          ) : (
            <>
              <button
                onClick={startRecording}
                className='flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors'
                disabled={isProcessing}
              >
                <Camera className='w-5 h-5' />
                Start Recording
              </button>

              {recordedVideoURL && (
                <button
                  onClick={handleReRecord}
                  className='flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors'
                >
                  <RefreshCw className='w-5 h-5' />
                  Re-record
                </button>
              )}

              <button
                onClick={playSampleSound}
                className='flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors'
                disabled={isProcessing}
              >
                <Music className='w-5 h-5' />
                Play Sample
              </button>

              {autotunedVideoURL && (
                <button
                  onClick={uploadAutotunedVideo}
                  className='flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors'
                  disabled={isProcessing}
                >
                  <Upload className='w-5 h-5' />
                  Upload Autotuned
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {isProcessing && (
        <div className='absolute inset-0 bg-black/50 flex items-center justify-center'>
          <div className='flex flex-col items-center gap-4 text-white'>
            <RefreshCw className='w-8 h-8 animate-spin' />
            <p>Processing video...</p>
          </div>
        </div>
      )}

      {!isRecording && recordedVideoURL && (
        <div className='absolute inset-0 bg-gray-900'>
          <video src={recordedVideoURL} controls className='w-full h-full' />
        </div>
      )}

      {!isRecording && autotunedVideoURL && (
        <div className='absolute inset-0 bg-gray-900'>
          <video src={autotunedVideoURL} controls className='w-full h-full' />
        </div>
      )}
    </div>
  );
};

VideoRecorder.propTypes = {
  instrument: PropTypes.shape({
    name: PropTypes.string.isRequired,
    family: PropTypes.string.isRequired,
    toLowerCase: PropTypes.func.isRequired
  }).isRequired,
  onVideoReady: PropTypes.func.isRequired
};

export default VideoRecorder;