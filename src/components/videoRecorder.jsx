/* eslint-disable react/prop-types */
import { useRef, useState, useEffect, useCallback } from 'react';
import { handleRecord, handleUploadedVideoAutotune } from '../js/handleRecordVideo';
import ControlButtons from './ControlButtons';
import LoadingSpinner from './LoadingSpinner';
import './styles.css';
import { isDrumTrack } from '../js/drumUtils';
import CountdownTimer from './CountdownTimer';
import VideoTrimmer from './VideoTrimmer';

const useRecordingState = (currentVideo) => {
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const recordingTimer = useRef(null);
  
  const [recordingState, setRecordingState] = useState({
    isRecording: false,
    showCountdown: false,
    isProcessing: false,
    recordingDuration: 0,
    recordedURL: null,
    autotunedURL: currentVideo || null,
    isCountingDown: false, // Add this new state
  });

  const cleanupMediaStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      cleanupMediaStream();
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
      if (recordingState.recordedURL) URL.revokeObjectURL(recordingState.recordedURL);
      if (recordingState.autotunedURL) URL.revokeObjectURL(recordingState.autotunedURL);
    };
  }, [cleanupMediaStream, recordingState.recordedURL, recordingState.autotunedURL]);

  return {
    videoRef,
    mediaStreamRef,
    recordingTimer,
    recordingState,
    setRecordingState,
    cleanupMediaStream,
  };
};

const VideoRecorder = ({ style, instrument, onVideoReady, minDuration, currentVideo }) => {
  const {
    videoRef,
    mediaStreamRef,
    recordingTimer,
    recordingState,
    setRecordingState,
    cleanupMediaStream,
  } = useRecordingState(currentVideo);

  const [isAutotuneEnabled, setIsAutotuneEnabled] = useState(true);
  const [isDrum] = useState(() => isDrumTrack(instrument));
  const [isUploadMode, setIsUploadMode] = useState(false);
  const [showTrimmer, setShowTrimmer] = useState(false);
  const countdownDuration = 3; // Can be made configurable later
  const duration = 60; // Example duration, replace with actual value
  const STEP = 1; // Example step value, replace with actual value
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(duration);
  
  const handleTimeUpdate = (value, isStart) => {
    if (isStart) {
      setStartTime(value);
    } else {
      setEndTime(value);
    }
  };
  
  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = time % 60;
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };
  
  const handleTrim = () => {
    // Implement trim logic here
    console.log('Trim applied from', startTime, 'to', endTime);
  };

  useEffect(() => {
    if (recordingState.autotunedURL && !recordingState.isProcessing) {
      onVideoReady?.(recordingState.autotunedURL, instrument);
    }
  }, [recordingState.autotunedURL, recordingState.isProcessing, instrument, onVideoReady]);

  // Update videoState when currentVideo changes
  useEffect(() => {
    if (currentVideo && currentVideo !== recordingState.autotunedURL) {
      setRecordingState(prev => ({
        ...prev,
        autotunedURL: currentVideo,
        isProcessing: false
      }));
    }
  }, [currentVideo]);

  useEffect(() => {
    // When recording starts, initialize the recording timer
    if (recordingState.isRecording) {
      setRecordingState(prev => ({ ...prev, recordingDuration: 0 }));
      const timer = setInterval(() => {
        setRecordingState(prev => ({ ...prev, recordingDuration: prev.recordingDuration + 1 }));
      }, 1000);
      recordingTimer.current = timer;
    } else {
      // Clear timer when recording stops
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
    }

    return () => {
      const currentTimer = recordingTimer.current;
      if (currentTimer) {
        clearInterval(currentTimer);
      }
    };
  }, [recordingState.isRecording]);

  const startRecording = async () => {
    try {
      // Clean up any existing streams before starting new recording
      cleanupMediaStream();
      setRecordingState(prev => ({ 
        ...prev, 
        showCountdown: true, 
        isCountingDown: true 
      }));
    } catch (error) {
      console.error('Recording failed:', error);
      alert('Failed to start recording. Please check your camera permissions.');
      setRecordingState(prev => ({ 
        ...prev, 
        isRecording: false, 
        isCountingDown: false 
      }));
    }
  };

  const handleCountdownComplete = useCallback(async () => {
    setRecordingState(prev => ({
      ...prev,
      showCountdown: false,
      isCountingDown: false,
      isProcessing: false,
      recordingDuration: 0,
    }));

    try {
      cleanupMediaStream();
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      mediaStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }

      setRecordingState(prev => ({ ...prev, isRecording: true }));

      const recordingDurationMs = (minDuration + 0.5) * 1000;

      await handleRecord(
        (recordedURL) => {
          setRecordingState(prev => ({ ...prev, recordedURL, isProcessing: true }));
        },
        (autotunedURL) => {
          const finalURL = isDrum ? recordingState.recordedURL : autotunedURL;
          setRecordingState(prev => ({
            ...prev,
            autotunedURL: finalURL,
            isProcessing: false
          }));
        },
        !isDrum && isAutotuneEnabled,
        recordingDurationMs
      );
    } catch (error) {
      console.error('Recording failed:', error);
      setRecordingState(prev => ({
        ...prev,
        isProcessing: false,
        isRecording: false,
      }));
    }
  }, [cleanupMediaStream, isDrum, isAutotuneEnabled, minDuration]);

  const stopRecording = useCallback(() => {
    console.log('Stopping recording at duration:', recordingState.recordingDuration);
    
    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }
    
    cleanupMediaStream();
    setRecordingState(prev => ({ ...prev, isRecording: false }));
  }, [recordingState.recordingDuration, cleanupMediaStream, recordingTimer]);

  const handleReRecord = useCallback(() => {
    cleanupMediaStream();
    setRecordingState(prev => ({
      ...prev,
      recordedURL: null,
      autotunedURL: null,
      isRecording: false,
      isProcessing: false,
      recordingDuration: 0
    }));
  }, [cleanupMediaStream]);

  const handleTrimComplete = (trimmedVideoUrl) => {
    setRecordingState(prev => ({ ...prev, autotunedURL: trimmedVideoUrl }));
    setShowTrimmer(false);
    onVideoReady?.(trimmedVideoUrl, instrument);
  };

  const handleFileUpload = async (event) => {
    console.log(`Uploading for instrument: ${instrument}`); // Debug log
    const file = event.target.files[0];
    if (!file) return;
  
    // Validate file type
    if (!file.type.startsWith('video/')) {
      alert('Please upload a valid video file');
      return;
    }
  
    setRecordingState(prev => ({ ...prev, isProcessing: true }));
    try {
      const uploadedVideoUrl = URL.createObjectURL(file);
      
      // If it's a drum track or autotune is disabled, use the uploaded video directly
      if (isDrum || !isAutotuneEnabled) {
        setRecordingState({
          recordedURL: uploadedVideoUrl,
          autotunedURL: uploadedVideoUrl
        });
        // Explicitly pass the instrument parameter to onVideoReady
        onVideoReady?.(uploadedVideoUrl, instrument);
      } else {
        // Handle autotune processing for uploaded video
        setRecordingState(prev => ({ ...prev, recordedURL: uploadedVideoUrl }));
        
        // Pass instrument to handleUploadedVideoAutotune
        await handleUploadedVideoAutotune(
          file,
          (autotunedURL) => {
            setRecordingState(prev => ({ ...prev, autotunedURL }));
            // Explicitly pass the instrument parameter to onVideoReady
            onVideoReady?.(autotunedURL, instrument);
          },
          instrument // Add instrument as parameter if needed by handleUploadedVideoAutotune
        );
      }
    } catch (error) {
      console.error('Error processing uploaded video:', error);
      alert('Failed to process the uploaded video');
    } finally {
      setRecordingState(prev => ({ ...prev, isProcessing: false }));
    }
  };

  const renderVideo = () => {
    if (recordingState.isRecording) {
      return (
        <div className="video-container">
          <video
            ref={videoRef}
            className='video-element'
            muted
            autoPlay
            playsInline
          ></video>
          <div className="recording-duration">
            Recording: {recordingState.recordingDuration}s / {minDuration}s minimum
          </div>
        </div>
      );
    }

    if (recordingState.autotunedURL) {
      return (
        <video
          key={recordingState.autotunedURL}
          ref={showTrimmer ? videoRef : null}
          src={recordingState.autotunedURL}
          className='video-element'
          controls={!showTrimmer}
          autoPlay
          playsInline
          onError={(e) => console.error('Video error:', e)}
        ></video>
      );
    }

    if (recordingState.recordedURL) {
      return (
        <video
          src={recordingState.recordedURL}
          className='video-element'
          controls
          autoPlay
          playsInline
        ></video>
      );
    }

    return (
      <div className='video-element'>
        <p>No video available</p>
      </div>
    );
  };

  const renderControls = () => {
    // Get the display name for the instrument
    const instrumentName = instrument.isDrum 
      ? `drum_${instrument.group}`
      : instrument.name;
      
    const uploadId = `video-upload-${instrumentName}`; // Use normalized name for ID
  
    return (
      <div className='controls-section'>
        <div className="mode-selector">
          <button 
            onClick={() => setIsUploadMode(false)}
            className={!isUploadMode ? 'active' : ''}
          >
            Record Video
          </button>
          <button 
            onClick={() => setIsUploadMode(true)}
            className={isUploadMode ? 'active' : ''}
          >
            Upload Video
          </button>
        </div>
  
        {isUploadMode ? (
          <div className="upload-controls">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              id={uploadId}
              style={{ display: 'none' }}
            />
            <label htmlFor={uploadId} className="upload-button">
              Choose Video File for {instrumentName} {/* Use the display name */}
            </label>
          </div>
        ) : (
          <ControlButtons
            isRecording={recordingState.isRecording}
            hasRecordedVideo={!!recordingState.recordedURL}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            onReRecord={handleReRecord}
            disabled={recordingState.isCountingDown} // Add this prop
            instrument={instrumentName} // Pass the string name instead of object
          />
        )}
  
        {recordingState.autotunedURL && !recordingState.isRecording && (
          <button onClick={() => setShowTrimmer(!showTrimmer)}>
            {showTrimmer ? 'Hide Trimmer' : 'Trim Video'}
          </button>
        )}
        <label>
          <input
            type="checkbox"
            checked={isAutotuneEnabled}
            onChange={(e) => setIsAutotuneEnabled(e.target.checked)}
          />
          <span>Enable Autotune</span> {/* Wrap text in span */}
        </label>
      </div>
    );
  };

  return (
    <>
      <div className='recorder-wrapper' style={style}>
        <div className='video-container'>
          {recordingState.showCountdown && !recordingState.isRecording && !isUploadMode && (
            <CountdownTimer 
              duration={countdownDuration} 
              onComplete={handleCountdownComplete} 
            />
          )}
          {renderVideo()}
          {recordingState.isProcessing && <LoadingSpinner />}
        </div>
        
        {showTrimmer && recordingState.autotunedURL && (
          <div className="trim-controls-container">
            <div className="trim-slider">
              <input
                type="range"
                min={0}
                max={duration}
                step={STEP}
                value={startTime}
                onChange={(e) => handleTimeUpdate(e.target.value, true)}
              />
              <input
                type="range"
                min={0}
                max={duration}
                step={STEP}
                value={endTime}
                onChange={(e) => handleTimeUpdate(e.target.value, false)}
              />
            </div>
            <div className="trim-times">
              <span>Start: {formatTime(startTime)}</span>
              <span>Duration: {formatTime(endTime - startTime)}</span>
              <span>End: {formatTime(endTime)}</span>
            </div>
            <button onClick={handleTrim} className="control-button">Apply Trim</button>
          </div>
        )}
        
        {recordingState.isRecording && (
          <div className="recording-duration" style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            background: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '5px',
            borderRadius: '4px',
            zIndex: 1000
          }}>
            Recording: {recordingState.recordingDuration}s / {minDuration}s minimum
          </div>
        )}
      </div>
      {renderControls()}
      {showTrimmer && recordingState.autotunedURL && (
        <VideoTrimmer
          videoUrl={recordingState.autotunedURL}
          onTrimComplete={handleTrimComplete}
        />
      )}
    </>
  );
};

export default VideoRecorder;
