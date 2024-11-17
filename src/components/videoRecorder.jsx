/* eslint-disable react/prop-types */
import { useRef, useState, useEffect, useCallback } from 'react';
import { handleRecord, handleUploadedVideoAutotune } from '../js/handleRecordVideo';
import ControlButtons from './ControlButtons';
import LoadingSpinner from './LoadingSpinner';
import './styles.css';
import { isDrumTrack } from '../js/drumUtils';
import CountdownTimer from './CountdownTimer';
import VideoTrimmer from './VideoTrimmer';

const VideoRecorder = ({ style, instrument, onVideoReady, minDuration, currentVideo }) => {
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const recordingTimer = useRef(null);

  // Initialize videoState with currentVideo if it exists
  const [videoState, setVideoState] = useState({
    recordedURL: null,
    autotunedURL: currentVideo || null,
  });
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAutotuneEnabled, setIsAutotuneEnabled] = useState(true); // Add state for autotune
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isDrum] = useState(() => isDrumTrack(instrument));
  const [showCountdown, setShowCountdown] = useState(false);
  const [showTrimmer, setShowTrimmer] = useState(false);
  const countdownDuration = 3; // Can be made configurable later
  const duration = 60; // Example duration, replace with actual value
  const STEP = 1; // Example step value, replace with actual value
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(duration);
  const [isUploadMode, setIsUploadMode] = useState(false);
  
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
    // Cleanup function for media streams and audio
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
      // Cleanup object URLs
      if (videoState.recordedURL) URL.revokeObjectURL(videoState.recordedURL);
      if (videoState.autotunedURL) URL.revokeObjectURL(videoState.autotunedURL);
    };
  }, [videoState]);

  useEffect(() => {
    if (isRecording) {
      startRecording().catch(console.error);
    }
  }, [isRecording]);

  useEffect(() => {
    if (videoState.autotunedURL) {
      console.log('Autotuned video URL updated:', videoState.autotunedURL);
      // Only call onVideoReady once per autotuned URL
      const currentURL = videoState.autotunedURL;
      onVideoReady?.(currentURL, instrument);
    }
  }, [videoState.autotunedURL, onVideoReady, instrument]);

  // Update videoState when currentVideo changes
  useEffect(() => {
    if (currentVideo && currentVideo !== videoState.autotunedURL) {
      setVideoState(prev => ({
        ...prev,
        autotunedURL: currentVideo
      }));
    }
  }, [currentVideo]);

  useEffect(() => {
    // When recording starts, initialize the recording timer
    if (isRecording) {
      setRecordingDuration(0);
      recordingTimer.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
    } else {
      // Clear timer when recording stops
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
    }

    return () => {
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      setShowCountdown(true);
    } catch (error) {
      console.error('Recording failed:', error);
      alert('Failed to start recording. Please check your camera permissions.');
    }
  };

  const handleCountdownComplete = async () => {
    setShowCountdown(false);
    setIsProcessing(true);
    setRecordingDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      mediaStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.play();
      }

      // Convert minDuration from seconds to milliseconds and add a small buffer
      const recordingDurationMs = (minDuration + 0.5) * 1000;
      console.log(`Starting recording for ${recordingDurationMs}ms`);

      // Start the recording timer
      recordingTimer.current = setInterval(() => {
        setRecordingDuration(prev => {
          // Stop recording if we've reached the minimum duration
          if (prev >= minDuration) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);

      await handleRecord(
        (recordedURL) => {
          console.log('Recorded video URL set');
          setVideoState((prev) => ({ ...prev, recordedURL }));
        },
        (autotunedURL) => {
          console.log('Autotuned video URL set');
          const finalURL = isDrum ? videoState.recordedURL : autotunedURL;
          setVideoState((prev) => {
            if (prev.autotunedURL !== finalURL) {
              return { ...prev, autotunedURL: finalURL };
            }
            return prev;
          });
        },
        !isDrum && isAutotuneEnabled,
        recordingDurationMs
      );
    } catch (error) {
      console.error('Recording failed:', error);
      setIsProcessing(false);
    }
  };

  const stopRecording = useCallback(() => {
    console.log('Stopping recording at duration:', recordingDuration);
    
    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }
  
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  
    setIsRecording(false);
  }, [recordingDuration]);

  const handleReRecord = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setVideoState({ recordedURL: null, autotunedURL: null });
    setIsRecording(false);
  }, []);

  const handleTrimComplete = (trimmedVideoUrl) => {
    setVideoState(prev => ({ ...prev, autotunedURL: trimmedVideoUrl }));
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
  
    setIsProcessing(true);
    try {
      const uploadedVideoUrl = URL.createObjectURL(file);
      
      // If it's a drum track or autotune is disabled, use the uploaded video directly
      if (isDrum || !isAutotuneEnabled) {
        setVideoState({
          recordedURL: uploadedVideoUrl,
          autotunedURL: uploadedVideoUrl
        });
        // Explicitly pass the instrument parameter to onVideoReady
        onVideoReady?.(uploadedVideoUrl, instrument);
      } else {
        // Handle autotune processing for uploaded video
        setVideoState(prev => ({ ...prev, recordedURL: uploadedVideoUrl }));
        
        // Pass instrument to handleUploadedVideoAutotune
        await handleUploadedVideoAutotune(
          file,
          (autotunedURL) => {
            setVideoState(prev => ({ ...prev, autotunedURL }));
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
      setIsProcessing(false);
    }
  };

  const renderVideo = () => {
    if (isRecording) {
      return (
        <video
          ref={videoRef}
          className='video-element'
          muted
          autoPlay
          playsInline
        ></video>
      );
    }

    if (videoState.autotunedURL) {
      return (
        <video
          key={videoState.autotunedURL}
          ref={showTrimmer ? videoRef : null}
          src={videoState.autotunedURL}
          className='video-element'
          controls={!showTrimmer}
          autoPlay
          playsInline
          onError={(e) => console.error('Video error:', e)}
        ></video>
      );
    }

    if (videoState.recordedURL) {
      return (
        <video
          src={videoState.recordedURL}
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
            isRecording={isRecording}
            hasRecordedVideo={!!videoState.recordedURL}
            onStartRecording={() => setIsRecording(true)}
            onStopRecording={stopRecording}
            onReRecord={handleReRecord}
            instrument={instrumentName} // Pass the string name instead of object
          />
        )}
  
        {videoState.autotunedURL && !isRecording && (
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
          {showCountdown && !isRecording && !isUploadMode && (
            <CountdownTimer 
              duration={countdownDuration} 
              onComplete={handleCountdownComplete} 
            />
          )}
          {renderVideo()}
          {isProcessing && <LoadingSpinner />}
        </div>
        
        {showTrimmer && videoState.autotunedURL && (
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
        
        {isRecording && (
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
            Recording: {recordingDuration}s / {minDuration}s minimum
          </div>
        )}
      </div>
      {renderControls()}
      {showTrimmer && videoState.autotunedURL && (
        <VideoTrimmer
          videoUrl={videoState.autotunedURL}
          onTrimComplete={handleTrimComplete}
        />
      )}
    </>
  );
};

export default VideoRecorder;
