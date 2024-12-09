/* eslint-disable react/prop-types */
import { useRef, useState, useEffect, useCallback } from 'react';
import { handleUploadedVideoAutotune } from '../js/handleRecordVideo';
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
    lastVideoSource: null, // 'recorded' or 'uploaded'
    hasVideo: !!currentVideo, // Add this to track if we have a video
  });

  const cleanupMediaStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // const handleDataAvailable = (event) => {
  //   if (event.data && event.data.size > 0) {
  //     const recordedBlob = new Blob([event.data], {
  //       type: 'video/webm;codecs=vp8,opus', // Explicit codec specification
  //     });
  //     console.log('Recording format:', recordedBlob.type); // Debug log
  //     setRecordingState((prev) => ({
  //       ...prev,
  //       recordedBlob,
  //       recordedURL: URL.createObjectURL(recordedBlob),
  //     }));
  //   }
  // };

  useEffect(() => {
    return () => {
      cleanupMediaStream();
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
      // Remove or comment out the following lines
      // if (recordingState.recordedURL) URL.revokeObjectURL(recordingState.recordedURL);
      // if (recordingState.autotunedURL) URL.revokeObjectURL(recordingState.autotunedURL);
    };
  }, [cleanupMediaStream]);

  return {
    videoRef,
    mediaStreamRef,
    recordingTimer,
    recordingState,
    setRecordingState,
    cleanupMediaStream,
  };
};

const VideoRecorder = ({
  onRecordingComplete,
  style,
  instrument,
  onVideoReady,
  minDuration,
  currentVideo,
}) => {
  const {
    videoRef,
    // eslint-disable-next-line no-unused-vars
    mediaStreamRef,
    recordingTimer,
    recordingState,
    setRecordingState,
    cleanupMediaStream,
  } = useRecordingState(currentVideo);

  const [isAutotuneEnabled, setIsAutotuneEnabled] = useState(false);
  const [isDrum] = useState(() => isDrumTrack(instrument));
  const [isUploadMode, setIsUploadMode] = useState(false);
  const [showTrimmer, setShowTrimmer] = useState(false);
  const countdownDuration = 3; // Can be made configurable later
  const duration = 60; // Example duration, replace with actual value
  const STEP = 1; // Example step value, replace with actual value
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(duration);

  console.log('minDuration:', minDuration); // Debug log
  console.log('typeof minDuration:', typeof minDuration); // Debug log

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

  const handleTrim = async () => {
    if (!videoRef.current) return;

    const trimmedVideo = await trimVideo(videoRef.current, startTime, endTime);
    if (trimmedVideo) {
      // Update state with trimmed video
      setRecordingState((prev) => ({
        ...prev,
        recordedURL: URL.createObjectURL(trimmedVideo),
        autotunedURL: URL.createObjectURL(trimmedVideo),
        recordingDuration: endTime - startTime,
      }));

      // Send trimmed video to backend
      onRecordingComplete(trimmedVideo, instrument);
      onVideoReady?.(URL.createObjectURL(trimmedVideo), instrument);
      setShowTrimmer(false);
    }
  };

  // Helper function to trim video (you'll need to implement this)
  const trimVideo = async (videoElement, start, end) => {
    // Implementation depends on your video processing library
    // This is a placeholder - you'll need to implement actual video trimming
    try {
      // Example using MediaRecorder to record the video element playing
      const stream = videoElement.captureStream();
      const mediaRecorder = new MediaRecorder(stream);
      const chunks = [];

      return new Promise((resolve) => {
        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
        mediaRecorder.onstop = () =>
          resolve(new Blob(chunks, { type: 'video/webm' }));

        videoElement.currentTime = start;
        videoElement.play();
        mediaRecorder.start();

        setTimeout(() => {
          mediaRecorder.stop();
          videoElement.pause();
        }, (end - start) * 1000);
      });
    } catch (error) {
      console.error('Error trimming video:', error);
      return null;
    }
  };

  useEffect(() => {
    if (recordingState.autotunedURL && !recordingState.isProcessing) {
      onVideoReady?.(recordingState.autotunedURL, instrument);
    }
  }, [
    recordingState.autotunedURL,
    recordingState.isProcessing,
    instrument,
    onVideoReady,
  ]);

  // Add a ref to keep track of the previous currentVideo
  // const prevCurrentVideo = useRef(currentVideo);

  // Modify the useEffect that calls onVideoReady
  useEffect(() => {
    if (
      recordingState.autotunedURL &&
      !recordingState.isProcessing &&
      recordingState.autotunedURL !== prevAutotunedURL.current
    ) {
      onVideoReady?.(recordingState.autotunedURL, instrument);
      prevAutotunedURL.current = recordingState.autotunedURL;
    }
  }, [
    recordingState.autotunedURL,
    recordingState.isProcessing,
    instrument,
    onVideoReady,
  ]);

  // Add a ref to keep track of the previous autotunedURL
  const prevAutotunedURL = useRef(recordingState.autotunedURL);

  useEffect(() => {
    // When recording starts, initialize the recording timer
    if (recordingState.isRecording) {
      setRecordingState((prev) => ({ ...prev, recordingDuration: 0 }));
      const timer = setInterval(() => {
        setRecordingState((prev) => ({
          ...prev,
          recordingDuration: prev.recordingDuration + 1,
        }));
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
      setRecordingState((prev) => ({
        ...prev,
        showCountdown: true,
        isCountingDown: true,
      }));
      // const stream = await navigator.mediaDevices.getUserMedia({
      //   audio: true,
      //   video: {
      //     width: { ideal: 640 },
      //     height: { ideal: 480 },
      //     frameRate: { ideal: 30 }
      //   }
      // });
  
      // const options = {
      //   mimeType: 'video/webm;codecs=vp8,opus',
      //   videoBitsPerSecond: 2500000, // 2.5 Mbps
      //   audioBitsPerSecond: 128000   // 128 kbps
      // };
      
      // const mediaRecorder = new MediaRecorder(stream, options);
    } catch (error) {
      console.error('Recording failed:', error);
      alert('Failed to start recording. Please check your camera permissions.');
      setRecordingState((prev) => ({
        ...prev,
        isRecording: false,
        isCountingDown: false,
      }));
    }
  };

  const handleRecord = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      return new Promise((resolve) => {
        const mediaRecorder = new MediaRecorder(stream);
        const chunks = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'video/webm' });
          console.log('Created video blob:', blob.size); // Debug log
          stream.getTracks().forEach((track) => track.stop());
          onRecordingComplete(blob, instrument); // Pass both blob and instrument
          resolve(blob);
        };

        mediaRecorder.start();
        setTimeout(() => mediaRecorder.stop(), (minDuration + 1) * 1000);
      });
    } catch (error) {
      console.error('Recording failed:', error);
      throw error;
    }
  }, [minDuration, instrument, onRecordingComplete]);

  const handleCountdownComplete = useCallback(async () => {
    setRecordingState((prev) => ({
      ...prev,
      showCountdown: false,
      isCountingDown: false,
      isRecording: true,
    }));

    try {
      const blob = await handleRecord();
      setRecordingState((prev) => ({
        ...prev,
        isRecording: false,
        recordedURL: URL.createObjectURL(blob),
      }));
    } catch (error) {
      console.error('Recording failed:', error);
      setRecordingState((prev) => ({
        ...prev,
        isRecording: false,
      }));
    }
  }, [handleRecord]);

  const handleRecordingFinished = useCallback(
    async (blob) => {
      if (!(blob instanceof Blob)) {
        console.error('Invalid recording blob');
        return;
      }

      console.log('Recording finished, blob size:', blob.size); // Debug log

      setRecordingState((prev) => ({
        ...prev,
        lastVideoSource: 'recorded',
      }));

      // Ensure we're passing both the blob and instrument
      onRecordingComplete(blob, instrument);

      // Create URL for preview
      const url = URL.createObjectURL(blob);
      onVideoReady?.(url, instrument);
    },
    [instrument, onRecordingComplete, onVideoReady]
  );

  const stopRecording = useCallback(() => {
    console.log(
      'Stopping recording at duration:',
      recordingState.recordingDuration
    );

    if (recordingTimer.current) {
      clearInterval(recordingTimer.current);
      recordingTimer.current = null;
    }

    cleanupMediaStream();
    setRecordingState((prev) => ({ ...prev, isRecording: false }));

    // The mediaRecorder.onstop event will trigger handleRecordingFinished
  }, [recordingState.recordingDuration, cleanupMediaStream]);

  const handleReRecord = useCallback(() => {
    // Revoke existing blob URLs
    if (recordingState.recordedURL) {
      URL.revokeObjectURL(recordingState.recordedURL);
    }
    if (recordingState.autotunedURL) {
      URL.revokeObjectURL(recordingState.autotunedURL);
    }

    cleanupMediaStream();
    setRecordingState((prev) => ({
      ...prev,
      recordedURL: null,
      autotunedURL: null,
      isRecording: false,
      isProcessing: false,
      recordingDuration: 0,
      hasVideo: false,
    }));
  }, [
    cleanupMediaStream,
    recordingState.recordedURL,
    recordingState.autotunedURL,
  ]);

  const handleTrimComplete = (trimmedVideoUrl) => {
    setRecordingState((prev) => ({ ...prev, autotunedURL: trimmedVideoUrl }));
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

    // Revoke existing blob URLs before setting new ones
    if (recordingState.recordedURL) {
      URL.revokeObjectURL(recordingState.recordedURL);
    }
    if (recordingState.autotunedURL) {
      URL.revokeObjectURL(recordingState.autotunedURL);
    }

    setRecordingState((prev) => ({ ...prev, isProcessing: true }));
    try {
      const uploadedVideoUrl = URL.createObjectURL(file);

      // Create a temporary video element to get duration
      const video = document.createElement('video');
      video.src = uploadedVideoUrl;

      await new Promise((resolve) => {
        video.addEventListener('loadedmetadata', () => {
          const duration = Math.round(video.duration);
          setRecordingState((prev) => ({
            ...prev,
            recordingDuration: duration,
            recordedURL: uploadedVideoUrl,
            lastVideoSource: 'uploaded',
            hasVideo: true, // Mark that we have a video
          }));
          resolve();
        });
      });

      // If it's a drum track or autotune is disabled, use the uploaded video directly
      if (isDrum || !isAutotuneEnabled) {
        setRecordingState({
          recordedURL: uploadedVideoUrl,
          autotunedURL: uploadedVideoUrl,
        });
        // Explicitly pass the instrument parameter to onVideoReady
        onVideoReady?.(uploadedVideoUrl, instrument);
        handleRecordingFinished(file); // Use the original file
      } else {
        // Handle autotune processing for uploaded video
        setRecordingState((prev) => ({
          ...prev,
          recordedURL: uploadedVideoUrl,
        }));

        // Pass instrument to handleUploadedVideoAutotune
        await handleUploadedVideoAutotune(
          file,
          (autotunedURL) => {
            setRecordingState((prev) => ({ ...prev, autotunedURL }));
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
      setRecordingState((prev) => ({ ...prev, isProcessing: false }));
    }
  };

  // const playSampleSound = useCallback(async () => {
  //   if (!audioEnabled) {
  //     console.log('Audio context not initialized');
  //     return;
  //   }

  //   try {
  //     // Your existing sample sound code
  //     // ...
  //   } catch (error) {
  //     console.error('Error playing sample:', error);
  //   }
  // }, [audioEnabled]); // Add audioEnabled to dependencies

  useEffect(() => {
    // Remove any Tone.js initialization from here
    // Only handle cleanup if needed
    return () => {
      // Your cleanup code
    };
  }, []);

  const renderVideo = () => {
    if (recordingState.isRecording) {
      return (
        <div className='video-container'>
          <video
            ref={videoRef}
            className='video-element'
            muted
            autoPlay
            playsInline
          ></video>
          <div className='recording-duration'>
            Recording: {recordingState.recordingDuration}s / {minDuration}s
            minimum
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

    const hasValidVideo =
      recordingState.hasVideo &&
      (recordingState.recordingDuration >= minDuration ||
        recordingState.lastVideoSource === 'uploaded');

    return (
      <div className='controls-section'>
        <div className='mode-selector'>
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
          <div className='upload-controls'>
            <input
              type='file'
              accept='video/*'
              onChange={handleFileUpload}
              id={uploadId}
              style={{ display: 'none' }}
            />
            <label htmlFor={uploadId} className='upload-button'>
              {recordingState.hasVideo ? 'Replace Video' : 'Choose Video File'}{' '}
              for {instrumentName}
            </label>
            {recordingState.recordingDuration > 0 && (
              <div className='video-duration'>
                Duration: {recordingState.recordingDuration}s
                {minDuration > 0 && ` / ${minDuration}s minimum`}
              </div>
            )}
          </div>
        ) : (
          <ControlButtons
            isRecording={recordingState.isRecording}
            hasRecordedVideo={hasValidVideo}
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
            type='checkbox'
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
          {recordingState.showCountdown &&
            !recordingState.isRecording &&
            !isUploadMode && (
              <CountdownTimer
                duration={countdownDuration}
                onComplete={handleCountdownComplete}
              />
            )}
          {renderVideo()}
          {recordingState.isProcessing && <LoadingSpinner />}
        </div>

        {showTrimmer && recordingState.autotunedURL && (
          <div className='trim-controls-container'>
            <div className='trim-slider'>
              <input
                type='range'
                min={0}
                max={duration}
                step={STEP}
                value={startTime}
                onChange={(e) => handleTimeUpdate(e.target.value, true)}
              />
              <input
                type='range'
                min={0}
                max={duration}
                step={STEP}
                value={endTime}
                onChange={(e) => handleTimeUpdate(e.target.value, false)}
              />
            </div>
            <div className='trim-times'>
              <span>Start: {formatTime(startTime)}</span>
              <span>Duration: {formatTime(endTime - startTime)}</span>
              <span>End: {formatTime(endTime)}</span>
            </div>
            <button onClick={handleTrim} className='control-button'>
              Apply Trim
            </button>
          </div>
        )}

        {recordingState.isRecording && (
          <div
            className='recording-duration'
            style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              background: 'rgba(0,0,0,0.7)',
              color: 'white',
              padding: '5px',
              borderRadius: '4px',
              zIndex: 1000,
            }}
          >
            Recording: {recordingState.recordingDuration}s / {minDuration}s
            minimum
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
