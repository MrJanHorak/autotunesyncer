/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable react/prop-types */
import { useRef, useState, useEffect, useCallback } from 'react';
import { handleUploadedVideoAutotune } from '../../js/handleRecordVideo';
import LoadingSpinner from '../LoadingSpinner/LoadingSpinner';
import SampleSoundButton from '../SampleSoundButton/SampleSoundButton';
import '../styles.css';
import { isDrumTrack } from '../../js/drumUtils';
import CountdownTimer from '../CountdownTimer/CountdownTimer';
import ToggleSwitch from '../ToggleSwitch/ToggleSwitch';

import './VideoRecorder.css';

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

  // Sync when the parent supplies a new currentVideo (e.g. clip restored from server).
  useEffect(() => {
    if (!currentVideo) return;
    setRecordingState((prev) => {
      if (prev.autotunedURL === currentVideo) return prev; // already set
      return { ...prev, autotunedURL: currentVideo, hasVideo: true };
    });
  }, [currentVideo]);

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
  midiData,
}) => {
  const {
    videoRef,
    mediaStreamRef,
    recordingTimer,
    recordingState,
    setRecordingState,
    cleanupMediaStream,
  } = useRecordingState(currentVideo);

  const [isAutotuneEnabled, setIsAutotuneEnabled] = useState(false);
  const [isDrum] = useState(() => isDrumTrack(instrument));
  const isPercussion = isDrum ||
    ['percussion', 'percussive'].includes(instrument.family?.toLowerCase());
  const [isUploadMode, setIsUploadMode] = useState(false);
  const [showTrimmer, setShowTrimmer] = useState(false);
  const [isTrimming, setIsTrimming] = useState(false);
  const [duration, setDuration] = useState(0);
  const STEP = 0.125;
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const countdownDuration = 3;

  const handleTimeUpdate = (value, isStart) => {
    const newTime = Number(value);
    if (isStart) {
      setStartTime(Math.min(newTime, endTime - STEP));
    } else {
      setEndTime(Math.max(newTime, startTime + STEP));
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const ms = Math.floor((time % 1) * 10);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}.${ms}`;
  };

  const trimVideo = async (videoElement, start, end) => {
    try {
      // Seek to start and wait for the seek to complete
      videoElement.pause();
      videoElement.currentTime = start;
      await new Promise((resolve) => {
        videoElement.addEventListener('seeked', resolve, { once: true });
      });

      const stream = videoElement.captureStream();
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
        ? 'video/webm;codecs=vp9,opus'
        : 'video/webm';
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      const chunks = [];

      return new Promise((resolve, reject) => {
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };
        mediaRecorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }));
        mediaRecorder.onerror = (e) => reject(e.error);

        videoElement.play().then(() => {
          mediaRecorder.start(100);
          setTimeout(() => {
            mediaRecorder.stop();
            videoElement.pause();
          }, (end - start) * 1000);
        }).catch(reject);
      });
    } catch (error) {
      console.error('Error trimming video:', error);
      return null;
    }
  };

  const handleTrim = async () => {
    if (!videoRef.current || isTrimming) return;
    setIsTrimming(true);
    try {
      const trimmedBlob = await trimVideo(videoRef.current, startTime, endTime);
      if (trimmedBlob) {
        const url = URL.createObjectURL(trimmedBlob);
        setRecordingState((prev) => ({
          ...prev,
          recordedURL: url,
          autotunedURL: url,
          recordingDuration: endTime - startTime,
          hasVideo: true,
        }));
        onRecordingComplete(trimmedBlob, instrument);
        onVideoReady?.(url, instrument);
        setShowTrimmer(false);
      }
    } finally {
      setIsTrimming(false);
    }
  };

  const handleAutotuneToggle = useCallback(
    (e) => {
      console.log('Toggle clicked, previous state:', isAutotuneEnabled);
      setIsAutotuneEnabled(e.target.checked);
      console.log('New state:', e.target.checked);
    },
    [isAutotuneEnabled],
  );

  // Initialize trim start/end from actual video duration when trimmer opens
  useEffect(() => {
    if (!showTrimmer || !videoRef.current) return;
    const video = videoRef.current;
    const init = () => {
      const d = video.duration;
      if (isFinite(d) && d > 0) {
        setDuration(d);
        setStartTime(0);
        setEndTime(d);
      }
    };
    if (video.readyState >= 1) {
      init();
    } else {
      video.addEventListener('loadedmetadata', init, { once: true });
    }
  }, [showTrimmer]);

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
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, aspectRatio: { ideal: 16 / 9 } },
        audio: true,
      });

      // Connect stream to the video element so the user can see themselves
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      return new Promise((resolve) => {
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
          ? 'video/webm;codecs=vp9,opus'
          : 'video/webm';
        const mediaRecorder = new MediaRecorder(stream, { mimeType });
        const chunks = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunks.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          // Detach live preview before switching to recorded playback
          if (videoRef.current) {
            videoRef.current.srcObject = null;
          }
          stream.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;

          const blob = new Blob(chunks, { type: 'video/webm' });
          onRecordingComplete(blob, instrument);
          resolve(blob);
        };

        mediaRecorder.start(100);
        setTimeout(() => mediaRecorder.stop(), (minDuration + 1) * 1000);
      });
    } catch (error) {
      console.error('Recording failed:', error);
      throw error;
    }
  }, [minDuration, instrument, onRecordingComplete, videoRef, mediaStreamRef]);

  const handleCountdownComplete = useCallback(async () => {
    setRecordingState((prev) => ({
      ...prev,
      showCountdown: false,
      isCountingDown: false,
      isRecording: true,
    }));

    try {
      const blob = await handleRecord();
      const url = URL.createObjectURL(blob);
      setRecordingState((prev) => ({
        ...prev,
        isRecording: false,
        recordedURL: url,
      }));
      // Notify parent so instrumentVideos updates (sidebar checkmarks, grid overlays)
      onVideoReady?.(url, instrument);
    } catch (error) {
      console.error('Recording failed:', error);
      setRecordingState((prev) => ({
        ...prev,
        isRecording: false,
      }));
    }
  }, [handleRecord, instrument, onVideoReady]);

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
    [instrument, onRecordingComplete, onVideoReady],
  );

  const stopRecording = useCallback(() => {
    console.log(
      'Stopping recording at duration:',
      recordingState.recordingDuration,
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
        setRecordingState((prev) => ({
          ...prev,
          recordedURL: uploadedVideoUrl,
          autotunedURL: uploadedVideoUrl,
          hasVideo: true,
          isProcessing: false,
        }));
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
          instrument, // Add instrument as parameter if needed by handleUploadedVideoAutotune
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
        <video
          ref={videoRef}
          className='video-element'
          muted
          autoPlay
          playsInline
        />
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
        />
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
        />
      );
    }

    return (
      <div className='video-element'>
        <p>No video available</p>
      </div>
    );
  };

  const renderControls = () => {
    const instrumentName = instrument.isDrum
      ? `drum_${instrument.group}`
      : instrument.name;

    const uploadId = `video-upload-${instrumentName}`;

    const hasValidVideo =
      recordingState.hasVideo &&
      (recordingState.recordingDuration >= minDuration ||
        recordingState.lastVideoSource === 'uploaded');

    return (
      <div className='controls-section'>
        <div className='action-buttons-wrapper'>
          {/* Button Slot 1: Primary Action */}
          {isUploadMode ? (
            <>
              <input
                type='file'
                accept='video/*'
                onChange={handleFileUpload}
                id={uploadId}
                style={{ display: 'none' }}
              />
              <label htmlFor={uploadId} className='control-button'>
                <svg
                  xmlns='http://www.w3.org/2000/svg'
                  fill='none'
                  viewBox='0 0 24 24'
                  stroke='currentColor'
                  style={{ width: '20px', height: '20px' }}
                >
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth='2'
                    d='M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12'
                  />
                </svg>
                {recordingState.hasVideo
                  ? 'Replace Video'
                  : 'Choose Video File'}
              </label>
            </>
          ) : (
            <button
              className='control-button'
              onClick={
                recordingState.isRecording ? stopRecording : startRecording
              }
              disabled={recordingState.isCountingDown}
            >
              {recordingState.isRecording
                ? 'Stop Recording'
                : 'Start Recording'}
            </button>
          )}

          {/* Button Slot 2: Secondary Action */}
          {isUploadMode ? (
            recordingState.recordingDuration > 0 ? (
              <div className='video-duration'>
                Duration: {recordingState.recordingDuration}s
                {minDuration > 0 && ` / ${minDuration}s minimum`}
              </div>
            ) : (
              <div className='button-spacer'></div>
            )
          ) : hasValidVideo ? (
            <button className='control-button' onClick={handleReRecord}>
              Re-record
            </button>
          ) : (
            <SampleSoundButton
              instrument={instrument}
              instrumentName={instrumentName}
              className='control-button'
              isDrumTrack={isDrum}
              midiData={midiData}
            />
          )}

          {/* Toggle Link */}
          <button
            onClick={() => setIsUploadMode(!isUploadMode)}
            className='upload-toggle'
          >
            <svg
              xmlns='http://www.w3.org/2000/svg'
              fill='none'
              viewBox='0 0 24 24'
              stroke='currentColor'
            >
              {isUploadMode ? (
                <>
                  <circle cx='12' cy='12' r='9' strokeWidth='2' />
                  <circle cx='12' cy='12' r='3' fill='currentColor' />
                </>
              ) : (
                <path
                  strokeLinecap='round'
                  strokeLinejoin='round'
                  strokeWidth='2'
                  d='M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12'
                />
              )}
            </svg>
            {isUploadMode ? 'Switch to recording' : 'or upload a video instead'}
          </button>
        </div>

        {recordingState.autotunedURL && !recordingState.isRecording && (
          <button onClick={() => setShowTrimmer(!showTrimmer)}>
            {showTrimmer ? 'Hide Trimmer' : 'Trim Video'}
          </button>
        )}
        {!isPercussion && (
          <div className='autotune-checkbox'>
            <ToggleSwitch
              checked={isAutotuneEnabled}
              onChange={handleAutotuneToggle}
              onText='ON'
              offText='OFF'
            />
            <span className='autotune-checkbox-text'>Enable Autotune</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      <div className='recorder-wrapper' style={style}>
        <div className='video-container'>
          {!recordingState.isRecording &&
            !recordingState.hasVideo &&
            minDuration > 0 && (
              <div className='duration-badge'>
                <svg
                  xmlns='http://www.w3.org/2000/svg'
                  fill='none'
                  viewBox='0 0 24 24'
                  stroke='currentColor'
                >
                  <circle cx='12' cy='12' r='10' strokeWidth='2' />
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth='2'
                    d='M12 6v6l4 2'
                  />
                </svg>
                {minDuration}s minimum
              </div>
            )}
          {recordingState.showCountdown &&
            !recordingState.isRecording &&
            !isUploadMode && (
              <CountdownTimer
                duration={countdownDuration}
                onComplete={handleCountdownComplete}
              />
            )}
          {renderVideo()}
          {recordingState.isRecording && (
            <div className='duration-badge' style={{ top: 'auto', bottom: '10px' }}>
              {recordingState.recordingDuration}s / {minDuration}s min
            </div>
          )}
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
            <button onClick={handleTrim} className='control-button' disabled={isTrimming}>
              {isTrimming ? 'Trimming…' : 'Apply Trim'}
            </button>
          </div>
        )}
      </div>
      {renderControls()}
    </>
  );
};

export default VideoRecorder;
