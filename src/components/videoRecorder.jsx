/* eslint-disable react/prop-types */
import { useRef, useState, useEffect, useCallback } from 'react';
import { handleRecord } from '../js/handleRecordVideo';
import ControlButtons from './ControlButtons';
import LoadingSpinner from './LoadingSpinner';
import './styles.css';

const VideoRecorder = ({ style, instrument, onVideoReady }) => {
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);

  const [videoState, setVideoState] = useState({
    recordedURL: null,
    autotunedURL: null,
  });
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    // Cleanup function for media streams and audio
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
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

  const startRecording = async () => {
    try {
      setIsProcessing(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      mediaStreamRef.current = stream;

      // Mute audio feedback
      const audioTracks = stream.getAudioTracks();
      audioTracks.forEach((track) => {
        track.enabled = true; // Keep enabled for recording
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true; // Mute the video element
        videoRef.current.play();
      }

      await handleRecord(
        (recordedURL) => {
          console.log('Recorded video URL set');
          setVideoState((prev) => ({ ...prev, recordedURL }));
        },
        (autotunedURL) => {
          console.log('Autotuned video URL set');
          // Only update if the URL is different
          setVideoState((prev) => {
            if (prev.autotunedURL !== autotunedURL) {
              return { ...prev, autotunedURL };
            }
            return prev;
          });
        }
      );
    } catch (error) {
      console.error('Recording failed:', error);
      alert('Failed to start recording. Please check your camera permissions.');
    } finally {
      setIsProcessing(false);
    }
  };

  const stopRecording = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsRecording(false);
  }, []);

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

  // const uploadAutotunedVideo = useCallback(async () => {
  //   if (!videoState.autotunedURL) return;

  //   try {
  //     setIsProcessing(true);
  //     const response = await fetch(videoState.autotunedURL);
  //     const autotunedVideoBlob = await response.blob();
  //     const autotunedVideoFile = new File(
  //       [autotunedVideoBlob],
  //       'autotuned-video.mp4'
  //     );
  //     await uploadVideo(autotunedVideoFile);
  //     alert('Autotuned video uploaded successfully!');
  //   } catch (error) {
  //     console.error('Upload failed:', error);
  //     alert('Failed to upload video. Please try again.');
  //   } finally {
  //     setIsProcessing(false);
  //   }
  // }, [videoState.autotunedURL]);

  return (
    <>
      <div className='recorder-wrapper' style={style}>
        <div className='video-container'>
          {isRecording ? (
            <video
              ref={videoRef}
              className='video-element'
              muted
              autoPlay
              playsInline
            ></video>
          ) : (
            <>
              {videoState.autotunedURL ? (
                <video
                  key={videoState.autotunedURL} // Add key to force reload
                  src={videoState.autotunedURL}
                  className='video-element'
                  controls
                  autoPlay
                  playsInline
                  onError={(e) => console.error('Video error:', e)}
                ></video>
              ) : videoState.recordedURL ? (
                <video
                  src={videoState.recordedURL}
                  className='video-element'
                  controls
                  autoPlay
                  playsInline
                ></video>
              ) : (
                <div className='video-element'>
                  <p>No video available</p>
                </div>
              )}
            </>
          )}

          {isProcessing && <LoadingSpinner />}
        </div>
      </div>
      <div className='controls-section'>
        <ControlButtons
          isRecording={isRecording}
          hasRecordedVideo={!!videoState.recordedURL}
          onStartRecording={() => setIsRecording(true)}
          onStopRecording={stopRecording}
          onReRecord={handleReRecord}
          instrument={instrument}
        />

        {/* {!isRecording && videoState.autotunedURL && (
          <div className="controls-container">
            <button className="control-button" onClick={uploadAutotunedVideo}>
              Upload Autotuned Video
            </button>
          </div>
        )} */}
      </div>
    </>
  );
};

export default VideoRecorder;
