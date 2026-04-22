/* eslint-disable react/prop-types */
import { useState, useEffect, useRef } from 'react';

const formatTime = (timeInSeconds) => {
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.floor(timeInSeconds % 60);
  const milliseconds = Math.floor((timeInSeconds % 1) * 1000);
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
};

const VideoTrimmer = ({ videoUrl, onTrimComplete }) => {
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [isTrimming, setIsTrimming] = useState(false);
  const videoRef = useRef(null);
  const STEP = 0.125; // 1/8th of a second

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.addEventListener('loadedmetadata', () => {
        const videoDuration = videoRef.current.duration;
        setDuration(videoDuration);
        setEndTime(videoDuration);
      });
    }
  }, [videoUrl]);

  const handleTimeUpdate = (value, isStart) => {
    const newTime = Number(value);
    if (isStart) {
      setStartTime(Math.min(newTime, endTime - STEP));
      if (videoRef.current) {
        videoRef.current.currentTime = newTime;
      }
    } else {
      setEndTime(Math.max(newTime, startTime + STEP));
      if (videoRef.current) {
        videoRef.current.currentTime = newTime;
      }
    }
  };

  const handleTrim = async () => {
    if (!videoRef.current || isTrimming) return;
    setIsTrimming(true);
    try {
      const video = videoRef.current;
      const stream = video.captureStream();
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
        ? 'video/webm;codecs=vp9,opus'
        : 'video/webm';
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      const chunks = [];

      await new Promise((resolve, reject) => {
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'video/webm' });
          onTrimComplete(URL.createObjectURL(blob));
          resolve();
        };
        mediaRecorder.onerror = (e) => reject(e.error);

        video.currentTime = startTime;
        video.play().then(() => {
          mediaRecorder.start();
          setTimeout(() => {
            mediaRecorder.stop();
            video.pause();
          }, (endTime - startTime) * 1000);
        }).catch(reject);
      });
    } catch (error) {
      console.error('Error trimming video:', error);
    } finally {
      setIsTrimming(false);
    }
  };

  return (
    <div className="video-trimmer">
      <video 
        ref={videoRef} 
        src={videoUrl} 
        controls 
        onTimeUpdate={(e) => {
          const time = e.target.currentTime;
          if (time < startTime || time > endTime) {
            e.target.currentTime = startTime;
          }
        }}
      />
      <div className="trim-controls">
        <div className="trim-slider">
          <input
            type="range"
            min={0}
            max={duration}
            step={STEP}
            value={startTime}
            onChange={(e) => handleTimeUpdate(e.target.value, true)}
            style={{width: '100%'}}
          />
          <input
            type="range"
            min={0}
            max={duration}
            step={STEP}
            value={endTime}
            onChange={(e) => handleTimeUpdate(e.target.value, false)}
            style={{width: '100%'}}
          />
        </div>
        <button onClick={handleTrim} disabled={isTrimming}>
          {isTrimming ? 'Trimming…' : 'Apply Trim'}
        </button>
      </div>
      <div className="trim-times">
        <span>Start: {formatTime(startTime)}</span>
        <span>Duration: {formatTime(endTime - startTime)}</span>
        <span>End: {formatTime(endTime)}</span>
      </div>
    </div>
  );
};

export default VideoTrimmer;
