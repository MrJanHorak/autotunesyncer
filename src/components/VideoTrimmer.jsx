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
    try {
      const response = await fetch(videoUrl);
      const blob = await response.blob();
      
      // Create a new MediaSource
      const mediaSource = new MediaSource();
      const sourceUrl = URL.createObjectURL(mediaSource);
      
      mediaSource.addEventListener('sourceopen', async () => {
        const sourceBuffer = mediaSource.addSourceBuffer('video/mp4');
        
        // Read the video data and trim it
        const reader = new FileReader();
        reader.onload = async (e) => {
          const videoData = new Uint8Array(e.target.result);
          // Here you would implement the actual trimming logic
          // This is a simplified version - in practice, you'd need a more robust solution
          sourceBuffer.appendBuffer(videoData);
          
          sourceBuffer.addEventListener('updateend', () => {
            mediaSource.endOfStream();
            onTrimComplete(sourceUrl);
          });
        };
        reader.readAsArrayBuffer(blob);
      });
    } catch (error) {
      console.error('Error trimming video:', error);
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
        <button onClick={handleTrim}>Apply Trim</button>
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
