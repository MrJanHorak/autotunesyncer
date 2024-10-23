import React from 'react';

const VideoPlayback = ({ videoUrl }) => {
  return (
    <div style={{ width: '100%', position: 'relative' }}>
      <video src={videoUrl} controls style={{ width: '100%' }}></video>
    </div>
  );
};

export default VideoPlayback;