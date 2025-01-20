/* eslint-disable react/prop-types */
import './styles.css';

const VideoDisplay = ({ videoURL }) => (
  <div className="video-container">
    <video src={videoURL} controls className="video-element"></video>
  </div>
);

export default VideoDisplay;
