import PropTypes from 'prop-types';

const VideoPlayback = ({ videoUrl }) => {
  return (
    <div style={{ width: '100%', position: 'relative' }}>
      <video src={videoUrl} controls style={{ width: '100%' }}></video>
    </div>
  );
};

VideoPlayback.propTypes = {
  videoUrl: PropTypes.string.isRequired,
};

export default VideoPlayback;